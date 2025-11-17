#!/usr/bin/env python3
"""Batch executor for deterministic multi-floor sweep simulations.

The grid is defined by BATCH_CONFIG and can be overridden via CLI.
Results are written to JSONL + CSV for downstream visualisation/ML.
"""
from __future__ import annotations

import argparse
import csv
import json
import concurrent.futures
from functools import partial
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sim_sweep_det import run_once  # type: ignore


BATCH_CONFIG: Dict[str, Any] = {
    "floors": "1-18",
    "layouts": "BASELINE,T,L",
    "occ": "5-10",
    "resp": "1-10",
}

LAYOUT_MAP: Dict[str, str] = {
    "BASELINE": "layout/baseline.json",
    "T": "layout/layout_A.json",
    "L": "layout/layout_B.json",
}

# Canonical key for a parameter combination
# (layout_label, floors, per_room, responders)
ComboKey = Tuple[str, int, int, int]


def _parse_range(spec: str) -> List[int]:
    """Parse simple range specs like '1-3' or '1,2,5'."""

    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec and "," not in spec:
        lo_str, hi_str = spec.split("-", 1)
        lo = int(lo_str.strip())
        hi = int(hi_str.strip())
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    values: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo = int(lo_str.strip())
            hi = int(hi_str.strip())
            if lo > hi:
                lo, hi = hi, lo
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def _parse_layouts(spec: str) -> List[Tuple[str, str]]:
    labels = [s.strip() for s in spec.split(",") if s.strip()]
    pairs: List[Tuple[str, str]] = []
    for label in labels:
        path = LAYOUT_MAP.get(label.upper())
        if path is None:
            # fallback to treat label as direct path
            pairs.append((label, label))
        else:
            pairs.append((label.upper(), path))
    return pairs


def _record_to_key(record: Dict[str, Any]) -> ComboKey | None:
    """Extract a canonical ComboKey from any record dict (JSONL/summary)."""

    layout_label = record.get("layout_label") or record.get("layout")
    if layout_label is None:
        return None
    # normalise label for robustness
    layout_label = str(layout_label).upper()

    floors_raw = record.get("floors") or record.get("floor") or 1
    per_room_raw = record.get("per_room")
    responders_raw = record.get("responders")
    if per_room_raw is None or responders_raw is None:
        return None
    try:
        floors = int(floors_raw)
        per_room = int(per_room_raw)
        responders = int(responders_raw)
    except (TypeError, ValueError):
        return None
    return (layout_label, floors, per_room, responders)


def _run_single(
    args: Tuple[str, str, int, int, int, int],
) -> Dict[str, Any]:
    """Worker wrapper for a single deterministic simulation."""

    layout_label, layout_path, floor, per_room, responders, max_steps = args
    result = run_once(
        layout_path=layout_path,
        num_responders=responders,
        per_room=per_room,
        max_steps=max_steps,
        floors=floor,
        seed=0,
        init_positions=None,
        frames_path=None,
        delay=0.0,
        log_every=max_steps + 1,  # suppress stdout spam
        cell_m=0.5,
        speed_solo=0.8,
        speed_escort=0.6,
        logger=None,
    )
    result["layout_label"] = layout_label
    result["floors"] = floor
    return result


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch deterministic multi-floor sweeps")
    parser.add_argument("--floors", default=BATCH_CONFIG["floors"], help="Floor range spec (e.g. '1-3')")
    parser.add_argument("--layouts", default=BATCH_CONFIG["layouts"], help="Layout labels (e.g. 'BASELINE,T,L')")
    parser.add_argument("--occ", default=BATCH_CONFIG["occ"], help="Per-room occupants range (e.g. '5-10')")
    parser.add_argument("--resp", default=BATCH_CONFIG["resp"], help="Responder count range (e.g. '1-10')")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max simulation steps per run")
    parser.add_argument("--output", default="output/batch_runs", help="Output root directory")
    args = parser.parse_args(argv)

    floors = _parse_range(args.floors)
    occ_values = _parse_range(args.occ)
    resp_values = _parse_range(args.resp)
    layout_pairs = _parse_layouts(args.layouts)

    if not floors or not occ_values or not resp_values or not layout_pairs:
        raise SystemExit("Batch grid is empty; check --floors/--occ/--resp/--layouts")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_root / "det_batch.jsonl"
    summary_json_path = out_root / "summary.json"
    summary_csv_path = out_root / "summary.csv"

    # --- checkpoint: load existing results (if any) ---------------------------
    existing_results: List[Dict[str, Any]] = []
    existing_keys: set[ComboKey] = set()

    # 1) primary checkpoint: JSONL with full run_once results
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except Exception:
                    continue
                key = _record_to_key(res)
                if key is not None:
                    existing_keys.add(key)
                existing_results.append(res)

    # 2) secondary checkpoint: summary.json (if present)
    if summary_json_path.exists():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        key = _record_to_key(rec)
                        if key is not None:
                            existing_keys.add(key)
        except Exception:
            # summary.json is purely a helper; ignore if malformed
            pass

    # 3) secondary checkpoint: summary.csv (if present)
    if summary_csv_path.exists():
        try:
            with open(summary_csv_path, "r", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    key = _record_to_key(row)
                    if key is not None:
                        existing_keys.add(key)
        except Exception:
            # summary.csv is purely a helper; ignore if malformed
            pass

    # --- build full task grid ------------------------------------------------
    tasks: List[Tuple[str, str, int, int, int, int]] = []
    for layout_label, layout_path in layout_pairs:
        norm_label = layout_label.upper()
        for responders in resp_values:
            for per_room in occ_values:
                for floor in floors:
                    combo_key = (norm_label, floor, per_room, responders)
                    if combo_key in existing_keys:
                        continue
                    tasks.append((layout_label, layout_path, floor, per_room, responders, args.max_steps))

    # --- run new tasks in parallel -------------------------------------------
    new_results: List[Dict[str, Any]] = []
    if tasks:
        max_workers = min(os.cpu_count() or 1, len(tasks))
        with open(jsonl_path, "a", encoding="utf-8") as jf, concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_run_single, tasks):
                jf.write(json.dumps(result, ensure_ascii=False) + "\n")
                jf.flush()
                new_results.append(result)

    all_results = existing_results + new_results

    # --- write JSONL (overwrite with full set) --------------------------------
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for res in all_results:
            jf.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Deterministic batch JSONL written to {jsonl_path}")
    print("Run 'make batchsum' to regenerate summary.json/summary.csv from det_batch.jsonl")


if __name__ == "__main__":
    main()

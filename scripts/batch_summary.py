#!/usr/bin/env python3
"""Regenerate batch summary JSON/CSV from det_batch.jsonl checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return results
    with jsonl_path.open("r", encoding="utf-8") as jf:
        for line in jf:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize deterministic batch results")
    parser.add_argument("--input", default="output/batch_runs/det_batch.jsonl", help="JSONL file with batch runs")
    parser.add_argument("--output", default="output/batch_runs", help="Directory for summary.json/summary.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"

    results = _load_results(input_path)
    summaries: List[Dict[str, Any]] = []
    for res in results:
        layout_label = res.get("layout_label")
        layout_path = res.get("layout") or res.get("layout_path")
        floors = res.get("floors") or res.get("floor") or 1
        per_room = res.get("per_room")
        responders = res.get("responders")
        room_order = "->".join(res.get("room_order", []))
        summaries.append(
            {
                "layout": layout_label,
                "layout_path": layout_path,
                "floors": floors,
                "per_room": per_room,
                "responders": responders,
                "max_steps": res.get("max_steps"),
                "time_steps": res.get("time"),
                "all_evacuated": res.get("all_evacuated"),
                "evacuated": res.get("evacuated"),
                "real_hms": res.get("real_hms"),
                "real_minutes": res.get("real_minutes"),
                "room_order": room_order,
            }
        )

    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "layout",
        "layout_path",
        "floors",
        "per_room",
        "responders",
        "max_steps",
        "time_steps",
        "all_evacuated",
        "evacuated",
        "real_hms",
        "real_minutes",
        "room_order",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Summary regenerated: {summary_json}, {summary_csv}")


if __name__ == "__main__":
    main()

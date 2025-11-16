"""Batch executor for MineEvac graph sweeps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import BatchSettings
from graph_evac import save_json
from graph_evac.io_utils import ensure_dir
from graph_evac.layout import expand_floors, load_layout
from src.main import execute_run


def _parse_int_list(values: List[str] | None) -> List[int] | None:
    if not values:
        return None
    return [int(value) for value in values]


def _sec_to_hms(seconds: float) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _layout_label(layout_path: str) -> str:
    """Map layout path to a short label such as BASELINE/T/L."""

    name = Path(layout_path).name
    mapping = {
        "baseline.json": "BASELINE",
        "layout_A.json": "T",
        "layout_B.json": "L",
    }
    if name in mapping:
        return mapping[name]
    return Path(name).stem.upper()


def _exit_side(node_id: str | None) -> str:
    """Best-effort mapping of exit / start node IDs to L/R."""

    if not node_id:
        return "L"
    s = str(node_id).lower()
    if "right" in s or s.endswith("_r") or s.endswith("e_r"):
        return "R"
    if "left" in s or s.endswith("_l") or s.endswith("e_l"):
        return "L"
    # Fallback: treat anything else as left for consistency
    return "L"


def _summarise_run(config, plan) -> Dict:
    """Build a single CSV-style summary row matching the expected schema."""

    layout_path = config.layout_path
    layout_label = _layout_label(layout_path)
    floors = config.floors

    # Layout metadata (per-room occupants)
    per_room_occ = None
    layout_data = None
    try:
        layout_data = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        occ_block = layout_data.get("occupants") or {}
        per_room_occ = occ_block.get("per_room")
        if per_room_occ is not None:
            try:
                per_room_occ = int(per_room_occ)
            except Exception:
                per_room_occ = None
    except Exception:
        layout_data = None

    responders_count = len(plan.responder_plans)

    # Derive room clear order from the first time any responder visits each room.
    room_first_visit: Dict[str, float] = {}
    for responder_plan in plan.responder_plans:
        for seg in responder_plan.segments:
            if seg.get("type") != "room":
                continue
            room_id = str(seg.get("target"))
            try:
                start_t = float(seg.get("start", 0.0))
            except Exception:
                start_t = 0.0
            if room_id not in room_first_visit or start_t < room_first_visit[room_id]:
                room_first_visit[room_id] = start_t
    def _pretty_room_id(room_id: str) -> str:
        if "_F" in room_id:
            base, floor_str = room_id.rsplit("_F", 1)
            try:
                floor_idx = int(floor_str)
            except Exception:
                return room_id
            return f"{base}_F{floor_idx + 1}"
        # Single-floor runs in examples still use _F1 suffix.
        return f"{room_id}_F1"

    ordered_rooms = [
        _pretty_room_id(room_id) for room_id, _ in sorted(room_first_visit.items(), key=lambda kv: kv[1])
    ]
    room_clear_order = "->".join(ordered_rooms)

    makespan_s = float(plan.makespan or 0.0)
    makespan_hms = _sec_to_hms(makespan_s)

    # Rebuild responders/exits from layout to infer start positions and exit sides.
    responders_start_node: Dict[str, str] = {}
    try:
        rooms_raw, responders_raw, exits_raw, _ = load_layout(layout_path)
        _, responders_exp, exits_exp = expand_floors(
            rooms_raw,
            responders_raw,
            exits_raw,
            config.floors,
            config.floor_spacing,
        )
        for resp in responders_exp:
            rid = str(resp.get("id"))
            start_node = resp.get("start_node")
            responders_start_node[rid] = start_node
    except Exception:
        responders_start_node = {}

    # Per-responder orders, start positions, and exit combo.
    responder_orders_parts: List[str] = []
    start_positions_parts: List[str] = []
    exit_sides: List[str] = []

    responder_plans = sorted(plan.responder_plans, key=lambda rp: str(rp.responder_id))
    for idx, responder_plan in enumerate(responder_plans, start=1):
        rid = str(responder_plan.responder_id)

        # Room visit order for this responder.
        if responder_plan.visit_order:
            path_str = "->".join(_pretty_room_id(str(r)) for r in responder_plan.visit_order)
        else:
            path_str = "N/A"
        responder_orders_parts.append(f"F{idx}:{path_str}")

        # Start side (E_L / E_R) from start_node.
        start_node = responders_start_node.get(rid)
        start_side = _exit_side(start_node)
        start_positions_parts.append(f"F{idx}:E_{start_side}")

        # Exit side from last egress segment (fallback to start side).
        last_exit_side = None
        for seg in responder_plan.segments:
            if seg.get("type") == "egress":
                last_exit_side = _exit_side(seg.get("target"))
        if last_exit_side is None:
            last_exit_side = start_side
        exit_sides.append(last_exit_side)

    exit_combo = "".join(exit_sides) if exit_sides else ""
    exit_combo_id = 1 if exit_combo.startswith("R") else 0

    responder_orders = " | ".join(responder_orders_parts)
    start_positions = ";".join(start_positions_parts)

    return {
        "layout": layout_label,
        "floors": floors,
        "per_room_occ": per_room_occ,
        "responders": responders_count,
        "exit_combo": exit_combo,
        "makespan_hms": makespan_hms,
        "room_clear_order": room_clear_order,
        "exit_combo_id": exit_combo_id,
        "makespan_s": makespan_s,
        "responder_orders": responder_orders,
        "start_positions": start_positions,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch MineEvac graph sweeps")
    parser.add_argument("--layout", help="Override layout path defined in configs.BatchSettings")
    parser.add_argument("--output", help="Override batch output root directory")
    parser.add_argument("--floors", nargs="*", help="Floor counts to evaluate")
    parser.add_argument("--redundancy", nargs="*", help="Redundancy modes to evaluate")
    parser.add_argument("--algorithms", nargs="*", help="Planner algorithms to evaluate")
    args = parser.parse_args(argv)

    settings = BatchSettings()
    if args.layout:
        settings.layout_path = args.layout
    if args.output:
        settings.output_root = args.output
    floor_override = _parse_int_list(args.floors)
    if floor_override:
        settings.floors = floor_override
    if args.redundancy:
        settings.redundancy_modes = args.redundancy
    if args.algorithms:
        settings.algorithms = args.algorithms

    summaries = []
    for config in settings.iter_configs():
        plan, _timeline = execute_run(config)
        summaries.append(_summarise_run(config, plan))

    out_dir = ensure_dir(settings.output_root)
    summary_path = Path(out_dir) / "summary.json"
    save_json(summaries, str(summary_path))

    csv_path = Path(out_dir) / "summary.csv"
    fieldnames = [
        "layout",
        "floors",
        "per_room_occ",
        "responders",
        "exit_combo",
        "makespan_hms",
        "room_clear_order",
        "exit_combo_id",
        "makespan_s",
        "responder_orders",
        "start_positions",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Batch summary written to {summary_path} and {csv_path}")
    print("Per-run sweep GIFs are available under each configuration's output directory (sweep.gif).")


if __name__ == "__main__":
    main()

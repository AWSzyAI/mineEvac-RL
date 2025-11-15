"""Batch executor for MineEvac graph sweeps."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import BatchSettings
from graph_evac import save_json
from graph_evac.io_utils import ensure_dir
from src.main import execute_run


def _parse_int_list(values: List[str] | None) -> List[int] | None:
    if not values:
        return None
    return [int(value) for value in values]


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
        plan, timeline = execute_run(config)
        summaries.append(
            {
                "run": config.label(),
                "layout": config.layout_path,
                "floors": config.floors,
                "redundancy": config.redundancy_mode,
                "algorithm": config.algorithm,
                "makespan": plan.makespan,
                "segments": sum(len(rp.segments) for rp in plan.responder_plans),
                "timeline_events": len(timeline),
                "output_dir": config.output_dir,
            }
        )

    out_dir = ensure_dir(settings.output_root)
    summary_path = Path(out_dir) / "summary.json"
    save_json(summaries, str(summary_path))

    csv_path = Path(out_dir) / "summary.csv"
    fieldnames = [
        "run",
        "layout",
        "floors",
        "redundancy",
        "algorithm",
        "makespan",
        "segments",
        "timeline_events",
        "output_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Batch summary written to {summary_path} and {csv_path}")


if __name__ == "__main__":
    main()

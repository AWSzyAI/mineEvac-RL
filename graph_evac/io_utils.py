"""IO utilities for MineEvac abstraction."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_timeline(timeline: Iterable[dict], path: str) -> None:
    fieldnames = ["responder_id", "segment_type", "target", "start", "end"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in timeline:
            writer.writerow(entry)


def write_run_log(path: str, *, config_dict: Mapping[str, Any], plan_summary: Mapping[str, Any]) -> None:
    """Persist a compact text log describing the run."""

    timestamp = datetime.utcnow().isoformat() + "Z"
    lines = [f"timestamp: {timestamp}"]

    lines.append("configuration:")
    for key, value in sorted(config_dict.items()):
        lines.append(f"  {key}: {value}")

    lines.append("plan:")
    makespan = plan_summary.get("makespan")
    if makespan is not None:
        lines.append(f"  makespan: {makespan:.2f}")
    responders = plan_summary.get("responders", [])
    for responder in responders:
        rid = responder.get("responder_id")
        total_time = responder.get("total_time", 0.0)
        segment_count = len(responder.get("segments", []))
        lines.append(f"  - {rid}: {total_time:.2f}s across {segment_count} segments")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["ensure_dir", "save_json", "save_timeline", "write_run_log"]

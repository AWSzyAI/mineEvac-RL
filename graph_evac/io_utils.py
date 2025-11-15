"""IO utilities for MineEvac abstraction."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


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


__all__ = ["ensure_dir", "save_json", "save_timeline"]

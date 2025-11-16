"""IO utilities for MineEvac abstraction."""
from __future__ import annotations

import csv
import json
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


def _sec_to_hms(seconds: float) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def write_run_log(path: str, *, config_dict: Mapping[str, Any], plan_summary: Mapping[str, Any]) -> None:
    """Persist a JSON run log compatible with the original deterministic schema.

    The structure mirrors ``logs/det_baseline.json`` from the legacy simulator so
    that downstream tooling expecting ``layout``, ``responders``, ``per_room``,
    ``room_order``, ``real_hms`` etc. continues to work.
    """

    layout_path = str(config_dict.get("layout_path", "layout/baseline.json"))
    layout_name = Path(layout_path).name

    # --- Layout-derived metadata -------------------------------------------------
    per_room = None
    room_ids_from_layout = []
    init_positions = []
    exits = []

    frame = None
    corridor = None
    layout_data = None
    try:
        layout_data = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        occupants_block = layout_data.get("occupants") or {}
        per_room = occupants_block.get("per_room")
        if per_room is not None:
            try:
                per_room = int(per_room)
            except Exception:
                per_room = None

        room_defs = layout_data.get("rooms")
        if room_defs is None:
            room_defs = list(layout_data.get("rooms_top", [])) + list(layout_data.get("rooms_bottom", []))
        for idx, room in enumerate(room_defs or []):
            room_ids_from_layout.append(room.get("id", f"R{idx}"))

        frame = layout_data.get("frame")
        corridor = layout_data.get("corridor")
    except Exception:
        layout_data = None

    # --- Plan-derived metadata ---------------------------------------------------
    responders = list(plan_summary.get("responders") or [])
    num_responders = len(responders)

    makespan = float(plan_summary.get("makespan") or 0.0)

    room_first_visit: dict[str, float] = {}
    for responder in responders:
        for seg in responder.get("segments", []):
            if seg.get("type") != "room":
                continue
            room_id = str(seg.get("target"))
            try:
                start = float(seg.get("start", 0.0))
            except Exception:
                start = 0.0
            if room_id not in room_first_visit or start < room_first_visit[room_id]:
                room_first_visit[room_id] = start

    room_order = [room_id for room_id, _ in sorted(room_first_visit.items(), key=lambda kv: kv[1])]

    # --- Derived evacuation summary ---------------------------------------------
    total_rooms = len(room_ids_from_layout)
    evacuated = None
    all_evacuated = None
    if per_room is not None and total_rooms:
        evacuated = per_room * len(room_order)
        total_possible = per_room * total_rooms
        if evacuated > total_possible:
            evacuated = total_possible
        all_evacuated = bool(len(room_order) >= total_rooms and evacuated == total_possible)

    # --- Init positions & exits (replicating deterministic conventions) ---------
    if layout_data and frame and corridor:
        x1 = int(corridor.get("x", 0))
        w = int(corridor.get("w", 0))
        z0 = int(corridor.get("z", 0))
        h = int(corridor.get("h", 0))
        if w > 0 and h > 0:
            x2 = x1 + w - 1
            corr_z_min = z0
            corr_z_max = z0 + h - 1
            mid_z_exit = corr_z_min + (corr_z_max - corr_z_min) // 2
            mid_z_init = z0 + h // 2

            n = max(1, num_responders or 0)
            start_x = x1 + 1
            end_x = max(start_x, x2 - 1)
            if n == 1:
                xs = [start_x]
            elif n == 2:
                xs = [start_x, end_x]
            else:
                span = end_x - start_x
                xs = [start_x + round(span * i / (n - 1)) for i in range(n)]
            init_positions = [[int(x), int(mid_z_init)] for x in xs]

            exits = [
                [int(frame.get("x1", x1)), int(mid_z_exit)],
                [int(frame.get("x2", x2)), int(mid_z_exit)],
            ]

    # --- Real-time mapping params (use deterministic defaults) ------------------
    cell_m = 0.5
    speed_solo = 0.8
    speed_escort = 0.6
    real_seconds = makespan
    real_minutes = round(real_seconds / 60.0, 2) if real_seconds is not None else 0.0

    log_payload: dict[str, Any] = {
        "layout": layout_name,
        "responders": num_responders,
        "per_room": per_room,
        "time": int(round(makespan)),
        "all_evacuated": all_evacuated,
        "room_order": room_order,
        "init_positions": init_positions,
        "exits": exits,
        "evacuated": evacuated,
        "real_hms": _sec_to_hms(real_seconds),
        "real_minutes": real_minutes,
        "cell_m": cell_m,
        "speed_solo_mps": speed_solo,
        "speed_escort_mps": speed_escort,
    }

    Path(path).write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["ensure_dir", "save_json", "save_timeline", "write_run_log"]

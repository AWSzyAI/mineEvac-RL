#!/usr/bin/env python3
"""Summarise RL eval_episode.jsonl into a det_baseline-style JSON log.

This is used to align `make run ALGO=ppo` outputs with the deterministic
pipeline: same JSON schema as logs/det_baseline.json so that downstream
analysis and reviewers can reuse existing tooling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _sec_to_hms(seconds: float) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _load_layout(path: str) -> Tuple[dict, int, List[List[int]]]:
    """Return (layout_json, per_room, exits_as_xy) for det-style logging."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    occ = data.get("occupants") or {}
    per_room = occ.get("per_room")
    if per_room is not None:
        try:
            per_room = int(per_room)
        except Exception:
            per_room = None

    frame = data.get("frame") or {}
    corridor = data.get("corridor") or {}
    x1 = int(frame.get("x1", corridor.get("x", 0)))
    x2 = int(frame.get("x2", corridor.get("x", 0)))
    cz = int(corridor.get("z", 0))
    ch = int(corridor.get("h", 1))
    corr_z_min = cz
    corr_z_max = cz + ch - 1
    mid_z = corr_z_min + (corr_z_max - corr_z_min) // 2
    exits = [[x1, mid_z], [x2, mid_z]]

    return data, per_room, exits


def _room_label(room_id: str) -> str:
    """Convert 'R0' -> 'R1' etc., mirroring deterministic logs."""

    if isinstance(room_id, str) and room_id.startswith("R"):
        try:
            k = int(room_id[1:]) + 1
            return f"R{k}"
        except Exception:
            return room_id
    return room_id


def summarise_eval_log(eval_log_path: str, layout_path: str, save_path: str) -> None:
    layout_json, per_room, exits = _load_layout(layout_path)

    # --- Scan eval_episode.jsonl ---------------------------------------------
    room_first_clear: Dict[str, int] = {}
    last_t: Optional[int] = None
    last_occupants: List[dict] = []

    with open(eval_log_path, "r", encoding="utf-8") as f:
        prev_cleared: Dict[str, bool] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            t = data.get("t")
            if t is None:
                # fall back to index if t is missing
                t = 0 if last_t is None else last_t + 1
            try:
                t_int = int(t)
            except Exception:
                t_int = 0
            last_t = t_int

            # Track room clear order based on room_cleared dict transitions.
            rc = data.get("room_cleared") or {}
            if not isinstance(rc, dict):
                rc = {}
            for room_id_str, cleared in rc.items():
                room_id = str(room_id_str)
                prev = prev_cleared.get(room_id, False)
                now = bool(cleared)
                if now and not prev and room_id not in room_first_clear:
                    room_first_clear[room_id] = t_int
                prev_cleared[room_id] = now

            # Keep last occupants snapshot to estimate evacuation count.
            occ_list = data.get("occupants") or []
            if isinstance(occ_list, list):
                last_occupants = occ_list

    if last_t is None:
        raise RuntimeError(f"No frames found in {eval_log_path}")

    time_steps = last_t

    # --- Derived quantities ---------------------------------------------------
    room_order = [
        _room_label(room_id) for room_id, _ in sorted(room_first_clear.items(), key=lambda kv: kv[1])
    ]

    evacuated = None
    all_evacuated = None
    if last_occupants:
        evac_flags = [bool(o.get("evacuated", False)) for o in last_occupants]
        evacuated = int(sum(1 for flag in evac_flags if flag))
        all_evacuated = bool(all(evac_flags))

    if per_room is not None and evacuated is None:
        # Fallback: assume everyone evacuated if rooms cleared.
        evacuated = per_room * len(layout_json.get("rooms_top", []) + layout_json.get("rooms_bottom", []))

    # Single-responder RL baseline.
    responders = 1

    # Init position: match MineEvacEnv.init_single_responder (left corridor +1, midline).
    corr = layout_json.get("corridor") or {}
    cx = int(corr.get("x", 0))
    cz = int(corr.get("z", 0))
    ch = int(corr.get("h", 1))
    corr_z_min = cz
    corr_z_max = cz + ch - 1
    mid_z = corr_z_min + (corr_z_max - corr_z_min) // 2
    init_positions = [[cx + 1, mid_z]]

    # --- Real-time mapping (simple: 1 step ~= 1 second) ----------------------
    cell_m = 0.5
    speed_solo = 0.8
    speed_escort = 0.6
    real_seconds = float(time_steps)
    real_minutes = round(real_seconds / 60.0, 2)

    payload = {
        "layout": Path(layout_path).name,
        "responders": responders,
        "per_room": per_room,
        "time": int(time_steps),
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

    Path(save_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise RL eval log to det-style JSON schema")
    parser.add_argument("--eval-log", required=True, help="Path to eval_episode.jsonl")
    parser.add_argument("--layout", required=True, help="Layout JSON path (e.g., layout/baseline.json)")
    parser.add_argument("--save", required=True, help="Output JSON path (e.g., logs/det_baseline.json)")
    args, _ = parser.parse_known_args()

    summarise_eval_log(args.eval_log, args.layout, args.save)


if __name__ == "__main__":
    main()


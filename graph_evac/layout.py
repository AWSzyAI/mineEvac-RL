"""Layout loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Coord = Tuple[float, float, float]


def _center_from_box(box: Dict[str, float]) -> Coord:
    x = box["x"] + box.get("w", 1) / 2.0
    z = box["z"] + box.get("h", 1) / 2.0
    return (x, 0.0, z)


def load_layout(path: str) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Load the simplified layout description."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    rooms: List[Dict] = []
    room_defs: Sequence[Dict]
    if "rooms" in data:
        room_defs = data["rooms"]
    else:
        top = data.get("rooms_top", [])
        bottom = data.get("rooms_bottom", [])
        room_defs = list(top) + list(bottom)
    for idx, box in enumerate(room_defs):
        rooms.append(
            {
                "id": box.get("id", f"R{idx}"),
                "coord": _center_from_box(box),
                "occupants": box.get("occupants", data.get("occupants", {}).get("per_room", 0)),
            }
        )

    exits: List[Dict]
    exit_defs = data.get("exits")
    if exit_defs:
        exits = [
            {
                "id": ex.get("id", f"E{idx}"),
                # Support both {coord:[x,z]} and legacy {position:[x,z]}
                "coord": tuple(
                    ex.get("coord")
                    or ex.get("position")
                    or (0.0, 0.0, 0.0)
                ),
            }
            for idx, ex in enumerate(exit_defs)
        ]
    else:
        frame = data.get("frame", {})
        corridor = data.get("corridor", {})
        if frame and corridor:
            mid_z = corridor.get("z", 0) + corridor.get("h", 0) / 2.0
            exits = [
                {"id": "E_left", "coord": (frame.get("x1", 0.0), 0.0, mid_z)},
                {"id": "E_right", "coord": (frame.get("x2", 0.0), 0.0, mid_z)},
            ]
        else:
            exits = [{"id": "E0", "coord": (0.0, 0.0, 0.0)}]

    occupants: List[Dict] = []
    next_occ_id = 0
    for room in rooms:
        for _ in range(int(room.get("occupants", 0))):
            occupants.append({"id": f"occ_{next_occ_id}", "room": room["id"]})
            next_occ_id += 1

    responders: List[Dict] = []
    responder_defs = data.get("responders") or []
    if responder_defs:
        for idx, resp in enumerate(responder_defs):
            start_node = resp.get("start_node")
            if start_node is None:
                start_node = resp.get("start_exit") or (exits[0]["id"] if exits else "E0")
            responders.append(
                {
                    "id": resp.get("id", f"responder_{idx}"),
                    "start_node": start_node,
                    "coord": tuple(resp.get("coord", exits[0]["coord"])),
                }
            )
    else:
        responders.append({"id": "responder_0", "start_node": exits[0]["id"], "coord": exits[0]["coord"]})
        if len(exits) > 1:
            responders.append(
                {
                    "id": "responder_1",
                    "start_node": exits[1]["id"],
                    "coord": exits[1]["coord"],
                }
            )

    return rooms, responders, exits, occupants


def expand_floors(
    rooms: Sequence[Dict],
    responders: Sequence[Dict],
    exits: Sequence[Dict],
    floors: int,
    spacing: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Replicate a single-floor layout to multiple floors."""

    if floors <= 1:
        for room in rooms:
            room.setdefault("floor", 0)
        for responder in responders:
            responder.setdefault("floor", 0)
        for exit_def in exits:
            exit_def.setdefault("floor", 0)
        return list(rooms), list(responders), list(exits)

    expanded_rooms: List[Dict] = []
    for floor in range(floors):
        for room in rooms:
            coord = room["coord"]
            if len(coord) == 3:
                x, y, z = coord
            else:  # pragma: no cover - fallback for legacy tuples
                x, z = coord
                y = 0.0
            expanded_rooms.append(
                {
                    **room,
                    "id": f"{room['id']}_F{floor}",
                    "coord": (x, y + floor * spacing, z),
                    "floor": floor,
                }
            )

    expanded_exits: List[Dict] = []
    for floor in range(floors):
        for exit_def in exits:
            coord = exit_def["coord"]
            if len(coord) == 3:
                x, y, z = coord
            else:
                x, z = coord
                y = 0.0
            expanded_exits.append(
                {
                    **exit_def,
                    "id": f"{exit_def['id']}_F{floor}",
                    "coord": (x, y + floor * spacing, z),
                    "floor": floor,
                }
            )

    expanded_responders: List[Dict] = []
    for floor in range(floors):
        for resp in responders:
            expanded_responders.append(
                {
                    **resp,
                    "id": f"{resp['id']}_F{floor}",
                    "start_node": f"{resp['start_node']}_F{floor}",
                    "floor": floor,
                }
            )

    return expanded_rooms, expanded_responders, expanded_exits


__all__ = ["load_layout", "expand_floors"]

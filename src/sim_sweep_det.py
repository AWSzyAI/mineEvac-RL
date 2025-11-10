#!/usr/bin/env python3
"""Deterministic multi-responder sweep simulator.

Goal
-----
- Given a layout JSON (baseline.json-like), place N responders at given initial
  corridor positions, simulate a sweep of rooms with a simple greedy policy,
  move occupants toward exits once their room is swept, and compute:
  * total time to evacuate everyone
  * room sweep order (first-entry times)

Notes
-----
- This purposely avoids Stable-Baselines and the RL env; it is a straight
  simulation so we can satisfy Task 1 and the baseline of Task 3.
- Movement is on the same integer grid as the env; door logic mirrors the
  layout doors: rooms connect to corridor through z=topZ and z=bottomZ walls at
  x in doors.xs; corridor spans layout["corridor"].
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Coord = Tuple[int, int]  # (x, z)


def load_layout(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Room:
    id: str
    x1: int
    z1: int
    x2: int
    z2: int

    def contains(self, p: Coord) -> bool:
        x, z = p
        return self.x1 <= x <= self.x2 and self.z1 <= z <= self.z2

    @property
    def center(self) -> Coord:
        return ((self.x1 + self.x2) // 2, (self.z1 + self.z2) // 2)


@dataclass
class Responder:
    id: int
    pos: Coord
    target: Optional[Coord] = None
    path: Optional[List[Coord]] = None
    busy_steps: int = 0


@dataclass
class Occupant:
    id: int
    pos: Coord
    origin_room: str
    evacuated: bool = False
    evac_time: Optional[int] = None
    attached_to: Optional[int] = None  # responder id if following


class SweepSim:
    def __init__(self, layout: dict, responders_init: List[Coord], per_room: int = 5, seed: int = 0):
        self.layout = layout
        self.rng = np.random.default_rng(seed)
        # corridor
        corr = layout["corridor"]
        self.corr_x_min = corr["x"]
        self.corr_x_max = corr["x"] + corr["w"] - 1
        self.corr_z_min = corr["z"]
        self.corr_z_max = corr["z"] + corr["h"] - 1
        # doors
        doors = layout.get("doors", {})
        self.door_xs: Set[int] = set(doors.get("xs", []))
        self.top_z = int(doors.get("topZ", self.corr_z_max + 1))
        self.bot_z = int(doors.get("bottomZ", self.corr_z_min - 1))
        # rooms
        self.rooms: List[Room] = []
        idx = 0
        for key in ("rooms_top", "rooms_bottom"):
            for r in layout.get(key, []):
                x1 = r["x"]; z1 = r["z"]; x2 = x1 + r["w"] - 1; z2 = z1 + r["h"] - 1
                self.rooms.append(Room(id=f"R{idx}", x1=x1, z1=z1, x2=x2, z2=z2))
                idx += 1
        # exits (corridor midline at frame ends)
        frame = layout["frame"]
        mid_z = self.corr_z_min + (self.corr_z_max - self.corr_z_min) // 2
        self.exits = [(frame["x1"], mid_z), (frame["x2"], mid_z)]

        # responders
        self.responders: List[Responder] = [Responder(i, p) for i, p in enumerate(responders_init)]

        # occupants: per room initially placed near center
        self.occupants: List[Occupant] = []
        oid = 0
        for room in self.rooms:
            cx, cz = room.center
            for k in range(per_room):
                pos = (cx, cz + (k % 2))
                self.occupants.append(Occupant(id=oid, pos=pos, origin_room=room.id))
                oid += 1

        # bookkeeping
        self.t = 0
        self.room_first_entry: Dict[str, Optional[int]] = {room.id: None for room in self.rooms}
        self.room_swept: Dict[str, bool] = {room.id: False for room in self.rooms}

    # -------- geometry helpers --------
    def in_corridor(self, p: Coord) -> bool:
        x, z = p
        return self.corr_x_min <= x <= self.corr_x_max and self.corr_z_min <= z <= self.corr_z_max

    def room_at(self, p: Coord) -> Optional[Room]:
        for r in self.rooms:
            if r.contains(p):
                return r
        return None

    def is_door_crossing(self, a: Coord, b: Coord) -> bool:
        ax, az = a; bx, bz = b
        # top wall interface: corridor z=topZ-1 <-> room z=topZ
        if {az, bz} == {self.top_z - 1, self.top_z} and ax == bx and ax in self.door_xs:
            return True
        # bottom wall interface: corridor z=bottomZ+1 <-> room z=bottomZ
        if {az, bz} == {self.bot_z + 1, self.bot_z} and ax == bx and ax in self.door_xs:
            return True
        return False

    def valid_step(self, a: Coord, b: Coord) -> bool:
        ax, az = a; bx, bz = b
        if abs(ax - bx) + abs(az - bz) != 1:
            return False
        in_corr_a = self.in_corridor(a)
        in_corr_b = self.in_corridor(b)
        in_room_a = self.room_at(a) is not None
        in_room_b = self.room_at(b) is not None
        if not (in_corr_b or in_room_b):
            return False
        if in_corr_a and in_room_b:
            return self.is_door_crossing(a, b)
        if in_room_a and in_corr_b:
            return self.is_door_crossing(a, b)
        if in_room_a and in_room_b:
            return self.room_at(a) == self.room_at(b)
        return True

    def neighbors(self, p: Coord):
        x, z = p
        for q in [(x+1,z), (x-1,z), (x,z+1), (x,z-1)]:
            if self.valid_step(p, q):
                yield q

    def shortest_path(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        if start == goal:
            return [start]
        from collections import deque
        q = deque([start])
        prev: Dict[Coord, Optional[Coord]] = {start: None}
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if v in prev:
                    continue
                prev[v] = u
                if v == goal:
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
        return None

    # -------- policy helpers --------
    def room_door_corridor_cell(self, room: Room) -> Coord:
        # choose the doorway x nearest to room center
        cx, cz = room.center
        if cz >= self.corr_z_max + 1:
            # top rooms open at z=top_z between (x in door_xs)
            x = min(self.door_xs, key=lambda dx: abs(dx - cx))
            return (x, self.top_z - 1)
        else:
            x = min(self.door_xs, key=lambda dx: abs(dx - cx))
            return (x, self.bot_z + 1)

    def attached_occupants(self, resp_id: int) -> List[Occupant]:
        return [o for o in self.occupants if (not o.evacuated) and o.attached_to == resp_id]

    def assign_next_targets(self):
        # If a responder is escorting occupants, send them to nearest exit;
        # otherwise pick unswept rooms greedily by shortest path.
        unswept = [r for r in self.rooms if not self.room_swept[r.id]]
        if not unswept:
            return
        for resp in self.responders:
            # If escorting, head to nearest exit
            if self.attached_occupants(resp.id):
                exit_goal = self.nearest_exit(resp.pos)
                path = self.shortest_path(resp.pos, exit_goal)
                if path and len(path) > 1:
                    resp.path = path[1:]
                continue
            if resp.path:  # already en route to a room/exit
                continue
            # choose closest room door from current position
            best = None
            best_path = None
            for room in unswept:
                door_cell = self.room_door_corridor_cell(room)
                path = self.shortest_path(resp.pos, door_cell)
                if not path:
                    continue
                if best is None or len(path) < len(best_path):
                    best = (room, door_cell)
                    best_path = path
            if best is not None:
                # after reaching door corridor cell, push one step into room entry cell (inside)
                room, door_cell = best
                # target room entry cell is the door cell on room side
                if room.z1 == self.top_z:
                    entry = (door_cell[0], self.top_z)
                elif room.z2 == self.bot_z:
                    entry = (door_cell[0], self.bot_z)
                else:
                    entry = room.center
                # path to door corridor cell, then to room entry
                path2 = self.shortest_path(door_cell, entry)
                full = best_path + (path2[1:] if path2 and len(path2) > 1 else [])
                resp.path = full[1:] if len(full) > 1 else []

    # -------- simulation step --------
    def step(self):
        self.t += 1
        # 1) assign targets as needed
        self.assign_next_targets()

        # 2) responders move one step along their paths
        for resp in self.responders:
            if resp.path and len(resp.path) > 0:
                nxt = resp.path.pop(0)
                if self.valid_step(resp.pos, nxt):
                    resp.pos = nxt

            # check room first entry
            room = self.room_at(resp.pos)
            if room is not None and self.room_first_entry[room.id] is None:
                self.room_first_entry[room.id] = self.t
                # attach all occupants in this room to this responder (escort mode)
                for o in self.occupants:
                    if not o.evacuated and o.attached_to is None and room.contains(o.pos):
                        o.attached_to = resp.id

            # if escorting and currently in a room, set path towards nearest exit
            if room is not None and self.attached_occupants(resp.id):
                exit_goal = self.nearest_exit(resp.pos)
                path = self.shortest_path(resp.pos, exit_goal)
                if path and len(path) > 1:
                    resp.path = path[1:]

        # 3) occupants follow evacuation when attached (no self-evac without escort)
        for o in self.occupants:
            if o.evacuated or o.attached_to is None:
                continue
            # target: if still in its origin room, head to that room's doorway corridor cell; otherwise head to nearest exit
            if self.room_at(o.pos) is not None:
                # move toward the door corridor cell corresponding to origin room
                origin = next(r for r in self.rooms if r.id == o.origin_room)
                door_corr = self.room_door_corridor_cell(origin)
                target = door_corr
            else:
                target = self.nearest_exit(o.pos)
            step = self.step_towards(o.pos, target)
            if step and self.valid_step(o.pos, step):
                o.pos = step
            # check evacuated
            if o.pos in self.exits:
                o.evacuated = True
                o.evac_time = self.t
                o.attached_to = None

        # 4) update swept flags: a room is swept when all its occupants have evacuated
        for room in self.rooms:
            if not self.room_swept[room.id]:
                if all(o.evacuated for o in self.occupants if o.origin_room == room.id):
                    self.room_swept[room.id] = True

    def snapshot(self) -> dict:
        """Legacy-style frame compatible with animate_sweep/visualize_heatmap.
        Uses 'responders' and 'occupants' arrays with x,y coordinates (y=z).
        """
        return {
            "time": self.t,
            "responders": [
                {"id": r.id, "x": r.pos[0], "y": r.pos[1]}
                for r in self.responders
            ],
            "occupants": [
                {"id": o.id, "x": o.pos[0], "y": o.pos[1], "evacuated": o.evacuated}
                for o in self.occupants
            ],
        }

    def step_towards(self, current: Coord, target: Coord) -> Optional[Coord]:
        if current == target:
            return current
        # greedy neighbor by Manhattan
        cx, cz = current
        candidates = [(cx+1, cz), (cx-1, cz), (cx, cz+1), (cx, cz-1)]
        candidates = [q for q in candidates if self.valid_step(current, q)]
        if not candidates:
            return None
        def md(a: Coord, b: Coord) -> int:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        base = md(current, target)
        candidates.sort(key=lambda q: md(q, target))
        for q in candidates:
            if md(q, target) <= base:
                return q
        return candidates[0]

    def nearest_exit(self, pos: Coord) -> Coord:
        best = None
        bestd = None
        for e in self.exits:
            d = abs(e[0]-pos[0]) + abs(e[1]-pos[1])
            if best is None or d < bestd:
                best = e; bestd = d
        return best

    def all_evacuated(self) -> bool:
        return all(o.evacuated for o in self.occupants)

    def room_order(self) -> List[str]:
        pairs = [(rid, t) for rid, t in self.room_first_entry.items() if t is not None]
        pairs.sort(key=lambda x: x[1])
        return [rid for rid, _ in pairs]


def default_init_positions(layout: dict, num: int) -> List[Coord]:
    corr = layout["corridor"]
    x1 = corr["x"]; x2 = corr["x"] + corr["w"] - 1
    mid_z = corr["z"] + corr["h"] // 2
    if num == 1:
        return [(x1+1, mid_z)]
    # two responders: near two ends
    if num == 2:
        return [(x1+1, mid_z), (x2-1, mid_z)]
    # spread evenly
    xs = np.linspace(x1+1, x2-1, num, dtype=int).tolist()
    return [(x, mid_z) for x in xs]


def run_once(layout_path: str, num_responders: int = 2, per_room: int = 5, max_steps: int = 3000, seed: int = 0, init_positions: Optional[List[Coord]] = None, frames_path: Optional[str] = None):
    layout = load_layout(layout_path)
    resp_init = init_positions if init_positions is not None else default_init_positions(layout, num_responders)
    sim = SweepSim(layout, resp_init, per_room=per_room, seed=seed)
    f = None
    try:
        if frames_path:
            os.makedirs(os.path.dirname(frames_path), exist_ok=True)
            f = open(frames_path, "w", encoding="utf-8")
        # initial snapshot (t=0)
        if f:
            f.write(json.dumps(sim.snapshot(), ensure_ascii=False) + "\n")
        while sim.t < max_steps and not sim.all_evacuated():
            sim.step()
            if f:
                f.write(json.dumps(sim.snapshot(), ensure_ascii=False) + "\n")
    finally:
        if f:
            f.close()
    result = {
        "layout": os.path.basename(layout_path),
        "responders": num_responders,
        "per_room": per_room,
        "time": sim.t,
        "all_evacuated": sim.all_evacuated(),
        "room_order": sim.room_order(),
        "init_positions": resp_init,
        "exits": sim.exits,
        "evacuated": sum(1 for o in sim.occupants if o.evacuated),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="layout/baseline.json")
    ap.add_argument("--responders", type=int, default=2)
    ap.add_argument("--per_room", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="logs/det_baseline.json")
    ap.add_argument("--frames", default=None, help="If set, write JSONL frames (legacy schema) for GIF/heatmap")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    result = run_once(args.layout, args.responders, args.per_room, args.max_steps, args.seed, frames_path=args.frames)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

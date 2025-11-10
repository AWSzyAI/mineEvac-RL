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
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Coord = Tuple[int, int]  # (x, z)


def _sec_to_hms(seconds: float) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


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
    pos: Tuple[float, float]  # continuous (x, z)
    target: Optional[Coord] = None
    path: Optional[List[Coord]] = None
    busy_steps: int = 0


@dataclass
class Occupant:
    id: int
    pos: Tuple[float, float]  # continuous
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

        # precompute wall cells (room perimeters), leaving door openings on corridor-facing walls
        self.wall_cells: Set[Coord] = self._build_wall_cells()

        # frame bounds (inclusive indices)
        frame = layout["frame"]
        self.x_min = int(frame["x1"]) ; self.x_max = int(frame["x2"])  # inclusive
        self.z_min = int(frame["z1"]) ; self.z_max = int(frame["z2"])  # inclusive

        # responders
        self.responders: List[Responder] = [Responder(i, (float(p[0]), float(p[1]))) for i, p in enumerate(responders_init)]
        # per-responder self-clearing memory (no sharing)
        self.cleared_by: Dict[int, Dict[str, bool]] = {r.id: {room.id: False for room in self.rooms} for r in self.responders}
        self.clear_order: Dict[int, List[str]] = {r.id: [] for r in self.responders}

        # occupants: per room initially placed near center
        self.occupants: List[Occupant] = []
        oid = 0
        for room in self.rooms:
            for _ in range(per_room):
                # sample interior, avoid wall cells
                for _tries in range(100):
                    rx = int(self.rng.integers(low=room.x1, high=room.x2 + 1))
                    rz = int(self.rng.integers(low=room.z1, high=room.z2 + 1))
                    if (rx, rz) not in self.wall_cells:
                        pos = (rx, rz)
                        break
                else:
                    pos = room.center
                self.occupants.append(Occupant(id=oid, pos=(float(pos[0]), float(pos[1])), origin_room=room.id))
                oid += 1

        # bookkeeping
        self.t = 0
        self.room_first_entry: Dict[str, Optional[int]] = {room.id: None for room in self.rooms}
        self.room_swept: Dict[str, bool] = {room.id: False for room in self.rooms}
        # stuck detection for responders
        self._resp_prev_d: Dict[int, Optional[float]] = {r.id: None for r in self.responders}
        self._resp_stuck: Dict[int, int] = {r.id: 0 for r in self.responders}
        self._resp_last_cell: Dict[int, Optional[Coord]] = {r.id: None for r in self.responders}
        self._resp_stagnant: Dict[int, int] = {r.id: 0 for r in self.responders}
        # escort flag for timing mapping (set each step)
        self._escort_active: bool = False

    # ---- scheduling helpers ----
    def _candidate_entries_for(self, rid: int, pos_cell: Coord) -> List[Tuple[Coord, float, Room]]:
        """Return list of (entry_cell, cost, room) for rooms not self-cleared by this responder, sorted by cost."""
        my_cleared = self.cleared_by[rid]
        todo = [room for room in self.rooms if not my_cleared[room.id]]
        cands: List[Tuple[Coord, float, Room]] = []
        if not todo or not self.cell_in_bounds(pos_cell):
            return cands
        for room in todo:
            entry = self.room_door_corridor_cell(room)
            ex, ez = entry
            if ez == self.top_z - 1:
                entry = (ex, self.top_z)
            elif ez == self.bot_z + 1:
                entry = (ex, self.bot_z)
            D = self.distance_field([entry])
            cost = D[pos_cell[1]-self.z_min, pos_cell[0]-self.x_min]
            if np.isfinite(cost):
                cands.append((entry, float(cost), room))
        cands.sort(key=lambda t: t[1])
        return cands

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
        # wall collision
        if a in self.wall_cells or b in self.wall_cells:
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

    def neighbors8_for_path(self, p: Coord):
        """8-neighbors for pathing; forbid diagonal corner-cuts across walls."""
        x, z = p
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                nx, nz = x + dx, z + dz
                q = (nx, nz)
                if not self.cell_in_bounds(q):
                    continue
                if self.is_wall_cell(q):
                    continue
                if dx != 0 and dz != 0:
                    # prevent walking through wall corners
                    if self.is_wall_cell((x + dx, z)) or self.is_wall_cell((x, z + dz)):
                        continue
                yield q

    # ---- continuous movement helpers ----
    def cell_in_bounds(self, p: Coord) -> bool:
        return self.x_min <= p[0] <= self.x_max and self.z_min <= p[1] <= self.z_max

    def is_wall_cell(self, p: Coord) -> bool:
        return p in self.wall_cells

    def cell_of(self, pos: Tuple[float, float]) -> Coord:
        return (int(np.floor(pos[0])), int(np.floor(pos[1])))

    def line_clear(self, a: Tuple[float, float], b: Tuple[float, float], step: float = 0.2) -> bool:
        ax, az = a; bx, bz = b
        dx, dz = bx - ax, bz - az
        L = float(np.hypot(dx, dz))
        n = max(1, int(np.ceil(L / max(1e-3, step))))
        for i in range(n + 1):
            t = i / n
            x = ax + t * dx
            z = az + t * dz
            if self.is_wall_cell(self.cell_of((x, z))):
                return False
        return True

    def distance_field(self, goals: List[Coord]) -> np.ndarray:
        W = self.x_max - self.x_min + 1
        H = self.z_max - self.z_min + 1
        INF = 1e9
        D = np.full((H, W), INF, dtype=float)
        import heapq
        h = []
        for gx, gz in goals:
            if not self.cell_in_bounds((gx, gz)) or self.is_wall_cell((gx, gz)):
                continue
            D[gz - self.z_min, gx - self.x_min] = 0.0
            heapq.heappush(h, (0.0, gx, gz))
        while h:
            d, x, z = heapq.heappop(h)
            if d > D[z - self.z_min, x - self.x_min]:
                continue
            for nx in (x-1, x, x+1):
                for nz in (z-1, z, z+1):
                    if nx == x and nz == z:
                        continue
                    if not self.cell_in_bounds((nx, nz)) or self.is_wall_cell((nx, nz)):
                        continue
                    # block diagonal corner cutting
                    if nx != x and nz != z:
                        if self.is_wall_cell((nx, z)) or self.is_wall_cell((x, nz)):
                            continue
                    w = float(np.hypot(nx - x, nz - z))
                    nd = d + w
                    ii = nz - self.z_min; jj = nx - self.x_min
                    if nd < D[ii, jj]:
                        D[ii, jj] = nd
                        heapq.heappush(h, (nd, nx, nz))
        return D

    def step_along_field(self, pos: Tuple[float, float], D: np.ndarray, step: float = 1.0) -> Tuple[float, float]:
        x, z = pos
        cx, cz = self.cell_of(pos)
        base = D[cz - self.z_min, cx - self.x_min] if self.cell_in_bounds((cx, cz)) else 1e9
        best_dir = None
        best_d = base
        for nx in (cx-1, cx, cx+1):
            for nz in (cz-1, cz, cz+1):
                if not self.cell_in_bounds((nx, nz)) or self.is_wall_cell((nx, nz)):
                    continue
                d = D[nz - self.z_min, nx - self.x_min]
                if d < best_d:
                    best_d = d
                    best_dir = (nx + 0.5 - x, nz + 0.5 - z)
        if best_dir is None:
            return pos
        dx, dz = best_dir
        L = float(np.hypot(dx, dz))
        if L < 1e-6:
            return pos
        dx /= L; dz /= L
        nx, nz = x + dx * step, z + dz * step
        # avoid crossing walls; shrink step if needed
        if not self.line_clear((x, z), (nx, nz)):
            lo, hi = 0.0, step
            for _ in range(12):
                mid = (lo + hi) / 2
                tx, tz = x + dx * mid, z + dz * mid
                if self.line_clear((x, z), (tx, tz)):
                    lo = mid
                else:
                    hi = mid
            nx, nz = x + dx * lo, z + dz * lo
        return (nx, nz)

    def shortest_path(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        """8-connected Dijkstra path (cost 1 or sqrt(2)) obeying walls/doors constraints."""
        if start == goal:
            return [start]
        import heapq
        INF = 1e18
        dist: Dict[Coord, float] = {start: 0.0}
        prev: Dict[Coord, Optional[Coord]] = {start: None}
        h = [(0.0, start)]
        while h:
            d, u = heapq.heappop(h)
            if u == goal:
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            if d > dist.get(u, INF):
                continue
            ux, uz = u
            for v in self.neighbors8_for_path(u):
                vx, vz = v
                w = float(np.hypot(vx - ux, vz - uz))
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(h, (nd, v))
        return None

    def _build_wall_cells(self) -> Set[Coord]:
        walls: Set[Coord] = set()
        for room in self.rooms:
            # horizontal edges
            for x in range(room.x1, room.x2 + 1):
                # top wall: only leave door openings on corridor-facing wall
                if room.z1 == self.top_z and x in self.door_xs:
                    pass
                else:
                    walls.add((x, room.z1))
                # bottom wall
                if room.z2 == self.bot_z and x in self.door_xs:
                    pass
                else:
                    walls.add((x, room.z2))
            # vertical edges (always walls)
            for z in range(room.z1, room.z2 + 1):
                walls.add((room.x1, z))
                walls.add((room.x2, z))
        return walls

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
        # per-responder schedule (no sharing): visit rooms until self-cleared
        for resp in self.responders:
            my_cleared = self.cleared_by[resp.id]
            my_todo = [r for r in self.rooms if not my_cleared[r.id]]
            # If escorting, go to nearest exit
            if self.attached_occupants(resp.id):
                cell = self.cell_of(resp.pos)
                resp.target = self.nearest_exit(cell)
                continue
            # If inside a room with remaining unattached occupants, aim for nearest occupant in same room
            cur_room = self.room_at(self.cell_of(resp.pos))
            if cur_room is not None:
                cand = [o for o in self.occupants if (not o.evacuated) and o.attached_to is None and cur_room.contains(self.cell_of(o.pos))]
                if cand:
                    tgt = min(cand, key=lambda o: np.hypot(o.pos[0]-resp.pos[0], o.pos[1]-resp.pos[1]))
                    resp.target = self.cell_of(tgt.pos)
                    continue
            # otherwise, head to the nearest room-side entry cell of a not-yet-self-cleared room (by reachable cost)
            if my_todo:
                pos_cell = self.cell_of(resp.pos)
                best = None
                best_cost = None
                for room in my_todo:
                    # room-side entry inside the room
                    entry = self.room_door_corridor_cell(room)
                    ex, ez = entry
                    if ez == self.top_z - 1:
                        entry = (ex, self.top_z)
                    elif ez == self.bot_z + 1:
                        entry = (ex, self.bot_z)
                    D = self.distance_field([entry])
                    if not self.cell_in_bounds(pos_cell):
                        continue
                    cost = D[pos_cell[1]-self.z_min, pos_cell[0]-self.x_min]
                    if not np.isfinite(cost):
                        continue
                    if best is None or cost < best_cost - 1e-6:
                        best = entry
                        best_cost = cost
                if best is None:
                    # fallback to straight-line heuristic if no reachable entry found
                    def _cand(r):
                        e = self.room_door_corridor_cell(r)
                        ex, ez = e
                        if ez == self.top_z - 1:
                            e = (ex, self.top_z)
                        elif ez == self.bot_z + 1:
                            e = (ex, self.bot_z)
                        return e
                    best = min(((_cand(r), r) for r in my_todo), key=lambda t: np.hypot((t[0][0]+0.5)-resp.pos[0], (t[0][1]+0.5)-resp.pos[1]), default=(None, None))[0]
                resp.target = best
            else:
                # all rooms self-cleared -> return to nearest exit
                cell = self.cell_of(resp.pos)
                resp.target = self.nearest_exit(cell)

    # -------- simulation step --------
    def step(self):
        self.t += 1
        # 1) assign targets as needed
        self.assign_next_targets()

        # 2) responders move one step toward target (8-neighbor discrete path first, continuous fallback)
        for resp in self.responders:
            # If previously escorting to exit but no longer attached, clear exit target ONLY if there are rooms left for this responder
            my_todo_now = [r for r in self.rooms if not self.cleared_by[resp.id][r.id]]
            if (not self.attached_occupants(resp.id)) and resp.target in self.exits and my_todo_now:
                resp.target = None
            # If target is a corridor-side door cell and we've reached it, retarget to the room-side entry cell
            if resp.target is not None:
                tx, tz = resp.target
                cx, cz = self.cell_of(resp.pos)
                if (cx, cz) == (tx, tz):
                    if tz == self.top_z - 1:
                        resp.target = (tx, self.top_z)
                    elif tz == self.bot_z + 1:
                        resp.target = (tx, self.bot_z)
            goals: List[Coord] = []
            if self.attached_occupants(resp.id):
                goals = self.exits
            elif resp.target is not None:
                goals = [resp.target]
            if goals:
                start_cell = self.cell_of(resp.pos)
                # choose a single goal for discrete path (nearest by Euclidean)
                if len(goals) > 1:
                    goal = min(goals, key=lambda g: np.hypot(g[0]-start_cell[0], g[1]-start_cell[1]))
                else:
                    goal = goals[0]
                # try discrete step first
                path = self.shortest_path(start_cell, goal)
                if path and len(path) > 1:
                    nx, nz = path[1]
                    resp.pos = (nx + 0.5, nz + 0.5)
                else:
                    # fallback: continuous step along distance field
                    D = self.distance_field([goal])
                    resp.pos = self.step_along_field(resp.pos, D, step=1.0)

            # stagnation guard: if cell didn't change for many steps and not escorting, retarget next best entry
            current_cell = self.cell_of(resp.pos)
            last = self._resp_last_cell.get(resp.id)
            if last == current_cell:
                self._resp_stagnant[resp.id] += 1
            else:
                self._resp_stagnant[resp.id] = 0
                self._resp_last_cell[resp.id] = current_cell
            if self._resp_stagnant[resp.id] >= 30 and not self.attached_occupants(resp.id):
                pos_cell = current_cell
                cands = self._candidate_entries_for(resp.id, pos_cell)
                # pick second best if available
                if cands:
                    chosen = cands[1][0] if len(cands) > 1 else cands[0][0]
                    resp.target = chosen
                    self._resp_stagnant[resp.id] = 0

            # check room first entry (by current cell)
            room = self.room_at(self.cell_of(resp.pos))
            if room is not None and self.room_first_entry[room.id] is None:
                self.room_first_entry[room.id] = self.t
            # if inside a room and no occupants currently inside, mark self-cleared for this responder
            if room is not None:
                if not any((not o.evacuated) and room.contains(self.cell_of(o.pos)) for o in self.occupants):
                    rid = resp.id
                    if not self.cleared_by[rid][room.id]:
                        self.cleared_by[rid][room.id] = True
                        self.clear_order[rid].append(room.id)
                        # If all rooms are now self-cleared, immediately set target to nearest exit
                        if all(self.cleared_by[rid].values()):
                            cell = self.cell_of(resp.pos)
                            resp.target = self.nearest_exit(cell)

            # attach only when responder reaches occupant cell AND responder is inside a room
            for o in self.occupants:
                if (not o.evacuated) and o.attached_to is None:
                    if self.room_at(self.cell_of(resp.pos)) is not None and self.cell_of(o.pos) == self.cell_of(resp.pos):
                        o.attached_to = resp.id
            # if escorting and currently in a room, keep target towards nearest exit
            if room is not None and self.attached_occupants(resp.id):
                cell = self.cell_of(resp.pos)
                resp.target = self.nearest_exit(cell)

        # 3) occupants follow: when attached, move toward their escorting responder; no self-evac without escort
        for o in self.occupants:
            if o.evacuated or o.attached_to is None:
                continue
            # follow the current responder position (cell target) using discrete 8-neighbor path first
            leader = next((r for r in self.responders if r.id == o.attached_to), None)
            if leader is not None:
                tgt_cell = self.cell_of(leader.pos)
                start_cell = self.cell_of(o.pos)
                path = self.shortest_path(start_cell, tgt_cell)
                if path and len(path) > 1:
                    nx, nz = path[1]
                    o.pos = (nx + 0.5, nz + 0.5)
                else:
                    D = self.distance_field([tgt_cell])
                    o.pos = self.step_along_field(o.pos, D, step=1.0)
            # check evacuated
            if self.cell_of(o.pos) in self.exits:
                o.evacuated = True
                o.evac_time = self.t
                o.attached_to = None

        # escort-active flag for timing: true if any responder is escorting now
        self._escort_active = any(bool(self.attached_occupants(r.id)) for r in self.responders)

        # 4) update swept flags: a room is swept when all its occupants have evacuated
        for room in self.rooms:
            if not self.room_swept[room.id]:
                if all(o.evacuated for o in self.occupants if o.origin_room == room.id):
                    self.room_swept[room.id] = True

    def snapshot(self) -> dict:
        """Legacy-style frame compatible with animate_sweep/visualize_heatmap.
        Uses 'responders' and 'occupants' arrays with x,y coordinates (y=z).
        """
        unswept = sum(1 for r in self.rooms if not self.room_swept[r.id])
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
            "unswept": unswept,
        }

    def step_towards(self, current: Coord, target: Coord) -> Optional[Coord]:
        # legacy helper; prefer 8-neighbor euclidean greedy
        if current == target:
            return current
        cx, cz = current
        best = None
        bestd = None
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                q = (cx+dx, cz+dz)
                if not self.valid_step(current, q):
                    continue
                d = np.hypot(q[0]-target[0], q[1]-target[1])
                if best is None or d < bestd:
                    best, bestd = q, d
        return best

    def nearest_exit(self, pos: Coord) -> Coord:
        best = None
        bestd = None
        for e in self.exits:
            d = np.hypot(e[0]-pos[0], e[1]-pos[1])
            if best is None or d < bestd:
                best = e; bestd = d
        return best

    def all_evacuated(self) -> bool:
        # episode ends only when everyone evacuated AND
        # each responder has self-cleared all rooms AND both are at exits
        everyone_out = all(o.evacuated for o in self.occupants)
        both_cleared = all(all(self.cleared_by[r.id].values()) for r in self.responders)
        at_exits = all(self.cell_of(r.pos) in self.exits for r in self.responders)
        return bool(everyone_out and both_cleared and at_exits)

    def room_order(self) -> List[str]:
        pairs = [(rid, t) for rid, t in self.room_first_entry.items() if t is not None]
        pairs.sort(key=lambda x: x[1])
        order = []
        for rid, _ in pairs:
            # unify to 1-based labels: R0->R1, R5->R6
            if isinstance(rid, str) and rid.startswith('R'):
                try:
                    k = int(rid[1:]) + 1
                    order.append(f"R{k}")
                except Exception:
                    order.append(rid)
            else:
                order.append(rid)
        return order


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


def run_once(layout_path: str,
             num_responders: int = 2,
             per_room: int = 5,
             max_steps: int = 3000,
             seed: int = 0,
             init_positions: Optional[List[Coord]] = None,
             frames_path: Optional[str] = None,
             delay: float = 0.0,
             log_every: int = 50,
             cell_m: float = 0.5,
             speed_solo: float = 0.8,
             speed_escort: float = 0.6):
    layout = load_layout(layout_path)
    resp_init = init_positions if init_positions is not None else default_init_positions(layout, num_responders)
    sim = SweepSim(layout, resp_init, per_room=per_room, seed=seed)
    f = None
    try:
        if frames_path:
            os.makedirs(os.path.dirname(frames_path), exist_ok=True)
            f = open(frames_path, "w", encoding="utf-8")
            print(f"[det] Writing frames to {frames_path} ...", flush=True)
        # initial snapshot (t=0)
        if f:
            snap0 = sim.snapshot()
            snap0["eta_seconds"] = 0.0
            snap0["eta_hms"] = _sec_to_hms(0.0)
            f.write(json.dumps(snap0, ensure_ascii=False) + "\n")
        print(f"[det] Simulation start: responders={num_responders}, per_room={per_room}, max_steps={max_steps}, seed={seed}", flush=True)
        # real-time accumulator (seconds)
        real_seconds = 0.0
        while sim.t < max_steps and not sim.all_evacuated():
            sim.step()
            if f:
                snap = sim.snapshot()
                snap["eta_seconds"] = real_seconds
                snap["eta_hms"] = _sec_to_hms(real_seconds)
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
            # map one sim step to real seconds based on escort activity
            dt = (cell_m / (speed_escort if sim._escort_active else speed_solo))
            real_seconds += dt
            if sim.t % max(1, log_every) == 0:
                evac = sum(1 for o in sim.occupants if o.evacuated)
                unswept = sum(1 for r in sim.rooms if not sim.room_swept[r.id])
                rpos = ", ".join([f"({r.pos[0]:.2f},{r.pos[1]:.2f})" for r in sim.responders])
                print(f"[det] t={sim.t:4d} | evac={evac}/{len(sim.occupants)} | unswept_rooms={unswept} | responders={rpos}", flush=True)
            if delay > 0:
                time.sleep(delay)
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
        "real_hms": _sec_to_hms(real_seconds),
        "real_minutes": round(real_seconds / 60.0, 2),
        "cell_m": cell_m,
        "speed_solo_mps": speed_solo,
        "speed_escort_mps": speed_escort,
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
    ap.add_argument("--delay", type=float, default=0.0, help="Per-step delay in seconds to make logs visible (e.g., 0.01)")
    ap.add_argument("--log-every", type=int, default=50, help="Print progress every N steps")
    ap.add_argument("--cell-m", type=float, default=0.5, help="Grid cell size in meters (default 0.5m)")
    ap.add_argument("--speed-solo", type=float, default=0.8, help="Responder speed without escort (m/s)")
    ap.add_argument("--speed-escort", type=float, default=0.6, help="Responder speed when escorting (m/s)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    result = run_once(
        layout_path=args.layout,
        num_responders=args.responders,
        per_room=args.per_room,
        max_steps=args.max_steps,
        seed=args.seed,
        init_positions=None,
        frames_path=args.frames,
        delay=args.delay,
        log_every=args.log_every,
        cell_m=args.cell_m,
        speed_solo=args.speed_solo,
        speed_escort=args.speed_escort,
    )
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[det] Simulation completed.", flush=True)
    print(json.dumps(result, ensure_ascii=False))
    print(f"[det] ≈ Real time: {result['real_hms']} (cell={result['cell_m']}m, v_solo={result['speed_solo_mps']} m/s, v_escort={result['speed_escort_mps']} m/s)")


if __name__ == "__main__":
    main()

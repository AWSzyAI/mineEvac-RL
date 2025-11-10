#!/usr/bin/env python3
"""Gymnasium-style RL environment for 1 responder on MineEvac-like grid.

This mirrors the deterministic escort rules:
- Occupants randomly distributed inside rooms; no self-evac.
- Responder must step onto an occupant cell to attach (escort).
- Attached occupants move toward their room doorway and then to nearest exit.

Action space: Discrete(5) -> [stay, +x, -x, +z, -z]
Observation: compact vector with position, progress, and simple guidance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

Coord = Tuple[int, int]


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


class SingleResponderEscortEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, layout_path: str, per_room: int = 5, max_steps: int = 500, seed: int = 0):
        super().__init__()
        self.layout_path = layout_path
        self.layout = load_layout(layout_path)
        self.per_room = per_room
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        corr = self.layout["corridor"]
        self.corr_x_min = corr["x"]; self.corr_x_max = corr["x"] + corr["w"] - 1
        self.corr_z_min = corr["z"]; self.corr_z_max = corr["z"] + corr["h"] - 1
        doors = self.layout.get("doors", {})
        self.door_xs: Set[int] = set(doors.get("xs", []))
        self.top_z = int(doors.get("topZ", self.corr_z_max + 1))
        self.bot_z = int(doors.get("bottomZ", self.corr_z_min - 1))

        self.rooms: List[Room] = []
        idx = 0
        for key in ("rooms_top", "rooms_bottom"):
            for r in self.layout.get(key, []):
                x1 = r["x"]; z1 = r["z"]; x2 = x1 + r["w"] - 1; z2 = z1 + r["h"] - 1
                self.rooms.append(Room(id=f"R{idx}", x1=x1, z1=z1, x2=x2, z2=z2))
                idx += 1

        frame = self.layout["frame"]
        mid_z = self.corr_z_min + (self.corr_z_max - self.corr_z_min) // 2
        self.exits = [(frame["x1"], mid_z), (frame["x2"], mid_z)]

        # wall cells: room perimeters; leave door openings on corridor-facing walls
        self.wall_cells: Set[Coord] = self._build_wall_cells()

        # action/obs
        self.action_space = spaces.Discrete(5)
        # obs: [x_norm, z_norm, evac_ratio, carrying, t_norm, door_dx, door_dz, in_room, occ_dx, occ_dz]
        self.obs_dim = 10
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    # -------- geometry --------
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
        if {az, bz} == {self.top_z - 1, self.top_z} and ax == bx and ax in self.door_xs:
            return True
        if {az, bz} == {self.bot_z + 1, self.bot_z} and ax == bx and ax in self.door_xs:
            return True
        return False

    def valid_step(self, a: Coord, b: Coord) -> bool:
        ax, az = a; bx, bz = b
        if abs(ax - bx) + abs(az - bz) != 1:
            return False
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

    def nearest_exit(self, pos: Coord) -> Coord:
        best = None; bestd = None
        for e in self.exits:
            d = abs(e[0]-pos[0]) + abs(e[1]-pos[1])
            if best is None or d < bestd:
                best = e; bestd = d
        return best

    def room_door_corridor_cell(self, room: Room) -> Coord:
        cx, cz = room.center
        if cz >= self.corr_z_max + 1:
            x = min(self.door_xs, key=lambda dx: abs(dx - cx))
            return (x, self.top_z - 1)
        else:
            x = min(self.door_xs, key=lambda dx: abs(dx - cx))
            return (x, self.bot_z + 1)

    # -------- env API --------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        # init responder at left corridor near mid
        mid_z = self.corr_z_min + (self.corr_z_max - self.corr_z_min)//2
        self.responder: Coord = (self.corr_x_min + 1, mid_z)
        # occupants randomly in rooms
        self.occupants: List[Dict] = []
        oid = 0
        for room in self.rooms:
            for _ in range(self.per_room):
                for _tries in range(100):
                    rx = int(self.rng.integers(low=room.x1, high=room.x2 + 1))
                    rz = int(self.rng.integers(low=room.z1, high=room.z2 + 1))
                    if (rx, rz) not in self.wall_cells:
                        break
                self.occupants.append({
                    "id": oid, "pos": (rx, rz), "origin": room.id,
                    "evac": False, "attached": False
                })
                oid += 1
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        # apply movement
        x, z = self.responder
        if action == 1: x += 1
        elif action == 2: x -= 1
        elif action == 3: z += 1
        elif action == 4: z -= 1
        new_pos = (x, z)
        moved = False
        if self.valid_step(self.responder, new_pos):
            if new_pos != self.responder:
                self.responder = new_pos
                moved = True

        # attach occupant if overlapping
        for o in self.occupants:
            if not o["evac"] and not o["attached"] and o["pos"] == self.responder:
                o["attached"] = True

        # move attached occupants
        for o in self.occupants:
            if o["evac"] or not o["attached"]:
                continue
            pos = o["pos"]
            r = self._room_at(pos)
            if r is not None:
                target = self.room_door_corridor_cell(r)
            else:
                target = self.nearest_exit(pos)
            step = self._step_towards(pos, target)
            if step and self.valid_step(pos, step):
                o["pos"] = step
            if o["pos"] in self.exits:
                o["evac"] = True
                o["attached"] = False

        # done flags
        terminated = all(o["evac"] for o in self.occupants)
        truncated = self.t >= self.max_steps

        # rewards
        reward = 0.0
        reward -= 0.2  # time penalty
        if moved:
            reward += 0.02
        # small per-attach
        reward += 0.5 * sum(1 for o in self.occupants if o["attached"] and o["pos"] == self.responder)
        # evac bonus
        ev_count = sum(1 for o in self.occupants if o["evac"])  # total evac
        reward += 0.0  # could add shaping per-step using delta, omitted for simplicity
        if terminated:
            reward += 200.0
        if truncated and not terminated:
            reward -= 200.0

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        x_min, x_max = self.corr_x_min, self.corr_x_max
        z_min, z_max = self.corr_z_min, self.corr_z_max
        span_x = max(1, x_max - x_min)
        span_z = max(1, z_max - z_min)
        x_norm = (self.responder[0] - x_min) / span_x * 2 - 1
        z_norm = (self.responder[1] - z_min) / span_z * 2 - 1
        evac_ratio = sum(1 for o in self.occupants if o["evac"]) / max(1, len(self.occupants))
        carrying = 1.0 if any(o["attached"] for o in self.occupants) else 0.0
        t_norm = min(1.0, self.t / self.max_steps) * 2 - 1
        # door vector to nearest door pos (both corridor-side positions)
        door_cells = [(x, self.top_z - 1) for x in self.door_xs] + [(x, self.bot_z + 1) for x in self.door_xs]
        if door_cells:
            d = min(door_cells, key=lambda d: abs(d[0]-self.responder[0]) + abs(d[1]-self.responder[1]))
            door_dx = (d[0] - self.responder[0]) / span_x
            door_dz = (d[1] - self.responder[1]) / span_z
        else:
            door_dx = door_dz = 0.0
        in_room = 1.0 if self._room_at(self.responder) is not None else 0.0
        # nearest unattached occupant vector (from resp), else zeros
        occs = [o for o in self.occupants if not o["evac"] and not o["attached"]]
        if occs:
            target = min(occs, key=lambda o: abs(o["pos"][0]-self.responder[0]) + abs(o["pos"][1]-self.responder[1]))
            occ_dx = (target["pos"][0] - self.responder[0]) / span_x
            occ_dz = (target["pos"][1] - self.responder[1]) / span_z
        else:
            occ_dx = occ_dz = 0.0
        vec = np.array([x_norm, z_norm, evac_ratio*2-1, carrying, t_norm, door_dx, door_dz, in_room, occ_dx, occ_dz], dtype=np.float32)
        return vec

    # helpers
    def _room_at(self, p: Coord) -> Optional[Room]:
        return self.room_at(p)

    def _step_towards(self, current: Coord, target: Coord) -> Optional[Coord]:
        if current == target:
            return current
        x, z = current
        neigh = [(x+1,z), (x-1,z), (x,z+1), (x,z-1)]
        neigh = [q for q in neigh if self.valid_step(current, q)]
        if not neigh:
            return None
        base = abs(x-target[0]) + abs(z-target[1])
        neigh.sort(key=lambda q: abs(q[0]-target[0]) + abs(q[1]-target[1]))
        for q in neigh:
            if abs(q[0]-target[0]) + abs(q[1]-target[1]) <= base:
                return q
        return neigh[0]

    def _build_wall_cells(self) -> Set[Coord]:
        walls: Set[Coord] = set()
        for room in self.rooms:
            # horizontal edges
            for x in range(room.x1, room.x2 + 1):
                # top
                if room.z1 == self.top_z and x in self.door_xs:
                    pass
                else:
                    walls.add((x, room.z1))
                # bottom
                if room.z2 == self.bot_z and x in self.door_xs:
                    pass
                else:
                    walls.add((x, room.z2))
            # vertical edges
            for z in range(room.z1, room.z2 + 1):
                walls.add((room.x1, z))
                walls.add((room.x2, z))
        return walls

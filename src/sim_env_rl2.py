#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

Coord = Tuple[int, int]


def load_layout(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
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
        return ((self.x1 + self.x2)//2, (self.z1 + self.z2)//2)


class TwoResponderEscortEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, layout_path: str, per_room: int = 5, max_steps: int = 500, seed: int = 0):
        super().__init__()
        self.layout = load_layout(layout_path)
        self.per_room = per_room
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        corr = self.layout['corridor']
        self.corr_x_min = corr['x']; self.corr_x_max = corr['x'] + corr['w'] - 1
        self.corr_z_min = corr['z']; self.corr_z_max = corr['z'] + corr['h'] - 1
        doors = self.layout.get('doors', {})
        self.door_xs: Set[int] = set(doors.get('xs', []))
        self.top_z = int(doors.get('topZ', self.corr_z_max + 1))
        self.bot_z = int(doors.get('bottomZ', self.corr_z_min - 1))

        self.rooms: List[Room] = []
        idx = 0
        for key in ('rooms_top', 'rooms_bottom'):
            for r in self.layout.get(key, []):
                x1 = r['x']; z1 = r['z']; x2 = x1 + r['w'] - 1; z2 = z1 + r['h'] - 1
                self.rooms.append(Room(id=f'R{idx}', x1=x1, z1=z1, x2=x2, z2=z2)); idx += 1
        frame = self.layout['frame']
        mid_z = self.corr_z_min + (self.corr_z_max - self.corr_z_min)//2
        self.exits = [(frame['x1'], mid_z), (frame['x2'], mid_z)]

        self.wall_cells: Set[Coord] = self._build_wall_cells()

        # actions: each responder has Discrete(5)
        self.action_space = spaces.MultiDiscrete([5, 5])
        # obs: [r1 x,z, r2 x,z, evac_ratio, t_norm, carry1, carry2, occ_dx1, occ_dz1, occ_dx2, occ_dz2]
        self.obs_dim = 12
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        mid_z = self.corr_z_min + (self.corr_z_max - self.corr_z_min)//2
        self.r1: Coord = (self.corr_x_min + 1, mid_z)
        self.r2: Coord = (self.corr_x_max - 1, mid_z)
        self.occupants: List[Dict] = []
        oid = 0
        for room in self.rooms:
            for _ in range(self.per_room):
                for _tries in range(100):
                    rx = int(self.rng.integers(low=room.x1, high=room.x2 + 1))
                    rz = int(self.rng.integers(low=room.z1, high=room.z2 + 1))
                    if (rx, rz) not in self.wall_cells:
                        break
                self.occupants.append({ 'id': oid, 'pos': (rx, rz), 'origin': room.id, 'evac': False, 'attached': -1 })
                oid += 1
        return self._obs(), {}

    # geometry
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
        in_corr_a = self.in_corridor(a); in_corr_b = self.in_corridor(b)
        in_room_a = self.room_at(a) is not None; in_room_b = self.room_at(b) is not None
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
        best=None; bestd=None
        for e in self.exits:
            d = abs(e[0]-pos[0]) + abs(e[1]-pos[1])
            if best is None or d < bestd:
                best=e; bestd=d
        return best
    def room_door_corridor_cell(self, room: Room) -> Coord:
        cx, cz = room.center
        if cz >= self.corr_z_max + 1:
            x = min(self.door_xs, key=lambda dx: abs(dx - cx))
            return (x, self.top_z - 1)
        else:
            x = min(self.door_xs, key=lambda dx: abs(dx - cx))
            return (x, self.bot_z + 1)

    # step
    def step(self, action: np.ndarray):
        self.t += 1
        a1, a2 = int(action[0]), int(action[1])
        self.r1 = self._apply_move(self.r1, a1)
        self.r2 = self._apply_move(self.r2, a2)
        # attach when overlap
        for o in self.occupants:
            if not o['evac'] and o['attached'] == -1:
                if o['pos'] == self.r1: o['attached'] = 0
                elif o['pos'] == self.r2: o['attached'] = 1
        # move attached
        for o in self.occupants:
            if o['evac'] or o['attached'] == -1:
                continue
            pos = o['pos']
            r = self.room_at(pos)
            if r is not None:
                target = self.room_door_corridor_cell(r)
            else:
                target = self.nearest_exit(pos)
            step = self._step_towards(pos, target)
            if step and self.valid_step(pos, step):
                o['pos'] = step
            if o['pos'] in self.exits:
                o['evac'] = True
                o['attached'] = -1
        terminated = all(o['evac'] for o in self.occupants)
        truncated = self.t >= self.max_steps
        reward = -0.3
        reward += 0.02 * sum(1 for mv in (a1, a2) if mv != 0)
        if terminated: reward += 200.0
        if truncated and not terminated: reward -= 200.0
        return self._obs(), reward, terminated, truncated, {}

    def _apply_move(self, p: Coord, a: int) -> Coord:
        x, z = p
        if a == 1: x += 1
        elif a == 2: x -= 1
        elif a == 3: z += 1
        elif a == 4: z -= 1
        q = (x, z)
        if self.valid_step(p, q):
            return q
        return p

    def _obs(self) -> np.ndarray:
        x_min, x_max = self.corr_x_min, self.corr_x_max
        z_min, z_max = self.corr_z_min, self.corr_z_max
        span_x = max(1, x_max - x_min); span_z = max(1, z_max - z_min)
        def norm(p: Coord):
            return ((p[0]-x_min)/span_x*2-1, (p[1]-z_min)/span_z*2-1)
        r1x, r1z = norm(self.r1)
        r2x, r2z = norm(self.r2)
        evac_ratio = sum(1 for o in self.occupants if o['evac'])/max(1,len(self.occupants))
        t_norm = min(1.0, self.t/self.max_steps)*2-1
        carry1 = 1.0 if any((not o['evac']) and o['attached']==0 for o in self.occupants) else 0.0
        carry2 = 1.0 if any((not o['evac']) and o['attached']==1 for o in self.occupants) else 0.0
        # nearest unattached for each responder
        def occ_vec(p: Coord):
            occs = [o for o in self.occupants if not o['evac'] and o['attached']==-1]
            if not occs:
                return 0.0, 0.0
            tgt = min(occs, key=lambda o: abs(o['pos'][0]-p[0]) + abs(o['pos'][1]-p[1]))
            return ((tgt['pos'][0]-p[0])/span_x, (tgt['pos'][1]-p[1])/span_z)
        o1dx, o1dz = occ_vec(self.r1)
        o2dx, o2dz = occ_vec(self.r2)
        vec = np.array([r1x, r1z, r2x, r2z, evac_ratio*2-1, t_norm, carry1, carry2, o1dx, o1dz, o2dx, o2dz], dtype=np.float32)
        return vec

    def _step_towards(self, current: Coord, target: Coord) -> Optional[Coord]:
        if current == target:
            return current
        x, z = current
        neigh = [(x+1,z),(x-1,z),(x,z+1),(x,z-1)]
        neigh = [q for q in neigh if self.valid_step(current,q)]
        if not neigh:
            return None
        base = abs(x-target[0])+abs(z-target[1])
        neigh.sort(key=lambda q: abs(q[0]-target[0])+abs(q[1]-target[1]))
        for q in neigh:
            if abs(q[0]-target[0])+abs(q[1]-target[1]) <= base:
                return q
        return neigh[0]

    def _build_wall_cells(self) -> Set[Coord]:
        walls: Set[Coord] = set()
        for room in self.rooms:
            for x in range(room.x1, room.x2+1):
                if room.z1 == self.top_z and x in self.door_xs:
                    pass
                else:
                    walls.add((x, room.z1))
                if room.z2 == self.bot_z and x in self.door_xs:
                    pass
                else:
                    walls.add((x, room.z2))
            for z in range(room.z1, room.z2+1):
                walls.add((room.x1, z))
                walls.add((room.x2, z))
        return walls


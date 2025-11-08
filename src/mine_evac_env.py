# filename: src/mine_evac_env.py
# 一个用于强化学习的 MineEvac 环境（Gymnasium 规范）
#
# 功能：
#   - 从 layout JSON (baseline.json 等) 加载布局
#   - 在抽象网格上模拟：
#       * 单个 responder 的移动（由 RL 控制）
#       * occupants 朝出口移动
#   - 在每一步计算奖励 & 终止条件
#   - 输出观测向量 obs，供 PPO / DQN 使用

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

Coord2D = Tuple[int, int]  # (x, z) coordinate


# ========= 基础数据结构 =========

@dataclass
class Room:
    id: str
    x1: int # 左下角
    z1: int # 左下角
    x2: int # 右上角
    z2: int # 右上角

    @property
    def center(self) -> Coord2D:
        cx = (self.x1 + self.x2) // 2
        cz = (self.z1 + self.z2) // 2
        return (cx, cz)

    def contains(self, p: Coord2D) -> bool: 
        """
        点p是否在房间内
        """
        x, z = p
        return self.x1 <= x <= self.x2 and self.z1 <= z <= self.z2


@dataclass
class Exit:
    id: str
    position: Coord2D


@dataclass
class Layout:
    name: str
    rooms: List[Room]
    exits: List[Exit]
    corridor_x_range: Tuple[int, int]
    corridor_z_range: Tuple[int, int]
    # 门洞：顶/底墙的房间侧 z 坐标，以及门所在的 x 列表
    # 例如 baseline.json 中 topZ=24, bottomZ=15, xs=[23,53,83]
    doors_top_z: int = 0
    doors_bottom_z: int = 0
    doors_xs: List[int] = None


@dataclass
class Occupant:
    id: int
    position: Coord2D
    target_exit: Exit
    evacuated: bool = False
    evac_time: Optional[int] = None
    self_evac: bool = True
    needs_assist: bool = False
    attached: bool = False


@dataclass
class Responder:
    id: int
    position: Coord2D
    init_position: Coord2D
    carrying: Optional[int] = None


# ========= 工具函数：加载 layout & 基础移动 =========

def load_layout_from_json(path: str) -> Tuple[Layout, dict]:
    """
    从 JSON (baseline / layout_1 / layout_2) 加载 Layout。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data["name"]

    # 走廊
    corridor = data["corridor"]
    cx, cz, cw, ch = corridor["x"], corridor["z"], corridor["w"], corridor["h"]
    corr_x_min = cx
    corr_x_max = cx + cw - 1
    corr_z_min = cz
    corr_z_max = cz + ch - 1

    corridor_x_range = (corr_x_min, corr_x_max)
    corridor_z_range = (corr_z_min, corr_z_max)

    # 房间（上 + 下）
    rooms: List[Room] = []
    room_id = 0

    def add_room_list(room_list):
        nonlocal room_id
        for r in room_list:
            x1 = r["x"]
            z1 = r["z"]
            x2 = x1 + r["w"] - 1
            z2 = z1 + r["h"] - 1
            rooms.append(Room(id=f"R{room_id}", x1=x1, z1=z1, x2=x2, z2=z2))
            room_id += 1

    add_room_list(data["rooms_top"])
    add_room_list(data["rooms_bottom"])

    # 出口：在 frame 两端、走廊中线
    frame = data["frame"]
    mid_z = cz + ch // 2
    exits = [
        Exit(id="E_left", position=(frame["x1"], mid_z)),
        Exit(id="E_right", position=(frame["x2"], mid_z)),
    ]

    doors = data.get("doors", {})
    doors_top_z = int(doors.get("topZ", cz + ch))
    doors_bottom_z = int(doors.get("bottomZ", cz - 1))
    doors_xs = list(doors.get("xs", []))

    layout = Layout(
        name=name,
        rooms=rooms,
        exits=exits,
        corridor_x_range=corridor_x_range,
        corridor_z_range=corridor_z_range,
        doors_top_z=doors_top_z,
        doors_bottom_z=doors_bottom_z,
        doors_xs=doors_xs,
    )
    return layout, data   # 把原始 JSON 一起返回（后面用 per_room）


def manhattan_step_towards(current: Coord2D, target: Coord2D) -> Coord2D:
    x, z = current
    tx, tz = target
    if x < tx:
        return (x + 1, z)
    if x > tx:
        return (x - 1, z)
    if z < tz:
        return (x, z + 1)
    if z > tz:
        return (x, z - 1)
    return (x, z)


def nearest_exit(layout: Layout, pos: Coord2D) -> Exit:
    x, z = pos
    best_e = None
    best_dist = None
    for e in layout.exits:
        ex, ez = e.position
        d = abs(ex - x) + abs(ez - z)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_e = e
    return best_e


def init_occupants(layout: Layout, per_room: int) -> List[Occupant]:
    occupants: List[Occupant] = []
    oid = 0
    for room in layout.rooms:
        cx, cz = room.center
        for k in range(per_room):
            # 稍微扰动一下位置
            pos = (cx, cz + (k % 2))
            e_star = nearest_exit(layout, pos)
            occupants.append(
                Occupant(
                    id=oid,
                    position=pos,
                    target_exit=e_star,
                    evacuated=False,
                    evac_time=None,
                )
            )
            oid += 1
    return occupants


def init_single_responder(layout: Layout) -> Responder:
    """
    baseline: 在走廊左端中线放一个 responder。
    后面你可以改成 2 个、或者从左右出口分别出发。
    """
    x_min, x_max = layout.corridor_x_range
    z_min, z_max = layout.corridor_z_range
    mid_z = (z_min + z_max) // 2
    init_pos = (x_min + 1, mid_z)
    return Responder(
        id=0,
        position=init_pos,
        init_position=init_pos,
    )


# ========= MineEvacEnv：Gymnasium 环境 =========

class MineEvacEnv(gym.Env):
    """
    单 responder 的 MineEvac 强化学习环境。

    - 动作空间：Discrete(5) [stay, +x, -x, +z, -z]
    - 状态：
        [x_norm, z_norm, evac_ratio, t_norm, visited_flags...]
    - 条件：
        * occupants 每步向最近出口曼哈顿移动
        * 房间第一次被进入时记录 τ(v)
        * 全部撤离且所有房间访问 -> terminated=True
        * 时间步 >= max_steps -> truncated=True
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, layout_path: str, max_steps: int = 500):
        super().__init__()

        self.layout_path = layout_path
        self.layout, self.layout_json = load_layout_from_json(layout_path)

        self.max_steps = max_steps
        self.t = 0

        # 状态变量
        self.responder: Optional[Responder] = None
        self.occupants: List[Occupant] = []
        self.tau: Dict[str, Optional[int]] = {room.id: None for room in self.layout.rooms}

        # 可见度设置（烟雾随时间变浓，可见半径递减）
        self.vis_max = 10
        self.vis_min = 3
        self.vis_decay = max(0.0, (self.vis_max - self.vis_min) / max(1, self.max_steps))

        # ----- 动作空间 -----
        # 0: stay, 1:+x, 2:-x, 3:+z, 4:-z
        self.action_space = spaces.Discrete(5)

        # ----- 观测空间 -----
        # obs = [x_norm, z_norm, evac_ratio, t_norm, visited_flags(N)]
        self.num_rooms = len(self.layout.rooms)
        obs_dim = 4 + self.num_rooms
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------ 必须实现：reset ------------

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self.t = 0
        # 初始化 responder
        self.responder = init_single_responder(self.layout)

        # 初始化 occupants
        occ_cfg = self.layout_json.get("occupants", {})
        per_room = occ_cfg.get("per_room", 5)
        self.occupants = init_occupants(self.layout, per_room=per_room)
        # 随机划分自发撤离 vs 需要救援
        ratio = float(occ_cfg.get("self_evac_ratio", 0.4))
        rng = np.random.default_rng()
        mask = rng.random(len(self.occupants)) < ratio
        for i, o in enumerate(self.occupants):
            o.self_evac = bool(mask[i])
            o.needs_assist = not o.self_evac
            o.attached = False

        # τ(v) 重置
        self.tau = {room.id: None for room in self.layout.rooms}

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------ 必须实现：step ------------

    def step(self, action: int):
        assert self.responder is not None

        self.t += 1

        # 1. 应用 responder 动作
        self._apply_action(action)

        # 2. 更新 τ(v)：第一次进入房间的时间
        for room in self.layout.rooms:
            if self.tau[room.id] is not None:
                continue
            if room.contains(self.responder.position):
                self.tau[room.id] = self.t

        # 3. occupants 行为（自主撤离 + 救援模式）
        evac_before = sum(1 for o in self.occupants if o.evacuated)

        # 3.1 responder 尝试附着最近的等待救援者（视距内）
        self._try_attach_needy_occupant()

        # 3.2 自主撤离人：沿门洞合法路径向出口推进
        for o in self.occupants:
            if o.evacuated or not o.self_evac:
                continue
            path = self._shortest_path(o.position, o.target_exit.position)
            if path and len(path) >= 2:
                o.position = path[1]
            if o.position == o.target_exit.position:
                o.evacuated = True
                o.evac_time = self.t

        # 3.3 被救援者：若附着则按照最短路向出口前进一步；未附着保持原地等待
        for o in self.occupants:
            if o.evacuated or not o.needs_assist:
                continue
            if o.attached and self.responder.carrying == o.id:
                path2 = self._shortest_path(o.position, o.target_exit.position)
                if path2 and len(path2) >= 2:
                    o.position = path2[1]
                if o.position == o.target_exit.position:
                    o.evacuated = True
                    o.evac_time = self.t
                    o.attached = False
                    if self.responder.carrying == o.id:
                        self.responder.carrying = None
            else:
                # 等待救援，不移动
                pass
        evac_after = sum(1 for o in self.occupants if o.evacuated)
        delta_evac = evac_after - evac_before

        # 4. 计算 terminated / truncated
        terminated, truncated = self._check_done()

        # 5. 奖励
        reward = self._compute_reward(
            delta_evac=delta_evac,
            terminated=terminated,
            truncated=truncated,
        )

        obs = self._get_obs()
        info = {
            "t": self.t,
            "tau": self.tau.copy(),
            "evacuated": evac_after,
            "responder_pos": list(self.responder.position),
            "occupants": [
                {
                    "id": o.id,
                    "x": o.position[0],
                    "z": o.position[1],
                    "evacuated": o.evacuated
                } for o in self.occupants
            ],
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    # ------------ 动作实现 ------------

    def _apply_action(self, action: int):
        x, z = self.responder.position
        if action == 1:      # +x
            x += 1
        elif action == 2:    # -x
            x -= 1
        elif action == 3:    # +z
            z += 1
        elif action == 4:    # -z
            z -= 1
        # 裁剪到 frame 内并检查门洞跨越合法性
        nx, nz = self._clip_inside_layout((x, z))
        if self._is_valid_move(self.responder.position, (nx, nz)):
            self.responder.position = (nx, nz)

    # ------------ 观测构造 ------------

    def _get_obs(self) -> np.ndarray:
        x_min, x_max = self.layout.corridor_x_range
        z_min, z_max = self.layout.corridor_z_range
        x_r, z_r = self.responder.position

        # 位置归一化
        x_norm = (x_r - x_min) / max(1, (x_max - x_min))
        z_norm = (z_r - z_min) / max(1, (z_max - z_min))

        # evac_ratio
        total = len(self.occupants)
        evac = sum(1 for o in self.occupants if o.evacuated)
        evac_ratio = evac / total if total > 0 else 0.0

        # 时间归一化
        t_norm = min(1.0, self.t / self.max_steps)

        # rooms visited flags
        visited = np.array(
            [1.0 if self.tau[room.id] is not None else 0.0
             for room in self.layout.rooms],
            dtype=np.float32
        )

        obs = np.concatenate(
            [
                np.array([x_norm, z_norm, evac_ratio, t_norm], dtype=np.float32),
                visited,
            ],
            axis=0
        )
        return obs

    # ------------ 终止条件 ------------

    def _check_done(self) -> Tuple[bool, bool]:
        all_evacuated = all(o.evacuated for o in self.occupants)
        all_rooms_visited = all(v is not None for v in self.tau.values())
        terminated = bool(all_evacuated and all_rooms_visited)
        truncated = bool(self.t >= self.max_steps)
        return terminated, truncated

    # ------------ 奖励函数 ------------

    def _compute_reward(self,
                        delta_evac: int,
                        terminated: bool,
                        truncated: bool) -> float:
        # 基本时间惩罚
        reward = -1.0

        # 新撤离的人数奖励
        reward += 0.5 * delta_evac

        # 每个新访问房间的奖励
        # 统计这一步 "新变为 visited" 的房间数
        new_rooms = sum(
            1 for v in self.tau.values()
            if v == self.t  # τ(v) 刚好等于当前时间步
        )
        reward += 2.0 * new_rooms

        # 成功结束奖励
        if terminated:
            reward += 100.0

        # 超时而未成功的额外惩罚
        if truncated and not terminated:
            reward -= 20.0

        return reward

    # ------------ 简单 render ------------

    def render(self):
        evac = sum(1 for o in self.occupants if o.evacuated)
        total = len(self.occupants)
        print(
            f"t={self.t}, "
            f"responder={self.responder.position}, "
            f"evac={evac}/{total}, "
            f"rooms_visited={sum(1 for v in self.tau.values() if v is not None)}/{self.num_rooms}"
        )

    # ------------ 门洞/墙 + BFS 寻路辅助 ------------

    def _clip_inside_layout(self, p: Coord2D) -> Coord2D:
        frame = self.layout_json.get("frame")
        if not frame:
            return p
        x, z = p
        x = max(frame["x1"], min(frame["x2"], x))
        z = max(frame["z1"], min(frame["z2"], z))
        return (x, z)

    def _is_in_corridor(self, p: Coord2D) -> bool:
        x, z = p
        x_min, x_max = self.layout.corridor_x_range
        z_min, z_max = self.layout.corridor_z_range
        return x_min <= x <= x_max and z_min <= z <= z_max

    def _is_in_any_room(self, p: Coord2D) -> bool:
        return any(r.contains(p) for r in self.layout.rooms)

    def _is_door_crossing(self, a: Coord2D, b: Coord2D) -> bool:
        ax, az = a
        bx, bz = b
        doors_xs = set(self.layout.doors_xs or [])
        # 顶墙: corridor z = top_z -1 与 room z = top_z
        if {az, bz} == {self.layout.doors_top_z - 1, self.layout.doors_top_z} and ax == bx and ax in doors_xs:
            return True
        # 底墙: corridor z = bottom_z +1 与 room z = bottom_z
        if {az, bz} == {self.layout.doors_bottom_z + 1, self.layout.doors_bottom_z} and ax == bx and ax in doors_xs:
            return True
        return False

    def _is_valid_move(self, a: Coord2D, b: Coord2D) -> bool:
        ax, az = a
        bx, bz = b
        if abs(ax - bx) + abs(az - bz) != 1:
            return False
        in_corr_a = self._is_in_corridor(a)
        in_corr_b = self._is_in_corridor(b)
        in_room_a = self._is_in_any_room(a)
        in_room_b = self._is_in_any_room(b)
        # 目标必须在某有效区域
        if not (in_corr_b or in_room_b):
            return False
        # 跨区域需要门洞
        if in_corr_a and in_room_b:
            return self._is_door_crossing(a, b)
        if in_room_a and in_corr_b:
            return self._is_door_crossing(a, b)
        return True

    def _neighbors(self, p: Coord2D):
        x, z = p
        for q in [(x+1, z), (x-1, z), (x, z+1), (x, z-1)]:
            q = self._clip_inside_layout(q)
            if self._is_valid_move(p, q):
                yield q

    def _shortest_path(self, start: Coord2D, goal: Coord2D) -> Optional[List[Coord2D]]:
        if start == goal:
            return [start]
        from collections import deque
        q = deque([start])
        prev: Dict[Coord2D, Optional[Coord2D]] = {start: None}
        while q:
            u = q.popleft()
            for v in self._neighbors(u):
                if v in prev:
                    continue
                prev[v] = u
                if v == goal:
                    # reconstruct
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
        return None

    def _try_attach_needy_occupant(self):
        if self.responder is None or self.responder.carrying is not None:
            return
        vis = max(self.vis_min, self.vis_max - int(self.vis_decay * self.t))
        candidates = [o for o in self.occupants if (not o.evacuated) and o.needs_assist and (not o.attached)]
        if not candidates:
            return
        best = None
        best_d = None
        for o in candidates:
            d = abs(o.position[0]-self.responder.position[0]) + abs(o.position[1]-self.responder.position[1])
            if d > vis:
                continue
            path = self._shortest_path(self.responder.position, o.position)
            if path is None or len(path)-1 > vis:
                continue
            if best_d is None or d < best_d:
                best_d = d
                best = o
        if best is not None:
            best.attached = True
            self.responder.carrying = best.id

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
from typing import List, Tuple, Dict, Optional, Iterable, Set

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from reward_config import reward_cfg

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
    door_target_room: Optional[Coord2D] = None
    door_target_corridor: Optional[Coord2D] = None


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

    def __init__(self, layout_path: str, max_steps: int = 2000):
        super().__init__()

        self.layout_path = layout_path
        self.layout, self.layout_json = load_layout_from_json(layout_path)

        self.max_steps = max_steps
        self.t = 0

        # 状态变量
        self.responder: Optional[Responder] = None
        self.occupants: List[Occupant] = []
        self.tau: Dict[str, Optional[int]] = {room.id: None for room in self.layout.rooms}
        self.room_cleared: Dict[str, bool] = {room.id: False for room in self.layout.rooms}
        self.room_entry_rewarded: Dict[str, bool] = {room.id: False for room in self.layout.rooms}
        self.room_occupancy: Dict[str, int] = {room.id: 0 for room in self.layout.rooms}
        self.total_needy: int = 0
        self.occupant_vis_radius = 2  # 5x5 sensing window
        self.visited_cells: Set[Coord2D] = set()
        self.door_half_width = 1  # allow 3-cell wide openings for easier room entry
        self.door_x_index_map = self._build_door_index_map()
        self.door_open_xs = set(self.door_x_index_map.keys())
        (
            self.door_corr_positions,
            self.door_room_positions,
        ) = self._enumerate_door_positions()
        self.door_positions: List[Coord2D] = self.door_corr_positions + self.door_room_positions
        self.door_corr_set = set(self.door_corr_positions)
        self.door_room_set = set(self.door_room_positions)
        self.door_position_set = set(self.door_positions)
        self.wall_cells: Set[Coord2D] = self._build_wall_cells()
        self.room_interior_bounds = self._compute_room_interior_bounds()
        self.door_idle_steps = 0

        # 可见度设置（烟雾随时间变浓，可见半径递减）
        self.vis_max = 10
        self.vis_min = 3
        self.vis_decay = max(0.0, (self.vis_max - self.vis_min) / max(1, self.max_steps))
        # 记录穿越过哪些门的 x（探索记忆）
        self.doors_visited: Set[int] = set()
        self.door_cross_rewarded: Set[int] = set()

        # ----- 动作空间 -----
        # 0: stay, 1:+x, 2:-x, 3:+z, 4:-z
        self.action_space = spaces.Discrete(5)

        # ----- 观测空间 -----
        # obs = [base(8 + carrying + vis_norm=10), per_room(3)*N, visited_room_flags(N), door_flags(D)]
        self.num_rooms = len(self.layout.rooms)
        self.num_doors = len(self.layout.doors_xs)
        self.per_room_feature_dim = 3
        # base: x_norm,z_norm,evac_ratio,needs_ratio,rooms_cleared_ratio,t_norm,door_dx,door_dz,carrying_flag,vis_norm => 10
        obs_dim = 10 + self.num_rooms * self.per_room_feature_dim + self.num_rooms + self.num_doors
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
        ratio = float(occ_cfg.get("self_evac_ratio", 0.0))
        rng = np.random.default_rng()
        flags = rng.random(len(self.occupants)) < ratio
        for i, o in enumerate(self.occupants):
            o.self_evac = bool(flags[i])
            o.needs_assist = not o.self_evac
            o.attached = False
            door_pair = self._door_target_for_position(o.position)
            if door_pair:
                o.door_target_room, o.door_target_corridor = door_pair
            else:
                o.door_target_room = None
                o.door_target_corridor = None
        self.total_needy = sum(1 for o in self.occupants if o.needs_assist)

        # τ(v) 重置
        self.tau = {room.id: None for room in self.layout.rooms}
        self.room_cleared = {room.id: False for room in self.layout.rooms}
        self.room_entry_rewarded = {room.id: False for room in self.layout.rooms}
        self.room_occupancy = {room.id: 0 for room in self.layout.rooms}
        self.visited_cells = {self.responder.position}
        self.doors_visited = set()
        self.door_cross_rewarded = set()
        self.door_idle_steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------ 必须实现：step ------------

    def step(self, action: int):
        assert self.responder is not None

        self.t += 1

        needs_before = sum(1 for o in self.occupants if o.needs_assist and not o.evacuated)
        # 1. 应用 responder 动作
        prev_pos = self.responder.position
        responder_moved, new_cell_visit, door_cross_idx, invalid_bump = self._apply_action(action)
        door_crossed = door_cross_idx is not None
        door_cross_bonus = False
        if door_cross_idx is not None and door_cross_idx not in self.door_cross_rewarded:
            self.door_cross_rewarded.add(door_cross_idx)
            door_cross_bonus = True

        room_entry_events = 0
        in_uncleared_room = False
        current_room = self._room_interior_at(self.responder.position)
        if current_room:
            if not self.room_entry_rewarded[current_room.id]:
                self.room_entry_rewarded[current_room.id] = True
                room_entry_events += 1
            in_uncleared_room = not self.room_cleared[current_room.id]

        in_corridor = self._is_in_corridor(self.responder.position)
        far_from_corridor = not in_corridor

        if door_crossed:
            self.door_idle_steps = 0
        elif self._is_at_doorway(self.responder.position):
            self.door_idle_steps += 1
        else:
            self.door_idle_steps = 0
        door_idle_penalty = 1 if self.door_idle_steps >= 3 else 0

        # 2. 更新 τ(v)：第一次进入房间的时间
        for room in self.layout.rooms:
            if self.tau[room.id] is not None:
                continue
            if self._room_interior_contains(room, self.responder.position):
                self.tau[room.id] = self.t

        # 3. occupants 行为（自主撤离 + 救援模式）
        evac_before = sum(1 for o in self.occupants if o.evacuated)
        rescued_delta = 0
        self_evac_delta = 0

        # 3.1 responder 尝试附着最近的等待救援者（视距内）
        attached_this_step = self._try_attach_needy_occupant()

        # 3.2 自主撤离人：允许部分自主撤离
        for o in self.occupants:
            if o.evacuated or not o.self_evac:
                continue
            before = o.evacuated
            done = self._move_self_evac_occupant(o)
            if done and not before:
                self_evac_delta += 1

        # 3.3 被救援者：若附着则按照最短路向出口前进一步；未附着保持原地等待
        for o in self.occupants:
            if o.evacuated or not o.needs_assist:
                continue
            if o.attached and self.responder.carrying == o.id:
                # 跟随：一步朝 responder 当前位置靠近
                next_step = self._local_step_towards(o.position, self.responder.position)
                if next_step and self._is_valid_move(o.position, next_step):
                    o.position = next_step
                if o.position == o.target_exit.position:
                    o.evacuated = True
                    o.evac_time = self.t
                    o.attached = False
                    if self.responder.carrying == o.id:
                        self.responder.carrying = None
                    if o.needs_assist:
                        rescued_delta += 1
            else:
                # 等待救援，不移动
                pass
        evac_after = sum(1 for o in self.occupants if o.evacuated)
        delta_evac = evac_after - evac_before
        needs_remaining = sum(1 for o in self.occupants if o.needs_assist and not o.evacuated)
        needs_delta = max(0, needs_before - needs_remaining)

        # 4. 房间清空状态 & 终止条件
        newly_cleared_rooms, room_counts = self._update_room_clear_status()
        terminated, truncated = self._check_done()

        # 5. 奖励
        # Potential shaping: door distance and room interior distance
        door_potential_delta = 0.0
        room_potential_delta = 0.0
        if responder_moved:
            door_potential_delta = self._door_distance(prev_pos) - self._door_distance(self.responder.position)
            room_potential_delta = self._room_interior_distance(prev_pos) - self._room_interior_distance(self.responder.position)

        reward = self._compute_reward(
            delta_evac=delta_evac,
            rescued_delta=rescued_delta,
            self_evac_delta=self_evac_delta,
            new_cleared_rooms=newly_cleared_rooms,
            terminated=terminated,
            truncated=truncated,
            room_counts=room_counts,
            needs_remaining=needs_remaining,
            responder_moved=responder_moved,
            new_cell_visit=new_cell_visit,
            room_entries=room_entry_events,
            attached=attached_this_step,
            in_uncleared_room=in_uncleared_room,
            needs_delta=needs_delta,
            far_from_corridor=far_from_corridor,
            door_cross_bonus=door_cross_bonus,
            door_idle_penalty=door_idle_penalty,
            door_potential_delta=door_potential_delta,
            room_potential_delta=room_potential_delta,
            invalid_bump=invalid_bump,
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
            "room_cleared": self.room_cleared.copy(),
            "room_occupancy": room_counts,
            "needs_remaining": needs_remaining,
            "responder_moved": responder_moved,
            "responder_new_cell": new_cell_visit,
            "room_entries": room_entry_events,
            "needs_delta": needs_delta,
            "door_crossed": door_crossed,
            "door_cross_index": door_cross_idx,
            "door_idle_steps": self.door_idle_steps,
        }
        return obs, reward, terminated, truncated, info

    # ------------ 动作实现 ------------

    def _apply_action(self, action: int) -> Tuple[bool, bool, Optional[int], bool]:
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
        old_pos = self.responder.position
        if self._is_valid_move(old_pos, (nx, nz)):
            new_pos = (nx, nz)
            if new_pos != old_pos:
                door_idx = self._door_index_if_crossing(old_pos, new_pos)
                door_crossed = door_idx is not None
                new_visit = new_pos not in self.visited_cells
                self.responder.position = new_pos
                if new_visit:
                    self.visited_cells.add(new_pos)
                # 记录门探索
                if door_crossed:
                    self.doors_visited.add(door_idx)
                return True, new_visit, door_idx, False
        # 非法移动/撞墙：仅当尝试移动（action!=0）且未发生位移时视为撞墙
        invalid_bump = (action != 0)
        return False, False, None, invalid_bump

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
        needs_remaining = sum(1 for o in self.occupants if o.needs_assist and not o.evacuated)
        needs_ratio = needs_remaining / max(1, self.total_needy)
        rooms_cleared_ratio = sum(1 for cleared in self.room_cleared.values() if cleared) / max(1, self.num_rooms)

        # 时间归一化
        t_norm = min(1.0, self.t / self.max_steps)

        door_dx, door_dz = self._nearest_door_vector((x_r, z_r))

        room_features = []
        span_x = max(1, (x_max - x_min))
        span_z = max(1, (z_max - z_min))
        for room in self.layout.rooms:
            cx, cz = room.center
            dx = (cx - x_r) / span_x
            dz = (cz - z_r) / span_z
            cleared = 1.0 if self.room_cleared[room.id] else 0.0
            room_features.extend([dx, dz, cleared])

        # rooms visited flags
        visited = np.array(
            [1.0 if self.tau[room.id] is not None else 0.0
             for room in self.layout.rooms],
            dtype=np.float32
        )

        carrying_flag = 1.0 if self.responder.carrying is not None else 0.0
        vis_radius = max(self.vis_min, self.vis_max - int(self.vis_decay * self.t))
        vis_norm = (vis_radius - self.vis_min) / max(1e-6, (self.vis_max - self.vis_min))
        door_flags = np.zeros(self.num_doors, dtype=np.float32)
        for d in self.doors_visited:
            if 0 <= d < self.num_doors:
                door_flags[d] = 1.0
        obs = np.concatenate(
            [
                np.array([x_norm, z_norm, evac_ratio, needs_ratio, rooms_cleared_ratio, t_norm, door_dx, door_dz, carrying_flag, vis_norm], dtype=np.float32),
                np.array(room_features, dtype=np.float32),
                visited,
                door_flags,
            ],
            axis=0
        )
        assert obs.shape[0] == self.observation_space.shape[0], (
            f"Observation length {obs.shape[0]} mismatches space {self.observation_space.shape[0]}"
        )
        return obs

    # ------------ 终止条件 ------------

    def _check_done(self) -> Tuple[bool, bool]:
        all_evacuated = all(o.evacuated for o in self.occupants)
        all_rooms_cleared = all(self.room_cleared.values())
        terminated = bool(all_evacuated and all_rooms_cleared)
        truncated = bool(self.t >= self.max_steps)
        return terminated, truncated

    # ------------ 奖励函数 ------------

    def _compute_reward(self,
                        delta_evac: int,
                        rescued_delta: int,
                        self_evac_delta: int,
                        new_cleared_rooms: int,
                        terminated: bool,
                        truncated: bool,
                        room_counts: Dict[str, int],
                        needs_remaining: int,
                        responder_moved: bool,
                        new_cell_visit: bool,
                        room_entries: int,
                        attached: bool,
                        in_uncleared_room: bool,
                        needs_delta: int,
                        far_from_corridor: bool,
                        door_cross_bonus: bool,
                        door_idle_penalty: int,
                        door_potential_delta: float,
                        room_potential_delta: float,
                        invalid_bump: bool) -> float:
        # 基本时间惩罚
        reward = -1.0

        # derive corridor flag from far_from_corridor
        in_corr = not far_from_corridor

        reward = 0.0
        if reward_cfg.time_penalty.enabled:
            reward += reward_cfg.time_penalty.weight

        if reward_cfg.delta_evac.enabled:
            reward += reward_cfg.delta_evac.weight * delta_evac
        if reward_cfg.self_evac_bonus.enabled:
            reward += reward_cfg.self_evac_bonus.weight * self_evac_delta
        if reward_cfg.rescued_bonus.enabled:
            reward += reward_cfg.rescued_bonus.weight * rescued_delta
        if reward_cfg.room_clear_bonus.enabled:
            reward += reward_cfg.room_clear_bonus.weight * new_cleared_rooms
        if reward_cfg.needs_remaining_penalty.enabled:
            reward += reward_cfg.needs_remaining_penalty.weight * needs_remaining

        if reward_cfg.responder_still_penalty.enabled and not responder_moved:
            reward += reward_cfg.responder_still_penalty.weight
        # new-cell bonus: prefer inside rooms, largely suppress in corridor
        if new_cell_visit:
            if in_corr and reward_cfg.new_cell_corridor_bonus.enabled:
                reward += reward_cfg.new_cell_corridor_bonus.weight
            elif (not in_corr) and reward_cfg.new_cell_room_bonus.enabled:
                reward += reward_cfg.new_cell_room_bonus.weight
        if reward_cfg.room_entry_bonus.enabled:
            reward += reward_cfg.room_entry_bonus.weight * room_entries
        if reward_cfg.attach_bonus.enabled and attached:
            reward += reward_cfg.attach_bonus.weight
        if reward_cfg.in_uncleared_room_bonus.enabled and in_uncleared_room:
            reward += reward_cfg.in_uncleared_room_bonus.weight
        if reward_cfg.needs_delta_bonus.enabled:
            reward += reward_cfg.needs_delta_bonus.weight * needs_delta
        if reward_cfg.far_from_corridor_bonus.enabled and far_from_corridor:
            reward += reward_cfg.far_from_corridor_bonus.weight
        if reward_cfg.corridor_step_penalty.enabled and in_corr:
            reward += reward_cfg.corridor_step_penalty.weight
        if reward_cfg.door_cross_bonus.enabled and door_cross_bonus:
            reward += reward_cfg.door_cross_bonus.weight
        if reward_cfg.door_idle_penalty.enabled and door_idle_penalty > 0:
            reward += reward_cfg.door_idle_penalty.weight * door_idle_penalty

        # Potential shaping
        if reward_cfg.door_potential.enabled and door_potential_delta != 0.0:
            reward += reward_cfg.door_potential.weight * door_potential_delta
        if reward_cfg.room_potential.enabled and room_potential_delta != 0.0:
            reward += reward_cfg.room_potential.weight * room_potential_delta
        if reward_cfg.invalid_bump_penalty.enabled and invalid_bump and not responder_moved:
            reward += reward_cfg.invalid_bump_penalty.weight

        if reward_cfg.success_bonus.enabled and terminated:
            reward += reward_cfg.success_bonus.weight

        if reward_cfg.truncation_penalty.enabled and truncated and not terminated:
            reward += reward_cfg.truncation_penalty.weight

        if terminated or truncated:
            remaining = sum(room_counts.values())
            rooms_uncleared = sum(1 for cleared in self.room_cleared.values() if not cleared)
            if reward_cfg.remaining_penalty.enabled:
                reward += reward_cfg.remaining_penalty.weight * remaining
            if reward_cfg.uncleared_penalty.enabled:
                reward += reward_cfg.uncleared_penalty.weight * rooms_uncleared

        return reward

    # ---------- Potential helpers ----------
    def _door_distance(self, p: Coord2D) -> float:
        """Manhattan distance to nearest doorway cell (either side)."""
        if not self.door_positions:
            return 0.0
        return float(min(self._manhattan_distance(p, q) for q in self.door_positions))

    def _room_interior_distance(self, p: Coord2D) -> float:
        """Manhattan distance to the nearest uncleared room's interior rectangle.

        If all rooms cleared, returns 0.
        """
        targets = []
        for room in self.layout.rooms:
            if self.room_cleared.get(room.id, False):
                continue
            x1, z1, x2, z2 = self.room_interior_bounds.get(room.id, (room.x1, room.z1, room.x2, room.z2))
            targets.append((x1, z1, x2, z2))
        if not targets:
            return 0.0
        x, z = p
        def dist_to_rect(rect):
            rx1, rz1, rx2, rz2 = rect
            dx = 0 if rx1 <= x <= rx2 else min(abs(x - rx1), abs(x - rx2))
            dz = 0 if rz1 <= z <= rz2 else min(abs(z - rz1), abs(z - rz2))
            return dx + dz
        return float(min(dist_to_rect(r) for r in targets))

    def _update_room_clear_status(self) -> Tuple[int, Dict[str, int]]:
        counts = {room.id: 0 for room in self.layout.rooms}
        for o in self.occupants:
            if o.evacuated:
                continue
            room = self._room_interior_at(o.position)
            if room is not None:
                counts[room.id] += 1

        newly_cleared = 0
        for room in self.layout.rooms:
            occupied = counts[room.id] > 0
            if not occupied and not self.room_cleared[room.id]:
                self.room_cleared[room.id] = True
                newly_cleared += 1
        self.room_occupancy = counts
        return newly_cleared, counts

    # ------------ Occupant helpers ------------

    def _build_door_index_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        doors = self.layout.doors_xs or []
        if not doors:
            return mapping
        frame = self.layout_json.get("frame", {})
        x_min = frame.get("x1", self.layout.corridor_x_range[0])
        x_max = frame.get("x2", self.layout.corridor_x_range[1])
        for idx, door_x in enumerate(doors):
            for offset in range(-self.door_half_width, self.door_half_width + 1):
                x = door_x + offset
                if x < x_min or x > x_max:
                    continue
                mapping[x] = idx
        return mapping

    def _nearest_open_door_x(self, x: int) -> Optional[int]:
        if not self.door_open_xs:
            return None
        return min(self.door_open_xs, key=lambda dx: abs(dx - x))

    def _door_target_for_position(self, pos: Coord2D) -> Optional[Tuple[Coord2D, Coord2D]]:
        if not self.door_open_xs:
            return None
        x, z = pos
        door_x = self._nearest_open_door_x(x)
        if door_x is None:
            return None
        if z >= self.layout.corridor_z_range[1] + 1:
            door_z_room = self.layout.doors_top_z
            door_z_corr = self.layout.doors_top_z - 1
        elif z <= self.layout.corridor_z_range[0] - 1:
            door_z_room = self.layout.doors_bottom_z
            door_z_corr = self.layout.doors_bottom_z + 1
        else:
            return None
        return (door_x, door_z_room), (door_x, door_z_corr)

    def _enumerate_door_positions(self) -> Tuple[List[Coord2D], List[Coord2D]]:
        corr_positions: List[Coord2D] = []
        room_positions: List[Coord2D] = []
        doors = sorted(self.door_open_xs) if self.door_open_xs else (self.layout.doors_xs or [])
        top_corr = self.layout.doors_top_z - 1
        top_room = self.layout.doors_top_z
        bottom_corr = self.layout.doors_bottom_z + 1
        bottom_room = self.layout.doors_bottom_z
        for x in doors:
            corr_positions.append((x, top_corr))
            room_positions.append((x, top_room))
            corr_positions.append((x, bottom_corr))
            room_positions.append((x, bottom_room))
        return corr_positions, room_positions

    def _build_wall_cells(self) -> Set[Coord2D]:
        walls: Set[Coord2D] = set()
        door_xs = self.door_open_xs or set()
        top_z = self.layout.doors_top_z
        bottom_z = self.layout.doors_bottom_z

        for room in self.layout.rooms:
            # horizontal edges
            z_top = room.z1
            z_bottom = room.z2
            for x in range(room.x1, room.x2 + 1):
                if (top_z is not None and z_top == top_z and x in door_xs):
                    pass  # doorway opening towards corridor
                else:
                    walls.add((x, z_top))
                if (bottom_z is not None and z_bottom == bottom_z and x in door_xs):
                    pass
                else:
                    walls.add((x, z_bottom))

            # vertical edges (left/right walls) – always solid
            for z in range(room.z1, room.z2 + 1):
                walls.add((room.x1, z))
                walls.add((room.x2, z))

        return walls

    def _compute_room_interior_bounds(self) -> Dict[str, Tuple[int, int, int, int]]:
        bounds: Dict[str, Tuple[int, int, int, int]] = {}
        top_z = self.layout.doors_top_z
        bottom_z = self.layout.doors_bottom_z
        for room in self.layout.rooms:
            x1, x2, z1, z2 = room.x1, room.x2, room.z1, room.z2
            new_z1, new_z2 = z1, z2
            if top_z is not None and z1 == top_z:
                new_z1 = min(z2, z1 + 1)
            if bottom_z is not None and z2 == bottom_z:
                new_z2 = max(z1, z2 - 1)
            if new_z1 > new_z2:
                new_z1, new_z2 = z1, z2
            bounds[room.id] = (x1, new_z1, x2, new_z2)
        return bounds

    def _room_interior_contains(self, room: Room, p: Coord2D) -> bool:
        bounds = self.room_interior_bounds.get(room.id)
        if not bounds:
            return room.contains(p)
        x, z = p
        x1, z1, x2, z2 = bounds
        return x1 <= x <= x2 and z1 <= z <= z2

    def _room_interior_at(self, p: Coord2D) -> Optional[Room]:
        for room in self.layout.rooms:
            if self._room_interior_contains(room, p):
                return room
        return None

    def _nearest_door_vector(self, current: Coord2D) -> Tuple[float, float]:
        if not self.door_positions:
            return 0.0, 0.0
        x_min, x_max = self.layout.corridor_x_range
        z_min, z_max = self.layout.corridor_z_range
        span_x = max(1, (x_max - x_min))
        span_z = max(1, (z_max - z_min))
        cx, cz = current
        best = min(self.door_positions, key=lambda d: abs(d[0]-cx) + abs(d[1]-cz))
        dx = (best[0] - cx) / span_x
        dz = (best[1] - cz) / span_z
        return dx, dz

    def _is_at_doorway(self, p: Coord2D) -> bool:
        return p in self.door_position_set

    def _chebyshev_distance(self, a: Coord2D, b: Coord2D) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _manhattan_distance(self, a: Coord2D, b: Coord2D) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _move_self_evac_occupant(self, occupant: Occupant) -> bool:
        target = self._decide_self_evac_target(occupant)
        if target is None:
            return False
        next_pos = self._local_step_towards(occupant.position, target)
        if next_pos is None:
            return False
        if not self._is_valid_move(occupant.position, next_pos):
            return False
        occupant.position = next_pos
        if occupant.position == occupant.target_exit.position:
            occupant.evacuated = True
            occupant.evac_time = self.t
            return True
        return False

    def _decide_self_evac_target(self, occupant: Occupant) -> Optional[Coord2D]:
        if self._is_in_corridor(occupant.position):
            return occupant.target_exit.position
        radius = self.occupant_vis_radius
        for exit_obj in self.layout.exits:
            if self._chebyshev_distance(exit_obj.position, occupant.position) <= radius:
                return exit_obj.position
        if self.responder and self._chebyshev_distance(self.responder.position, occupant.position) <= radius:
            return self.responder.position
        door_pair = self._door_target_for_position(occupant.position)
        if door_pair:
            occupant.door_target_room, occupant.door_target_corridor = door_pair
        if occupant.door_target_room and occupant.position != occupant.door_target_room:
            return occupant.door_target_room
        if occupant.door_target_room and occupant.door_target_corridor and occupant.position == occupant.door_target_room:
            return occupant.door_target_corridor
        if occupant.door_target_corridor and not self._is_in_corridor(occupant.position):
            return occupant.door_target_corridor
        return occupant.target_exit.position

    def _local_step_towards(self, current: Coord2D, target: Coord2D) -> Optional[Coord2D]:
        if current == target:
            return current
        neighbors = list(self._neighbors(current))
        if not neighbors:
            return None
        base_dist = self._manhattan_distance(current, target)
        neighbors.sort(key=lambda p: self._manhattan_distance(p, target))
        for nb in neighbors:
            if self._manhattan_distance(nb, target) <= base_dist:
                return nb
        return neighbors[0]

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
    def _room_at(self, p: Coord2D) -> Optional[Room]:
        for room in self.layout.rooms:
            if room.contains(p):
                return room
        return None

    def _is_door_crossing(self, a: Coord2D, b: Coord2D) -> bool:
        ax, az = a
        bx, bz = b
        doors_xs = self.door_open_xs or set()
        # 顶墙: corridor z = top_z -1 与 room z = top_z
        if {az, bz} == {self.layout.doors_top_z - 1, self.layout.doors_top_z} and ax == bx and ax in doors_xs:
            return True
        # 底墙: corridor z = bottom_z +1 与 room z = bottom_z
        if {az, bz} == {self.layout.doors_bottom_z + 1, self.layout.doors_bottom_z} and ax == bx and ax in doors_xs:
            return True
        return False

    def _door_index_if_crossing(self, a: Coord2D, b: Coord2D) -> Optional[int]:
        if not self._is_door_crossing(a, b):
            return None
        ax, _ = a
        bx, _ = b
        door_x = ax if ax in (self.door_open_xs or set()) else bx
        if door_x not in self.door_x_index_map:
            return None
        return self.door_x_index_map[door_x]

    def _is_valid_move(self, a: Coord2D, b: Coord2D) -> bool:
        ax, az = a
        bx, bz = b
        if abs(ax - bx) + abs(az - bz) != 1:
            return False
        if (ax, az) in self.wall_cells or (bx, bz) in self.wall_cells:
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
        if in_room_a and in_room_b:
            room_a = self._room_at(a)
            room_b = self._room_at(b)
            if room_a is None or room_b is None:
                return False
            return room_a.id == room_b.id
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

    def _try_attach_needy_occupant(self) -> bool:
        if self.responder is None or self.responder.carrying is not None:
            return False
        for o in self.occupants:
            if o.evacuated or not o.needs_assist or o.attached:
                continue
            if o.position == self.responder.position:
                o.attached = True
                self.responder.carrying = o.id
                return True
        return False

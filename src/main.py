# filename: src/main.py
# MineEvac baseline simulation (pure Python, no MineStudio / Minecraft)
#
# 功能：
#   1. 从 layout/baseline.json 读取布局 ℓ
#   2. 构造 rooms, exits, occupants, responders
#   3. 在抽象网格上模拟撤离，计算：
#        - τ(v): 每个房间第一次被 responder 进入的时间
#        - σ:    按 τ(v) 升序的扫房间顺序
#        - T_evac: 所有人撤离所需时间
#   4. 可选：画出布局俯视图，证明确实读对了 JSON
#   5. 导出结果到 sim_result_baseline.json

import os
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

# 如果你没有安装 matplotlib，可以把下面两行和后面的 debug_plot_layout 函数一起注释掉
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors   # ← 新增这一行


Coord2D = Tuple[int, int]  # (x, z)


# ===================== 数据结构 =====================

@dataclass
class Room:
    id: str
    x1: int
    z1: int
    x2: int
    z2: int

    @property
    def center(self) -> Coord2D:
        cx = (self.x1 + self.x2) // 2
        cz = (self.z1 + self.z2) // 2
        return (cx, cz)

    def contains(self, pos: Coord2D) -> bool:
        x, z = pos
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


@dataclass
class Occupant:
    id: int
    position: Coord2D
    target_exit: Exit
    evacuated: bool = False
    evac_time: Optional[int] = None


@dataclass
class Responder:
    id: int
    position: Coord2D
    init_position: Coord2D
    route: List[Room] = field(default_factory=list)
    current_target_idx: int = 0


# ===================== 布局加载：baseline.json → Layout =====================

def load_layout_from_json(path: str) -> Layout:
    """
    从 baseline.json 解析出 Layout 结构 ℓ。
    使用字段：
      - corridor: {x,z,w,h}
      - rooms_top / rooms_bottom: [{x,z,w,h}, ...]
      - frame: {x1,z1,x2,z2} 用于确定整体范围 & 出口位置
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data["name"]

    # 走廊
    corridor = data["corridor"]
    corr_x_min = corridor["x"]
    corr_x_max = corridor["x"] + corridor["w"] - 1
    corr_z_min = corridor["z"]
    corr_z_max = corridor["z"] + corridor["h"] - 1
    corridor_x_range = (corr_x_min, corr_x_max)
    corridor_z_range = (corr_z_min, corr_z_max)

    # 房间（上排+下排）
    rooms: List[Room] = []
    room_id = 0

    def add_room_list(room_list):
        nonlocal room_id, rooms
        for r_def in room_list:
            x1 = r_def["x"]
            z1 = r_def["z"]
            x2 = x1 + r_def["w"] - 1
            z2 = z1 + r_def["h"] - 1
            rooms.append(Room(id=f"R{room_id}", x1=x1, z1=z1, x2=x2, z2=z2))
            room_id += 1

    add_room_list(data["rooms_top"])
    add_room_list(data["rooms_bottom"])

    # 出口：notes 中说在走廊两端中线
    frame = data["frame"]
    mid_z = (corr_z_min + corr_z_max) // 2
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

    if layout.doors_xs and (layout.doors_top_z <= layout.corridor_z_range[0] or layout.doors_bottom_z >= layout.corridor_z_range[1]):
        print(f"[WARN] door z ranges may be inverted: top_z={layout.doors_top_z} bottom_z={layout.doors_bottom_z} corridor={layout.corridor_z_range}")

    return layout, data


# ===================== 可视化：证明确实读到了 JSON =====================
def debug_plot_layout(data: dict):
    """
    用离散方块视角画 layout：
      - 每个整数 (x,z) 是一个方块
      - 房间 / 走廊 / 墙 / 门 都按 JSON 的含义逐格上色
      - 墙和房间不会再错一格
    """
    frame = data["frame"]
    corridor = data["corridor"]
    doors = data.get("doors", None)

    x1, z1, x2, z2 = frame["x1"], frame["z1"], frame["x2"], frame["z2"]

    # ---------- 颜色映射 ----------
    color_map = {
        "orange_wool": "#FFA500",
        "green_wool": "#32CD32",
        "pink_wool": "#FFC0CB",
        "cyan_wool": "#00FFFF",
        "purple_wool": "#9370DB",
        "blue_wool": "#1E90FF",
        "white_concrete": "#E8E8E8",
        "smooth_stone": "#D0D0D0",
        "gray_concrete": "#B0B0B0",
    }

    def block_color(block_name: str):
        return color_map.get(block_name, "#DDDDDD")

    # ---------- 先构造一个 (z, x) 的 tile map ----------
    W = x2 - x1 + 1
    H = z2 - z1 + 1
    # 用字符串标记类型，方便 debug
    tile_type = np.full((H, W), "empty", dtype=object)
    tile_color = np.full((H, W), "#FFFFFF", dtype=object)

    def idx_x(x: int) -> int:
        return x - x1

    def idx_z(z: int) -> int:
        return z - z1

    # 走廊：z in [cz, cz+h-1], x in [cx, cx+w-1]
    cx, cz, cw, ch = corridor["x"], corridor["z"], corridor["w"], corridor["h"]
    corr_block = data.get("corridor_floor", "white_concrete")
    corr_col = block_color(corr_block)
    for x in range(cx, cx + cw):
        for z in range(cz, cz + ch):
            tile_type[idx_z(z), idx_x(x)] = "corridor"
            tile_color[idx_z(z), idx_x(x)] = corr_col

    # 房间：上排 / 下排
    def fill_room_list(room_list):
        for r in room_list:
            rx, rz, rw, rh = r["x"], r["z"], r["w"], r["h"]
            col = block_color(r.get("block", "white_concrete"))
            for x in range(rx, rx + rw):
                for z in range(rz, rz + rh):
                    tile_type[idx_z(z), idx_x(x)] = "room"
                    tile_color[idx_z(z), idx_x(x)] = col

    fill_room_list(data["rooms_top"])
    fill_room_list(data["rooms_bottom"])

    # 墙 + 门：按 notes
    # alignment:
    #  - corridor z = 16..23
    #  - rooms_top z = 24..35, 外墙 z = topZ = 24
    #  - rooms_bottom z = 1..15, 外墙 z = bottomZ = 15
    if doors is not None:
        topZ = doors["topZ"]
        bottomZ = doors["bottomZ"]
        door_xs = set(doors["xs"])

        # 墙颜色（用 wall.material）
        wall_material = data.get("wall", {}).get("material", "white_concrete")
        wall_color = "#8B4513"  # 棕色，看得清楚一点；你也可以用 block_color(wall_material)

        # 上排：在 z = topZ 这一行，房间和走廊之间画墙，门位置留空
        for r in data["rooms_top"]:
            rx, rz, rw, rh = r["x"], r["z"], r["w"], r["h"]
            # 房间横向 [rx, rx+rw-1]
            for x in range(rx, rx + rw):
                if x in door_xs:
                    # 门：不画墙，保持原 tile（房间/走廊），即“空位”
                    continue
                z = topZ
                tile_type[idx_z(z), idx_x(x)] = "wall"
                tile_color[idx_z(z), idx_x(x)] = wall_color

        # 下排：在 z = bottomZ 这一行
        for r in data["rooms_bottom"]:
            rx, rz, rw, rh = r["x"], r["z"], r["w"], r["h"]
            for x in range(rx, rx + rw):
                if x in door_xs:
                    continue
                z = bottomZ
                tile_type[idx_z(z), idx_x(x)] = "wall"
                tile_color[idx_z(z), idx_x(x)] = wall_color

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Layout JSON (discrete tiles): {data['name']}")

    # imshow 以 z 方向向上，x 方向向右
    # 需要把颜色转成 RGBA 矩阵
    rgb_array = np.empty((H, W, 3))
    for i in range(H):
        for j in range(W):
            rgb = mcolors.to_rgb(tile_color[i, j])
            rgb_array[i, j, :] = rgb

    # 注意 imshow 的坐标系，origin="lower" 让 z1 在底部
    ax.imshow(
        rgb_array,
        origin="lower",
        extent=(x1, x2 + 1, z1, z2 + 1),
        interpolation="nearest",
    )

    # 每 10 个方块显示一格刻度
    ax.set_xticks(range(x1, x2 + 1, 10))
    ax.set_yticks(range(z1, z2 + 1, 10))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    ax.set_xlabel("x (blocks)")
    ax.set_ylabel("z (blocks)")

    # 出口：位置在 notes: (8, 中线), (97, 中线)
    mid_z = cz + ch // 2
    exits = [
        ("E_left", (frame["x1"], mid_z)),
        ("E_right", (frame["x2"], mid_z)),
    ]
    for eid, (ex, ez) in exits:
        ax.plot(ex + 0.5, ez + 0.5, "go", markersize=6)  # +0.5 放在格子中心
        ax.text(ex + 0.5, ez + 0.5, eid,
                ha="center", va="bottom", color="green", fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"✅ JSON '{data['name']}' loaded.")
    print("  rooms_top   :", len(data['rooms_top']))
    print("  rooms_bottom:", len(data['rooms_bottom']))
    print("  corridor    :", corridor)
    print("  doors       :", doors)

# ===================== 基本函数：移动 & 最近出口 =====================

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


# ===================== 初始化 occupants / responders =====================

def init_occupants(layout: Layout, per_room: int) -> List[Occupant]:
    occupants: List[Occupant] = []
    oid = 0
    for room in layout.rooms:
        cx, cz = room.center
        for k in range(per_room):
            pos = (cx, cz + (k % 2))  # 稍微错开一点点
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


def init_responders(layout: Layout, num_responders: int) -> List[Responder]:
    rooms_sorted = sorted(layout.rooms, key=lambda r: r.center[0])
    responders: List[Responder] = []

    x_min, x_max = layout.corridor_x_range
    z_min, z_max = layout.corridor_z_range
    mid_z = (z_min + z_max) // 2

    # 在走廊左端附近排开初始位置
    for rid in range(num_responders):
        init_pos = (x_min + 1 + rid, mid_z)
        responders.append(
            Responder(
                id=rid,
                position=init_pos,
                init_position=init_pos,
                route=[],
                current_target_idx=0,
            )
        )

    # 轮流分配房间给 responder：R0, R1, R0, R1, ...
    for idx, room in enumerate(rooms_sorted):
        r = responders[idx % num_responders]
        r.route.append(room)

    return responders


# ===================== 仿真：τ(v), σ, T_evac =====================

def simulate_evacuation(layout: Layout,
                        occupants: List[Occupant],
                        responders: List[Responder],
                        max_steps: int = 10_000):
    """
    离散时间 t=0,1,2,... 模拟：
      - responders 按 route 依次走房间中心
      - occupants 朝各自 target_exit 曼哈顿移动
    输出：
      - tau: dict room.id -> τ(v)
      - sweep_order: σ
      - evac_time: T_evac
    """

    tau: Dict[str, Optional[int]] = {room.id: None for room in layout.rooms}

    def all_evacuated() -> bool:
        return all(o.evacuated for o in occupants)

    t = 0

    while t < max_steps and not all_evacuated():
        # --- responders 移动 ---
        for r in responders:
            if not r.route:
                continue
            if r.current_target_idx >= len(r.route):
                continue  # route 走完，不再动

            target_room = r.route[r.current_target_idx]
            target_pos = target_room.center
            new_pos = manhattan_step_towards(r.position, target_pos)
            r.position = new_pos

            if new_pos == target_pos:
                r.current_target_idx += 1

        # --- 记录 τ(v) ---
        for room in layout.rooms:
            if tau[room.id] is not None:
                continue
            for r in responders:
                if room.contains(r.position):
                    tau[room.id] = t
                    break

        # --- occupants 朝出口移动 ---
        for o in occupants:
            if o.evacuated:
                continue
            ex, ez = o.target_exit.position
            o.position = manhattan_step_towards(o.position, (ex, ez))
            if o.position == (ex, ez):
                o.evacuated = True
                o.evac_time = t

        t += 1

    # --- 收尾：σ 与 T_evac ---
    large_T = max_steps + 1
    tau_filled = {
        rid: (tau[rid] if tau[rid] is not None else large_T)
        for rid in tau.keys()
    }
    rooms_by_tau = sorted(layout.rooms, key=lambda r: tau_filled[r.id])
    sweep_order = [r.id for r in rooms_by_tau]

    evac_times = [o.evac_time for o in occupants if o.evac_time is not None]
    if evac_times:
        evac_time = max(evac_times)
    else:
        evac_time = large_T

    return tau, sweep_order, evac_time


# ===================== 导出结果 =====================

def dump_sim_result(layout: Layout,
                    tau: Dict[str, Optional[int]],
                    sweep_order: List[str],
                    evac_time: int,
                    out_path: str):
    # 标注那些未被 sweep 到的房间：在 id 后面添加 '*'
    sweep_order_marked = [
        (rid + "*") if (tau.get(rid) is None) else rid
        for rid in sweep_order
    ]

    payload = {
        "layout": layout.name,
        "rooms": [
            {"id": r.id, "center": list(r.center)}
            for r in layout.rooms
        ],
        "tau": {rid: (tau[rid] if tau[rid] is not None else None)
                for rid in tau.keys()},
        # 导出的 sweep_order 使用带标注的版本
        "sweep_order": sweep_order_marked,
        "evac_time": evac_time,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Simulation result saved to {out_path}")


# ===================== 主入口 =====================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout_dir = os.path.join(project_root, "layout")
    baseline_path = os.path.join(layout_dir, "baseline.json")

    layout, raw_json = load_layout_from_json(baseline_path)

    # --- 可视化一次布局，确保 JSON 读对了 ---
    try:
        debug_plot_layout(raw_json)
    except Exception as e:
        print(f"[WARN] debug_plot_layout failed: {e}")

    # occupants.per_room 从 JSON 中读取（默认 5）
    per_room = raw_json.get("occupants", {}).get("per_room", 5)

    occupants = init_occupants(layout, per_room=per_room)
    responders = init_responders(layout, num_responders=2)

    tau, sweep_order, evac_time = simulate_evacuation(layout, occupants, responders)

    print("=== MineEvac Simulation Result (baseline) ===")
    print(f"layout: {layout.name}")
    print("τ(v):")
    for room in layout.rooms:
        print(f"  {room.id}: {tau[room.id]}")
    print("\nσ (sweep_order):")
    # 打印时也显示未扫到房间的 '*' 标注，便于人工检查
    sweep_order_marked = [(rid + "*") if (tau.get(rid) is None) else rid
                          for rid in sweep_order]
    print("  ", " -> ".join(sweep_order_marked))
    print(f"\nT_evac (evacuation time): {evac_time}")

    out_path = os.path.join(project_root, "../output/sim_result_baseline.json")
    # 修正输出路径：使用项目内的 output 目录，而不是跳出到父目录
    out_path = os.path.join(project_root, "output", "sim_result_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump_sim_result(layout, tau, sweep_order, evac_time, out_path)


if __name__ == "__main__":
    main()

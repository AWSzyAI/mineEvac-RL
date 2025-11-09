#!/usr/bin/env python3
"""轨迹热图可视化

支持两种日志格式：
1) 旧版 trajectories.jsonl：每行包含 occupants / responders 列表，坐标键为 x,y
2) 新版 eval_episode.jsonl：每行包含 responder_pos 单点，occupants 列表坐标键为 x,z，附带 reward

自动检测 'responder_pos' 判断新版格式，统一抽取 (x,y) 或 (x,z) 为平面坐标。

用法示例：
    python3 scripts/visualize_heatmap.py --path logs/eval_episode.jsonl --bins 60 --save output/heatmap.png
"""
import argparse
import json
import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def load_positions(path: str, state_filter: str = None):
    """加载 occupant / responder 位置，兼容新版与旧版 JSONL。

    返回：xs_o, ys_o, xs_r, ys_r (均为 np.array)
    """
    occ_xs: List[float] = []
    occ_ys: List[float] = []
    resp_xs: List[float] = []
    resp_ys: List[float] = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data: Dict = json.loads(line)

            # 新版格式检测：有 'responder_pos'
            if 'responder_pos' in data:
                # occupants: x,z
                for occ in data.get('occupants', []):
                    # 新版不再有 state，保持兼容：如果传 state_filter 就跳过过滤逻辑
                    if state_filter and occ.get('state') != state_filter:
                        continue
                    x = occ.get('x')
                    z = occ.get('z')  # 替代 y
                    if x is None or z is None:
                        continue
                    occ_xs.append(float(x))
                    occ_ys.append(float(z))
                # 单 responder
                rpos = data.get('responder_pos')
                if isinstance(rpos, (list, tuple)) and len(rpos) == 2:
                    rx, rz = rpos
                    if rx is not None and rz is not None:
                        resp_xs.append(float(rx))
                        resp_ys.append(float(rz))
            else:
                # 旧版格式
                for occ in data.get("occupants", []):
                    if state_filter and occ.get("state") != state_filter:
                        continue
                    x = occ.get("x")
                    y = occ.get("y")
                    if x is None or y is None:
                        continue
                    occ_xs.append(float(x))
                    occ_ys.append(float(y))
                for r in data.get("responders", []):
                    rx = r.get("x")
                    ry = r.get("y")
                    if rx is None or ry is None:
                        continue
                    resp_xs.append(float(rx))
                    resp_ys.append(float(ry))

    return np.array(occ_xs), np.array(occ_ys), np.array(resp_xs), np.array(resp_ys)


def load_layout(layout_path: Optional[str]) -> Optional[dict]:
    if not layout_path:
        return None
    if not os.path.exists(layout_path):
        raise FileNotFoundError(f"Layout file not found: {layout_path}")
    with open(layout_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_rooms_with_ids(layout: dict):
    idx = 1
    for key in ("rooms_top", "rooms_bottom"):
        for room in layout.get(key, []):
            yield room, f"R{idx}"
            idx += 1


def _wall_segments(room: dict, door_xs: set) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    x0 = room["x"]
    x1 = room["x"] + room["w"]
    for x in range(x0, x1):
        if x in door_xs:
            if start is not None:
                segments.append((start, x))
                start = None
        else:
            if start is None:
                start = x
    if start is not None:
        segments.append((start, x1))
    return segments


def _draw_wall_rows(ax, layout: dict):
    doors = layout.get("doors", {})
    door_xs = set(doors.get("xs", []))
    top_z = doors.get("topZ")
    bottom_z = doors.get("bottomZ")
    wall_color = "#8B5A2B"
    if top_z is not None:
        for room in layout.get("rooms_top", []):
            for x0, x1 in _wall_segments(room, door_xs):
                ax.add_patch(Rectangle(
                    (x0, top_z),
                    x1 - x0,
                    1,
                    facecolor=wall_color,
                    edgecolor="none",
                    alpha=0.6,
                ))
    if bottom_z is not None:
        for room in layout.get("rooms_bottom", []):
            for x0, x1 in _wall_segments(room, door_xs):
                ax.add_patch(Rectangle(
                    (x0, bottom_z),
                    x1 - x0,
                    1,
                    facecolor=wall_color,
                    edgecolor="none",
                    alpha=0.6,
                ))
    door_color = "#FFD700"
    for x in door_xs:
        if top_z is not None:
            ax.add_patch(Rectangle((x, top_z), 1, 1, facecolor=door_color, edgecolor="none", alpha=0.9))
        if bottom_z is not None:
            ax.add_patch(Rectangle((x, bottom_z), 1, 1, facecolor=door_color, edgecolor="none", alpha=0.9))


def _draw_room_labels(ax, layout: dict):
    for room, label in _iter_rooms_with_ids(layout):
        cx = room["x"] + room["w"] / 2
        cz = room["z"] + room["h"] / 2
        ax.text(
            cx,
            cz,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        )


def draw_layout(ax, layout: dict):
    if not layout:
        return

    frame = layout.get("frame")
    if frame:
        ax.add_patch(Rectangle(
            (frame["x1"], frame["z1"]),
            frame["x2"] - frame["x1"],
            frame["z2"] - frame["z1"],
            fill=False,
            edgecolor="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        ))

    def _draw_rooms(room_list, color):
        for room in room_list:
            ax.add_patch(Rectangle(
                (room["x"], room["z"]),
                room["w"],
                room["h"],
                fill=False,
                edgecolor=color,
                linewidth=1.2,
            ))

    rooms_top = layout.get("rooms_top", [])
    rooms_bottom = layout.get("rooms_bottom", [])
    _draw_rooms(rooms_top, "orange")
    _draw_rooms(rooms_bottom, "cyan")

    _draw_wall_rows(ax, layout)
    _draw_room_labels(ax, layout)

    corr = layout.get("corridor")
    if corr:
        ax.add_patch(Rectangle(
            (corr["x"], corr["z"]),
            corr["w"],
            corr["h"],
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
        ))

    doors = layout.get("doors", {})
    doors_xs = doors.get("xs", [])
    top_z = doors.get("topZ")
    bottom_z = doors.get("bottomZ")

    corr_mid = None
    if corr:
        corr_mid = corr["z"] + corr["h"] / 2
    elif frame:
        corr_mid = frame["z1"] + (frame["z2"] - frame["z1"]) / 2

    exits = []
    if frame and corr_mid is not None:
        exits = [
            (frame["x1"], corr_mid),
            (frame["x2"], corr_mid),
        ]
    if exits:
        xs, ys = zip(*exits)
        ax.scatter(xs, ys, c="lime", marker="X", s=80, label="Exits")
        ax.legend(loc="upper right", fontsize="small", frameon=False)


def _compute_extent(xs_list: List[np.ndarray], ys_list: List[np.ndarray], layout: Optional[dict]):
    xs_all = []
    ys_all = []
    for xs in xs_list:
        if xs.size:
            xs_all.extend([np.nanmin(xs), np.nanmax(xs)])
    for ys in ys_list:
        if ys.size:
            ys_all.extend([np.nanmin(ys), np.nanmax(ys)])
    if layout and layout.get("frame"):
        frame = layout["frame"]
        xs_all.extend([frame["x1"], frame["x2"]])
        ys_all.extend([frame["z1"], frame["z2"]])
    xmin, xmax = (float(min(xs_all)), float(max(xs_all))) if xs_all else (0.0, 1.0)
    ymin, ymax = (float(min(ys_all)), float(max(ys_all))) if ys_all else (0.0, 1.0)
    return xmin, xmax, ymin, ymax


def plot_heatmaps(xs_occ, ys_occ, xs_res, ys_res, entity: str, bins: int = 50, save: str = None, layout: Optional[dict] = None):
    # 共享 extent
    datasets = []
    if entity in ("occupants", "both"):
        datasets.append(("Occupants (hot)", xs_occ, ys_occ, "hot", "Occupant density (log)"))
    if entity in ("responder", "both"):
        datasets.append(("Responder (Blues)", xs_res, ys_res, "Blues", "Responder density (log)"))

    if not datasets:
        raise ValueError("entity must be one of: responder, occupants, both")

    xmin, xmax, ymin, ymax = _compute_extent([d[1] for d in datasets], [d[2] for d in datasets], layout)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6 if len(datasets) == 1 else 12), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.subplots_adjust(hspace=0.12)

    vmax = 1.0
    heat_data = []
    for _, xs, ys, _, _ in datasets:
        if xs.size and ys.size:
            heat, _, _ = np.histogram2d(xs, ys, bins=(bins, bins))
            heat_data.append(heat.T)
            vmax = max(vmax, heat.max() + 1.0)
        else:
            heat_data.append(None)

    for ax, (title, _, _, cmap, cbar_label), heat in zip(axes, datasets, heat_data):
        if heat is not None and np.any(heat):
            im = ax.imshow(
                heat + 1.0,
                origin="lower",
                cmap=cmap,
                norm=LogNorm(vmin=1, vmax=vmax),
                interpolation="nearest",
                extent=[xmin, xmax, ymin, ymax],
                aspect="auto",
            )
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xmin, xmax)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                pad=0.01,
                fraction=0.05,
                label=cbar_label,
            )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect("equal")
        draw_layout(ax, layout)

    plt.suptitle("Trajectories Heatmaps")

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Saved heatmap to {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="logs/trajectories.jsonl", help="Path to log JSONL (trajectories or eval_episode)")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins per axis")
    parser.add_argument("--state", default=None, help="Filter by occupant.state (optional)")
    parser.add_argument("--save", default=None, help="If set, save image to this path instead of showing")
    parser.add_argument("--layout", default="layout/baseline.json", help="Layout JSON for drawing walls/doors/exits")
    parser.add_argument(
        "--entity",
        choices=["responder", "occupants", "both"],
        default="both",
        help="Which trajectories to aggregate into the heatmap",
    )
    args = parser.parse_args()

    xs_o, ys_o, xs_r, ys_r = load_positions(args.path, state_filter=args.state)
    layout = load_layout(args.layout) if args.layout else None
    plot_heatmaps(xs_o, ys_o, xs_r, ys_r, entity=args.entity, bins=args.bins, save=args.save, layout=layout)


if __name__ == "__main__":
    main()

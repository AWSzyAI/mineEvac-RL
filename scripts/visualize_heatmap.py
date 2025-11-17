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
import math


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
            try:
                data: Dict = json.loads(line)
            except Exception:
                # Skip malformed JSON lines (robustness for mixed logs)
                continue

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
    # Fallback: derive from corridor if doors missing
    if (top_z is None or bottom_z is None) and layout.get("corridor"):
        corr = layout["corridor"]
        z0 = int(corr.get("z", 0))
        h = int(corr.get("h", 0))
        if top_z is None:
            top_z = z0 + h  # first row above corridor
        if bottom_z is None:
            bottom_z = z0 - 1  # last row below corridor
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
    # doors left unfilled (white)


def _draw_room_perimeters(ax, layout: dict):
    """Draw each room's outermost ring of walls.

    - Left/right vertical edges for all rooms
    - Far horizontal edge (away from corridor) for each room row
    - Corridor-facing edge is left to _draw_wall_rows to handle door gaps
    """
    wall_color = "#8B5A2B"
    doors = layout.get("doors", {})
    top_z = doors.get("topZ")
    bottom_z = doors.get("bottomZ")
    # Fallback: derive from corridor if doors missing
    if (top_z is None or bottom_z is None) and layout.get("corridor"):
        corr = layout["corridor"]
        z0 = int(corr.get("z", 0))
        h = int(corr.get("h", 0))
        if top_z is None:
            top_z = z0 + h
        if bottom_z is None:
            bottom_z = z0 - 1

    def add_perimeter(room: dict, corridor_side_z: int):
        rx, rz, rw, rh = room["x"], room["z"], room["w"], room["h"]
        left_x = rx
        right_x = rx + rw - 1
        bottom = rz
        top = rz + rh - 1

        # vertical edges
        ax.add_patch(Rectangle((left_x, rz), 1, rh, facecolor=wall_color, edgecolor="none", alpha=0.6))
        ax.add_patch(Rectangle((right_x, rz), 1, rh, facecolor=wall_color, edgecolor="none", alpha=0.6))

        # horizontal edges, skip corridor-facing side (will be drawn with door gaps)
        if bottom != corridor_side_z:
            ax.add_patch(Rectangle((rx, bottom), rw, 1, facecolor=wall_color, edgecolor="none", alpha=0.6))
        if top != corridor_side_z:
            ax.add_patch(Rectangle((rx, top), rw, 1, facecolor=wall_color, edgecolor="none", alpha=0.6))

    # top row rooms: corridor-facing edge is at topZ
    if top_z is not None:
        for room in layout.get("rooms_top", []):
            add_perimeter(room, top_z)
    # bottom row rooms: corridor-facing edge is at bottomZ
    if bottom_z is not None:
        for room in layout.get("rooms_bottom", []):
            add_perimeter(room, bottom_z)


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

    # Draw room perimeters and corridor-facing wall rows (doors create openings)
    _draw_room_perimeters(ax, layout)
    _draw_wall_rows(ax, layout)
    # no labels, no corridor outlines, keep non-wall cells white


def _compute_extent(xs_list: List[np.ndarray], ys_list: List[np.ndarray], layout: Optional[dict]):
    """Return plotting extent [xmin, xmax, ymin, ymax].

    Prefer the tight building bounds (rooms + corridor) so the bottom wall
    sits flush with the canvas. Fallback to frame if geometry missing, else
    derive from data with a small margin.
    """
    if layout:
        # Collect geometry bounds from rooms and corridor
        xs_min = []
        xs_max = []
        ys_min = []
        ys_max = []

        for key in ("rooms_top", "rooms_bottom"):
            for r in layout.get(key, []) or []:
                xs_min.append(r["x"])            # cell edge on the left
                xs_max.append(r["x"] + r["w"])  # right edge beyond last cell
                ys_min.append(r["z"])            # bottom edge
                ys_max.append(r["z"] + r["h"])  # top edge beyond last cell

        corr = layout.get("corridor")
        if corr:
            xs_min.append(corr["x"])           
            xs_max.append(corr["x"] + corr["w"]) 
            ys_min.append(corr["z"])           
            ys_max.append(corr["z"] + corr["h"]) 

        if xs_min and xs_max and ys_min and ys_max:
            xmin = float(min(xs_min))
            xmax = float(max(xs_max))
            ymin = float(min(ys_min))
            ymax = float(max(ys_max))
            return xmin, xmax, ymin, ymax

        # If rooms/corridor absent, fall back to frame
        if layout.get("frame"):
            f = layout["frame"]
            xmin = float(f["x1"])            # left edge of leftmost cell
            xmax = float(f["x2"] + 1)         # right edge beyond rightmost cell
            ymin = float(f["z1"])            # bottom edge of bottom cell
            ymax = float(f["z2"] + 1)         # top edge beyond topmost cell
            return xmin, xmax, ymin, ymax

    xs_all = []
    ys_all = []
    for xs in xs_list:
        if xs.size:
            xs_all.extend([np.nanmin(xs), np.nanmax(xs)])
    for ys in ys_list:
        if ys.size:
            ys_all.extend([np.nanmin(ys), np.nanmax(ys)])
    xmin, xmax = (float(min(xs_all)), float(max(xs_all))) if xs_all else (0.0, 1.0)
    ymin, ymax = (float(min(ys_all)), float(max(ys_all))) if ys_all else (0.0, 1.0)
    # add 1-cell margin when layout is absent
    return xmin - 1.0, xmax + 1.0, ymin - 1.0, ymax + 1.0


def plot_heatmaps(xs_occ, ys_occ, xs_res, ys_res, entity: str, bins: int = 50, save: str = None, layout: Optional[dict] = None):
    # Determine integer-aligned grid from layout frame if available, else from data
    if layout and layout.get("frame"):
        f = layout["frame"]
        x1, x2 = int(f["x1"]), int(f["x2"])  # inclusive indices
        z1, z2 = int(f["z1"]), int(f["z2"])  # inclusive indices
    else:
        all_x = np.concatenate([a for a in (xs_occ, xs_res) if a.size]) if xs_occ.size or xs_res.size else np.array([0, 1])
        all_y = np.concatenate([a for a in (ys_occ, ys_res) if a.size]) if ys_occ.size or ys_res.size else np.array([0, 1])
        x1, x2 = int(np.floor(np.nanmin(all_x))), int(np.ceil(np.nanmax(all_x)))
        z1, z2 = int(np.floor(np.nanmin(all_y))), int(np.ceil(np.nanmax(all_y)))
    W = max(1, x2 - x1 + 1)
    H = max(1, z2 - z1 + 1)

    def accumulate(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        grid = np.zeros((H, W), dtype=np.int32)
        if xs.size and ys.size:
            j = np.floor(xs).astype(int) - x1
            i = np.floor(ys).astype(int) - z1
            mask = (i >= 0) & (i < H) & (j >= 0) & (j < W)
            if np.any(mask):
                np.add.at(grid, (i[mask], j[mask]), 1)
        return grid

    show_occ = entity in ("occupants", "both")
    show_res = entity in ("responder", "both")

    rows = (1 if (show_occ ^ show_res) else 2) if (show_occ and show_res) else 1
    labels = []
    grids = []
    cmaps = []
    norms = []
    if show_occ:
        g = accumulate(xs_occ, ys_occ)
        grids.append(g)
        labels.append("Occupants")
        cmaps.append("Reds")
        norms.append(LogNorm(vmin=1, vmax=max(1, g.max())))
    if show_res:
        g = accumulate(xs_res, ys_res)
        grids.append(g)
        labels.append("Responder")
        cmaps.append("Blues")
        norms.append(LogNorm(vmin=1, vmax=max(1, g.max())))

    nplots = len(grids)
    fig, axes = plt.subplots(nplots, 1, figsize=(10, 6 if nplots == 1 else 12), squeeze=False)
    axes = axes.ravel()
    # remove all extra margins so cells touch canvas edges
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.0)

    x_edges = np.arange(x1, x2 + 2)
    y_edges = np.arange(z1, z2 + 2)

    for ax, title, g, cmap, norm in zip(axes, labels, grids, cmaps, norms):
        # use pcolormesh to color every cell; add 1 to show cells with at least 1 visit in log norm
        vals = g.copy().astype(float)
        vals[vals == 0] = np.nan  # leave unvisited cells transparent
        m = ax.pcolormesh(x_edges, y_edges, vals, cmap=cmap, norm=norm, shading='flat', edgecolors='none')
        ax.set_xlim(x1, x2 + 1)
        ax.set_ylim(z1, z2 + 1)
        ax.set_aspect('equal', adjustable='box')
        try:
            ax.set_box_aspect((z2 + 1 - z1) / max(1, (x2 + 1 - x1)))
        except Exception:
            pass
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        # no ticks or spines; avoid extra padding
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        draw_layout(ax, layout)
        # optional: remove titles/labels to keep edges flush
        # ax.set_title(title)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        cbar = fig.colorbar(m, ax=ax, orientation='horizontal', pad=0.01, fraction=0.05)
        cbar.set_label('Visits (log)')

    # plt.suptitle('Trajectories Heatmaps')

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0, facecolor='white')
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

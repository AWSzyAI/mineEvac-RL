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
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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


def plot_two_heatmaps(xs1, ys1, xs2, ys2, bins: int = 50, save: str = None):
    # 共享 extent
    all_x = np.concatenate([xs1, xs2]) if xs1.size or xs2.size else np.array([0.0])
    all_y = np.concatenate([ys1, ys2]) if ys1.size or ys2.size else np.array([0.0])
    xmin, xmax = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Occupants
    if xs1.size:
        heat1, xedges, yedges = np.histogram2d(xs1, ys1, bins=(bins, bins))
        heat1 = heat1.T
        im1 = axes[0].imshow(
            heat1 + 1.0,  # +1 to avoid zeros for LogNorm
            origin="lower",
            cmap="hot",
            norm=LogNorm(vmin=1, vmax=max(1.0, heat1.max() + 1.0)),
            interpolation="nearest",
            extent=[xmin, xmax, ymin, ymax],
        )
        fig.colorbar(im1, ax=axes[0], label="Occupant density (log)")
    else:
        axes[0].text(0.5, 0.5, "No occupant data", ha="center", va="center")

    axes[0].set_title("Occupants (hot)")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")

    # Responders
    if xs2.size:
        heat2, xedges2, yedges2 = np.histogram2d(xs2, ys2, bins=(bins, bins))
        heat2 = heat2.T
        im2 = axes[1].imshow(
            heat2 + 1.0,
            origin="lower",
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=max(1.0, heat2.max() + 1.0)),
            interpolation="nearest",
            extent=[xmin, xmax, ymin, ymax],
        )
        fig.colorbar(im2, ax=axes[1], label="Responder density (log)")
    else:
        axes[1].text(0.5, 0.5, "No responder data", ha="center", va="center")

    axes[1].set_title("Responders (Blues)")
    axes[1].set_xlabel("X Position")
    axes[1].set_ylabel("Y Position")

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
    args = parser.parse_args()

    xs_o, ys_o, xs_r, ys_r = load_positions(args.path, state_filter=args.state)
    plot_two_heatmaps(xs_o, ys_o, xs_r, ys_r, bins=args.bins, save=args.save)


if __name__ == "__main__":
    main()

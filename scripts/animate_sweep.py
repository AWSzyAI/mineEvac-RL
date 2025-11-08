#!/usr/bin/env python3
"""Animate sweep / evacuation dynamics with layout overlay & reward.

Input log format (eval_episode.jsonl recommended): each line JSON with keys:
    {
        "t": int,
        "reward": float,
        "cumulative_reward": float,
        "responder_pos": [x, z],
        "occupants": [ {"id": int, "x": float, "z": float, "evacuated": bool}, ... ]
    }

If using raw trajectories.jsonl (older format) it will fall back (no reward display).

Layout overlay: rectangles for corridor & rooms; exits marked as green points.

Usage:
    python3 scripts/animate_sweep.py \
            --path logs/eval_episode.jsonl \
            --layout layout/baseline.json \
            --save output/sweep_anim.gif \
            --fps 12 --skip 1 --trail 80
"""
import argparse
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from matplotlib.animation import PillowWriter
    PILLOW_AVAILABLE = True
except Exception:
    PILLOW_AVAILABLE = False


def load_frames(path: str) -> List[dict]:
    """Load frames supporting two formats: legacy trajectories.jsonl & eval_episode.jsonl."""
    frames: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Try new format first
            if 'responder_pos' in data:
                # new eval format
                occ_list = data.get('occupants', [])
                occ_xy = np.array([[float(o.get('x', np.nan)), float(o.get('z', np.nan))] for o in occ_list], dtype=float)
                res_pos = data.get('responder_pos', [])
                if res_pos:
                    res_xy = np.array([[float(res_pos[0]), float(res_pos[1])]])
                    res_id = [0]
                else:
                    res_xy = np.empty((0, 2))
                    res_id = []
                frames.append({
                    'time': data.get('t', len(frames)),
                    'occ_xy': occ_xy,
                    'res_xy': res_xy,
                    'res_id': res_id,
                    'reward': data.get('reward'),
                    'cumulative_reward': data.get('cumulative_reward'),
                })
            else:
                # legacy format
                occ = data.get('occupants', [])
                occ_xy = np.array([[float(o.get('x', np.nan)), float(o.get('y', np.nan))] for o in occ], dtype=float)
                res = data.get('responders', [])
                res_xy = np.array([[float(r.get('x', np.nan)), float(r.get('y', np.nan))] for r in res], dtype=float)
                res_id = [r.get('id') for r in res]
                frames.append({
                    'time': data.get('time', len(frames)),
                    'occ_xy': occ_xy,
                    'res_xy': res_xy,
                    'res_id': res_id,
                    'reward': None,
                    'cumulative_reward': None,
                })
    return frames


def compute_extent(frames: List[dict]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for fr in frames:
        if fr['occ_xy'].size:
            xs.append(fr['occ_xy'][:, 0])
            ys.append(fr['occ_xy'][:, 1])
        if fr['res_xy'].size:
            xs.append(fr['res_xy'][:, 0])
            ys.append(fr['res_xy'][:, 1])
    if not xs:
        # default extent
        return 0.0, 1.0, 0.0, 1.0
    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    xmin, xmax = float(np.nanmin(allx)), float(np.nanmax(allx))
    ymin, ymax = float(np.nanmin(ally)), float(np.nanmax(ally))
    # add small margins
    dx = max(1e-6, (xmax - xmin) * 0.05)
    dy = max(1e-6, (ymax - ymin) * 0.05)
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def load_layout(layout_path: Optional[str]):
    if not layout_path:
        return None
    if not os.path.exists(layout_path):
        return None
    with open(layout_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def animate(path: str, save: str, fps: int, skip: int, trail: int, dpi: int, layout_path: Optional[str]) -> str:
    frames = load_frames(path)
    if not frames:
        raise RuntimeError(f"No frames loaded from {path}")

    xmin, xmax, ymin, ymax = compute_extent(frames)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Sweep dynamics (layout + reward)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # ---- layout overlay ----
    layout_json = load_layout(layout_path)
    if layout_json:
        # corridor
        corridor = layout_json.get('corridor')
        if corridor:
            cx, cz, cw, ch = corridor['x'], corridor['z'], corridor['w'], corridor['h']
            rect = plt.Rectangle((cx, cz), cw, ch, facecolor='#F5F5F5', edgecolor='#AAAAAA', alpha=0.4)
            ax.add_patch(rect)
        # rooms
        def draw_rooms(key, color_edge):
            for r in layout_json.get(key, []):
                rx, rz, rw, rh = r['x'], r['z'], r['w'], r['h']
                room_rect = plt.Rectangle((rx, rz), rw, rh, facecolor='none', edgecolor=color_edge, linewidth=1.2)
                ax.add_patch(room_rect)
        draw_rooms('rooms_top', '#FF8C00')
        draw_rooms('rooms_bottom', '#1E90FF')
        # doors
        doors = layout_json.get('doors')
        if doors and corridor:
            top_z = doors.get('topZ')
            bottom_z = doors.get('bottomZ')
            xs = doors.get('xs', [])
            # draw small markers at seam locations
            for x in xs:
                if top_z is not None:
                    ax.plot([x-0.6, x+0.6], [top_z-0.2, top_z-0.2], color='green', linewidth=3, alpha=0.9)
                if bottom_z is not None:
                    ax.plot([x-0.6, x+0.6], [bottom_z+0.2, bottom_z+0.2], color='green', linewidth=3, alpha=0.9)
            ax.text(xs[0] if xs else cx, cz+ch+0.8, 'Doors', color='green', fontsize=8, ha='left')
        # exits
        frame = layout_json.get('frame')
        if frame and corridor:
            mid_z = corridor['z'] + corridor['h']//2
            ax.plot(frame['x1'], mid_z, 'go', markersize=6)
            ax.plot(frame['x2'], mid_z, 'go', markersize=6)
            ax.text(frame['x1'], mid_z+0.5, 'ExitL', color='green', fontsize=8, ha='center')
            ax.text(frame['x2'], mid_z+0.5, 'ExitR', color='green', fontsize=8, ha='center')

    scat_occ = ax.scatter([], [], c='red', s=8, alpha=0.7, label='occupants')
    scat_res = ax.scatter([], [], c='blue', s=25, alpha=0.9, label='responders')
    ax.legend(loc='upper right')

    # responder trails: rid -> Line2D object and a deque of points
    trail_deques: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trail))
    trail_lines: Dict[int, any] = {}

    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', ha='left', fontsize=10)

    frame_indices = list(range(0, len(frames), max(1, skip)))

    def init():
        scat_occ.set_offsets(np.empty((0, 2)))
        scat_res.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        # clear trails
        for line in trail_lines.values():
            line.remove()
        trail_lines.clear()
        trail_deques.clear()
        return scat_occ, scat_res, time_text

    def update(idx):
        fr = frames[frame_indices[idx]]
        occ_xy = fr['occ_xy']
        res_xy = fr['res_xy']
        res_id = fr['res_id']

        # update scatters
        if occ_xy.size:
            scat_occ.set_offsets(occ_xy)
        else:
            scat_occ.set_offsets(np.empty((0, 2)))

        if res_xy.size:
            scat_res.set_offsets(res_xy)
        else:
            scat_res.set_offsets(np.empty((0, 2)))

        # update trails by responder id
        # ensure we have line objects for each responder id present
        for rid, pos in zip(res_id, res_xy):
            if rid is None:
                # skip if id missing
                continue
            dq = trail_deques[rid]
            dq.append(tuple(pos))
            xs = [p[0] for p in dq]
            ys = [p[1] for p in dq]
            if rid not in trail_lines:
                (line,) = ax.plot(xs, ys, color='navy', alpha=0.6, linewidth=1.5)
                trail_lines[rid] = line
            else:
                line = trail_lines[rid]
                line.set_data(xs, ys)

        # Reward display if available
        if fr['reward'] is not None:
            time_text.set_text(f"t={fr['time']}  r={fr['reward']:.2f}  Rsum={fr['cumulative_reward']:.2f}")
        else:
            time_text.set_text(f"t={fr['time']}")
        return scat_occ, scat_res, time_text, *trail_lines.values()

    ani = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, interval=max(10, int(1000/max(1,fps))), blit=False)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        if save.lower().endswith('.gif') and PILLOW_AVAILABLE:
            writer = PillowWriter(fps=fps)
            ani.save(save, writer=writer, dpi=dpi)
        else:
            # fallback: try to save as mp4 using default writer; or per-frame PNGs
            try:
                ani.save(save, dpi=dpi)
            except Exception:
                # last resort: dump frames
                frames_dir = os.path.join(os.path.dirname(save), 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                for i in range(len(frame_indices)):
                    update(i)
                    fig.savefig(os.path.join(frames_dir, f'frame_{i:04d}.png'), dpi=dpi)
        print(f"Saved animation to {save}")
    else:
        plt.show()

    plt.close(fig)
    return save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='logs/trajectories.jsonl', help='Path to trajectories JSONL')
    parser.add_argument('--save', default='output/sweep_anim.gif', help='Output GIF/MP4 path')
    parser.add_argument('--fps', type=int, default=12, help='Frames per second')
    parser.add_argument('--skip', type=int, default=1, help='Use every Nth frame to speed up')
    parser.add_argument('--trail', type=int, default=60, help='Responder trail length (frames)')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    parser.add_argument('--layout', default='layout/baseline.json', help='Layout JSON for overlay')
    args = parser.parse_args()

    animate(args.path, args.save, fps=args.fps, skip=args.skip, trail=args.trail, dpi=args.dpi, layout_path=args.layout)


if __name__ == '__main__':
    main()

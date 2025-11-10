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
            --fps 12 --skip 1

Notes:
- Previously, responders drew line trails. This has been replaced by a
  real‑time per‑cell heatmap: each time a responder/occupant visits a grid
  cell, that cell's intensity increases.
- Visual style: a unified grayscale base shows total visit intensity, with
  semi‑transparent colored overlays indicating entity types (responders: blue,
  occupants: red). Current frame positions remain as blue/red dots.
"""
import argparse
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import math

try:
    from matplotlib.animation import PillowWriter
    PILLOW_AVAILABLE = True
except Exception:
    PILLOW_AVAILABLE = False


# --- Simple ETA mapping: seconds = steps * (cell_m / speed) ---
# Defaults consistent with deterministic sim mapping (may be adjusted if needed)
CELL_M = 0.5          # meters per grid cell
SPEED_SOLO = 0.8      # m/s when not escorting (rough assumption for ETA)

def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


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
                # unswept from room_cleared if present
                unswept = None
                rc = data.get('room_cleared')
                if isinstance(rc, dict):
                    try:
                        unswept = int(sum(1 for v in rc.values() if not v))
                    except Exception:
                        unswept = None
                frames.append({
                    'time': data.get('t', len(frames)),
                    'occ_xy': occ_xy,
                    'res_xy': res_xy,
                    'res_id': res_id,
                    'reward': data.get('reward'),
                    'cumulative_reward': data.get('cumulative_reward'),
                    'unswept': unswept,
                    'eta_seconds': data.get('eta_seconds'),
                    'eta_hms': data.get('eta_hms'),
                })
            else:
                # legacy format (deterministic simulator)
                occ = data.get('occupants', [])
                occ_xy = np.array([[float(o.get('x', np.nan)), float(o.get('y', np.nan))] for o in occ], dtype=float)
                occ_evac = [bool(o.get('evacuated', False)) for o in occ]
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
                    'evac_count': int(sum(1 for e in occ_evac if e)),
                    'occ_total': int(len(occ_evac)),
                    'unswept': data.get('unswept'),
                    'eta_seconds': data.get('eta_seconds'),
                    'eta_hms': data.get('eta_hms'),
                })
    return frames


def compute_extent(frames: List[dict], layout_json: Optional[dict]) -> Tuple[float, float, float, float]:
    """Return tight extent aligned to layout grid edges.

    Prefer the building geometry (rooms + corridor):
      [min_x, max_x] x [min_z, max_z]
    where max_x/max_z are right/top edges (i.e., +1 from the last cell), so
    1-cell-thick walls on the boundary are fully visible and the bottom wall
    sits flush with the canvas. If layout is missing, fall back to data with
    a small symmetric margin.
    """
    if layout_json:
        xs_min, xs_max, ys_min, ys_max = [], [], [], []
        for key in ('rooms_top', 'rooms_bottom'):
            for r in layout_json.get(key, []) or []:
                xs_min.append(r['x'])
                xs_max.append(r['x'] + r['w'])
                ys_min.append(r['z'])
                ys_max.append(r['z'] + r['h'])
        corr = layout_json.get('corridor')
        if corr:
            xs_min.append(corr['x'])
            xs_max.append(corr['x'] + corr['w'])
            ys_min.append(corr['z'])
            ys_max.append(corr['z'] + corr['h'])
        if xs_min and xs_max and ys_min and ys_max:
            xmin = float(min(xs_min))
            xmax = float(max(xs_max))
            ymin = float(min(ys_min))
            ymax = float(max(ys_max))
            return xmin, xmax, ymin, ymax
        # fallback to frame if present
        if layout_json.get('frame'):
            f = layout_json['frame']
            return float(f['x1']), float(f['x2'] + 1), float(f['z1']), float(f['z2'] + 1)

    # data-driven fallback with small margin
    xs, ys = [], []
    for fr in frames:
        if fr['occ_xy'].size:
            xs.append(fr['occ_xy'][:, 0])
            ys.append(fr['occ_xy'][:, 1])
        if fr['res_xy'].size:
            xs.append(fr['res_xy'][:, 0])
            ys.append(fr['res_xy'][:, 1])
    if xs:
        allx = np.concatenate(xs)
        ally = np.concatenate(ys)
        xmin, xmax = float(np.nanmin(allx)), float(np.nanmax(allx))
        ymin, ymax = float(np.nanmin(ally)), float(np.nanmax(ally))
    else:
        xmin = ymin = 0.0
        xmax = ymax = 1.0
    margin = max(1.0, 0.02 * max(xmax - xmin, ymax - ymin))
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def load_layout(layout_path: Optional[str]):
    if not layout_path:
        return None
    if not os.path.exists(layout_path):
        return None
    with open(layout_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _iter_rooms_with_ids(layout: dict):
    idx = 1
    for key in ("rooms_top", "rooms_bottom"):
        for room in layout.get(key, []):
            yield room, f"R{idx}"
            idx += 1


def _wall_segments(room: dict, door_xs: set):
    segments = []
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
                ax.add_patch(Rectangle((x0, top_z), x1 - x0, 1, facecolor=wall_color, edgecolor="none", alpha=0.55))
    if bottom_z is not None:
        for room in layout.get("rooms_bottom", []):
            for x0, x1 in _wall_segments(room, door_xs):
                ax.add_patch(Rectangle((x0, bottom_z), x1 - x0, 1, facecolor=wall_color, edgecolor="none", alpha=0.55))
    # doors are left unfilled (white) to keep background clean


def _draw_room_perimeters(ax, layout: dict):
    """Draw a 1-cell wall ring around each room (except the corridor-facing side, which is drawn with door gaps)."""
    wall_color = "#8B5A2B"
    doors = layout.get("doors", {})
    top_z = doors.get("topZ")
    bottom_z = doors.get("bottomZ")

    def add_perimeter(room: dict, corridor_side_z: int):
        rx, rz, rw, rh = room["x"], room["z"], room["w"], room["h"]
        left_x = rx
        right_x = rx + rw - 1
        bottom = rz
        top = rz + rh - 1
        # verticals
        ax.add_patch(Rectangle((left_x, rz), 1, rh, facecolor=wall_color, edgecolor="none", alpha=0.55))
        ax.add_patch(Rectangle((right_x, rz), 1, rh, facecolor=wall_color, edgecolor="none", alpha=0.55))
        # far horizontal
        if bottom != corridor_side_z:
            ax.add_patch(Rectangle((rx, bottom), rw, 1, facecolor=wall_color, edgecolor="none", alpha=0.55))
        if top != corridor_side_z:
            ax.add_patch(Rectangle((rx, top), rw, 1, facecolor=wall_color, edgecolor="none", alpha=0.55))

    if top_z is not None:
        for room in layout.get("rooms_top", []):
            add_perimeter(room, top_z)
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
            fontsize=7,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=5,
        )


def animate(path: str, save: str, fps: int, skip: int, trail: int, dpi: int, layout_path: Optional[str], cell_m: float, speed_solo: float, summary_path: Optional[str] = None) -> str:
    frames = load_frames(path)
    if not frames:
        raise RuntimeError(f"No frames loaded from {path}")

    layout_json = load_layout(layout_path)
    xmin, xmax, ymin, ymax = compute_extent(frames, layout_json)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    # keep cells square regardless of figure size
    try:
        ax.set_box_aspect((ymax - ymin) / (xmax - xmin))
    except Exception:
        pass
    # remove axes labels/title to keep canvas tight for animation frames
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    # pure white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    # remove ticks/spines and outer margins so content touches canvas
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    try:
        # small margins; leave a thin band on top for header and a thin band at bottom for footer
        fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.97)
    except Exception:
        pass

    # ---- layout overlay ----
    if layout_json:
        # walls and labels
        _draw_room_perimeters(ax, layout_json)
        _draw_wall_rows(ax, layout_json)
        _draw_room_labels(ax, layout_json)
        # small exit labels
        frame = layout_json.get('frame')
        corridor = layout_json.get('corridor')
        if frame and corridor:
            mid_z = corridor['z'] + corridor['h']//2
            ax.text(frame['x1']+0.3, mid_z+0.3, 'ExitL', fontsize=6, color='green', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.7), zorder=6)
            ax.text(frame['x2']-0.3, mid_z+0.3, 'ExitR', fontsize=6, color='green', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.7), zorder=6)

    # live heatmaps (occupants: Reds, responders: Blues) accumulating visits per cell
    # Define a grid aligned to cell edges so 1x1 cells map cleanly to pixels.
    grid_x0 = int(np.floor(xmin))
    grid_y0 = int(np.floor(ymin))
    grid_w = int(np.ceil(xmax) - grid_x0)
    grid_h = int(np.ceil(ymax) - grid_y0)

    # Enforce exact square cells by snapping limits to integer grid edges and locking box aspect
    ax.set_xlim(grid_x0, grid_x0 + grid_w)
    ax.set_ylim(grid_y0, grid_y0 + grid_h)
    try:
        ax.set_aspect('equal', adjustable='box')
        ax.set_box_aspect(grid_h / max(1, grid_w))
    except Exception:
        pass

    heat_occ = np.zeros((grid_h, grid_w), dtype=np.float32)
    heat_res = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Single RGBA overlay that accumulates permanently (no fading)
    # Use pcolormesh so each 1x1 cell is a perfect square fully filled
    mix_rgba = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
    x_edges = np.arange(grid_x0, grid_x0 + grid_w + 1)
    y_edges = np.arange(grid_y0, grid_y0 + grid_h + 1)
    pmesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.zeros((grid_h, grid_w), dtype=np.float32),
        shading='flat',
        edgecolors='none',
        zorder=10.0,
    )
    pmesh.set_array(None)
    pmesh.set_facecolors(mix_rgba.reshape(-1, 4))

    # live markers as circles centered at cell centers; diameter = cell diagonal (sqrt(2))
    occ_patches = []
    res_patches = []
    res_texts = []
    radius = math.sqrt(2) / 2.0
    # legend removed to avoid colored lines

    # Header text: just above the axes, left-aligned, tightly attached
    header_text = ax.text(0.005, 1.005, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=10, zorder=20)

    # Footer room order (static): try to load from summary JSON if provided or detect default
    room_order_str = None
    summary_json = None
    cand = summary_path
    if cand is None:
        # best-effort default (for det): logs/det_baseline.json next to frames
        cand = os.path.join(os.path.dirname(path), 'det_baseline.json')
    try:
        if cand and os.path.exists(cand):
            with open(cand, 'r', encoding='utf-8') as sf:
                summary_json = json.load(sf)
        if isinstance(summary_json, dict) and isinstance(summary_json.get('room_order'), list):
            seq = []
            for x in summary_json['room_order']:
                # unify to 1-based: R0->R1, R5->R6
                sx = str(x)
                if sx.startswith('R'):
                    try:
                        k = int(sx[1:]) + 1
                        sx = f"R{k}"
                    except Exception:
                        pass
                seq.append(sx)
            room_order_str = ' → '.join(seq)
    except Exception:
        room_order_str = None

    footer_text = None
    if room_order_str:
        footer_text = ax.text(0.005, -0.055, f"order: {room_order_str}", transform=ax.transAxes, ha='left', va='top', fontsize=9, zorder=20,
                              bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.7))

    frame_indices = list(range(0, len(frames), max(1, skip)))

    def init():
        # remove any previous patches
        for p in occ_patches:
            try:
                p.remove()
            except Exception:
                pass
        occ_patches.clear()
        for p in res_patches:
            try:
                p.remove()
            except Exception:
                pass
        res_patches.clear()
        for t in res_texts:
            try:
                t.remove()
            except Exception:
                pass
        res_texts.clear()
        header_text.set_text('')
        # reset accumulators and overlay
        heat_occ[:] = 0.0
        heat_res[:] = 0.0
        mix_rgba[:] = 0.0
        pmesh.set_facecolors(mix_rgba.reshape(-1, 4))
        return pmesh, header_text

    def update(idx):
        fr = frames[frame_indices[idx]]
        occ_xy = fr['occ_xy']
        res_xy = fr['res_xy']
        res_id = fr['res_id']

        # update marker circles (centers at cell centers, diameter = sqrt(2))
        for p in occ_patches:
            try:
                p.remove()
            except Exception:
                pass
        occ_patches.clear()
        for p in res_patches:
            try:
                p.remove()
            except Exception:
                pass
        res_patches.clear()
        for t in res_texts:
            try:
                t.remove()
            except Exception:
                pass
        res_texts.clear()
        if occ_xy.size:
            centers = occ_xy + 0.5
            for (cx, cy) in centers:
                c = Circle((cx, cy), radius=radius, facecolor='red', edgecolor='red', linewidth=0.8, zorder=12.0, alpha=0.9)
                ax.add_patch(c)
                occ_patches.append(c)
        if res_xy.size:
            centers = res_xy + 0.5
            for (cx, cy), rid in zip(centers, res_id):
                c = Circle((cx, cy), radius=radius, facecolor='blue', edgecolor='blue', linewidth=0.8, zorder=12.1, alpha=0.9)
                ax.add_patch(c)
                res_patches.append(c)
                # small id label near responder (r1, r2, ...)
                txt = ax.text(cx, cy-0.35, f"r{int(rid)+1}", fontsize=6, color='blue', ha='center', va='top', zorder=12.2,
                              bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.7))
                res_texts.append(txt)

        # accumulate per-cell heat (one count per entity per frame)
        if occ_xy.size:
            # To grid indices (row=i for y, col=j for x)
            js = np.floor(occ_xy[:, 0]).astype(int) - grid_x0
            is_ = np.floor(occ_xy[:, 1]).astype(int) - grid_y0
            mask = (is_ >= 0) & (is_ < grid_h) & (js >= 0) & (js < grid_w)
            if np.any(mask):
                np.add.at(heat_occ, (is_[mask], js[mask]), 1.0)
        if res_xy.size:
            js = np.floor(res_xy[:, 0]).astype(int) - grid_x0
            is_ = np.floor(res_xy[:, 1]).astype(int) - grid_y0
            mask = (is_ >= 0) & (is_ < grid_h) & (js >= 0) & (js < grid_w)
            if np.any(mask):
                np.add.at(heat_res, (is_[mask], js[mask]), 1.0)

        # Build a monotonic, non-fading color overlay from absolute visit counts
        # Scale factors control how fast color deepens; tune as needed.
        K_OCC = 0.08
        K_RES = 0.15
        r = np.clip(heat_occ * K_OCC, 0.0, 1.0)
        b = np.clip(heat_res * K_RES, 0.0, 1.0)
        a = np.clip(np.maximum(r, b), 0.0, 1.0)
        mix_rgba[..., 0] = r
        mix_rgba[..., 1] = 0.0
        mix_rgba[..., 2] = b
        mix_rgba[..., 3] = a
        pmesh.set_facecolors(mix_rgba.reshape(-1, 4))

        # ETA in hh:mm:ss: prefer per-frame eta_seconds/eta_hms if present; else fallback to simple mapping
        eta_secs = fr.get('eta_seconds')
        eta_hms = fr.get('eta_hms')
        if eta_hms is None:
            if eta_secs is None:
                eta_secs = float(fr['time']) * (cell_m / max(1e-6, speed_solo))
            eta_str = _format_eta(float(eta_secs))
        else:
            eta_str = str(eta_hms)
        # Header text: RL reward if available; else det stats; always append ETA
        if fr['reward'] is not None:
            if fr.get('unswept') is not None:
                header_text.set_text(f"t={fr['time']}  {eta_str}  r={fr['reward']:.2f}  Rsum={fr['cumulative_reward']:.2f}  unswept={fr['unswept']}")
            else:
                header_text.set_text(f"t={fr['time']}  {eta_str}  r={fr['reward']:.2f}  Rsum={fr['cumulative_reward']:.2f}")
        else:
            evac = fr.get('evac_count')
            tot = fr.get('occ_total')
            uns = fr.get('unswept')
            if evac is not None and tot is not None and tot > 0:
                loss = tot - evac
                if uns is not None:
                    header_text.set_text(f"t={fr['time']}  {eta_str}  evac={evac}/{tot}  loss={loss}  unswept={uns}")
                else:
                    header_text.set_text(f"t={fr['time']}  {eta_str}  evac={evac}/{tot}  loss={loss}")
            else:
                header_text.set_text(f"t={fr['time']}  {eta_str}")
        return pmesh, header_text

    ani = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, interval=max(10, int(1000/max(1,fps))), blit=False)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        if save.lower().endswith('.gif') and PILLOW_AVAILABLE:
            writer = PillowWriter(fps=fps)
            ani.save(
                save,
                writer=writer,
                dpi=dpi,
                savefig_kwargs={
                    "facecolor": "white",
                    "edgecolor": "none",
                },
            )
        else:
            # fallback: try to save as mp4 using default writer; or per-frame PNGs
            try:
                ani.save(
                    save,
                    dpi=dpi,
                    savefig_kwargs={
                        "facecolor": "white",
                        "edgecolor": "none",
                    },
                )
            except Exception:
                # last resort: dump frames
                frames_dir = os.path.join(os.path.dirname(save), 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                for i in range(len(frame_indices)):
                    update(i)
                    fig.savefig(os.path.join(frames_dir, f'frame_{i:04d}.png'), dpi=dpi)
        # Also save three static frames (first, middle, final) as heatmap images.
        # Each still reflects the cumulative state up to that time (like the GIF),
        # not just a single isolated frame.
        try:
            if frame_indices:
                out_dir = os.path.dirname(save)
                def save_upto(index: int, out_path: str):
                    # reset state and accumulate up to the given index
                    init()
                    for j in range(0, index + 1):
                        update(j)
                    fig.savefig(out_path, dpi=max(dpi, 200), bbox_inches='tight', pad_inches=0, facecolor='white')

                first_idx = 0
                mid_idx = len(frame_indices) // 2
                last_idx = len(frame_indices) - 1

                first_path = os.path.join(out_dir, 'heatmap_traj_first.png')
                mid_path = os.path.join(out_dir, 'heatmap_traj_mid.png')
                last_path = os.path.join(out_dir, 'heatmap_traj.png')

                save_upto(first_idx, first_path)
                save_upto(mid_idx, mid_path)
                save_upto(last_idx, last_path)
                print(f"Saved heatmap frames to {first_path}, {mid_path}, {last_path}")
        except Exception:
            pass
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
    # trail kept for backward CLI compatibility; ignored (heatmap replaces tails)
    parser.add_argument('--trail', type=int, default=0, help='Deprecated: trails removed; heatmap used instead')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    parser.add_argument('--layout', default='layout/baseline.json', help='Layout JSON for overlay')
    parser.add_argument('--summary', default=None, help='Optional summary JSON (det) to read room_order for footer')
    parser.add_argument('--cell-m', type=float, default=0.5, help='Grid cell size in meters for ETA mapping (default: 0.5)')
    parser.add_argument('--speed-solo', type=float, default=0.8, help='Solo speed (m/s) for ETA mapping (default: 0.8)')
    args = parser.parse_args()

    animate(
        args.path,
        args.save,
        fps=args.fps,
        skip=args.skip,
        trail=args.trail,
        dpi=args.dpi,
        layout_path=args.layout,
        cell_m=args.cell_m,
        speed_solo=args.speed_solo,
        summary_path=args.summary,
    )


if __name__ == '__main__':
    main()

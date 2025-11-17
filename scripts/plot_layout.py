#!/usr/bin/env python3
"""Render the layout JSON as a static floor map for quick inspection."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LayoutDict = Dict[str, object]
Coord = Tuple[float, float]


@dataclass
class LayoutRect:
    """Represents a rectangular feature in the layout."""

    name: str
    x: float
    z: float
    w: float
    h: float
    category: str
    block: Optional[str]


CATEGORY_COLORS = {
    "room_top": "#f6d5c5",
    "room_bottom": "#d7ebf9",
    "room": "#ece3f7",
    "corridor": "#f5f5f5",
    "feature": "#f0f0f0",
}

BLOCK_COLOR_MAP = {
    "orange_wool": "#fb8b24",
    "green_wool": "#2b8447",
    "pink_wool": "#f48fb1",
    "cyan_wool": "#26c6da",
    "purple_wool": "#7e57c2",
    "blue_wool": "#1e88e5",
    "yellow_wool": "#fdd835",
    "red_wool": "#f44336",
    "light_blue_wool": "#81d4fa",
    "light_gray_wool": "#cfd8dc",
    "gray_wool": "#9e9e9e",
    "white_wool": "#fafafa",
    "white_concrete": "#f5f5f5",
    "black_concrete": "#424242",
    "light_blue_concrete": "#b3e5fc",
}


def load_layout(path: str) -> LayoutDict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _rect_from_entry(entry: Dict, name: str, category: str) -> LayoutRect:
    geom = _extract_geometry(entry)
    if geom is None:
        raise ValueError(f"No geometry available for entry {name}")
    x, z, w, h = geom
    return LayoutRect(
        name=name,
        x=x,
        z=z,
        w=w,
        h=h,
        category=category,
        block=entry.get("block"),
    )


def _iter_rectangles(list_data: Sequence[Dict], category: str, prefix: str) -> Iterable[LayoutRect]:
    for idx, entry in enumerate(list_data):
        label = entry.get("name") or entry.get("id") or f"{prefix}_{idx + 1}"
        try:
            yield _rect_from_entry(entry, label, category)
        except ValueError:
            continue


def _extract_geometry(entry: Dict) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(entry, dict):
        return None
    coords = ("x", "z", "w", "h")
    if all(key in entry for key in coords):
        return tuple(float(entry[key]) for key in coords)
    shape = entry.get("shape")
    if isinstance(shape, dict) and all(key in shape for key in coords):
        return tuple(float(shape[key]) for key in coords)
    return None


def rectangles_from_layout(layout: LayoutDict) -> List[LayoutRect]:
    rects: List[LayoutRect] = []

    rects.extend(_iter_rectangles(layout.get("rooms_top", []), "room_top", "room_top"))
    rects.extend(_iter_rectangles(layout.get("rooms_bottom", []), "room_bottom", "room_bottom"))
    rects.extend(_iter_rectangles(layout.get("rooms", []), "room", "room"))

    corridor = layout.get("corridor")
    if corridor:
        rects.append(_rect_from_entry(corridor, corridor.get("name", "corridor"), "corridor"))

    for idx, corridor in enumerate(layout.get("corridors", [])):
        label = corridor.get("name") or f"corridor_{idx+1}"
        rects.append(_rect_from_entry(corridor, label, "corridor"))

    handled = {"rooms_top", "rooms_bottom", "rooms", "corridor", "corridors"}
    for key, value in layout.items():
        if key in handled:
            continue
        if isinstance(value, dict):
            label = value.get("name") or key
            try:
                rects.append(_rect_from_entry(value, label, "feature"))
            except ValueError:
                pass
        elif isinstance(value, list):
            for idx, entry in enumerate(value):
                if not isinstance(entry, dict):
                    continue
                label = entry.get("name") or f"{key}_{idx + 1}"
                try:
                    rects.append(_rect_from_entry(entry, label, "feature"))
                except ValueError:
                    continue
    return rects


def _door_coordinates(doors: Dict) -> List[Coord]:
    coords: List[Coord] = []
    xs = doors.get("xs", [])
    top_z = doors.get("topZ")
    bottom_z = doors.get("bottomZ")
    for x in xs:
        if top_z is not None:
            coords.append((float(x), float(top_z)))
        if bottom_z is not None:
            coords.append((float(x), float(bottom_z)))
    return coords


def door_points(layout: LayoutDict) -> List[Coord]:
    raw = layout.get("doors")
    if not raw:
        return []
    if isinstance(raw, dict):
        return _door_coordinates(raw)
    coords = []
    for entry in raw:
        x = entry.get("x")
        z = entry.get("z")
        if x is not None and z is not None:
            coords.append((float(x), float(z)))
    return coords


def _point_locations(layout: LayoutDict, key: str) -> List[Coord]:
    points: List[Coord] = []
    for entry in layout.get(key, []):
        if not isinstance(entry, dict):
            continue
        x = entry.get("x")
        z = entry.get("z")
        if x is None or z is None:
            coord = entry.get("coord")
            if isinstance(coord, Sequence) and not isinstance(coord, str) and len(coord) >= 2:
                x, z = coord[0], coord[1]
        if x is not None and z is not None:
            points.append((float(x), float(z)))
    return points


def layout_bounds(rects: Sequence[LayoutRect], frame: Optional[Dict]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    zs: List[float] = []
    for rect in rects:
        xs.append(rect.x)
        xs.append(rect.x + rect.w)
        zs.append(rect.z)
        zs.append(rect.z + rect.h)
    if frame:
        x1 = float(frame.get("x1", 0))
        x2 = float(frame.get("x2", x1))
        z1 = float(frame.get("z1", 0))
        z2 = float(frame.get("z2", z1))
        xs.extend([x1, x2 + 1])
        zs.extend([z1, z2 + 1])
    if not xs or not zs:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs), max(xs), min(zs), max(zs)


def _category_color(rect: LayoutRect) -> str:
    block = rect.block
    if block:
        color = BLOCK_COLOR_MAP.get(block.lower())
        if color:
            return color
    return CATEGORY_COLORS.get(rect.category, "#d3d3d3")


def plot_layout(
    layout: LayoutDict,
    rects: Sequence[LayoutRect],
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 200,
) -> None:
    frame = layout.get("frame")
    x_min, x_max, z_min, z_max = layout_bounds(rects, frame)

    width = max(1.0, x_max - x_min)
    height = max(1.0, z_max - z_min)
    figsize = (
        min(20.0, max(6.0, width / 8.0)),
        min(14.0, max(4.0, height / 8.0)),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(x_min - 1.0, x_max + 1.0)
    ax.set_ylim(z_min - 1.0, z_max + 1.0)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Layout: {layout.get('name', 'Unnamed')}", fontsize=14)

    for rect in rects:
        ax.add_patch(
            Rectangle(
                (rect.x, rect.z),
                rect.w,
                rect.h,
                facecolor=_category_color(rect),
                edgecolor="#444444",
                linewidth=0.8,
                alpha=0.9,
            )
        )
        if rect.name:
            cx = rect.x + rect.w / 2
            cz = rect.z + rect.h / 2
            ax.text(
                cx,
                cz,
                rect.name,
                ha="center",
                va="center",
                fontsize=7,
                color="#222222",
                clip_on=True,
            )

    if frame:
        x1 = float(frame.get("x1", 0))
        z1 = float(frame.get("z1", 0))
        x2 = float(frame.get("x2", x1))
        z2 = float(frame.get("z2", z1))
        ax.add_patch(
            Rectangle(
                (x1, z1),
                (x2 - x1) + 1,
                (z2 - z1) + 1,
                fill=False,
                edgecolor="#222222",
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
            )
        )

    door_pts = door_points(layout)
    if door_pts:
        xs, zs = zip(*door_pts)
        ax.scatter(xs, zs, marker="s", s=50, color="#a52a2a", edgecolor="black", zorder=5)

    exit_pts = _point_locations(layout, "exits")
    if exit_pts:
        xs, zs = zip(*exit_pts)
        ax.scatter(xs, zs, marker="X", s=120, color="#2e7d32", edgecolor="black", linewidths=1.2, zorder=6)

    entrance_pts = _point_locations(layout, "entrances")
    if entrance_pts:
        xs, zs = zip(*entrance_pts)
        ax.scatter(xs, zs, marker="o", s=80, facecolor="#ffeb3b", edgecolor="black", zorder=6)

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved layout map to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render layout JSON as a floor map.")
    parser.add_argument("--layout", default="layout/baseline.json", help="Path to layout JSON")
    parser.add_argument("--save", default=None, help="Path to save the PNG output")
    parser.add_argument("--show", action="store_true", help="Display the figure after rendering")
    parser.add_argument("--dpi", type=int, default=200, help="Resolution for saved image")
    args = parser.parse_args()

    if not args.save and not args.show:
        args.show = True

    layout = load_layout(args.layout)
    rects = rectangles_from_layout(layout)
    plot_layout(layout, rects, save_path=args.save, show=args.show, dpi=args.dpi)


if __name__ == "__main__":
    main()

"""Visualization helpers that convert timelines into lightweight GIFs."""
from __future__ import annotations

import io
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

SEGMENT_COLOURS: Dict[str, str] = {
    "move": "#1f77b4",
    "room": "#ff7f0e",
    "egress": "#2ca02c",
}


def _group_segments(timeline: Sequence[dict]) -> DefaultDict[str, List[dict]]:
    grouped: DefaultDict[str, List[dict]] = defaultdict(list)
    for entry in timeline:
        grouped[entry["responder_id"]].append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda seg: seg["start"])
    return grouped


def _draw_frame(grouped: Dict[str, List[dict]], current_time: float, makespan: float, dpi: int) -> Image.Image:
    responders = list(sorted(grouped))
    fig_height = max(2.5, 1.2 * len(responders))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    for idx, responder in enumerate(responders):
        segments = grouped[responder]
        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            width = max(0.0, end - start)
            colour = SEGMENT_COLOURS.get(segment["segment_type"], "#bbbbbb")
            ax.broken_barh([(start, width)], (idx - 0.35, 0.7), facecolors=colour, alpha=0.25, edgecolor="none")
            if current_time > start:
                progress = min(current_time, end) - start
                if progress > 0:
                    ax.broken_barh([(start, progress)], (idx - 0.35, 0.7), facecolors=colour, alpha=0.9)
        ax.text(
            makespan * 1.01 + 0.5,
            idx,
            responder,
            color="white",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.axvline(current_time, color="#ffffff", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlim(0.0, max(1.0, makespan * 1.05))
    ax.set_ylim(-0.6, len(responders) - 0.4)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Responder", color="white")
    ax.set_title("Responder progress", color="white", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
    ax.tick_params(colors="white")
    ax.set_yticks([])
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf).convert("P")
    buf.close()
    return frame


def render_gantt_gif(
    timeline: Sequence[dict],
    output_path: str,
    frame_count: int = 40,
    dpi: int = 120,
) -> None:
    """Render a GIF showing responder progress through the timeline."""

    if not timeline:
        return

    grouped = _group_segments(timeline)
    makespan = max(float(entry["end"]) for entry in timeline)
    if makespan <= 0.0:
        makespan = 1.0

    frames: List[Image.Image] = []
    denom = max(frame_count - 1, 1)
    for idx in range(frame_count):
        current_time = makespan * idx / denom
        frames.append(_draw_frame(grouped, current_time, makespan, dpi))

    first, *rest = frames
    first.save(output_path, format="GIF", save_all=True, append_images=rest, duration=120, loop=0)


__all__ = ["render_gantt_gif"]

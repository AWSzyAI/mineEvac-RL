#!/usr/bin/env python3
"""Extract key frames from a GIF into PNGs.

For a given GIF, save:
- first frame
- frame at 1/4 of total frames
- frame at 3/4 of total frames
- last frame

Frames are not re-rendered; they are copied directly from the GIF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def extract_key_frames(gif_path: str, prefix: str) -> None:
    gif = Image.open(gif_path)
    n = getattr(gif, "n_frames", 1)
    if n <= 0:
        return

    idx_first = 0
    idx_q1 = max(0, n // 4)
    idx_q3 = max(0, (3 * n) // 4)
    idx_last = max(0, n - 1)

    frames = [
        (idx_first, "first"),
        (idx_q1, "q1"),
        (idx_q3, "q3"),
        (idx_last, "last"),
    ]

    prefix_path = Path(prefix)
    prefix_dir = prefix_path.parent
    prefix_dir.mkdir(parents=True, exist_ok=True)

    for idx, tag in frames:
        if idx < 0 or idx >= n:
            continue
        try:
            gif.seek(idx)
        except EOFError:
            continue
        frame = gif.convert("RGBA")
        out_path = prefix_dir / f"{prefix_path.name}_{tag}.png"
        frame.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract key frames from a GIF into PNGs.")
    parser.add_argument("--gif", required=True, help="Input GIF path.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Output prefix (directory + base name, without extension).",
    )
    args = parser.parse_args()

    extract_key_frames(args.gif, args.prefix)


if __name__ == "__main__":
    main()


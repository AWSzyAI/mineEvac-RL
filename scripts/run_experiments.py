#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Tuple

from pathlib import Path

# ensure src/ is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.append(str(SRC_DIR))

from sim_sweep_det import run_once, load_layout, default_init_positions


def alt_inits(layout_path: str, responders: int) -> List[Tuple[str, List[Tuple[int,int]]]]:
    layout = load_layout(layout_path)
    # ends
    ends = default_init_positions(layout, responders)
    # center
    corr = layout["corridor"]
    mid_x = corr["x"] + corr["w"] // 2
    mid_z = corr["z"] + corr["h"] // 2
    if responders == 1:
        center = [(mid_x, mid_z)]
    else:
        dx = max(2, corr["w"] // (responders + 1))
        xs = [mid_x + (i - (responders-1)/2)*dx for i in range(responders)]
        center = [(int(x), mid_z) for x in xs]
    # left-packed
    left = []
    for i in range(responders):
        left.append((corr["x"] + 1 + i*2, mid_z))
    return [("ends", ends), ("center", center), ("left", left)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="*", default=[
        "layout/baseline.json", "layout/layout_A.json", "layout/layout_B.json"
    ])
    ap.add_argument("--responders", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--per_room", nargs="*", type=int, default=[3, 5])
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--out", default="logs/det_experiments.jsonl")
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    md_path = os.path.splitext(args.out)[0] + ".md"

    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        for lp in args.layouts:
            for rnum in args.responders:
                for pr in args.per_room:
                    for label, inits in alt_inits(lp, rnum):
                        res = run_once(lp, num_responders=rnum, per_room=pr, max_steps=args.max_steps, init_positions=inits)
                        # override init positions label (we used default inside run_once)
                        res["init_label"] = label
                        json.dump(res, f, ensure_ascii=False)
                        f.write("\n")
                        rows.append(res)

    # markdown summary
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("# Deterministic Sweep Experiments\n\n")
        for r in rows:
            mf.write(f"- {r['layout']} | responders={r['responders']} | per_room={r['per_room']} | time={r['time']} | evacuated={r['evacuated']} | order={','.join(r['room_order'])}\n")
    print(f"Saved: {args.out} and {md_path}")


if __name__ == "__main__":
    main()

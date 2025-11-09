#!/usr/bin/env python3
"""
Summarise training/evaluation logs into log/report.md.

Usage:
    python scripts/generate_report.py \
        --progress logs/ppo_baseline/progress.csv \
        --monitor logs/ppo_baseline/monitor.csv \
        --eval logs/eval_episode.jsonl \
        --output log/report.md
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_progress_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            numeric_row = {k: _safe_float(v) for k, v in row.items()}
            rows.append(numeric_row)
    return rows


def summarise_progress(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not rows:
        return {}
    last = rows[-1]
    ep_rewards = [r["rollout/ep_rew_mean"] for r in rows if r.get("rollout/ep_rew_mean") is not None]
    return {
        "final_timesteps": last.get("time/total_timesteps"),
        "final_iterations": last.get("time/iterations"),
        "final_episode_reward": last.get("rollout/ep_rew_mean"),
        "best_episode_reward": max(ep_rewards) if ep_rewards else None,
        "worst_episode_reward": min(ep_rewards) if ep_rewards else None,
        "avg_episode_reward": mean(ep_rewards) if ep_rewards else None,
        "final_learning_rate": last.get("train/learning_rate"),
        "final_value_loss": last.get("train/value_loss"),
        "final_entropy": last.get("train/entropy_loss"),
    }


def load_monitor_csv(path: Path) -> List[Tuple[float, float]]:
    metrics: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()  # json metadata
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            rew = _safe_float(parts[0])
            length = _safe_float(parts[1])
            if rew is None or length is None:
                continue
            metrics.append((rew, length))
    return metrics


def summarise_monitor(entries: List[Tuple[float, float]]) -> Dict[str, Optional[float]]:
    if not entries:
        return {}
    rewards = [r for r, _ in entries]
    lengths = [l for _, l in entries]
    return {
        "episodes": len(entries),
        "reward_mean": mean(rewards),
        "reward_best": max(rewards),
        "reward_worst": min(rewards),
        "length_mean": mean(lengths),
    }


def load_eval_jsonl(path: Path) -> List[dict]:
    frames: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def summarise_eval(frames: List[dict]) -> Dict[str, Optional[float]]:
    if not frames:
        return {}
    last = frames[-1]
    occupants = last.get("occupants", [])
    evac_count = sum(1 for occ in occupants if occ.get("evacuated"))
    tau = last.get("tau") or {}
    visited = [(room_id, step) for room_id, step in tau.items() if step is not None]
    visited.sort(key=lambda item: item[1])
    visited_order = " -> ".join(room for room, _ in visited) if visited else None
    missing_rooms = [room_id for room_id, step in tau.items() if step is None]
    return {
        "steps": len(frames),
        "final_cumulative_reward": last.get("cumulative_reward"),
        "final_step_reward": last.get("reward"),
        "final_responder_pos": last.get("responder_pos"),
        "occupants_total": len(occupants),
        "occupants_evacuated": evac_count,
        "rooms_visited_order": visited_order,
        "rooms_not_visited": ", ".join(missing_rooms) if missing_rooms else None,
    }


def format_section(title: str, rows: Dict[str, Optional[float]]) -> str:
    if not rows:
        return f"## {title}\n\n_无可用数据_\n"
    lines = [f"## {title}", ""]
    for key, value in rows.items():
        pretty_key = key.replace("_", " ").capitalize()
        if isinstance(value, float):
            line = f"- **{pretty_key}**: {value:.4f}"
        else:
            line = f"- **{pretty_key}**: {value}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate training/eval report.")
    parser.add_argument("--progress", default="logs/ppo_baseline/progress.csv")
    parser.add_argument("--monitor", default="logs/ppo_baseline/monitor.csv")
    parser.add_argument("--eval", dest="eval_path", default="logs/eval_episode.jsonl")
    parser.add_argument("--output", default="log/report.md")
    args = parser.parse_args()

    progress_rows = load_progress_csv(Path(args.progress)) if Path(args.progress).exists() else []
    monitor_entries = load_monitor_csv(Path(args.monitor)) if Path(args.monitor).exists() else []
    eval_frames = load_eval_jsonl(Path(args.eval_path)) if Path(args.eval_path).exists() else []

    progress_summary = summarise_progress(progress_rows)
    monitor_summary = summarise_monitor(monitor_entries)
    eval_summary = summarise_eval(eval_frames)

    report_lines = [
        "# 训练与评估报告",
        "",
        format_section("训练进展 (progress.csv)", progress_summary),
        format_section("Episodes 汇总 (monitor.csv)", monitor_summary),
        format_section("评估回合 (eval_episode.jsonl)", eval_summary),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✅ 保存报告到 {output_path}")


if __name__ == "__main__":
    main()

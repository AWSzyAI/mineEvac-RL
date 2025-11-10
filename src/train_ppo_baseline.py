# filename: src/train_ppo_baseline.py

import json
import argparse
from contextlib import suppress
import os
import subprocess
import sys
from typing import Tuple

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

from mine_evac_env import MineEvacEnv


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_env_fn(layout_path: str, max_steps: int, seed_offset: int = 0):
    def _init():
        env = MineEvacEnv(layout_path=layout_path, max_steps=max_steps)
        env.reset(seed=seed_offset)
        return env

    return _init

def run_deterministic_eval(model_path: str, layout_path: str, device: str, project_root: str) -> str:
    eval_env = MineEvacEnv(layout_path=layout_path, max_steps=500)
    obs, _ = eval_env.reset()
    model = PPO.load(model_path, env=eval_env, device=device)

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    eval_log_path = os.path.join(log_dir, "eval_episode.jsonl")
    cum_reward = 0.0
    with open(eval_log_path, "w", encoding="utf-8") as f:
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = eval_env.step(int(action))
            cum_reward += float(reward)
            line = {
                "t": step_info.get("t"),
                "reward": float(reward),
                "cumulative_reward": float(cum_reward),
                "responder_pos": step_info.get("responder_pos"),
                "occupants": step_info.get("occupants"),
                "tau": step_info.get("tau"),
                "room_cleared": step_info.get("room_cleared"),
                "room_occupancy": step_info.get("room_occupancy"),
                "needs_remaining": step_info.get("needs_remaining"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if terminated or truncated:
                break
    return eval_log_path


def generate_visuals(eval_log_path: str, layout_path: str, project_root: str) -> Tuple[str, str]:
    scripts_dir = os.path.join(project_root, "scripts")
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    heatmap_path = os.path.join(output_dir, "heatmap.png")
    gif_path = os.path.join(output_dir, "sweep_anim.gif")

    commands = [
        [
            sys.executable,
            os.path.join(scripts_dir, "visualize_heatmap.py"),
            "--path",
            eval_log_path,
            "--layout",
            layout_path,
            "--save",
            heatmap_path,
        ],
        [
            sys.executable,
            os.path.join(scripts_dir, "animate_sweep.py"),
            "--path",
            eval_log_path,
            "--layout",
            layout_path,
            "--save",
            gif_path,
        ],
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️ Failed to generate visualization via {' '.join(cmd)}: {exc}")

    print(f"🖼 Heatmap saved to {heatmap_path}")
    print(f"🎞 GIF saved to {gif_path}")
    return heatmap_path, gif_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default=None, help='Path to layout JSON (default: layout/baseline.json)')
    ap.add_argument('--timesteps', type=int, default=400_000, help='Training timesteps (ignored if --eval-only)')
    ap.add_argument('--eval-only', action='store_true', help='Skip training; evaluate best/last model and generate visuals')
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout_path = args.layout or os.path.join(project_root, "layout", "baseline.json")
    log_dir = os.path.join(project_root, "logs", "ppo_baseline")
    os.makedirs(log_dir, exist_ok=True)

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = _get_device()

    total_envs = max(1, (os.cpu_count() or 1) - 1)
    if total_envs == 1:
        env = DummyVecEnv([_make_env_fn(layout_path, max_steps=500)])
    else:
        env = SubprocVecEnv([
            _make_env_fn(layout_path, max_steps=500, seed_offset=i)
            for i in range(total_envs)
        ])
    env = VecMonitor(env, os.path.join(log_dir, "monitor.csv"))
    rollout_steps = max(32, 2048 // total_envs)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=rollout_steps,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.97,
        device=device,
        ent_coef=0.01,
    )

    new_logger = configure(log_dir, ["stdout"])
    model.set_logger(new_logger)

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ppo_mine_evac_baseline")

    interrupted = False
    if not args.eval_only:
        try:
            # Set up evaluation during training to save the best model
            eval_env = DummyVecEnv([_make_env_fn(layout_path, max_steps=500)])
            eval_env = VecMonitor(eval_env)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=models_dir,
                log_path=os.path.join(log_dir, "eval"),
                eval_freq=max(rollout_steps, 10_000),
                n_eval_episodes=3,
                deterministic=True,
                render=False,
            )

            model.learn(total_timesteps=args.timesteps, callback=eval_callback)
        except KeyboardInterrupt:
            interrupted = True
            print("⚠️ Training interrupted by user, saving current weights...")
        finally:
            with suppress(EOFError, BrokenPipeError):
                env.close()
            model.save(model_path)
            print(f"✅ Model saved to {model_path}.zip")

    # Prefer best model (saved by EvalCallback) if available
    best_model_zip = os.path.join(models_dir, "best_model.zip")
    model_to_eval = best_model_zip if os.path.exists(best_model_zip) else model_path
    if os.path.exists(best_model_zip):
        print(f"🌟 Using best_model.zip for evaluation: {best_model_zip}")
    else:
        print(f"ℹ️ best_model.zip not found; evaluating last checkpoint: {model_path}")

    eval_log_path = run_deterministic_eval(model_to_eval, layout_path, device, project_root)
    print(f"📄 Eval episode logged to {eval_log_path}")
    generate_visuals(eval_log_path, layout_path, project_root)
    if interrupted:
        print("💡 Training was interrupted, but evaluation + visualizations still completed.")


if __name__ == "__main__":
    main()

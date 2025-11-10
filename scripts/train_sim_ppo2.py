#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / 'src'))
from sim_env_rl2 import TwoResponderEscortEnv


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default='layout/baseline.json')
    ap.add_argument('--per_room', type=int, default=5)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--timesteps', type=int, default=400000)
    ap.add_argument('--save', default='models/n_r_2.zip')
    ap.add_argument('--logdir', default='logs/sim_rl2')
    args = ap.parse_args()

    device = get_device()

    def make_env():
        def _init():
            return TwoResponderEscortEnv(layout_path=args.layout, per_room=args.per_room, max_steps=args.max_steps)
        return _init

    env = DummyVecEnv([make_env()])
    model = PPO('MlpPolicy', env, verbose=1, n_steps=1024, batch_size=256, gamma=0.995, learning_rate=3e-4, ent_coef=0.01, device=device)
    os.makedirs(args.logdir, exist_ok=True)
    model.set_logger(configure(args.logdir, ["stdout"]))

    # evaluation callback to save best_model.zip during training
    eval_env = DummyVecEnv([make_env()])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(args.save).parent),
        log_path=args.logdir,
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model.save(args.save)
    print(f"Saved model to {args.save}")
    best_path = Path(args.save).parent / 'best_model.zip'
    if best_path.exists():
        print(f"Best model saved to {best_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / 'src'))
from sim_env_rl import SingleResponderEscortEnv


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
    ap.add_argument('--timesteps', type=int, default=300000)
    ap.add_argument('--save', default='models/n_r_1.zip')
    ap.add_argument('--logdir', default='logs/sim_rl')
    args = ap.parse_args()

    device = get_device()

    def make_env():
        def _init():
            return SingleResponderEscortEnv(layout_path=args.layout, per_room=args.per_room, max_steps=args.max_steps)
        return _init

    env = DummyVecEnv([make_env()])

    model = PPO(
        'MlpPolicy', env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
        ent_coef=0.01,
        device=device,
    )

    os.makedirs(args.logdir, exist_ok=True)
    logger = configure(args.logdir, ["stdout"])
    model.set_logger(logger)

    model.learn(total_timesteps=args.timesteps)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model.save(args.save)
    print(f"Saved model to {args.save}")


if __name__ == '__main__':
    main()


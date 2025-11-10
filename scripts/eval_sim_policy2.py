#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / 'src'))
from sim_env_rl2 import TwoResponderEscortEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default='layout/baseline.json')
    ap.add_argument('--per_room', type=int, default=5)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--policy', default=None, help='Policy path; default: models/best_model.zip or models/n_r_2.zip')
    ap.add_argument('--frames', default='logs/policy2_eval_episode.jsonl')
    ap.add_argument('--save', default='logs/policy2_eval_summary.json')
    args = ap.parse_args()

    env = TwoResponderEscortEnv(layout_path=args.layout, per_room=args.per_room, max_steps=args.max_steps)
    # auto-select best model if not given
    policy_path = args.policy
    if policy_path is None:
        repo_root = REPO_ROOT
        cand_best = repo_root / 'models' / 'best_model.zip'
        cand_last = repo_root / 'models' / 'n_r_2.zip'
        if cand_best.exists():
            policy_path = str(cand_best)
        elif cand_last.exists():
            policy_path = str(cand_last)
        else:
            raise FileNotFoundError('No policy provided and neither models/best_model.zip nor models/n_r_2.zip found')
    model = PPO.load(policy_path, device='cpu')
    obs, _ = env.reset()
    os.makedirs(os.path.dirname(args.frames), exist_ok=True)
    f = open(args.frames, 'w', encoding='utf-8')

    def snapshot(t):
        return {
            'time': t,
            'responders': [
                { 'id': 0, 'x': env.r1[0], 'y': env.r1[1] },
                { 'id': 1, 'x': env.r2[0], 'y': env.r2[1] }
            ],
            'occupants': [
                { 'id': o['id'], 'x': o['pos'][0], 'y': o['pos'][1], 'evacuated': o['evac'] }
                for o in env.occupants
            ]
        }

    f.write(json.dumps(snapshot(0), ensure_ascii=False) + '\n')
    cum_reward = 0.0
    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        cum_reward += float(reward)
        f.write(json.dumps(snapshot(env.t), ensure_ascii=False) + '\n')
        if terminated or truncated:
            break
    f.close()

    result = {
        'time': env.t,
        'all_evacuated': all(o['evac'] for o in env.occupants),
        'evacuated': sum(1 for o in env.occupants if o['evac']),
        'layout': os.path.basename(args.layout),
        'policy': os.path.basename(policy_path),
        'cumulative_reward': cum_reward
    }
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w', encoding='utf-8') as sf:
        json.dump(result, sf, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

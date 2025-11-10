#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / 'src'))
from sim_env_rl import SingleResponderEscortEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default='layout/baseline.json')
    ap.add_argument('--per_room', type=int, default=5)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--frames', default='logs/policy_eval_episode.jsonl')
    ap.add_argument('--save', default='logs/policy_eval_summary.json')
    args = ap.parse_args()

    env = SingleResponderEscortEnv(layout_path=args.layout, per_room=args.per_room, max_steps=args.max_steps)
    model = PPO.load(args.policy, device='cpu')

    obs, _ = env.reset()
    os.makedirs(os.path.dirname(args.frames), exist_ok=True)
    f = open(args.frames, 'w', encoding='utf-8')

    def snapshot(t):
        return {
            'time': t,
            'responders': [{ 'id': 0, 'x': env.responder[0], 'y': env.responder[1] }],
            'occupants': [
                { 'id': o['id'], 'x': o['pos'][0], 'y': o['pos'][1], 'evacuated': o['evac'] }
                for o in env.occupants
            ]
        }

    f.write(json.dumps(snapshot(0), ensure_ascii=False) + '\n')
    # track room first entry order
    room_first: dict[str, int] = {r.id: None for r in env.rooms}
    cum_reward = 0.0
    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        cum_reward += float(reward)
        f.write(json.dumps(snapshot(env.t), ensure_ascii=False) + '\n')
        # update room entry
        cur_room = None
        for r in env.rooms:
            x, z = env.responder
            if r.contains((x, z)):
                cur_room = r
                break
        if cur_room is not None and room_first[cur_room.id] is None:
            room_first[cur_room.id] = env.t
        if terminated or truncated:
            break
    f.close()

    order = [rid for rid, t in sorted(((rid, t) for rid, t in room_first.items() if t is not None), key=lambda p: p[1])]
    result = {
        'time': env.t,
        'all_evacuated': all(o['evac'] for o in env.occupants),
        'evacuated': sum(1 for o in env.occupants if o['evac']),
        'room_order': order,
        'layout': os.path.basename(args.layout),
        'policy': os.path.basename(args.policy),
        'cumulative_reward': cum_reward
    }
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w', encoding='utf-8') as sf:
        json.dump(result, sf, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

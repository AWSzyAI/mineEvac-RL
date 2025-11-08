# filename: src/train_ppo_baseline.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from mine_evac_env import MineEvacEnv

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout_path = os.path.join(project_root, "layout", "baseline.json")

    env = MineEvacEnv(layout_path=layout_path, max_steps=500)

    # 日志目录
    log_dir = os.path.join(project_root, "logs", "ppo_baseline")
    os.makedirs(log_dir, exist_ok=True)

    # 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    # 配置 logger（可选）
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    # 训练
    model.learn(total_timesteps=200_000)

    # 保存模型
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ppo_mine_evac_baseline")
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # ====== Evaluation (deterministic) with logging ======
    import json
    import numpy as np

    # fresh env for eval
    eval_env = MineEvacEnv(layout_path=layout_path, max_steps=500)
    obs, info = eval_env.reset()
    model = PPO.load(model_path, env=eval_env)

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    eval_log_path = os.path.join(log_dir, "eval_episode.jsonl")
    cum_reward = 0.0
    with open(eval_log_path, "w", encoding="utf-8") as f:
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = eval_env.step(int(action))
            cum_reward += float(reward)
            # write one line
            line = {
                "t": step_info.get("t"),
                "reward": float(reward),
                "cumulative_reward": float(cum_reward),
                "responder_pos": step_info.get("responder_pos"),
                "occupants": step_info.get("occupants"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if terminated or truncated:
                break
    print(f"📄 Eval episode logged to {eval_log_path}")


if __name__ == "__main__":
    main()

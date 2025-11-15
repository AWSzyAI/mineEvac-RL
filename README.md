# MineEvac Graph Abstraction

The repository now exposes a lightweight, fully modular graph-based evacuation abstraction that mirrors the reference MineEvac layouts.  The implementation lives under the ``graph_evac`` package and is intentionally structured so that algorithms, layout parsing, simulation, IO, and visualisation are completely decoupled.

## Quick start

```bash
make install   # install python deps
make run       # simulate once (plan.json, timeline.csv/json, log, GIF)
make batch     # sweep the parameter grid defined in configs.py
make clean     # remove artifacts (artifacts/, batch_runs/, logs/)
```

``make run`` triggers ``src/main.py`` which in turn stitches together the graph pipeline, emits ``plan.json`` plus the CSV/JSON timeline, writes a textual ``run.log`` and renders a lightweight Gantt GIF under ``artifacts/``. ``make batch`` calls ``scripts/run_batch.py`` to iterate over every configuration emitted by ``configs.BatchSettings`` so that different redundancy/floor settings can be compared without editing code.

## High level pipeline

1. ``configs.Config`` collects every runtime parameter and can be overridden through environment variables (``GRAPH_EVAC_*``).
2. ``graph_evac.layout.load_layout`` reads any JSON file from ``layout/`` and ``graph_evac.layout.expand_floors`` replicates the footprint to multiple floors.
3. ``graph_evac.problem.EvacuationProblem`` stores the strongly typed rooms/responders/exits that are shared by all algorithms.
4. ``graph_evac.planner.plan_sweep`` dispatches to the requested algorithm (the greedy baseline today, ILP hooks reserved for future work).
5. ``graph_evac.simulator.simulate_sweep`` produces responder timelines that can be exported to CSV/JSON with ``graph_evac.io_utils`` and visualised through ``graph_evac.visuals.render_gantt_gif``.

The CLI at ``src/main.py`` stitches the whole flow together.  The ``if __name__ == '__main__'`` guard intentionally only keeps pseudo-code comments so that the same file can serve as the blueprint for LaTeX pseudo-code generation.  To execute the pipeline with custom switches call ``run_from_cli`` explicitly:

```bash
python3 -c "from src.main import run_from_cli; run_from_cli(['--layout', 'layout/baseline.json', '--floors', '2', '--redundancy', 'double_check', '--output', 'artifacts/custom'])"
```

## Configuration

All settings now live in ``configs.py``.  ``Config`` describes a single deterministic run (output directory, filenames, redundancy mode, etc.) while ``BatchSettings`` exposes the parameter grid consumed by ``make batch``.  Override any field via environment variables prefixed with ``GRAPH_EVAC_`` (for example ``GRAPH_EVAC_FLOORS=3 make run``) without editing the source tree.

The remainder of the README below (original PPO training quick-reference) is preserved verbatim for backwards compatibility with the RL baseline.

pip install "stable-baselines3[extra]" gymnasium

**训练日志字段说明（一次 log 的含义）**

以下是 Stable-Baselines3(PPO) 在训练时打印的一段典型日志示例，各字段含义与解读如下：

示例
-------------------------------------------
| time/                   |               |
|    fps                  | 275           |
|    iterations           | 107           |
|    time_elapsed         | 397           |
|    total_timesteps      | 109568        |
| train/                  |               |
|    approx_kl            | 0.00070979795 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.56         |
|    explained_variance   | 0.483         |
|    learning_rate        | 0.0003        |
|    loss                 | 325           |
|    n_updates            | 1060          |
|    policy_gradient_loss | -0.000857     |
|    value_loss           | 697           |
-------------------------------------------

说明
- time/fps: 每秒环境步数（吞吐）。数值越高训练越快。
- time/iterations: 迭代轮数（每轮收集一批 rollout 再优化）。本项目设置下每轮大约收集 ~2048 个环境步。
- time/time_elapsed: 自训练开始累计用时（秒）。
- time/total_timesteps: 到目前为止与环境交互的总步数（样本量）。

- train/approx_kl: 旧策略与新策略之间的近似 KL 散度。过大表示更新过猛（可减小学习率或 clip_range）；一直很小可能更新偏保守。
- train/clip_fraction: 在 PPO 比例裁剪中被裁剪的样本占比。典型范围 ~0.1–0.3；长期为 0 可能意味着更新幅度很小或优势较弱。
- train/clip_range: PPO 的裁剪阈值 epsilon（默认 0.2）。
- train/entropy_loss: 策略熵的负值（越接近 0 表示越确定性；越负表示越随机）。5 动作下最大熵约 ln(5)≈1.609，因此早期常见 ≈ -1.6。
- train/explained_variance: 价值函数对回报的判定系数 R²（越接近 1 越好；≈0 表示几乎没学到；<0 表示劣于常数预测）。
- train/learning_rate: 当前学习率。
- train/loss: PPO 总损失（policy_loss + vf_coef*value_loss + ent_coef*entropy_term）。受 value_loss 影响较大，数值大不一定表示坏，但趋势应稳定。
- train/n_updates: 到目前为止执行的优化步数（梯度更新次数）。
- train/policy_gradient_loss: 仅策略梯度项（通常为负，因目标是最大化）。
- train/value_loss: 价值网络的 MSE（越小越好，过大说明 critic 拟合困难或噪声大）。

解读建议
- KL 与 clip_fraction 联动判断更新幅度：KL 过大或 clip_fraction 过高→更新太猛；二者长期过低→可能学得太保守或优势为零。
- explained_variance 稳步上升是 critic 学到的信号；长期低位需检查奖励塑形或观测设计。
- loss/value_loss 关注趋势而非绝对值；若持续爆涨/振荡，通常降低学习率、增大 batch、或调低奖励尺度。

备注
- 训练过程的逐回合回报记录在 `logs/ppo_baseline/monitor.csv`（字段 r,l,t），而上面的表格是训练中每隔若干步汇总的一次训练状态快照（打印到 STDOUT）。

## 常用命令用法

环境准备
```bash
python3 -m pip install -r requirements.txt
```

训练 PPO（含评估与自动出图）
```bash
python3 src/train_ppo_baseline.py
```
- 产物：
  - 模型：`models/ppo_mine_evac_baseline`
  - 评估日志：`logs/eval_episode.jsonl`
  - 热图：`output/heatmap.png`
 - 动画 GIF：`output/sweep_anim.gif`

只评估“最佳”MineEvacEnv 模型并出图（不训练）
```bash
# 使用训练阶段 EvalCallback 产出的 best_model.zip
python3 src/train_ppo_baseline.py --eval-only
# 或使用 Makefile 快捷方式：
make show-best
```

仅评估已有模型并出图（跳过训练）
```bash
python3 - <<'PY'
import os
from src.train_ppo_baseline import run_deterministic_eval, generate_visuals, _get_device
pr=os.getcwd(); lp=os.path.join(pr,'layout','baseline.json'); mp=os.path.join(pr,'models','ppo_mine_evac_baseline')
ev=run_deterministic_eval(mp, lp, _get_device(), pr)
generate_visuals(ev, lp, pr)
print('eval_log=', ev)
PY
```

从任意日志生成热图
```bash
python3 scripts/visualize_heatmap.py \
  --path logs/eval_episode.jsonl \
  --layout layout/baseline.json \
  --entity both \
  --bins 60 \
  --save output/heatmap.png
```
- 参数：
  - `--path` 日志：支持新版 `eval_episode.jsonl` 或旧版 `trajectories.jsonl`
  - `--layout` 布局 JSON（用于墙/门/房间叠加与坐标对齐）
  - `--entity` 绘制对象：`responder` | `occupants` | `both`
  - `--bins` 直方图分箱数（默认 50）
  - `--save` 输出文件路径（省略则 `plt.show()`）

从任意日志生成动画 GIF/MP4
```bash
python3 scripts/animate_sweep.py \
  --path logs/eval_episode.jsonl \
  --layout layout/baseline.json \
  --fps 12 --skip 1 --trail 80 --dpi 150 \
  --save output/sweep_anim.gif
```
- 参数：
  - `--path` 日志路径（同上）
  - `--layout` 布局 JSON 覆盖	s
  - `--fps` 帧率，`--skip` 每隔 N 帧抽样，`--trail` 尾迹长度，`--dpi` 输出分辨率
  - `--save` 输出 GIF/MP4，若省略则窗口播放

确定性（非 RL）扫楼仿真 + 可视化
```bash
# 单次确定性仿真（可选写出帧日志以便可视化）
python3 src/sim_sweep_det.py \
  --layout layout/baseline.json \
  --responders 2 \
  --per_room 5 \
  --max_steps 1500 \
  --frames logs/det_eval_episode.jsonl \
  --save logs/det_baseline.json

# 将帧日志可视化为动画
python3 scripts/animate_sweep.py \
  --path logs/det_eval_episode.jsonl \
  --layout layout/baseline.json \
  --save output/det_sweep.gif
```

批量确定性实验（不同布局/人数/初始点）
```bash
python3 scripts/run_experiments.py \
  --layouts layout/baseline.json layout/layout_A.json layout/layout_B.json \
  --responders 1 2 \
  --per_room 3 5 \
  --max_steps 3000 \
  --out logs/det_experiments.jsonl
```
- 产物：`logs/det_experiments.jsonl` 与同名 `.md` 摘要。

快速检查
```bash
tail -n 5 logs/ppo_baseline/monitor.csv         # 训练每回合回报
tail -n 5 logs/eval_episode.jsonl               # 评估逐步奖励与累计回报
ls -la output                                   # 出图产物
```

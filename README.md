# MineEvac：矿井疏散模拟与可视化（中文说明）

本仓库围绕“矿井/建筑疏散”提供了三条互相关联但彼此解耦的工作流：

- **图抽象 + 贪心调度**：在简化的房间–出口图上规划扫楼顺序，导出计划、时间线和 GIF。
- **确定性网格仿真（det）**：在原始 Minecraft 风格网格上模拟多名救援者与被疏散人员，生成“带地图背景”的动画和热力图。
- **强化学习（PPO）基线**：在网格环境中训练 / 评估 RL 策略，并生成对应的可视化。

阅读完本 README，你应该能够：

1. 安装依赖；
2. 运行一次图抽象贪心仿真并查看输出；
3. 复现原始地图风格的确定性 GIF；
4. 训练 / 评估一个 PPO 基线；
5. 使用批量脚本生成用于分析的结果表。

---

## 1. 环境准备

1. 安装 Python（推荐 3.10+）。
2. 在仓库根目录安装依赖：

   ```bash
   make install
   # 等价于：
   # python3 -m pip install --upgrade pip
   # python3 -m pip install -r requirements.txt
   ```

---

## 2. 图抽象 + 贪心调度：`make run`

这部分是**最快上手、最轻量**的版本，不依赖强化学习，只在简化的房间–出口图上做贪心规划。

### 2.1 一次运行

在根目录执行：

```bash
make run
```

其中：

- 默认 `ALGO=greedy`，走图抽象贪心管线（`src/main.py`）。
- 设置 `ALGO=ppo`（且已安装 Stable-Baselines3）时，将直接调用 PPO 基线进行评估并生成 `output/heatmap.png` 与 `output/sweep_anim.gif`，效果类似 `make show-rl`。

默认行为：

- 布局：`layout/baseline.json`
- 楼层数：`1`
- 算法：`greedy`
- 输出目录：`outputs/`

运行结束后，在 `outputs/` 中可以看到：

- `plan.json`：每个救援者的房间分配、访问顺序、时间段；
- `timeline.csv` / `timeline.json`：时间线（谁在什么时候做什么）；
- `run.log`：**JSON 结构的运行日志**（兼容历史 `logs/det_baseline.json` 的字段含义）；
- `timeline.gif`：简洁的 Gantt 风格时间线 GIF。

`outputs/run.log` 的结构示例（数值会随布局改变）：

```json
{
  "layout": "baseline.json",
  "responders": 2,
  "per_room": 5,
  "time": 270,
  "all_evacuated": true,
  "room_order": ["R2","R3","R5","R0","R4","R1"],
  "init_positions": [[9,20],[96,20]],
  "exits": [[8,19],[97,19]],
  "evacuated": 30,
  "real_hms": "00:04:30",
  "real_minutes": 4.5,
  "cell_m": 0.5,
  "speed_solo_mps": 0.8,
  "speed_escort_mps": 0.6
}
```

审稿人如果希望看到“原始 run.log JSON 结构”，可以直接使用该文件。

### 2.2 修改参数（布局 / 楼层 / 冗余模式等）

`make run` 底层调用 `src/main.py` 的 `run_from_cli`，核心配置在 `configs.Config` 中。你可以通过 `RUN_ARGS` 或环境变量覆盖默认值。

使用 `RUN_ARGS` 的例子：

```bash
# 使用 school_layout.json，2 层楼，冗余模式 double_check，输出到 outputs/school_run
make run RUN_ARGS="--layout layout/school_layout.json --floors 2 --redundancy double_check --output outputs/school_run"
```

`src/main.py` 支持的一些关键参数：

- `--layout`：布局 JSON 路径，默认 `layout/baseline.json`
- `--floors`：复制楼层数，默认 `1`
- `--floor-spacing`：楼层间距（仅影响 3D 坐标）
- `--algorithm`：规划算法，目前支持 `greedy`
- `--redundancy`：冗余模式，例如 `assignment` / `double_check` / `per_responder_all_rooms`
- `--no-sim`：仅规划，不生成时间线和 GIF
- `--output`：输出目录，默认 `outputs`

你也可以通过环境变量覆盖配置，例如：

```bash
GRAPH_EVAC_FLOORS=3 GRAPH_EVAC_REDUNDANCY_MODE=double_check make run
```

---

## 3. 确定性网格仿真 + 地图风格 GIF：`make det` / `make det-gif`

这部分复原了原始「地图背景 + 房间 / 走廊 + 移动点 + 热力图」的可视化风格，用于对照与展示。

### 3.1 单次确定性仿真：`make det`

```bash
make det
```

等价于调用：

```bash
python3 src/sim_sweep_det.py \
  --layout layout/baseline.json \
  --responders 2 \
  --per_room 5 \
  --max_steps 1500 \
  --frames logs/det_eval_episode.jsonl \
  --save logs/det_baseline.json \
  --delay 0.01 \
  --log-every 50
```

产物：

- `logs/det_baseline.json`：**确定性 JSON 运行日志**，字段结构与历史提交 `a0880de` 中的 `logs/det_baseline.json` 一致；
- `logs/det_eval_episode.jsonl`：逐步帧日志，用于热力图和动画。

`logs/det_baseline.json` 的典型内容：

```json
{
  "layout": "baseline.json",
  "responders": 2,
  "per_room": 5,
  "time": 1161,
  "all_evacuated": true,
  "room_order": ["R3","R1","R4","R6","R2","R5"],
  "init_positions": [[9,20],[96,20]],
  "exits": [[8,19],[97,19]],
  "evacuated": 30,
  "real_hms": "00:15:09",
  "real_minutes": 15.15,
  "cell_m": 0.5,
  "speed_solo_mps": 0.8,
  "speed_escort_mps": 0.6
}
```

### 3.2 从确定性帧生成地图 GIF：`make det-gif`

```bash
make det-gif
```

等价于：

```bash
python3 scripts/visualize_heatmap.py \
  --path logs/det_eval_episode.jsonl \
  --layout layout/baseline.json \
  --entity responder \
  --bins 80 \
  --save output/det_heatmap.png

python3 scripts/animate_sweep.py \
  --path logs/det_eval_episode.jsonl \
  --layout layout/baseline.json \
  --save output/det_sweep.gif \
  --fps 12 \
  --skip 1 \
  --trail 80
```

产物：

- `output/det_heatmap.png`：带布局的访问热力图；
- `output/det_sweep.gif`：地图风格的疏散动画，与 `example/det_sweep.gif` 风格相同。

---

## 4. 强化学习（PPO）基线：训练 / 评估 / 可视化

RL 相关逻辑集中在：

- 环境：`src/mine_evac_env.py`
- 训练与评估：`src/train_ppo_baseline.py`

### 4.1 仅评估已有模型并生成 GIF（推荐先用）

如果已经有现成模型（`models/best_model.zip` 或 `models/ppo_mine_evac_baseline.zip`），可以直接评估：

```bash
python3 src/train_ppo_baseline.py --eval-only
```

或使用 Make 封装：

```bash
make show-rl
```

完成后得到：

- 评估日志：`logs/eval_episode.jsonl`
- 热力图：`output/heatmap.png`
- 地图 GIF：`output/sweep_anim.gif`

这三者可与确定性管线的 `logs/det_eval_episode.jsonl`、`output/det_heatmap.png`、`output/det_sweep.gif` 对照分析。

### 4.2 训练 PPO 模型

完整训练（耗时视机器而定）：

```bash
make train
# 或自定义步数：
python3 src/train_ppo_baseline.py --timesteps 400000
```

训练过程：

1. 创建 MineEvac 环境（图结构 + 网格动态）；
2. 使用 Stable-Baselines3 的 `PPO` 算法训练；
3. 周期性评估并保存 `best_model.zip`；
4. 结束后自动进行一轮确定性评估并生成热图和 GIF。

典型产物：

- 模型：
  - `models/ppo_mine_evac_baseline.zip`
  - `models/best_model.zip`
- 训练监控：`logs/ppo_baseline/monitor.csv`
- 评估日志：`logs/eval_episode.jsonl`
- 可视化：`output/heatmap.png`, `output/sweep_anim.gif`

---

## 5. 批量图抽象分析：`make batch`

批量图抽象 sweeps 由 `scripts/run_batch.py` 驱动，配置集中在 `configs.BatchSettings`。

### 5.1 默认批量运行

```bash
make batch
```

默认配置（见 `configs.py` 中的 `BatchSettings`）大致为：

- 布局：`layout/baseline.json`
- 楼层数列表：例如 `[1, 2]`
- 冗余模式：`["assignment","double_check","per_responder_all_rooms"]`
- 算法：`["greedy"]`
- 输出根目录：`batch_runs/`

脚本会为每个配置调用 `execute_run(config)`（与 `make run` 相同的图抽象管线），并把结果汇总为：

- `batch_runs/summary.json`
- `batch_runs/summary.csv`

每一行包含 run 标签、布局路径、楼层数、冗余模式、算法、makespan 等信息，可直接导入 Excel / pandas 进行分析。

### 5.2 自定义批量参数

你可以直接调用脚本并传入参数：

```bash
python3 scripts/run_batch.py \
  --layout layout/baseline.json \
  --floors 1 2 3 \
  --redundancy assignment double_check \
  --algorithms greedy
```

支持的参数（见 `scripts/run_batch.py`）：

- `--layout`：覆盖 `BatchSettings.layout_path`
- `--output`：覆盖输出根目录
- `--floors`：覆盖楼层数列表（空格分隔）
- `--redundancy`：冗余模式列表
- `--algorithms`：算法列表

如需生成你自己的 CSV 格式（例如包含 `layout,floors,per_room_occ,responders,...` 等列），可以在 `scripts/run_batch.py` 内部根据 `config + plan + layout JSON` 自行组合并写出。

---

## 6. 目录结构速览

- `layout/`：布局定义（`baseline.json`, `layout_A.json`, `layout_B.json`, `school_layout.json` 等）
- `graph_evac/`：
  - `layout.py`：从布局 JSON 抽取房间、出口、救援者；
  - `problem.py`：图抽象问题定义；
  - `greedy.py`：贪心扫楼算法；
  - `simulator.py`：把计划展开为时间线；
  - `visuals.py`：`render_gantt_gif`，生成时间线 GIF；
  - `io_utils.py`：JSON / CSV 保存与 `run.log` JSON 生成。
- `src/`：
  - `main.py`：图抽象 CLI（被 `make run` 调用）；
  - `sim_sweep_det.py`：确定性网格仿真（被 `make det` 调用）；
  - `mine_evac_env.py`：RL 环境定义；
  - `train_ppo_baseline.py`：PPO 训练与评估。
- `scripts/`：
  - `animate_sweep.py`：地图风格动画 GIF 生成；
  - `visualize_heatmap.py`：热力图生成；
  - `run_batch.py`：批量图抽象 sweeps；
  - 其他 RL 扩展训练 / 评估 / 报告脚本。
- `outputs/`：`make run` 的默认输出目录（plan/timeline/run.log/timeline.gif）。
- `logs/`：训练与评估日志（包括确定性与 RL）。
- `output/`：确定性与 RL 可视化产物（热力图和地图 GIF）。

---

## 7. 常用命令速查表

安装依赖：

```bash
make install
```

图抽象贪心（单次）：

```bash
make run
make run RUN_ARGS="--layout layout/school_layout.json --floors 2 --redundancy double_check"
```

确定性网格基线 + 地图 GIF：

```bash
make det        # 生成 logs/det_baseline.json, logs/det_eval_episode.jsonl
make det-gif    # 生成 output/det_heatmap.png, output/det_sweep.gif
```

强化学习基线：

```bash
make train      # 训练 PPO 模型
make show-rl    # 评估 best_model 并生成 output/heatmap.png, output/sweep_anim.gif
# 或直接：
python3 src/train_ppo_baseline.py --eval-only
```

批量图抽象分析：

```bash
make batch
# 或自定义：
python3 scripts/run_batch.py --floors 1 2 --redundancy assignment double_check
```

清理所有产物：

```bash
make clean
```

---

如果你只想“跑起来看结果”，建议按以下顺序尝试：

1. `make run`：看 `outputs/` 下的 `plan.json`、`run.log` 和 `timeline.gif`；
2. `make det && make det-gif`：对比 `logs/det_baseline.json` 与 `output/det_sweep.gif`；
3. `make show-rl`：体验 PPO 策略生成的热力图和地图 GIF。



---

```bash
FLOORS=2 make run ALGO=greedy LAYOUT_FILE=layout/baseline.json RESPONDERS=2 PER_ROOM=5
FLOORS=2 RESPONDERS=2 PER_ROOM=5 LAYOUT_FILE=layout/baseline.json make det-gif
```

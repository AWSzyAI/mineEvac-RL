## Makefile for mineEvac-RL project (simplified interface)
# 使用 zsh / macOS

PYTHON ?= python

.PHONY: help run show report clean

help:
	@echo "Targets:"
	@echo "  run   - train model + eval episode + generate heatmap & GIF"
	@echo "  show  - regenerate visuals (heatmap + GIF) from existing eval log"
	@echo "  report- summarise training/eval logs into log/report.md"
	@echo "  det   - deterministic two-responder sweep on baseline layout"
	@echo "  exp   - run deterministic experiments on baseline/layout_A/layout_B"
	@echo "  train - train MineEvacEnv PPO (best_model saved)"
	@echo "  show-rl - evaluate best MineEvacEnv PPO and visualize"
	@echo "  train2 - train 2-responder RL (best_model saved)"
	@echo "  show-rl2 - evaluate best 2-responder policy and visualize"
	@echo "  show-best - evaluate best MineEvacEnv PPO (best_model.zip) and visualize"
	@echo "  clean - remove generated artifacts (logs/models/output/__pycache__)"

run:
	@echo "[1/4] Training RL model & logging eval episode"
	$(PYTHON) src/train_ppo_baseline.py
	@echo "[2/4] Generating heatmap from eval log occupants/responder"
	$(PYTHON) scripts/visualize_heatmap.py --path logs/eval_episode.jsonl --bins 80 --save output/heatmap.png
	@echo "[3/4] Generating sweep animation with reward overlay"
	$(PYTHON) scripts/animate_sweep.py --path logs/eval_episode.jsonl --layout layout/baseline.json --save output/sweep_anim.gif --fps 12 --skip 1 --trail 80
	@echo "[4/4] Artifacts ready: output/heatmap.png, output/sweep_anim.gif"

show:
	@echo "Regenerating visuals from existing logs/eval_episode.jsonl"
	$(PYTHON) scripts/visualize_heatmap.py --path logs/eval_episode.jsonl --bins 80 --save output/heatmap.png
	$(PYTHON) scripts/animate_sweep.py --path logs/eval_episode.jsonl --layout layout/baseline.json --save output/sweep_anim.gif --fps 12 --skip 1 --trail 80
	@echo "Visuals updated."

report:
	@echo "Generating Markdown report from logs"
	$(PYTHON) scripts/generate_report.py --output logs/report.md
	@echo "Report available at log/report.md"

det:
	@echo "Running deterministic sweep on baseline.json with 2 responders"
	$(PYTHON) src/sim_sweep_det.py --layout layout/baseline.json --responders 2 --per_room 5 --save logs/det_baseline.json --frames logs/det_eval_episode.jsonl
	@echo "Saved logs/det_baseline.json"

exp:
	@echo "Running deterministic experiments (baseline, layout_A, layout_B)"
	$(PYTHON) scripts/run_experiments.py --layouts layout/baseline.json layout/layout_A.json layout/layout_B.json --responders 2 --per_room 5 --out logs/det_experiments.jsonl
	@echo "Summary at logs/det_experiments.md"

.PHONY: det-gif
det-gif:
	@echo "Generating visuals from deterministic frames (logs/det_eval_episode.jsonl)"
	$(PYTHON) scripts/visualize_heatmap.py --path logs/det_eval_episode.jsonl --layout layout/baseline.json --entity responder --bins 80 --save output/det_heatmap.png
	$(PYTHON) scripts/animate_sweep.py --path logs/det_eval_episode.jsonl --layout layout/baseline.json --save output/det_sweep.gif --fps 12 --skip 1 --trail 80
	@echo "Artifacts: output/det_heatmap.png, output/det_sweep.gif"

clean:
	@echo "Cleaning artifacts..."
	find . -type d -name "__pycache__" -print -exec rm -rf {} + || true
	rm -rf logs/* models/*  || true
.PHONY: show-best
show-best:
	@echo "Evaluating best MineEvacEnv PPO model (best_model.zip) and generating visuals"
	$(PYTHON) src/train_ppo_baseline.py --eval-only
	@echo "Artifacts: output/heatmap.png, output/sweep_anim.gif"
.PHONY: train show-rl
train:
	@echo "Training MineEvacEnv PPO on baseline.json with best-model selection"
	$(PYTHON) src/train_ppo_baseline.py --timesteps 400000
	@echo "Models: models/best_model.zip, models/ppo_mine_evac_baseline.zip"

show-rl:
	@echo "Evaluating best MineEvacEnv PPO model (best_model.zip) and generating visuals"
	$(PYTHON) src/train_ppo_baseline.py --eval-only
	@echo "Artifacts: output/heatmap.png, output/sweep_anim.gif"

.PHONY: train2 show-rl2
train2:
	@echo "Training two-responder RL policy (TwoResponderEscortEnv) on baseline.json with best-model selection"
	$(PYTHON) scripts/train_sim_ppo2.py --layout layout/baseline.json --per_room 5 --max_steps 500 --timesteps 400000 --save models/n_r_2.zip --logdir logs/sim_rl2
	@echo "Models: models/best_model.zip (if present), models/n_r_2.zip"

show-rl2:
	@echo "Evaluating best available two-responder policy and generating visuals"
	$(PYTHON) scripts/eval_sim_policy2.py --layout layout/baseline.json --frames logs/policy2_eval_episode.jsonl --save logs/policy2_eval_summary.json
	$(PYTHON) scripts/visualize_heatmap.py --path logs/policy2_eval_episode.jsonl --layout layout/baseline.json --entity responder --bins 80 --save output/policy2_heatmap.png
	$(PYTHON) scripts/animate_sweep.py --path logs/policy2_eval_episode.jsonl --layout layout/baseline.json --save output/policy2_sweep.gif --fps 12 --skip 1 --trail 80
	@echo "Artifacts: output/policy2_heatmap.png, output/policy2_sweep.gif"

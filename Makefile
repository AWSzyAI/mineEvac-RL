SHELL := /bin/bash
PYTHON ?= python3
RUN_ARGS ?=
BATCH_ARGS ?=
ALGO ?= greedy
FLOORS ?= 1
LAYOUT_FILE ?= layout/baseline.json
RESPONDERS ?= 2
PER_ROOM ?= 5

LAYOUT_NAME := $(basename $(notdir $(LAYOUT_FILE)))
RUN_LABEL := $(ALGO)_$(LAYOUT_NAME)_res$(RESPONDERS)_per$(PER_ROOM)_floors$(FLOORS)
RUN_DIR := output/$(RUN_LABEL)
LOG_DIR := $(RUN_DIR)/logs
VIS_DIR := $(RUN_DIR)/visuals

.ONESHELL:

.PHONY: help install run run_greedy run_ppo batch clean \
	det det-gif exp report \
	train show-rl train2 show-rl2 show-best show

help:
	@echo "Targets:"
	@echo "  install   - install python dependencies"
	@echo "  run       - single sweep (ALGO=greedy|ppo, override args via RUN_ARGS)"
	@echo "  batch     - deterministic multi-floor sweep grid (for data/ML analysis)"
	@echo "  det       - deterministic grid sweep on baseline layout (logs/baseline_greedy_result.json)"
	@echo "  det-gif   - map-based GIF + heatmap from deterministic frames"
	@echo "  show-best - evaluate best RL baseline and regenerate visuals"
	@echo "  train     - train MineEvacEnv PPO baseline"
	@echo "  train2    - train two-responder RL policy"
	@echo "  show      - regenerate RL visuals from logs/eval_episode.jsonl"
	@echo "  exp       - deterministic experiments on baseline/layout_A/layout_B"
	@echo "  report    - summarise training/eval logs into logs/report.md"
	@echo "  clean     - remove generated artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

prepare_run_dir:
	@mkdir -p $(LOG_DIR) $(VIS_DIR)
	@echo "Using run directory: $(RUN_DIR)"

ifeq ($(ALGO),ppo)
run: run_ppo
else
run: run_greedy
endif

run_greedy: prepare_run_dir
	@echo "Running deterministic grid sweep + map GIF (equivalent to 'make det && make det-gif')..."
	$(MAKE) det
	$(MAKE) det-gif

run_ppo: prepare_run_dir
	@echo "Running MineEvac PPO baseline (eval-only, with GIFs)..."
	@$(PYTHON) src/train_ppo_baseline.py --eval-only $(RUN_ARGS) 2>&1 | tee $(LOG_DIR)/run_ppo.log
	@echo "Summarising RL eval into baseline_ppo_result.json..."
	$(PYTHON) scripts/rl_eval_to_det_log.py --eval-log logs/eval_episode.jsonl --layout layout/baseline.json --save logs/baseline_ppo_result.json
	@cp logs/baseline_ppo_result.json $(LOG_DIR)/baseline_ppo_result.json
	@echo "Aligning RL visuals with deterministic filenames..."
	@cp output/heatmap.png $(VIS_DIR)/det_heatmap.png
	@cp output/sweep_anim.gif $(VIS_DIR)/det_sweep.gif
	$(PYTHON) scripts/extract_gif_frames.py --gif $(VIS_DIR)/det_sweep.gif --prefix $(VIS_DIR)/det_sweep

batch:
	@echo "Running deterministic multi-floor batch sweeps..."
	$(PYTHON) scripts/run_det_batch.py $(BATCH_ARGS)

# ---- Original deterministic + RL pipelines (map-based GIFs) ----

det:
	@echo "Running deterministic sweep on baseline.json with 2 responders"
	$(PYTHON) src/sim_sweep_det.py \
		--layout $(LAYOUT_FILE) \
		--responders $(RESPONDERS) \
		--per_room $(PER_ROOM) \
		--max_steps 5000 \
		--floors $(FLOORS) \
		--save $(LOG_DIR)/baseline_greedy_result.json \
		--frames $(LOG_DIR)/det_eval_episode.jsonl \
		--delay 0.01 \
		--log-every 50
	@echo "Saved $(LOG_DIR)/baseline_greedy_result.json (JSON run result with original schema)"

det-gif:
	@echo "Generating map-based visuals from deterministic frames (logs/det_eval_episode.jsonl)"
	$(PYTHON) scripts/visualize_heatmap.py \
		--path $(LOG_DIR)/det_eval_episode.jsonl \
		--layout $(LAYOUT_FILE) \
		--entity responder \
		--bins 80 \
		--save $(VIS_DIR)/det_heatmap.png
	$(PYTHON) scripts/animate_sweep.py \
		--path $(LOG_DIR)/det_eval_episode.jsonl \
		--layout $(LAYOUT_FILE) \
		--floors $(FLOORS) \
		--save $(VIS_DIR)/det_sweep.gif \
		--fps 12 \
		--skip 1 \
		--trail 80
	$(PYTHON) scripts/extract_gif_frames.py --gif $(VIS_DIR)/det_sweep.gif --prefix $(VIS_DIR)/det_sweep
	@echo "Artifacts: $(VIS_DIR)/det_heatmap.png, $(VIS_DIR)/det_sweep.gif and key PNG frames"

show:
	@echo "Regenerating RL visuals from existing logs/eval_episode.jsonl"
	$(PYTHON) scripts/visualize_heatmap.py \
		--path logs/eval_episode.jsonl \
		--bins 80 \
		--save output/heatmap.png
	$(PYTHON) scripts/animate_sweep.py \
		--path logs/eval_episode.jsonl \
		--layout layout/baseline.json \
		--save output/sweep_anim.gif \
		--fps 12 \
		--skip 1 \
		--trail 80
	@echo "Visuals updated: output/heatmap.png, output/sweep_anim.gif"

report:
	@echo "Generating Markdown report from logs"
	$(PYTHON) scripts/generate_report.py --output logs/report.md
	@echo "Report available at logs/report.md"

exp:
	@echo "Running deterministic experiments (baseline, layout_A, layout_B)"
	$(PYTHON) scripts/run_experiments.py \
		--layouts layout/baseline.json layout/layout_A.json layout/layout_B.json \
		--responders 2 \
		--per_room 5 \
		--out logs/det_experiments.jsonl
	@echo "Summary at logs/det_experiments.md"

train:
	@echo "Training MineEvacEnv PPO on baseline.json with best-model selection"
	$(PYTHON) src/train_ppo_baseline.py --timesteps 400000
	@echo "Models: models/best_model.zip, models/ppo_mine_evac_baseline.zip"

show-rl:
	@echo "Evaluating best MineEvacEnv PPO model (best_model.zip) and generating visuals"
	$(PYTHON) src/train_ppo_baseline.py --eval-only
	@echo "Artifacts: output/heatmap.png, output/sweep_anim.gif"

train2:
	@echo "Training two-responder RL policy (TwoResponderEscortEnv) on baseline.json with best-model selection"
	$(PYTHON) scripts/train_sim_ppo2.py \
		--layout layout/baseline.json \
		--per_room 5 \
		--max_steps 500 \
		--timesteps 400000 \
		--save models/n_r_2.zip \
		--logdir logs/sim_rl2
	@echo "Models: models/best_model.zip (if present), models/n_r_2.zip"

show-rl2:
	@echo "Evaluating best available two-responder policy and generating visuals"
	$(PYTHON) scripts/eval_sim_policy2.py \
		--layout layout/baseline.json \
		--frames logs/policy2_eval_episode.jsonl \
		--save logs/policy2_eval_summary.json
	$(PYTHON) scripts/visualize_heatmap.py \
		--path logs/policy2_eval_episode.jsonl \
		--layout layout/baseline.json \
		--entity responder \
		--bins 80 \
		--save output/policy2_heatmap.png
	$(PYTHON) scripts/animate_sweep.py \
		--path logs/policy2_eval_episode.jsonl \
		--layout layout/baseline.json \
		--save output/policy2_sweep.gif \
		--fps 12 \
		--skip 1 \
		--trail 80
	@echo "Artifacts: output/policy2_heatmap.png, output/policy2_sweep.gif"

clean:
	@echo "Removing generated artifacts"
	rm -rf artifacts batch_runs outputs graph_outputs
	rm -rf logs models output || true
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

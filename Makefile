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

## Makefile for mineEvac-RL project (simplified interface)
# 使用 zsh / macOS

.PHONY: help run show clean

help:
	@echo "Targets:"
	@echo "  run   - train model + eval episode + generate heatmap & GIF"
	@echo "  show  - regenerate visuals (heatmap + GIF) from existing eval log"
	@echo "  clean - remove generated artifacts (logs/models/output/__pycache__)"

run:
	@echo "[1/4] Training RL model & logging eval episode"
	python3 src/train_ppo_baseline.py
	@echo "[2/4] Generating heatmap from eval log occupants/responder"
	python3 scripts/visualize_heatmap.py --path logs/eval_episode.jsonl --bins 80 --save output/heatmap.png
	@echo "[3/4] Generating sweep animation with reward overlay"
	python3 scripts/animate_sweep.py --path logs/eval_episode.jsonl --layout layout/baseline.json --save output/sweep_anim.gif --fps 12 --skip 1 --trail 80
	@echo "[4/4] Artifacts ready: output/heatmap.png, output/sweep_anim.gif"

show:
	@echo "Regenerating visuals from existing logs/eval_episode.jsonl"
	python3 scripts/visualize_heatmap.py --path logs/eval_episode.jsonl --bins 80 --save output/heatmap.png
	python3 scripts/animate_sweep.py --path logs/eval_episode.jsonl --layout layout/baseline.json --save output/sweep_anim.gif --fps 12 --skip 1 --trail 80
	@echo "Visuals updated."

clean:
	@echo "Cleaning artifacts..."
	find . -type d -name "__pycache__" -print -exec rm -rf {} + || true
	rm -rf logs/* models/* output/* || true

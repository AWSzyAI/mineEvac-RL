SHELL := /bin/bash
PYTHON ?= python3
RUN_ARGS ?=
BATCH_ARGS ?=

.ONESHELL:

.PHONY: help install run clean batch

help:
	@echo "Targets:"
	@echo "  install - install python dependencies"
	@echo "  run     - simulate once using src/main.py (override args via RUN_ARGS)"
	@echo "  batch   - iterate the grid defined in configs.BatchSettings"
	@echo "  clean   - remove generated artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	@echo "Running MineEvac graph abstraction..."
	RUN_ARGS_STR="$(RUN_ARGS)" $(PYTHON) - <<-'PY'
	import os
	import shlex
	from src.main import run_from_cli

	args = shlex.split(os.environ.get("RUN_ARGS_STR", ""))
	run_from_cli(args or None)
	PY
	@echo "Artifacts stored under the configured output directory (default: artifacts/)"

batch:
	@echo "Running batch sweeps defined in configs.BatchSettings..."
	$(PYTHON) scripts/run_batch.py $(BATCH_ARGS)

clean:
	@echo "Removing generated artifacts"
	rm -rf artifacts batch_runs outputs graph_outputs
	rm -rf logs models output || true
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

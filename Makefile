# Action-Conditioned World Models & Predictive Control Makefile

PYTHON = .venv/Scripts/python.exe
PYTEST = .venv/Scripts/pytest.exe
RUFF = .venv/Scripts/ruff.exe

.PHONY: help setup test test-all lint format smoke run demo clean

help:
	@echo "Available commands:"
	@echo "  make setup     - Verify environment and run headless smoke test"
	@echo "  make lint      - Lint codebase using ruff"
	@echo "  make format    - Format codebase using ruff"
	@echo "  make test      - Run fast non-simulator unit tests"
	@echo "  make test-all  - Run all tests including MetaDrive simulator tests"
	@echo "  make smoke     - Run end-to-end smoke pipeline (~5 min)"
	@echo "  make run       - Run full default benchmark pipeline"
	@echo "  make demo      - Render MPC closed-loop driving GIF"

setup:
	$(PYTHON) scripts/setup_check.py

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

test:
	$(PYTEST) -m "not sim" -v

test-all:
	$(PYTEST) -v

smoke:
	$(PYTHON) scripts/run_pipeline.py --config configs/smoke.yaml

run:
	$(PYTHON) scripts/run_pipeline.py --config configs/default.yaml

demo:
	$(PYTHON) scripts/demo.py --controller mpc --map SC --seed 42 --output docs/img/demo.gif

clean:
	rm -rf .pytest_cache data/test_* *.log

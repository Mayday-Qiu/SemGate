.PHONY: demo

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)

demo:
	$(PYTHON) scripts/demo_run.py --all

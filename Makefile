.PHONY: help install dev test test-parity lint format type clean fixtures docs docs-serve notebook

help:
	@echo "pyFIES — common dev targets"
	@echo "  make install      # install in editable mode"
	@echo "  make dev          # install with dev extras + pre-commit hooks"
	@echo "  make test         # run unit tests (skips R parity)"
	@echo "  make test-parity  # run all tests including R fixture parity"
	@echo "  make lint         # ruff check"
	@echo "  make format       # ruff format"
	@echo "  make type         # mypy"
	@echo "  make fixtures     # regenerate R reference fixtures (requires R)"
	@echo "  make docs         # build the documentation site"
	@echo "  make docs-serve   # serve docs locally with live-reload at :8000"
	@echo "  make notebook     # rebuild and re-execute the demo notebook"
	@echo "  make clean        # remove build/cache artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -m "not parity" -ra

test-parity:
	pytest -ra

lint:
	ruff check src tests

format:
	ruff format src tests

type:
	mypy src

fixtures:
	Rscript scripts/generate_r_fixtures.R

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

notebook:
	python scripts/build_demo_notebook.py
	jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=120 \
		notebooks/01_parity_demo.ipynb

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

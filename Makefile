.PHONY: install dev test lint typecheck fmt e2e audit lock clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,test]"

test:
	python -m pytest tests/ -q

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src/evalforge

e2e:
	bash scripts/e2e.sh

lock:
	pip-compile --output-file=requirements.lock pyproject.toml
	pip-compile --output-file=requirements-dev.lock --extra=dev --extra=test pyproject.toml

audit:
	pip-audit -r requirements.lock

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info .evalforge
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

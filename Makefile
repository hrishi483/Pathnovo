.PHONY: install run test lint

install:
	python -m pip install -e ".[dev]"

run:
	python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest

lint:
	python -m ruff check src tests eval

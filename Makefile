# En Windows sin make: `uv run python -m tipologia.pipeline.cli <etapa>`
PY = uv run python -m tipologia.pipeline.cli

.PHONY: install ingest validate features cluster stability test lint

install:
	uv sync

ingest:
	$(PY) ingest

validate:
	$(PY) validate

features:
	$(PY) features

cluster:
	$(PY) cluster

stability:
	$(PY) stability

test:
	uv run pytest

lint:
	uv run ruff check src tests

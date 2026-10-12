# En Windows sin make: `uv run python -m tipologia.pipeline.cli <etapa>`
PY = uv run python -m tipologia.pipeline.cli

.PHONY: install ingest validate features cluster stability profile export pipeline app test lint

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

profile:
	$(PY) profile

export:
	$(PY) export

# ingest + validate + features + cluster + stability + profile + export.
# Idempotente: si los datos, la configuración y el código no cambiaron, no recomputa nada.
pipeline: ingest validate
	$(PY) pipeline

app:
	uv run streamlit run app/Inicio.py

test:
	uv run pytest

lint:
	uv run ruff check src tests app

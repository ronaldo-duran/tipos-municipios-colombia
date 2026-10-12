# ¿Cuántos tipos de municipio hay en Colombia?

Tipología de los 1.102 municipios de Colombia mediante clustering sobre indicadores socioeconómicos de TerriData (DNP).
Proyecto en construcción por fases; ver el historial de commits.

> Descripción estadística de territorios a partir de datos oficiales. No implica causalidad ni jerarquía de calidad.

## Cómo ejecutar

```bash
uv sync
uv run python -m tipologia.pipeline.cli ingest
uv run python -m tipologia.pipeline.cli validate
uv run pytest
```

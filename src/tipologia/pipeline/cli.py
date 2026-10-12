"""Interfaz de línea de comandos: `python -m tipologia.pipeline.cli <etapa>`."""

from __future__ import annotations

import argparse
import logging
import sys

from tipologia.config import cargar_config
from tipologia.logs import configurar_logging

log = logging.getLogger(__name__)

ETAPAS = ["ingest", "validate", "features", "cluster", "stability", "profile", "temporal", "export", "pipeline"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline de la tipología municipal")
    parser.add_argument("etapa", choices=ETAPAS)
    parser.add_argument("--forzar", action="store_true", help="recomputa aunque existan resultados")
    parser.add_argument("--offline", action="store_true", help="no consulta las fuentes remotas, usa la caché")
    args = parser.parse_args(argv)
    configurar_logging()
    cfg = cargar_config()

    if args.etapa == "ingest":
        from tipologia.data import ingest
        ingest.ejecutar(cfg, forzar=args.forzar, offline=args.offline)
        return 0
    if args.etapa == "validate":
        from tipologia.data import validate
        rep = validate.ejecutar(cfg)
        return 1 if rep.errores else 0

    from tipologia.pipeline import run
    return run.ejecutar_etapa(args.etapa, cfg, forzar=args.forzar, offline=args.offline)


if __name__ == "__main__":
    sys.exit(main())

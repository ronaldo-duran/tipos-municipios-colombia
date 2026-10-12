"""Configuración de logging del proyecto."""

from __future__ import annotations

import logging
import sys


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Formato común para todos los módulos."""
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

"""Lectura y escritura de artefactos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _a_json(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, Path):
        return o.as_posix()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def guardar_json(obj: Any, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(obj, indent=2, ensure_ascii=False, default=_a_json, allow_nan=False) if _sin_nan(obj) else \
        json.dumps(_limpiar(obj), indent=2, ensure_ascii=False, default=_a_json)
    ruta.write_text(texto, encoding="utf-8", newline="\n")


def leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _sin_nan(obj: Any) -> bool:
    try:
        json.dumps(obj, default=_a_json, allow_nan=False)
        return True
    except ValueError:
        return False


def _limpiar(obj: Any) -> Any:
    """Reemplaza NaN/inf por None para que el JSON sea estándar."""
    if isinstance(obj, dict):
        return {str(k): _limpiar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpiar(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)) and not np.isfinite(obj):
        return None
    return obj

"""Etapa export: tabla de asignaciones consolidada, metadata.json, puntero LATEST y poda de ejecuciones."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tipologia.config import Config
from tipologia.features.prepare import cargar_datos_base
from tipologia.pipeline.artefactos import actualizar_latest, escribir_metadata, podar

log = logging.getLogger(__name__)


def tabla_asignaciones(cfg: Config, destino: Path) -> pd.DataFrame:
    """Una fila por municipio incluido, con tipo, probabilidades, marcas de frontera y contexto."""
    asig = pd.read_parquet(destino / "asignaciones_base.parquet")
    mun = pd.read_parquet(destino / "municipios.parquet")
    nombres = pd.read_csv(destino / "nombres_tipos.csv").set_index("tipo")
    datos = cargar_datos_base(cfg)
    anio = cfg["datos"]["anio_referencia"]
    pob = datos.poblacion(cfg, anio)
    df = mun[["nombre", "cod_dpto", "dpto", "region"]].rename(columns={"nombre": "municipio", "dpto": "departamento"})
    df = df.join(asig)
    df.insert(df.columns.get_loc("tipo") + 1, "nombre_tipo", df["tipo"].map(nombres["nombre"]))
    df["poblacion"] = pob.reindex(df.index).round().astype("Int64")
    df["cat_ruralidad_dnp"] = mun["cat_ruralidad"]
    df["tipologia_dnp_2026"] = mun["tipologia_dnp_2026"]
    alt = pd.read_parquet(destino / "alternativas.parquet")
    for c in alt.columns[1:]:
        df[f"tipo_alternativo_{c}"] = alt[c]
    for archivo in ("espacial_local.parquet", "espacial_restringido.parquet"):
        if (destino / archivo).exists():
            df = df.join(pd.read_parquet(destino / archivo))
    if (destino / "temporal_asignaciones.parquet").exists():
        t = pd.read_parquet(destino / "temporal_asignaciones.parquet")
        df = df.join(t[[c for c in t.columns if c.startswith("tipo_")]].add_prefix("temporal_"))
    df["anio_referencia"] = anio
    df.index.name = "cod"
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(4)
    return df


def etapa_export(cfg: Config, destino: Path, podar_anteriores: bool = True) -> dict:
    df = tabla_asignaciones(cfg, destino)
    df.to_csv(destino / "asignaciones.csv", encoding="utf-8", lineterminator="\n")
    df.to_parquet(destino / "asignaciones.parquet")
    meta = escribir_metadata(cfg, destino)
    actualizar_latest(cfg, destino.name)
    borradas = podar(cfg, destino.name) if podar_anteriores else []
    log.info("export: %s (%d artefactos); ejecuciones podadas: %s", destino.name, len(meta["artefactos_sha256"]), borradas)
    return {"ejecucion": destino.name, "artefactos": len(meta["artefactos_sha256"]), "podadas": borradas}

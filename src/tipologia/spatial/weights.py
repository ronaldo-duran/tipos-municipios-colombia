"""Vecindad espacial entre municipios (contigüidad tipo reina a partir del MGN 2025, solo en local)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import KNN, Queen, W

log = logging.getLogger(__name__)


def vecindad_reina(geo: gpd.GeoDataFrame, codigos: pd.Index, columna: str = "cod") -> pd.DataFrame:
    """Lista de adyacencias (cod, vecino) por contigüidad reina entre los municipios dados.

    Los municipios sin vecinos (islas, como Providencia) se conectan con su vecino más cercano por
    centroide para que todos tengan al menos un vecino.
    """
    g = geo[geo[columna].isin(codigos)].drop_duplicates(columna).set_index(columna).loc[sorted(set(codigos) & set(geo[columna]))]
    g = g.to_crs(9377)  # MAGNA-SIRGAS / Origen-Nacional (metros)
    wq = Queen.from_dataframe(g, use_index=True, silence_warnings=True)
    vecinos = {k: list(v) for k, v in wq.neighbors.items()}
    islas = [k for k, v in vecinos.items() if not v]
    if islas:
        cent = g.geometry.centroid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            knn = KNN.from_array(np.column_stack([cent.x, cent.y]), k=1)
        ids = list(g.index)
        for isla in islas:
            j = knn.neighbors[ids.index(isla)][0]
            vecinos[isla] = [ids[j]]
            vecinos[ids[j]] = sorted(set(vecinos[ids[j]]) | {isla})
        log.info("Municipios sin vecinos conectados por cercanía: %s", islas)
    filas = [(k, v) for k, vs in vecinos.items() for v in vs]
    return pd.DataFrame(filas, columns=["cod", "vecino"]).sort_values(["cod", "vecino"]).reset_index(drop=True)


def pesos_desde_lista(adyacencias: pd.DataFrame, codigos: pd.Index) -> W:
    """Matriz de pesos (estandarizada por filas) restringida a `codigos`."""
    cods = set(codigos)
    ady = adyacencias[adyacencias["cod"].isin(cods) & adyacencias["vecino"].isin(cods)]
    vecinos = {c: [] for c in codigos}
    for c, v in ady.itertuples(index=False):
        vecinos[c].append(v)
    w = W(vecinos, silence_warnings=True, id_order=list(codigos))
    w.transform = "r"
    return w


def guardar_vecindad(ady: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ady.to_csv(ruta, index=False)

"""Autocorrelación espacial: ¿los tipos forman zonas contiguas o están dispersos?"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from esda.join_counts import Join_Counts
from esda.moran import Moran, Moran_Local
from libpysal.weights import W

log = logging.getLogger(__name__)

CUADRANTES = {1: "alto-alto", 2: "bajo-alto", 3: "bajo-bajo", 4: "alto-bajo"}


def moran_por_tipo(w: W, tipos: pd.Series, permutaciones: int, semilla: int) -> pd.DataFrame:
    """Moran global y join counts (pares vecinos del mismo tipo) para el indicador binario de cada tipo."""
    filas = []
    orden = pd.Index(w.id_order)
    t = tipos.reindex(orden)
    for tipo in sorted(t.unique()):
        y = (t == tipo).astype(float).to_numpy()
        np.random.seed(semilla)
        mi = Moran(y, w, permutations=permutaciones)
        np.random.seed(semilla)
        jc = Join_Counts(y.astype(int), w, permutations=permutaciones)
        filas.append({"tipo": tipo, "moran_I": float(mi.I), "moran_esperado": float(mi.EI), "moran_p": float(mi.p_sim),
                      "pares_mismo_tipo_obs": float(jc.bb), "pares_mismo_tipo_esperados": float(jc.mean_bb),
                      "join_count_p": float(jc.p_sim_bb)})
    return pd.DataFrame(filas)


def moran_variable(w: W, y: pd.Series, permutaciones: int, semilla: int) -> dict:
    yv = y.reindex(pd.Index(w.id_order)).to_numpy(dtype=float)
    np.random.seed(semilla)
    mi = Moran(yv, w, permutations=permutaciones)
    return {"moran_I": float(mi.I), "moran_p": float(mi.p_sim), "z": float(mi.z_sim)}


def lisa(w: W, y: pd.Series, permutaciones: int, semilla: int, alfa: float) -> pd.DataFrame:
    """Moran local: cuadrante y significancia por municipio (con corrección por permutaciones condicionales)."""
    orden = pd.Index(w.id_order)
    yv = y.reindex(orden).to_numpy(dtype=float)
    np.random.seed(semilla)
    ml = Moran_Local(yv, w, permutations=permutaciones, seed=semilla)
    cuad = pd.Series(ml.q, index=orden).map(CUADRANTES)
    sig = ml.p_sim < alfa
    return pd.DataFrame({"lisa_I": ml.Is, "lisa_p": ml.p_sim, "lisa_cuadrante": cuad,
                         "lisa_categoria": np.where(sig, cuad, "no significativo")}, index=orden)


def vecinos_mismo_tipo(w: W, tipos: pd.Series) -> tuple[pd.Series, dict]:
    """Proporción de vecinos del mismo tipo por municipio, frente a lo esperado si los tipos estuvieran
    repartidos al azar (suma de las proporciones al cuadrado)."""
    t = tipos.reindex(pd.Index(w.id_order))
    prop = {}
    for cod, vs in w.neighbors.items():
        prop[cod] = float(np.mean([t[v] == t[cod] for v in vs])) if vs else np.nan
    s = pd.Series(prop).reindex(t.index)
    p = t.value_counts(normalize=True)
    return s, {"observado": float(s.mean()), "esperado_al_azar": float((p ** 2).sum())}

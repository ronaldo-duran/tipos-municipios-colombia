"""Transformaciones, winsorización, escalado y ponderación por dominio."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from tipologia.config import Indicador

log = logging.getLogger(__name__)


def transformar(df: pd.DataFrame, indicadores: list[Indicador]) -> pd.DataFrame:
    """Aplica log o log1p según el catálogo. Valores no positivos en log quedan como NaN (se avisa)."""
    out = df.copy()
    for ind in indicadores:
        if ind.id not in out:
            continue
        if ind.transformacion == "log":
            malos = (out[ind.id] <= 0).sum()
            if malos:
                log.warning("%s: %d valores <= 0 no admiten log; quedan como faltantes", ind.id, malos)
            out[ind.id] = np.log(out[ind.id].where(out[ind.id] > 0))
        elif ind.transformacion == "log1p":
            out[ind.id] = np.log1p(out[ind.id].clip(lower=0))
    return out


@dataclass
class Winsorizacion:
    limites: dict[str, tuple[float, float]] = field(default_factory=dict)
    recortados: dict[str, int] = field(default_factory=dict)


def winsorizar(df: pd.DataFrame, inferior: float, superior: float) -> tuple[pd.DataFrame, Winsorizacion]:
    """Recorta cada columna a sus percentiles [inferior, superior], ignorando faltantes."""
    out = df.copy()
    info = Winsorizacion()
    for c in out.columns:
        lo, hi = out[c].quantile([inferior, superior]).to_numpy()
        n = int(((out[c] < lo) | (out[c] > hi)).sum())
        out[c] = out[c].clip(lo, hi)
        info.limites[c] = (float(lo), float(hi))
        info.recortados[c] = n
    return out, info


@dataclass
class Escalado:
    metodo: str
    centro: pd.Series
    escala: pd.Series

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.centro) / self.escala


def ajustar_escalado(df: pd.DataFrame, metodo: str = "estandar") -> Escalado:
    """Estándar (media, desviación) o robusto (mediana, rango intercuartílico)."""
    if metodo == "estandar":
        centro, escala = df.mean(), df.std(ddof=0)
    elif metodo == "robusto":
        centro = df.median()
        escala = df.quantile(0.75) - df.quantile(0.25)
    else:
        raise ValueError(f"Escalado desconocido: {metodo}")
    escala = escala.replace(0, 1.0).fillna(1.0)
    return Escalado(metodo, centro, escala)


@dataclass
class Ponderacion:
    metodo: str
    pesos: pd.Series                      # peso por columna de la matriz resultante
    dominio_columna: dict[str, str]       # dominio de cada columna resultante
    detalle: dict = field(default_factory=dict)


def ponderar(Z: pd.DataFrame, dominios: dict[str, str], metodo: str = "dominio",
             varianza_bloque: float = 0.8, semilla: int = 0) -> tuple[pd.DataFrame, Ponderacion]:
    """Evita que un dominio domine solo por tener más indicadores.

    - indicador: cada indicador pesa 1 (los dominios con más indicadores pesan más).
    - dominio: cada indicador se multiplica por 1/sqrt(k_d), así cada dominio aporta la misma varianza total.
    - pca_bloque: PCA dentro de cada dominio; se conservan los componentes que explican `varianza_bloque`
      y se reescalan para que cada dominio aporte varianza total 1.
    Z debe venir estandarizada.
    """
    cols_por_dom: dict[str, list[str]] = {}
    for c in Z.columns:
        cols_por_dom.setdefault(dominios[c], []).append(c)
    if metodo == "indicador":
        pesos = pd.Series(1.0, index=Z.columns)
        return Z.copy(), Ponderacion(metodo, pesos, {c: dominios[c] for c in Z.columns})
    if metodo == "dominio":
        pesos = pd.Series({c: 1 / np.sqrt(len(cols_por_dom[dominios[c]])) for c in Z.columns})
        return Z * pesos, Ponderacion(metodo, pesos, {c: dominios[c] for c in Z.columns})
    if metodo == "pca_bloque":
        bloques, dom_col, detalle = [], {}, {}
        for dom, cols in cols_por_dom.items():
            X = Z[cols].to_numpy()
            if len(cols) == 1:
                S = X / X.std(ddof=0)
                nombres = [f"{dom}_pc1"]
                evr = [1.0]
            else:
                pca = PCA(random_state=semilla).fit(X)
                k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), varianza_bloque) + 1)
                S = pca.transform(X)[:, :k]
                S = S / np.sqrt(S.var(axis=0, ddof=0).sum())
                nombres = [f"{dom}_pc{i + 1}" for i in range(k)]
                evr = pca.explained_variance_ratio_[:k].round(3).tolist()
            bloques.append(pd.DataFrame(S, index=Z.index, columns=nombres))
            dom_col.update({n: dom for n in nombres})
            detalle[dom] = {"componentes": len(nombres), "varianza_explicada": evr}
        out = pd.concat(bloques, axis=1)
        return out, Ponderacion(metodo, pd.Series(1.0, index=out.columns), dom_col, detalle)
    raise ValueError(f"Ponderación desconocida: {metodo}")


def varianza_por_dominio(X: pd.DataFrame, dominio_columna: dict[str, str]) -> pd.Series:
    """Varianza total que aporta cada dominio a la matriz ponderada."""
    v = X.var(ddof=0)
    return v.groupby(pd.Series(dominio_columna)).sum()

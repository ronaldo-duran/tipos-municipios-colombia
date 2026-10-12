"""Imputación de faltantes. Nunca usa información del clustering (solo variables y departamento)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

log = logging.getLogger(__name__)

METODOS = ("iterativa", "mediana_departamento", "knn", "casos_completos")


def imputar(df: pd.DataFrame, metodo: str, grupos: pd.Series | None = None, semilla: int = 0,
            knn_vecinos: int = 10, max_iter: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (datos imputados, máscara de celdas imputadas).

    - mediana_departamento: mediana del departamento; si el departamento no tiene dato, mediana nacional.
    - knn: KNNImputer sobre los datos estandarizados.
    - iterativa: IterativeImputer (BayesianRidge) sobre los datos estandarizados.
    - casos_completos: elimina los municipios con algún faltante.
    """
    mascara = df.isna()
    if not mascara.any().any():
        return df.copy(), mascara
    if metodo == "casos_completos":
        completos = ~mascara.any(axis=1)
        log.info("Casos completos: %d de %d municipios", int(completos.sum()), len(df))
        return df.loc[completos].copy(), mascara.loc[completos]
    if metodo == "mediana_departamento":
        if grupos is None:
            raise ValueError("mediana_departamento necesita la serie de departamentos")
        g = grupos.reindex(df.index)
        out = df.copy()
        for c in df.columns:
            med_dep = df[c].groupby(g).transform("median")
            out[c] = df[c].fillna(med_dep).fillna(df[c].median())
        return out, mascara
    media, desv = df.mean(), df.std(ddof=0).replace(0, 1.0)
    Z = (df - media) / desv
    if metodo == "knn":
        imp = KNNImputer(n_neighbors=knn_vecinos)
    elif metodo == "iterativa":
        imp = IterativeImputer(max_iter=max_iter, random_state=semilla, sample_posterior=False, tol=1e-3)
    else:
        raise ValueError(f"Imputación desconocida: {metodo}")
    Zi = pd.DataFrame(imp.fit_transform(Z.to_numpy()), index=df.index, columns=df.columns)
    out = Zi * desv + media
    # Las celdas observadas no cambian (protege contra redondeos).
    out = out.where(mascara, df)
    return out, mascara


def recortar_a_rango_observado(imputado: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Evita que la imputación produzca valores fuera del rango observado de cada indicador."""
    lo, hi = original.min(), original.max()
    return imputado.clip(lower=lo, upper=hi, axis=1)


def resumen_imputacion(mascara: pd.DataFrame) -> pd.DataFrame:
    """Celdas imputadas por indicador."""
    return pd.DataFrame({"imputados": mascara.sum(), "pct": (mascara.mean() * 100).round(2)})


def indicadores_imputados_por_municipio(mascara: pd.DataFrame) -> pd.Series:
    """Lista de indicadores imputados por municipio (texto)."""
    return mascara.apply(lambda r: ", ".join(r.index[r.to_numpy(dtype=bool)]), axis=1).replace("", np.nan)

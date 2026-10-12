"""Algoritmos de clustering y criterios internos para elegir el número de grupos."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

log = logging.getLogger(__name__)

ALGORITMOS_K = ("kmeans", "gmm", "ward")


@dataclass
class Ajuste:
    algoritmo: str
    k: int
    etiquetas: np.ndarray
    modelo: object
    probabilidades: np.ndarray | None = None


def ajustar(algoritmo: str, X: np.ndarray, k: int, semilla: int, n_init_kmeans: int = 20,
            n_init_gmm: int = 5, covarianza: str = "full", min_cluster_size: int = 15,
            min_samples: int = 5) -> Ajuste:
    """Ajusta un algoritmo con k grupos (para HDBSCAN, k se ignora y manda min_cluster_size)."""
    X = np.asarray(X)
    if algoritmo == "kmeans":
        m = KMeans(n_clusters=k, n_init=n_init_kmeans, random_state=semilla).fit(X)
        return Ajuste(algoritmo, k, m.labels_, m)
    if algoritmo == "gmm":
        m = GaussianMixture(n_components=k, covariance_type=covarianza, n_init=n_init_gmm,
                            random_state=semilla, reg_covar=1e-4, max_iter=500).fit(X)
        proba = m.predict_proba(X)
        return Ajuste(algoritmo, k, proba.argmax(axis=1), m, proba)
    if algoritmo == "ward":
        m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        return Ajuste(algoritmo, k, m.labels_, m)
    if algoritmo == "hdbscan":
        m = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit(X)
        lab = m.labels_
        return Ajuste(algoritmo, int(len(set(lab)) - (1 if -1 in lab else 0)), lab, m, getattr(m, "probabilities_", None))
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def metricas_internas(X: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    """Silhouette (↑), Davies-Bouldin (↓) y Calinski-Harabasz (↑). Ignora el ruido (-1)."""
    X = np.asarray(X)
    m = etiquetas >= 0
    if len(set(etiquetas[m])) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(X[m], etiquetas[m])),
        "davies_bouldin": float(davies_bouldin_score(X[m], etiquetas[m])),
        "calinski_harabasz": float(calinski_harabasz_score(X[m], etiquetas[m])),
    }


def _log_wk(X: np.ndarray, etiquetas: np.ndarray) -> float:
    w = 0.0
    for c in np.unique(etiquetas):
        P = X[etiquetas == c]
        w += ((P - P.mean(axis=0)) ** 2).sum()
    return float(np.log(w))


def gap_statistic(X: np.ndarray, ks: list[int], referencias: int, semilla: int, n_init: int = 10) -> pd.DataFrame:
    """Gap statistic (Tibshirani et al., 2001) con referencias uniformes en la caja de los componentes
    principales. Devuelve gap(k), s(k) y el k recomendado por la regla gap(k) >= gap(k+1) - s(k+1)."""
    X = np.asarray(X)
    rng = np.random.default_rng(semilla)
    U, _, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    Xp = (X - X.mean(axis=0)) @ Vt.T
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)
    filas = []
    for k in ks:
        obs = _log_wk(X, KMeans(k, n_init=n_init, random_state=semilla).fit_predict(X))
        ref = []
        for _ in range(referencias):
            R = rng.uniform(lo, hi, size=Xp.shape) @ Vt + X.mean(axis=0)
            ref.append(_log_wk(R, KMeans(k, n_init=3, random_state=semilla).fit_predict(R)))
        ref = np.array(ref)
        filas.append({"k": k, "gap": ref.mean() - obs, "s": ref.std(ddof=0) * np.sqrt(1 + 1 / referencias)})
    df = pd.DataFrame(filas)
    df["cumple_regla"] = False
    for i in range(len(df) - 1):
        df.loc[i, "cumple_regla"] = bool(df.loc[i, "gap"] >= df.loc[i + 1, "gap"] - df.loc[i + 1, "s"])
    return df


def seleccion_k(X: pd.DataFrame, ks: list[int], semilla: int, cfg_cluster: dict) -> pd.DataFrame:
    """Métricas internas para cada algoritmo y k. Incluye BIC/AIC de GMM."""
    Xa = X.to_numpy()
    filas = []
    for alg in cfg_cluster["algoritmos"]:
        for k in ks:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = ajustar(alg, Xa, k, semilla, cfg_cluster["kmeans_n_init"], cfg_cluster["gmm_n_init"],
                            cfg_cluster["gmm_covarianza"])
            fila = {"algoritmo": alg, "k": k, **metricas_internas(Xa, a.etiquetas),
                    "tamano_min": int(np.bincount(a.etiquetas).min()),
                    "tamano_max": int(np.bincount(a.etiquetas).max())}
            if alg == "gmm":
                fila["bic"] = float(a.modelo.bic(Xa))
                fila["aic"] = float(a.modelo.aic(Xa))
            if alg == "kmeans":
                fila["inercia"] = float(a.modelo.inertia_)
            filas.append(fila)
        log.info("Selección de k: %s listo", alg)
    return pd.DataFrame(filas)


def barrido_hdbscan(X: pd.DataFrame, tamanos: list[int], min_samples: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """HDBSCAN con distintos tamaños mínimos de grupo: número de grupos y % de ruido."""
    Xa = X.to_numpy()
    filas, etiquetas = [], {}
    for t in tamanos:
        a = ajustar("hdbscan", Xa, 0, 0, min_cluster_size=t, min_samples=min_samples)
        etiquetas[t] = a.etiquetas
        filas.append({"min_cluster_size": t, "n_grupos": a.k, "pct_ruido": float((a.etiquetas < 0).mean() * 100),
                      **metricas_internas(Xa, a.etiquetas)})
    return pd.DataFrame(filas), etiquetas


def acuerdo(etiquetas: dict[str, np.ndarray]) -> pd.DataFrame:
    """ARI y AMI entre todas las particiones (formato largo)."""
    nombres = list(etiquetas)
    filas = []
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            filas.append({"a": a, "b": b, "ari": float(adjusted_rand_score(etiquetas[a], etiquetas[b])),
                          "ami": float(adjusted_mutual_info_score(etiquetas[a], etiquetas[b]))})
    return pd.DataFrame(filas)


def orden_por_tamano(etiquetas: np.ndarray) -> np.ndarray:
    """Reetiqueta los grupos de mayor a menor tamaño (0 = el más grande). El ruido (-1) se conserva.

    Ordenar por tamaño evita sugerir una jerarquía de 'mejor' a 'peor'.
    """
    validas = etiquetas[etiquetas >= 0]
    cuentas = pd.Series(validas).value_counts()
    orden = sorted(cuentas.index, key=lambda c: (-cuentas[c], c))
    mapa = {c: i for i, c in enumerate(orden)}
    return np.array([mapa.get(e, -1) for e in etiquetas])

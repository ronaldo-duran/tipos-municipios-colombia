"""Reducción de dimensionalidad: PCA (análisis y entrada del clustering) y UMAP (solo visualización)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


@dataclass
class ResultadoPCA:
    modelo: PCA
    scores: pd.DataFrame          # componentes retenidos (entrada del clustering)
    varianza: pd.DataFrame        # varianza explicada por componente (todas)
    cargas: pd.DataFrame          # cargas (componentes retenidos x variables)
    n_componentes: int
    signos: np.ndarray            # signo aplicado a cada componente (para que sea determinista)

    def scores_todos(self, X: pd.DataFrame, n: int) -> pd.DataFrame:
        """Primeros n componentes (aunque no se retengan) con los mismos signos."""
        S = self.modelo.transform(X.to_numpy())[:, :n] * self.signos[:n]
        return pd.DataFrame(S, index=X.index, columns=[f"PC{i + 1}" for i in range(n)])


def ajustar_pca(X: pd.DataFrame, varianza_objetivo: float, semilla: int) -> ResultadoPCA:
    """PCA completo; retiene los componentes necesarios para explicar `varianza_objetivo`."""
    pca = PCA(random_state=semilla, svd_solver="full").fit(X.to_numpy())
    acum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(acum, varianza_objetivo) + 1)
    S = pca.transform(X.to_numpy())
    # Signo determinista: la carga de mayor valor absoluto de cada componente es positiva.
    signos = np.sign(pca.components_[np.arange(len(pca.components_)), np.abs(pca.components_).argmax(axis=1)])
    S = S * signos
    comps = pca.components_ * signos[:, None]
    nombres = [f"PC{i + 1}" for i in range(len(acum))]
    scores = pd.DataFrame(S[:, :k], index=X.index, columns=nombres[:k])
    varianza = pd.DataFrame({"componente": nombres, "varianza_explicada": pca.explained_variance_ratio_,
                             "acumulada": acum})
    cargas = pd.DataFrame(comps[:k], index=nombres[:k], columns=X.columns)
    log.info("PCA: %d componentes explican %.1f%% de la varianza", k, acum[k - 1] * 100)
    return ResultadoPCA(pca, scores, varianza, cargas, k, signos)


def entrada_clustering(X: pd.DataFrame, entrada: str, varianza_objetivo: float, semilla: int) -> tuple[pd.DataFrame, ResultadoPCA | None]:
    """Matriz sobre la que se clusteriza: scores del PCA o variables ponderadas."""
    if entrada == "pca":
        r = ajustar_pca(X, varianza_objetivo, semilla)
        return r.scores, r
    if entrada == "variables":
        return X.copy(), None
    raise ValueError(f"Entrada desconocida: {entrada}")


def umap_2d(X: pd.DataFrame, n_vecinos: int, min_dist: float, semilla: int) -> pd.DataFrame:
    """Proyección UMAP 2D solo para visualización exploratoria.

    Sus distancias no son interpretables y no se usa para decidir clusters.
    """
    import umap

    emb = umap.UMAP(n_neighbors=n_vecinos, min_dist=min_dist, random_state=semilla, n_jobs=1).fit_transform(X.to_numpy())
    return pd.DataFrame(emb, index=X.index, columns=["umap_1", "umap_2"])

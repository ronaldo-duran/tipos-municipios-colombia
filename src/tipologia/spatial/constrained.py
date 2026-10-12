"""Contraste opcional: clustering jerárquico de Ward con restricción de contigüidad espacial."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from libpysal.weights import W
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from tipologia.clustering.models import orden_por_tamano


def ward_restringido(S: pd.DataFrame, w: W, k: int) -> pd.Series:
    """Regiones contiguas homogéneas: Ward solo puede unir municipios vecinos."""
    orden = pd.Index(w.id_order)
    conect = w.sparse.copy()
    conect.data[:] = 1
    with warnings.catch_warnings():
        # Si la vecindad tiene componentes desconectados (islas), sklearn los completa: se documenta.
        warnings.simplefilter("ignore")
        m = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=conect).fit(S.reindex(orden).to_numpy())
    return pd.Series(orden_por_tamano(m.labels_), index=orden)


def comparar_con_base(base: pd.Series, restringido: pd.Series) -> dict:
    comunes = base.index.intersection(restringido.index)
    return {"ari": float(adjusted_rand_score(base.loc[comunes], restringido.loc[comunes])),
            "ami": float(adjusted_mutual_info_score(base.loc[comunes], restringido.loc[comunes])),
            "tamanos": restringido.value_counts().sort_index().to_dict(),
            "n_regiones": int(restringido.nunique()),
            "tamano_mediano": float(np.median(restringido.value_counts()))}

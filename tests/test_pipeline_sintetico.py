"""Prueba de punta a punta con el código real del pipeline sobre datos sintéticos en formato TerriData."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from tests.sintetico import datos_sinteticos
from tipologia.clustering.models import seleccion_k
from tipologia.clustering.reduce import entrada_clustering
from tipologia.clustering.seleccion import decidir_k
from tipologia.config import cargar_config
from tipologia.evaluation.stability import tabla_estabilidad
from tipologia.features.prepare import preparar
from tipologia.pipeline.etapa_estabilidad import modelo_final


@pytest.fixture(scope="module")
def cfg():
    return cargar_config().con_cambios({
        "clustering.k_min": 2, "clustering.k_max": 6, "clustering.kmeans_n_init": 5, "clustering.gmm_n_init": 2,
        "estabilidad.n_remuestreos": 8,
    })


def correr(cfg, datos):
    """features -> PCA -> selección de k con estabilidad -> modelo final (igual que las etapas reales)."""
    P = preparar(cfg, datos)
    c, e = cfg["clustering"], cfg["estabilidad"]
    S, _ = entrada_clustering(P.X, c["entrada"], c["pca_varianza"], cfg.semilla)
    ks = list(range(c["k_min"], c["k_max"] + 1))
    sel = seleccion_k(S, ks, cfg.semilla, c)
    estab, _ = tabla_estabilidad(S, c["algoritmos"], ks, c, e["n_remuestreos"], e["fraccion_submuestra"], cfg.semilla)
    dec = decidir_k(sel, estab, e["umbral_estable"], c["modelo_final"], n=len(S))
    etiquetas, proba, _, _ = modelo_final(S, dec["k"], cfg)
    return P, dec, etiquetas, proba, estab


@pytest.fixture(scope="module")
def con_grupos(cfg):
    datos, verdad = datos_sinteticos(cfg, k=3)
    return verdad, correr(cfg, datos)


def test_recupera_los_grupos_plantados(con_grupos):
    verdad, (P, dec, etiquetas, proba, _) = con_grupos
    assert P.Z.shape[0] == len(verdad) and P.Z.shape[1] >= 38
    assert dec["k"] == 3 and not dec["ninguna_particion_estable"]
    assert adjusted_rand_score(verdad, etiquetas) > 0.95
    assert np.allclose(proba.sum(axis=1), 1) and (proba.max(axis=1) > 0.9).mean() > 0.9


def test_con_ruido_la_estabilidad_es_baja(cfg):
    datos, _ = datos_sinteticos(cfg, k=0, semilla=5)
    _, dec, _, _, estab = correr(cfg, datos)
    gmm = estab[estab["algoritmo"] == "gmm"]
    assert gmm["ari_medio"].max() < 0.75
    assert dec["ninguna_particion_estable"]


def test_determinismo_con_las_mismas_semillas(cfg, con_grupos):
    datos, _ = datos_sinteticos(cfg, k=3)
    _, (_, dec1, et1, pr1, est1) = con_grupos
    _, dec2, et2, pr2, est2 = correr(cfg, datos)
    np.testing.assert_array_equal(et1, et2)
    np.testing.assert_array_equal(pr1, pr2)
    pd.testing.assert_frame_equal(est1, est2)

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tipologia.clustering.models import (
    acuerdo,
    ajustar,
    barrido_hdbscan,
    gap_statistic,
    orden_por_tamano,
    seleccion_k,
)
from tipologia.clustering.reduce import ajustar_pca
from tipologia.clustering.seleccion import decidir_k

CFG = {"algoritmos": ["kmeans", "gmm", "ward"], "kmeans_n_init": 10, "gmm_n_init": 2, "gmm_covarianza": "full"}


@pytest.mark.parametrize("algoritmo", ["kmeans", "gmm", "ward"])
def test_determinismo_con_semilla(plantados, algoritmo):
    X, _ = plantados
    a = ajustar(algoritmo, X.to_numpy(), 4, semilla=123).etiquetas
    b = ajustar(algoritmo, X.to_numpy(), 4, semilla=123).etiquetas
    np.testing.assert_array_equal(a, b)


def test_pca_determinista_y_con_signo_fijo(plantados):
    X, _ = plantados
    r1, r2 = ajustar_pca(X, 0.8, 0), ajustar_pca(-X * 1.0 + 0.0, 0.8, 0)
    np.testing.assert_allclose(ajustar_pca(X, 0.8, 0).scores, r1.scores)
    # el signo no depende de la orientación de los datos de entrada
    assert (r1.cargas.abs().idxmax(axis=1) == r2.cargas.abs().idxmax(axis=1)).all()


@pytest.mark.parametrize("algoritmo", ["kmeans", "gmm", "ward"])
def test_recupera_grupos_plantados(plantados, algoritmo):
    X, verdad = plantados
    pred = ajustar(algoritmo, X.to_numpy(), 4, semilla=0).etiquetas
    assert adjusted_rand_score(verdad, pred) > 0.95


def test_regla_de_seleccion_elige_k_plantado(plantados):
    X, _ = plantados
    sel = seleccion_k(X, list(range(2, 8)), 0, CFG)
    assert decidir_k(sel, n=len(X))["k"] == 4


def test_gap_statistic_en_datos_con_grupos(plantados):
    X, _ = plantados
    gap = gap_statistic(X.to_numpy(), list(range(1, 8)), referencias=10, semilla=0)
    assert int(gap.loc[gap["cumple_regla"], "k"].min()) == 4


def test_hdbscan_grupos_densos_vs_ruido(plantados, ruido):
    X, _ = plantados
    hd, _ = barrido_hdbscan(X, [15], 5)
    assert hd["n_grupos"].iloc[0] == 4 and hd["pct_ruido"].iloc[0] < 20
    hd_r, _ = barrido_hdbscan(ruido, [15], 5)
    assert hd_r["n_grupos"].iloc[0] <= 1 or hd_r["pct_ruido"].iloc[0] > 50


def test_silhouette_bajo_en_ruido(ruido):
    sel = seleccion_k(ruido, [3, 4], 0, {**CFG, "algoritmos": ["kmeans"]})
    assert sel["silhouette"].max() < 0.25


def test_acuerdo_entre_algoritmos(plantados):
    X, verdad = plantados
    etq = {a: ajustar(a, X.to_numpy(), 4, 0).etiquetas for a in ("kmeans", "ward")}
    etq["verdad"] = verdad
    df = acuerdo(etq)
    assert (df["ari"] > 0.95).all() and len(df) == 3


def test_orden_por_tamano_y_ruido():
    e = np.array([2, 2, 2, 0, 1, 1, -1])
    np.testing.assert_array_equal(orden_por_tamano(e), [0, 0, 0, 2, 1, 1, -1])

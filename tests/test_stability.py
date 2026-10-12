import numpy as np
import pandas as pd
import pytest

from tests.conftest import generar_plantados, generar_ruido
from tipologia.clustering.models import ajustar
from tipologia.evaluation.sensitivity import alinear, etiqueta_tipo
from tipologia.evaluation.stability import (
    estabilidad,
    perturbacion_por_tamano,
    referencia_nula_gaussiana,
    resumen_coasociacion,
)

CFG = {"kmeans_n_init": 10, "gmm_n_init": 2, "gmm_covarianza": "full"}


@pytest.fixture(scope="module")
def datos():
    X, verdad = generar_plantados(n=400)
    return X, verdad, generar_ruido(n=400)


@pytest.mark.parametrize("algoritmo", ["kmeans", "gmm", "ward"])
def test_estabilidad_alta_con_grupos_plantados_y_baja_con_ruido(datos, algoritmo):
    X, _, R = datos
    alta = estabilidad(X.to_numpy(), algoritmo, 4, CFG, 10, 0.8, 0)
    baja = estabilidad(R.to_numpy(), algoritmo, 4, CFG, 10, 0.8, 0)
    assert alta.ari.mean() > 0.95 and alta.jaccard["jaccard_medio"].min() > 0.9
    assert baja.ari.mean() < 0.5


def test_coasociacion_y_pac(datos):
    X, _, R = datos
    for D, pac_max in ((X, 0.01), (R, 1.0)):
        ref = ajustar("kmeans", D.to_numpy(), 4, 0).etiquetas
        r = estabilidad(D.to_numpy(), "kmeans", 4, CFG, 10, 0.8, 0, referencia=ref, coasociar=True)
        res, indiv = resumen_coasociacion(r.coasociacion, r.pares, ref)
        assert res["pac"] <= pac_max
    ref = ajustar("kmeans", X.to_numpy(), 4, 0).etiquetas
    r = estabilidad(X.to_numpy(), "kmeans", 4, CFG, 10, 0.8, 0, referencia=ref, coasociar=True)
    _, indiv = resumen_coasociacion(r.coasociacion, r.pares, ref)
    assert indiv.min() > 0.99


def test_datos_con_grupos_superan_la_referencia_nula(datos):
    X, _, _ = datos
    nula = referencia_nula_gaussiana(X, "gmm", 4, CFG, replicas=2, semilla=0, n_remuestreos=5)
    obs = ajustar("gmm", X.to_numpy(), 4, 0).etiquetas
    from tipologia.clustering.models import metricas_internas
    assert metricas_internas(X.to_numpy(), obs)["silhouette"] > nula["silhouette"].max() + 0.2


def test_alinear_etiquetas_permutadas():
    base = pd.Series([0, 0, 1, 1, 2, 2], index=list("abcdef"))
    otra = pd.Series([5, 5, 3, 3, 4, 4], index=list("abcdef"))
    alineada, coinc = alinear(base, otra)
    assert coinc == 1.0 and alineada.tolist() == base.tolist()


def test_perturbacion_sin_ruido_no_cambia_nada():
    Z = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=list("abc"),
                     index=[f"{i:05d}" for i in range(50)])
    base = (Z["a"] > 0).astype(int).to_numpy()
    pob = pd.Series(np.linspace(1000, 100000, 50), index=Z.index)
    t, tasa = perturbacion_por_tamano(Z, pd.Series(1.0, index=Z.columns), lambda X: X.to_numpy(),
                                      lambda S: (S[:, 0] > 0).astype(int), base, pob, ["a"], 0.0, 5, 0)
    assert tasa.max() == 0.0
    t, tasa = perturbacion_por_tamano(Z, pd.Series(1.0, index=Z.columns), lambda X: X.to_numpy(),
                                      lambda S: (S[:, 0] > 0).astype(int), base, pob, ["a"], 1.0, 20, 0)
    # con ruido, los municipios pequeños cambian más que los grandes
    assert t["tasa_cambio_media"].iloc[0] >= t["tasa_cambio_media"].iloc[-1]


def test_etiquetas_por_tamano_no_por_merito():
    assert [etiqueta_tipo(i) for i in range(3)] == ["A", "B", "C"]

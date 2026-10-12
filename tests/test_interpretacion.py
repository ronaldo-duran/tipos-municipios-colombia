import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tests.conftest import generar_plantados
from tipologia.evaluation.external import (
    comparar_ordinal,
    comparar_particion,
    cramer_v,
    gamma_goodman_kruskal,
    validacion_externa,
)
from tipologia.evaluation.profile import descriptores, perfiles_z, representativos, similares, tamanos
from tipologia.evaluation.surrogate import arbol_sustituto
from tipologia.spatial.autocorr import moran_por_tipo, vecinos_mismo_tipo
from tipologia.spatial.constrained import ward_restringido
from tipologia.spatial.weights import pesos_desde_lista, vecindad_reina


def test_gamma_y_cramer_en_asociacion_perfecta():
    x = np.array([0, 0, 1, 1, 2, 2])
    assert gamma_goodman_kruskal(x, x) == pytest.approx(1.0)
    assert gamma_goodman_kruskal(x, 2 - x) == pytest.approx(-1.0)
    t = pd.crosstab(x, x)
    assert cramer_v(t) > 0.8


def test_comparacion_ordinal_no_usa_ari_y_separa_especiales():
    tipos = pd.Series(list("AAABBBCCC"), index=[f"{i:05d}" for i in range(9)])
    ref = pd.Series(["1", "1", "2", "3", "3", "4", "5", "5", "Bogotá"], index=tipos.index)
    out = comparar_ordinal(tipos, ref, ["1", "2", "3", "4", "5"], ["Bogotá"], ["A", "B", "C"])
    assert out["n_ordinal"] == 8 and "Bogotá" in out["tabla"].columns
    assert out["kendall_tau_b"] > 0.8 and out["gamma"] > 0.8
    assert "ari" not in out


def test_particion_y_validacion_externa():
    tipos = pd.Series(list("AABB"), index=list("abcd"))
    assert comparar_particion(tipos, pd.Series(list("xxyy"), index=list("abcd")))["ari"] == 1.0
    v = validacion_externa(tipos, pd.Series([1.0, 1.1, 5.0, 5.2], index=list("abcd")))
    assert v["eta2"] > 0.9


def test_perfiles_ignoran_imputados_y_representativos():
    X, verdad = generar_plantados(n=200, k=2)
    tipos = pd.Series(np.where(verdad == 0, "A", "B"), index=X.index)
    imp = pd.DataFrame(False, index=X.index, columns=X.columns)
    Zm = X.copy()
    Zm.iloc[0, 0] = 1e6
    imp.iloc[0, 0] = True
    pz = perfiles_z(Zm, imp, tipos)
    assert abs(pz.loc["A", "v0"]) < 100
    rep = representativos(X, tipos, pd.Series(1.0, index=X.index), pd.Series(X.index, index=X.index), 3)
    assert rep.groupby("tipo")["es_medoide"].sum().tolist() == [1, 1]
    sim = similares(X, 5)
    assert len(sim) == len(X) * 5 and not (sim["cod"] == sim["cod_similar"]).any()
    t = tamanos(tipos, pd.Series(100.0, index=X.index))
    assert t["pct_poblacion"].sum() == pytest.approx(100.0)
    d = descriptores(pz, {c: c for c in X.columns})
    assert set(d.index) == {"A", "B"}


def test_arbol_sustituto_recupera_particion_plantada():
    X, verdad = generar_plantados(n=300, k=3)
    tipos = pd.Series(np.array(list("ABC"))[verdad], index=X.index)
    out = arbol_sustituto(X, tipos, 3, 3, 0, {})
    assert out["exactitud_cv_media"] > 0.95


@pytest.fixture
def grilla():
    """Grilla de 6x6 celdas con códigos; las 18 de la izquierda son tipo A y las de la derecha B."""
    filas = []
    for i in range(6):
        for j in range(6):
            x0, y0 = round(-75 + j * 0.1, 6), round(5 + i * 0.1, 6)
            x1, y1 = round(-75 + (j + 1) * 0.1, 6), round(5 + (i + 1) * 0.1, 6)
            filas.append({"cod": f"{i * 6 + j + 1:05d}", "geometry": box(x0, y0, x1, y1),
                          "tipo": "A" if j < 3 else "B"})
    return gpd.GeoDataFrame(filas, crs=4326)


def test_vecindad_y_autocorrelacion_de_tipos_contiguos(grilla):
    cods = pd.Index(grilla["cod"])
    ady = vecindad_reina(grilla, cods)
    w = pesos_desde_lista(ady, cods)
    assert w.n == 36 and min(w.cardinalities.values()) == 3 and w.n_components == 1
    tipos = grilla.set_index("cod")["tipo"]
    mt = moran_por_tipo(w, tipos, 99, 0)
    assert (mt["moran_I"] > 0.5).all() and (mt["moran_p"] < 0.05).all()
    _, res = vecinos_mismo_tipo(w, tipos)
    assert res["observado"] > res["esperado_al_azar"]
    S = pd.DataFrame({"x": np.where(tipos == "A", 0.0, 1.0) + np.random.default_rng(0).normal(scale=0.01, size=36)}, index=cods)
    r = ward_restringido(S, w, 2)
    assert pd.crosstab(r, tipos).max(axis=1).sum() == 36

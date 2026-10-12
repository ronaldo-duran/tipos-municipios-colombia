import numpy as np
import pandas as pd
import pytest

from tipologia.config import Indicador
from tipologia.features.build import construir, regla_exclusion
from tipologia.features.impute import imputar, recortar_a_rango_observado
from tipologia.features.transform import (
    ajustar_escalado,
    ponderar,
    transformar,
    varianza_por_dominio,
    winsorizar,
)


def _ind(id_, **kw):
    base = dict(id=id_, nombre=id_, dominio="d", anio="ref", uso=("entrada",))
    base.update(kw)
    return Indicador(**base)


# --- transformaciones ---------------------------------------------------------

def test_transformar_log_y_log1p():
    df = pd.DataFrame({"a": [1.0, np.e, 0.0], "b": [0.0, np.e - 1, -3.0]})
    out = transformar(df, [_ind("a", transformacion="log"), _ind("b", transformacion="log1p")])
    assert out["a"].iloc[1] == pytest.approx(1.0)
    assert np.isnan(out["a"].iloc[2])                 # log de 0 no existe: queda faltante
    assert out["b"].tolist() == pytest.approx([0.0, 1.0, 0.0])  # negativos se recortan a 0 antes de log1p


def test_winsorizar_recorta_a_percentiles_y_cuenta():
    df = pd.DataFrame({"x": list(range(100)) + [10_000.0]})
    out, info = winsorizar(df, 0.01, 0.99)
    assert out["x"].max() < 10_000 and info.recortados["x"] >= 1
    assert out["x"].min() == pytest.approx(df["x"].quantile(0.01))


def test_escalado_estandar_y_robusto():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(5, 2, 500), "y": rng.lognormal(0, 1, 500)})
    z = ajustar_escalado(df, "estandar").aplicar(df)
    assert z.mean().abs().max() < 1e-9 and np.allclose(z.std(ddof=0), 1)
    r = ajustar_escalado(df, "robusto").aplicar(df)
    assert r.median().abs().max() < 1e-9


# --- ponderación por dominio ---------------------------------------------------

@pytest.fixture
def z_dominios():
    rng = np.random.default_rng(1)
    Z = pd.DataFrame(rng.normal(size=(300, 7)), columns=[f"v{i}" for i in range(7)])
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    dominios = {"v0": "A", "v1": "A", "v2": "A", "v3": "A", "v4": "A", "v5": "B", "v6": "C"}
    return Z, dominios


def test_ponderacion_por_dominio_iguala_la_varianza(z_dominios):
    Z, dom = z_dominios
    X, pond = ponderar(Z, dom, "dominio")
    v = varianza_por_dominio(X, pond.dominio_columna)
    assert np.allclose(v.to_numpy(), 1.0)
    assert pond.pesos["v0"] == pytest.approx(1 / np.sqrt(5))


def test_ponderacion_por_indicador_favorece_dominios_grandes(z_dominios):
    Z, dom = z_dominios
    X, pond = ponderar(Z, dom, "indicador")
    v = varianza_por_dominio(X, pond.dominio_columna)
    assert v["A"] == pytest.approx(5.0) and v["B"] == pytest.approx(1.0)


def test_pca_por_bloque_iguala_la_varianza(z_dominios):
    Z, dom = z_dominios
    X, pond = ponderar(Z, dom, "pca_bloque", varianza_bloque=0.8)
    v = varianza_por_dominio(X, pond.dominio_columna)
    assert np.allclose(v.to_numpy(), 1.0)
    assert pond.detalle["A"]["componentes"] <= 5


# --- imputación ----------------------------------------------------------------

@pytest.fixture
def con_faltantes():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": 2 * x + rng.normal(scale=0.1, size=200), "c": rng.normal(size=200)},
                      index=[f"{i:05d}" for i in range(1, 201)])
    df.iloc[:20, 1] = np.nan
    dep = pd.Series(["A"] * 100 + ["B"] * 100, index=df.index)
    return df, dep


@pytest.mark.parametrize("metodo", ["iterativa", "knn", "mediana_departamento"])
def test_imputacion_marca_celdas_y_no_toca_observados(con_faltantes, metodo):
    df, dep = con_faltantes
    out, mascara = imputar(df, metodo, grupos=dep, semilla=0)
    assert not out.isna().any().any()
    assert mascara["b"].sum() == 20 and mascara[["a", "c"]].sum().sum() == 0
    pd.testing.assert_frame_equal(out[~mascara], df[~mascara])


def test_imputacion_iterativa_aprovecha_correlaciones(con_faltantes):
    df, dep = con_faltantes
    verdad = 2 * df["a"].iloc[:20]
    it, _ = imputar(df, "iterativa", semilla=0)
    med, _ = imputar(df, "mediana_departamento", grupos=dep)
    err_it = (it["b"].iloc[:20] - verdad).abs().mean()
    err_med = (med["b"].iloc[:20] - verdad).abs().mean()
    assert err_it < err_med


def test_mediana_por_departamento_usa_el_grupo():
    df = pd.DataFrame({"x": [1.0, 3.0, np.nan, 10.0, 30.0, np.nan]}, index=list("abcdef"))
    dep = pd.Series(["A", "A", "A", "B", "B", "B"], index=df.index)
    out, _ = imputar(df, "mediana_departamento", grupos=dep)
    assert out.loc["c", "x"] == 2.0 and out.loc["f", "x"] == 20.0


def test_casos_completos_elimina_filas(con_faltantes):
    df, _ = con_faltantes
    out, _ = imputar(df, "casos_completos")
    assert len(out) == 180


def test_recorte_al_rango_observado():
    orig = pd.DataFrame({"x": [0.0, 1.0, np.nan]})
    imp = pd.DataFrame({"x": [0.0, 1.0, 5.0]})
    assert recortar_a_rango_observado(imp, orig)["x"].max() == 1.0


# --- construcción de la matriz -------------------------------------------------

@pytest.fixture
def largo():
    filas = []
    for cod, pob in [("05001", 1000.0), ("05002", 500.0), ("05004", 200.0)]:
        for anio in (2021, 2022, 2023):
            filas.append((cod, "POB", anio, pob))
    filas += [
        ("05001", "X", 2023, 10.0), ("05002", "X", 2022, 7.0),            # 05002 sin dato 2023 -> vecino 2022
        ("05001", "N", 2022, 50.0), ("05002", "N", 2022, 25.0), ("05004", "N", 2021, 4.0),
        ("05001", "C", 2018, 1.0), ("05002", "C", 2018, 2.0), ("05004", "C", 2018, 3.0),
    ]
    return pd.DataFrame(filas, columns=["cod_ent", "cod_ind", "anio", "valor"])


def test_reglas_de_anio_razon_y_vecino(largo):
    inds = [
        _ind("x", codigo="X", vecino=1),
        _ind("tasa", formula="razon", numerador=("N",), denominador="poblacion", factor=100, anio="rezago:1"),
        _ind("censo", codigo="C", anio="fijo:2018"),
    ]
    universo = pd.Index(["05001", "05002", "05004"])
    val, anios, vecino, falta = construir(largo, inds, 2023, universo, "POB", "AREA")
    assert val.loc["05002", "x"] == 7.0 and vecino.loc["05002", "x"] and falta.loc["05002", "x"]
    assert np.isnan(val.loc["05004", "x"]) and not vecino.loc["05004", "x"]
    # razón con población del mismo año del numerador (2022); 05004 tiene N solo en 2021 (fuera del rezago)
    assert val.loc["05001", "tasa"] == pytest.approx(5.0) and anios.loc["05001", "tasa"] == 2022
    assert np.isnan(val.loc["05004", "tasa"])
    assert val["censo"].tolist() == [1.0, 2.0, 3.0]


def test_regla_de_exclusion_publica_motivo():
    falta = pd.DataFrame([[False] * 10, [True] * 2 + [False] * 8, [True] * 3 + [False] * 7],
                         index=["05001", "05002", "05004"], columns=[f"i{k}" for k in range(10)])
    incl, excl = regla_exclusion(falta, 0.2)
    assert list(incl) == ["05001", "05002"]
    assert excl["cod"].tolist() == ["05004"] and "faltan 3 de 10" in excl["motivo"].iloc[0]

import pandas as pd
import pytest

from tipologia.data.divipola import (
    duplicados,
    es_codigo_valido,
    es_municipio,
    normalizar_codigo,
    normalizar_serie,
)


@pytest.mark.parametrize(
    "entrada, esperado",
    [
        (5001, "05001"),
        ("5001", "05001"),
        ("05001", "05001"),
        (5001.0, "05001"),
        ("5001.0", "05001"),
        ("CO05001", "05001"),
        (" 11001 ", "11001"),
        ("99773", "99773"),
    ],
)
def test_normaliza_ceros_a_la_izquierda(entrada, esperado):
    assert normalizar_codigo(entrada) == esperado


@pytest.mark.parametrize("entrada", [None, float("nan"), "", "abc", "123456", "00123", "5O01"])
def test_rechaza_codigos_invalidos(entrada):
    assert normalizar_codigo(entrada) is None


def test_municipio_vs_departamento_y_nacion():
    assert es_municipio("05001")
    assert not es_municipio("05000")
    assert not es_municipio("01001")
    assert es_codigo_valido("05000")


def test_duplicados_detecta_codigos_repetidos():
    s = normalizar_serie(pd.Series([5001, "05001", "05002", "CO05003"]))
    assert duplicados(s) == ["05001"]

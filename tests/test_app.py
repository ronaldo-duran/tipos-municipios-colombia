"""Humo de la app: cada página carga sin errores con los artefactos versionados en outputs/."""

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = ["app/Inicio.py", *sorted(str(p.relative_to(RAIZ)) for p in (RAIZ / "app" / "pages").glob("*.py"))]

pytestmark = pytest.mark.skipif(not (RAIZ / "outputs" / "LATEST").exists(), reason="sin artefactos en outputs/")


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_carga_sin_errores(pagina):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / pagina), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    textos = " ".join(i.value for i in at.info)
    assert "No implica causalidad ni jerarquía de calidad" in textos

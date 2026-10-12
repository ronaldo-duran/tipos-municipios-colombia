"""Datos sintéticos compartidos por los tests."""

import numpy as np
import pandas as pd
import pytest


def generar_plantados(n: int = 600, k: int = 4, d: int = 8, separacion: float = 6.0, semilla: int = 7):
    """k grupos gaussianos bien separados en d dimensiones. Devuelve (DataFrame, etiquetas verdaderas)."""
    rng = np.random.default_rng(semilla)
    centros = rng.normal(scale=separacion, size=(k, d))
    etiquetas = np.repeat(np.arange(k), n // k)
    X = centros[etiquetas] + rng.normal(size=(len(etiquetas), d))
    idx = [f"{i:05d}" for i in range(1, len(etiquetas) + 1)]
    return pd.DataFrame(X, index=idx, columns=[f"v{j}" for j in range(d)]), etiquetas


def generar_ruido(n: int = 600, d: int = 8, semilla: int = 7) -> pd.DataFrame:
    """Ruido uniforme sin estructura de grupos."""
    rng = np.random.default_rng(semilla)
    idx = [f"{i:05d}" for i in range(1, n + 1)]
    return pd.DataFrame(rng.uniform(size=(n, d)), index=idx, columns=[f"v{j}" for j in range(d)])


@pytest.fixture
def plantados():
    return generar_plantados()


@pytest.fixture
def ruido():
    return generar_ruido()

"""Normalización y validación de la llave DIVIPOLA (5 dígitos)."""

from __future__ import annotations

import re

import pandas as pd

_PATRON = re.compile(r"^\d{5}$")


def normalizar_codigo(valor: object) -> str | None:
    """Convierte un código municipal a texto de 5 dígitos con ceros a la izquierda.

    Acepta enteros (5001), textos con o sin cero ("05001", "5001"), flotantes de Excel (5001.0)
    y el prefijo "CO" de los P-codes de HDX. Devuelve None si no es un código válido.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip().upper()
    if texto.startswith("CO"):
        texto = texto[2:]
    if re.fullmatch(r"\d+\.0+", texto):
        texto = texto.split(".")[0]
    if not texto.isdigit() or len(texto) > 5:
        return None
    texto = texto.zfill(5)
    return texto if es_codigo_valido(texto) else None


def es_codigo_valido(codigo: str) -> bool:
    """True si el código tiene 5 dígitos y un departamento plausible (01-99)."""
    return bool(_PATRON.match(codigo or "")) and codigo[:2] != "00"


def es_municipio(codigo: str) -> bool:
    """Los códigos XX000 son departamentos y 01001 es la nación en TerriData."""
    return es_codigo_valido(codigo) and not codigo.endswith("000") and codigo != "01001"


def normalizar_serie(serie: pd.Series) -> pd.Series:
    """Aplica normalizar_codigo a una serie completa."""
    return serie.map(normalizar_codigo)


def duplicados(serie: pd.Series) -> list[str]:
    """Códigos repetidos en una serie."""
    return sorted(serie[serie.duplicated(keep=False)].dropna().unique().tolist())

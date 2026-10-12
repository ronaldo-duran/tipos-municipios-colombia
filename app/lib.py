"""Utilidades compartidas por las páginas de la app: carga cacheada de artefactos ya calculados."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
OUTPUTS = RAIZ / "outputs"
GEO = RAIZ / "app" / "data" / "municipios.geojson"

AVISO = "Descripción estadística de territorios a partir de datos oficiales. No implica causalidad ni jerarquía de calidad."
SIN_DATO = "Sin dato"
COLORES = ["#2a78b8", "#e08a2c", "#3f9b62", "#b8519a", "#7a6bb8", "#b84a3c", "#5f9ea0", "#9c8a35"]
GRIS = "#c9c9c9"


@st.cache_data(show_spinner=False)
def ejecucion() -> str:
    """Nombre de la ejecución vigente (outputs/LATEST)."""
    return (OUTPUTS / "LATEST").read_text(encoding="utf-8").strip()


def ruta(nombre: str) -> Path:
    return OUTPUTS / ejecucion() / nombre


@st.cache_data(show_spinner=False)
def csv(nombre: str, **kw) -> pd.DataFrame:
    return pd.read_csv(ruta(nombre), **kw)


@st.cache_data(show_spinner=False)
def parquet(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(ruta(nombre))


@st.cache_data(show_spinner=False)
def js(nombre: str) -> dict:
    return json.loads(ruta(nombre).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def texto(nombre: str) -> str:
    return ruta(nombre).read_text(encoding="utf-8")


def existe(nombre: str) -> bool:
    return ruta(nombre).exists()


@st.cache_resource(show_spinner=False)
def geojson() -> dict:
    return json.loads(GEO.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def asignaciones() -> pd.DataFrame:
    return pd.read_csv(ruta("asignaciones.csv"), dtype={"cod": str}).set_index("cod")


@st.cache_data(show_spinner=False)
def indicadores() -> pd.DataFrame:
    return pd.read_csv(ruta("indicadores.csv")).set_index("id")


@st.cache_data(show_spinner=False)
def nombres_tipos() -> pd.DataFrame:
    return pd.read_csv(ruta("nombres_tipos.csv")).set_index("tipo")


def colores_tipos() -> dict[str, str]:
    tipos = sorted(nombres_tipos().index)
    mapa = {t: COLORES[i % len(COLORES)] for i, t in enumerate(tipos)}
    mapa[SIN_DATO] = GRIS
    return mapa


def etiqueta(t: str) -> str:
    """'A · nombre del tipo'."""
    n = nombres_tipos()
    return f"{t} · {n.loc[t, 'nombre']}" if t in n.index else t


def meta() -> dict:
    return js("metadata.json")


def encabezado(titulo: str) -> None:
    """Título, aviso obligatorio y año de referencia en todas las páginas."""
    st.set_page_config(page_title=f"{titulo} · Tipos de municipio", page_icon="🗺️", layout="wide")
    m = meta()
    st.title(titulo)
    st.info(f"**{AVISO}**  \nAño de referencia: **{m['anio_referencia']}** · Datos TerriData (DNP) versión "
            f"**{m['fuentes']['terridata']['version']}** · Pobreza, NBI, vivienda y servicios básicos provienen del "
            f"**censo 2018**.", icon="ℹ️")


def pie() -> None:
    m = meta()
    st.divider()
    st.caption(f"Ejecución `{m['ejecucion']}` · código {m['codigo']['version_paquete']} "
               f"({(m['codigo']['commit_git'] or '')[:7]}) · semilla {m['semilla']}. Fuentes: TerriData (DNP), "
               "DIVIPOLA (DANE, datos.gov.co), Tipologías municipales 2026 (DNP) y límites COD-AB de HDX "
               "(DANE, CC BY-IGO). Una tipología describe territorios, no a las personas que viven en ellos.")


def descarga_asignaciones(clave: str = "descarga") -> None:
    datos = (OUTPUTS / ejecucion() / "asignaciones.csv").read_bytes()
    st.download_button("Descargar asignaciones (CSV)", datos, file_name=f"tipologia_municipios_{ejecucion()}.csv",
                       mime="text/csv", key=clave)


def mapa(df, color: str, *, discreto: dict[str, str] | None = None, orden: list[str] | None = None,
         continuo: str | None = None, rango: tuple[float, float] | None = None, hover: dict | None = None,
         hover_name: str | None = None, alto: int = 650, etiquetas: dict | None = None):
    """Coropleta en SVG (sin teselas ni WebGL), ajustada a los límites de Colombia."""
    import plotly.express as px

    kw = dict(geojson=geojson(), locations="cod", featureidkey="properties.cod", color=color,
              hover_name=hover_name, hover_data=hover, labels=etiquetas or {})
    if discreto is not None:
        kw.update(color_discrete_map=discreto, category_orders={color: orden} if orden else None)
    if continuo is not None:
        kw.update(color_continuous_scale=continuo, range_color=rango)
    fig = px.choropleth(df.reset_index() if "cod" not in df.columns else df, **{k: v for k, v in kw.items() if v is not None})
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_traces(marker_line_width=0.15, marker_line_color="white")
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=0, b=0), legend=dict(title="", orientation="h", y=-0.02),
                      paper_bgcolor="rgba(0,0,0,0)", geo=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def formato_poblacion(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f} M".replace(".", ",")
    return f"{int(x):,}".replace(",", ".")

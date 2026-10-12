"""Etapa validate: chequeos de llaves, duplicados, cobertura, rangos y unión con geometrías."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from tipologia.config import Config
from tipologia.data.divipola import duplicados, es_codigo_valido, es_municipio

log = logging.getLogger(__name__)

# Rangos plausibles por indicador (en unidades originales, antes de transformar).
# Las coberturas que usan proyecciones de población pueden superar 100 %.
RANGOS_POR_DEFECTO: dict[str, tuple[float, float]] = {"%": (0.0, 100.0)}
RANGOS_ESPECIFICOS: dict[str, tuple[float, float]] = {
    "pob_log": (500, 9_000_000),
    "dens_log": (0.01, 50_000),
    "cob_media": (0, 160),
    "cob_sup": (0, 400),
    "transito_sup": (0, 100),
    "cob_subsid": (0, 250),
    "vac_penta": (0, 250),
    "contrib_pc": (0, 150),
    "var_interc": (-90, 500),
    "internet": (0, 100),
    "fec_adol": (0, 400),
    "ind_envej": (0, 1000),
}


@dataclass
class Chequeo:
    nombre: str
    estado: str                       # ok | aviso | error
    detalle: str = ""
    datos: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reporte:
    chequeos: list[Chequeo] = field(default_factory=list)

    def agregar(self, nombre: str, ok: bool, detalle: str = "", nivel: str = "error", **datos: Any) -> None:
        estado = "ok" if ok else nivel
        self.chequeos.append(Chequeo(nombre, estado, detalle, datos))
        getattr(log, "info" if ok else ("warning" if nivel == "aviso" else "error"))("[%s] %s %s", estado, nombre, detalle)

    @property
    def errores(self) -> list[Chequeo]:
        return [c for c in self.chequeos if c.estado == "error"]

    def a_dict(self) -> dict:
        return {"chequeos": [asdict(c) for c in self.chequeos],
                "n_errores": len(self.errores),
                "n_avisos": sum(c.estado == "aviso" for c in self.chequeos)}

    def guardar(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(self.a_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def validar_llaves(codigos: pd.Series, nombre: str, reporte: Reporte) -> None:
    """Formato de 5 dígitos y unicidad."""
    malos = sorted(c for c in codigos.dropna().unique() if not es_codigo_valido(c))
    reporte.agregar(f"{nombre}: formato DIVIPOLA", not malos and codigos.notna().all(),
                    f"{len(malos)} códigos inválidos, {int(codigos.isna().sum())} vacíos", invalidos=malos[:50])
    dups = duplicados(codigos)
    reporte.agregar(f"{nombre}: códigos únicos", not dups, f"{len(dups)} repetidos", repetidos=dups[:50])


def validar_terridata(parquet: Path, divipola: pd.DataFrame, reporte: Reporte) -> set[str]:
    """Chequeos sobre el parquet largo de TerriData. Devuelve el conjunto de municipios."""
    con = duckdb.connect()
    p = parquet.as_posix()
    ents = con.execute(f"SELECT DISTINCT cod_ent, entidad FROM '{p}'").df()
    validar_llaves(ents["cod_ent"].drop_duplicates(), "TerriData entidades", reporte)
    multi = ents.groupby("cod_ent").entidad.nunique()
    reporte.agregar("TerriData: un nombre por código", bool((multi == 1).all()),
                    f"{int((multi > 1).sum())} códigos con varios nombres")
    filas, distintas = con.execute(f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT * FROM '{p}')) FROM '{p}'").fetchone()
    reporte.agregar("TerriData: sin filas repetidas", filas == distintas, f"{filas - distintas} filas repetidas")
    sin_parsear = con.execute(f"SELECT count(*) FROM '{p}' WHERE dato_raw IS NOT NULL AND valor IS NULL").fetchone()[0]
    reporte.agregar("TerriData: valores numéricos interpretables", sin_parsear == 0, f"{sin_parsear} sin interpretar")
    munis = {c for c in ents["cod_ent"] if es_municipio(c)}
    oficiales = divipola.loc[divipola["tipo_municipio"] == "Municipio", "cod_mpio"]
    faltan = sorted(set(oficiales) - munis)
    sobran = sorted(munis - set(divipola["cod_mpio"]))
    reporte.agregar("TerriData vs DIVIPOLA: municipios faltantes", not faltan,
                    f"{len(faltan)} municipios oficiales sin datos: {faltan}", nivel="aviso", faltantes=faltan)
    reporte.agregar("TerriData vs DIVIPOLA: códigos no oficiales", not sobran, f"{len(sobran)}: {sobran}")
    return munis


def validar_geometrias(ruta: Path, columna: str, municipios: set[str], nombre: str, reporte: Reporte) -> list[str]:
    """Unión de los municipios con datos contra una capa de geometrías. Devuelve los que no unen."""
    if ruta.suffix == ".zip":
        raise ValueError("Use leer_geometrias para capas comprimidas")
    g = gpd.read_file(ruta, columns=[columna]) if ruta.suffix != ".geojson" else gpd.read_file(ruta)
    cods = set(g[columna].astype(str))
    sin_geo = sorted(municipios - cods)
    reporte.agregar(f"{nombre}: municipios con geometría", not sin_geo,
                    f"{len(sin_geo)} sin geometría (se muestran 'sin dato'): {sin_geo}", nivel="aviso", sin_geometria=sin_geo)
    return sin_geo


def validar_rangos(wide: pd.DataFrame, unidades: dict[str, str], reporte: Reporte) -> dict[str, list[str]]:
    """Valores fuera de rangos plausibles (en unidades originales). No modifica los datos."""
    fuera: dict[str, list[str]] = {}
    for col in wide.columns:
        rango = RANGOS_ESPECIFICOS.get(col) or RANGOS_POR_DEFECTO.get(unidades.get(col, ""))
        if rango is None:
            continue
        v = wide[col]
        mask = v.notna() & ((v < rango[0]) | (v > rango[1]))
        if mask.any():
            fuera[col] = sorted(v.index[mask].tolist())
    n = sum(len(v) for v in fuera.values())
    reporte.agregar("Rangos plausibles por indicador", n == 0,
                    f"{n} valores fuera de rango en {len(fuera)} indicadores (se conservan y se reportan)",
                    nivel="aviso", fuera_de_rango={k: v[:30] for k, v in fuera.items()})
    return fuera


def validar_cobertura(wide: pd.DataFrame, minimo: float, reporte: Reporte) -> pd.Series:
    """Cobertura por indicador en el año de referencia."""
    cob = wide.notna().mean()
    bajos = cob[cob < minimo]
    reporte.agregar("Cobertura mínima por indicador", bajos.empty,
                    f"{len(bajos)} indicadores bajo {minimo:.0%}: {bajos.round(3).to_dict()}", nivel="aviso")
    return cob


def validar_matriz_final(X: pd.DataFrame, reporte: Reporte) -> None:
    """La matriz que entra al clustering no puede tener faltantes, infinitos ni duplicados."""
    reporte.agregar("Matriz final: sin faltantes", not X.isna().any().any(), f"{int(X.isna().sum().sum())} NaN")
    reporte.agregar("Matriz final: valores finitos", bool(np.isfinite(X.to_numpy()).all()))
    reporte.agregar("Matriz final: municipios únicos", X.index.is_unique)


def ejecutar(cfg: Config) -> Reporte:
    """Validación de las fuentes ingeridas."""
    from tipologia.data.ingest import leer_geometrias

    proc = cfg.ruta("processed")
    reporte = Reporte()
    divi = pd.read_parquet(proc / "divipola.parquet")
    validar_llaves(divi["cod_mpio"], "DIVIPOLA", reporte)
    munis = validar_terridata(proc / "terridata_long.parquet", divi, reporte)
    dnp = pd.read_parquet(proc / "dnp_tipologias.parquet")
    validar_llaves(dnp["cod"], "Tipología DNP", reporte)
    sin_dnp = sorted(munis - set(dnp["cod"]))
    reporte.agregar("Tipología DNP: cubre los municipios con datos", not sin_dnp, f"faltan {sin_dnp}", nivel="aviso")
    geo_app = cfg.ruta("app_geo")
    if geo_app.exists():
        validar_geometrias(geo_app, "cod", munis, "Geometrías app (HDX)", reporte)
    mgn_man = cfg.ruta("raw") / "mgn" / "manifest.json"
    if mgn_man.exists():
        m = json.loads(mgn_man.read_text(encoding="utf-8"))
        zip_path = cfg.ruta("raw") / "mgn" / m["version"] / m["archivo"]
        c = cfg["fuentes"]["mgn"]
        g = leer_geometrias(zip_path, c["capa"], c["columna_codigo"])
        sin = sorted(munis - set(g["cod"]))
        reporte.agregar("MGN 2025: municipios con geometría", not sin, f"{len(sin)} sin geometría: {sin}", nivel="aviso")
        reporte.agregar("MGN 2025: geometrías válidas", bool(g.geometry.is_valid.all()),
                        f"{int((~g.geometry.is_valid).sum())} inválidas", nivel="aviso")
    reporte.guardar(proc / "validacion_fuentes.json")
    return reporte

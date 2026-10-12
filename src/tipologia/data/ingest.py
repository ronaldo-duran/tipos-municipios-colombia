"""Etapa ingest: descarga (con caché) y normalización de todas las fuentes."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import shapely

from tipologia.config import Config
from tipologia.data.divipola import normalizar_serie
from tipologia.data.sources import HttpFileSource, SocrataSource, TerriDataPortalSource, sha256_archivo

log = logging.getLogger(__name__)

COLUMNAS_TERRIDATA = {
    "Código Departamento": "cod_dpto",
    "Departamento": "dpto",
    "Código Entidad": "cod_ent",
    "Entidad": "entidad",
    "Dimensión": "dimension",
    "Subcategoría": "subcat",
    "Indicador": "indicador",
    "Código Indicador": "cod_ind",
    "Dato Numérico": "dato_raw",
    "Dato Cualitativo": "dato_cual",
    "Año": "anio",
    "Mes": "mes",
    "Fuente": "fuente",
    "Unidad de Medida": "unidad",
}


def consulta_limpieza(origen: str) -> str:
    """SQL que renombra columnas, interpreta números (miles con punto, decimal con coma) y quita
    filas repetidas idénticas. `origen` es una expresión de tabla de DuckDB."""
    cols = ",\n  ".join(f'"{k}" AS {v}' for k, v in COLUMNAS_TERRIDATA.items())
    return f"""
SELECT DISTINCT
  lpad(cod_dpto, 2, '0') AS cod_dpto, dpto, lpad(cod_ent, 5, '0') AS cod_ent, entidad,
  dimension, subcat, indicador, cod_ind, dato_raw,
  TRY_CAST(replace(replace(dato_raw, '.', ''), ',', '.') AS DOUBLE) AS valor,
  NULLIF(trim(dato_cual), '') AS dato_cual,
  TRY_CAST(anio AS INTEGER) AS anio, TRY_CAST(mes AS INTEGER) AS mes, fuente, unidad
FROM (SELECT
  {cols}
FROM {origen})
"""


def leer_csv_sql(ruta_txt: Path) -> str:
    """Expresión read_csv para el archivo plano de TerriData."""
    p = ruta_txt.as_posix()
    return (f"read_csv('{p}', delim='|', header=true, all_varchar=true, quote='', escape='', "
            f"strict_mode=false, null_padding=true)")


def convertir_terridata(zip_path: Path, destino: Path, archivo_interno: str = "TerriData.txt") -> dict:
    """Convierte el zip de TerriData a parquet limpio y deduplicado. Devuelve un resumen."""
    tmp_dir = destino.parent / "_tmp_terridata"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    txt = tmp_dir / archivo_interno
    try:
        log.info("Extrayendo %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as z, z.open(archivo_interno) as src, open(txt, "wb") as dst:
            shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
        con = duckdb.connect()
        con.execute(f"SET temp_directory='{(tmp_dir / 'duck').as_posix()}'")
        origen = leer_csv_sql(txt)
        n_crudo = con.execute(f"SELECT count(*) FROM {origen}").fetchone()[0]
        log.info("Convirtiendo %d filas a parquet", n_crudo)
        con.execute(f"COPY ({consulta_limpieza(origen)}) TO '{destino.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        resumen = resumen_terridata(con, destino)
        resumen["filas_crudas"] = int(n_crudo)
        resumen["filas_duplicadas_eliminadas"] = int(n_crudo - resumen["filas"])
        return resumen
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def resumen_terridata(con: duckdb.DuckDBPyConnection, parquet: Path) -> dict:
    """Conteos básicos y conflictos (misma entidad, indicador y periodo con valores distintos)."""
    p = parquet.as_posix()
    filas, entidades, indicadores = con.execute(
        f"SELECT count(*), count(DISTINCT cod_ent), count(DISTINCT cod_ind) FROM '{p}'").fetchone()
    conflictos = con.execute(f"""
        SELECT count(*) FROM (SELECT cod_ent, cod_ind, anio, mes FROM '{p}'
        GROUP BY ALL HAVING count(DISTINCT coalesce(valor, -1e300)) > 1)""").fetchone()[0]
    sin_parsear = con.execute(
        f"SELECT count(*) FROM '{p}' WHERE dato_raw IS NOT NULL AND valor IS NULL").fetchone()[0]
    return {"filas": int(filas), "entidades": int(entidades), "indicadores": int(indicadores),
            "grupos_en_conflicto": int(conflictos), "valores_sin_parsear": int(sin_parsear)}


def ingest_terridata(cfg: Config, forzar: bool = False, offline: bool = False) -> dict:
    """Descarga (si hay versión nueva) y convierte TerriData. Idempotente por checksum."""
    fuente = TerriDataPortalSource(cfg["fuentes"]["terridata"]["url"], cfg.ruta("raw"))
    zip_path = fuente.obtener(offline=offline)
    man = fuente.manifiesto()
    destino = cfg.ruta("processed") / "terridata_long.parquet"
    meta_path = destino.with_suffix(".meta.json")
    if destino.exists() and meta_path.exists() and not forzar:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256_fuente") == man.sha256:
            log.info("TerriData ya convertido para esta versión (%s); no recomputo", man.version)
            return meta
    destino.parent.mkdir(parents=True, exist_ok=True)
    resumen = convertir_terridata(zip_path, destino, cfg["fuentes"]["terridata"]["archivo_interno"])
    meta = {"version_fuente": man.version, "sha256_fuente": man.sha256, "tamano_fuente": man.tamano,
            "url": man.url, "sha256_parquet": sha256_archivo(destino), **resumen}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("TerriData: %s", meta)
    return meta


def ingest_divipola(cfg: Config, forzar: bool = False, offline: bool = False) -> Path:
    """DIVIPOLA oficial desde la API de datos.gov.co."""
    c = cfg["fuentes"]["divipola"]
    fuente = SocrataSource("divipola", c["dominio"], c["dataset"], cfg.ruta("raw"))
    csv = fuente.obtener(forzar=forzar, offline=offline)
    df = pd.read_csv(csv, dtype=str)
    df["cod_mpio"] = normalizar_serie(df["cod_mpio"])
    df["cod_dpto"] = df["cod_dpto"].str.zfill(2)
    destino = cfg.ruta("processed") / "divipola.parquet"
    df.to_parquet(destino, index=False)
    return destino


def ingest_dnp(cfg: Config, forzar: bool = False, offline: bool = False) -> Path:
    """Tipologías del DNP (referencia externa; nunca entra al clustering)."""
    c = cfg["fuentes"]["dnp_tipologias"]
    fuente = HttpFileSource("dnp_tipologias", c["url"], cfg.ruta("raw"), archivo="ResultadosTipologias2026.xlsx")
    xlsx = fuente.obtener(forzar=forzar, offline=offline)
    df = pd.read_excel(xlsx, sheet_name=c["hoja"], header=1, dtype={"CodDANE_txt": str})
    df["cod"] = normalizar_serie(df["CodDANE_txt"])
    keep = {
        "cod": "cod", "Municipio": "municipio_dnp", "Tipología_2026": "tipologia_dnp_2026",
        "Tipología_2025": "tipologia_dnp_2025", "Cat_Ruralidad 2024": "cat_ruralidad",
        "Ciudad_capital": "capital", "PDET": "pdet", "Cat 617_2025": "cat_617",
    }
    out = df[list(keep)].rename(columns=keep)
    for col in ("tipologia_dnp_2026", "tipologia_dnp_2025", "cat_ruralidad", "cat_617"):
        out[col] = out[col].astype(str).str.strip().replace({"nan": None, "N.D.": None})
    destino = cfg.ruta("processed") / "dnp_tipologias.parquet"
    out.to_parquet(destino, index=False)
    return destino


def leer_geometrias(zip_path: Path, capa: str, columna: str) -> gpd.GeoDataFrame:
    """Lee una capa de un zip y normaliza la columna de código a DIVIPOLA de 5 dígitos."""
    g = gpd.read_file(f"zip://{zip_path.as_posix()}!{capa}")
    g["cod"] = normalizar_serie(g[columna])
    return g.to_crs(4326)


def ingest_geo_app(cfg: Config, forzar: bool = False, offline: bool = False) -> Path:
    """Límites municipales simplificados para la app (HDX COD-AB, CC BY-IGO)."""
    c = cfg["fuentes"]["geo_app"]
    fuente = HttpFileSource("geo_app", c["url"], cfg.ruta("raw"))
    destino = cfg.ruta("app_geo")
    meta_path = destino.with_suffix(".meta.json")
    try:
        zip_path = fuente.obtener(forzar=forzar, offline=offline)
    except (FileNotFoundError, OSError) as e:
        if destino.exists():
            log.warning("Sin fuente de geometrías (%s); conservo %s", e, destino)
            return destino
        raise
    man = fuente.manifiesto()
    if destino.exists() and meta_path.exists() and not forzar:
        if json.loads(meta_path.read_text(encoding="utf-8")).get("sha256_fuente") == man.sha256:
            return destino
    g = leer_geometrias(zip_path, c["capa"], c["columna_codigo"])
    g = g[["cod", "ADM2_ES", "ADM1_ES", "geometry"]].rename(columns={"ADM2_ES": "nombre", "ADM1_ES": "dpto"})
    g["geometry"] = shapely.set_precision(
        g.geometry.simplify(c["tolerancia_simplificacion"], preserve_topology=True).values, 1e-4)
    g = g[~g.geometry.is_empty].sort_values("cod")
    destino.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(destino, driver="GeoJSON", layer_options={"COORDINATE_PRECISION": 4, "RFC7946": "YES"})
    meta = {"fuente": "HDX COD-AB Colombia (DANE, MGN 2018)", "licencia": c["licencia"], "url": man.url,
            "version_fuente": man.version, "sha256_fuente": man.sha256, "poligonos": int(len(g)),
            "tolerancia_simplificacion_grados": c["tolerancia_simplificacion"]}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Geometrías de la app: %d polígonos, %.1f MB", len(g), destino.stat().st_size / 1e6)
    return destino


def ingest_mgn(cfg: Config, forzar: bool = False, offline: bool = False) -> Path | None:
    """MGN 2025 del DANE, solo en local para la vecindad espacial (no se redistribuye)."""
    c = cfg["fuentes"]["mgn"]
    fuente = HttpFileSource("mgn", c["url"], cfg.ruta("raw"))
    try:
        return fuente.obtener(forzar=forzar, offline=offline)
    except (FileNotFoundError, OSError) as e:
        log.warning("MGN no disponible (%s); la vecindad espacial usará el archivo de vecinos versionado", e)
        return None


def manifiestos(cfg: Config) -> dict[str, dict]:
    """Manifiestos de todas las fuentes en caché (para metadata.json)."""
    out = {}
    for nombre in ("terridata", "divipola", "dnp_tipologias", "geo_app", "mgn"):
        p = cfg.ruta("raw") / nombre / "manifest.json"
        if p.exists():
            out[nombre] = json.loads(p.read_text(encoding="utf-8"))
    return out


def ejecutar(cfg: Config, forzar: bool = False, offline: bool = False) -> dict:
    """Corre la ingesta completa."""
    resumen = {"terridata": ingest_terridata(cfg, forzar, offline)}
    resumen["divipola"] = str(ingest_divipola(cfg, forzar, offline))
    resumen["dnp"] = str(ingest_dnp(cfg, forzar, offline))
    resumen["geo_app"] = str(ingest_geo_app(cfg, forzar, offline))
    mgn = ingest_mgn(cfg, forzar, offline)
    resumen["mgn"] = str(mgn) if mgn else None
    return resumen

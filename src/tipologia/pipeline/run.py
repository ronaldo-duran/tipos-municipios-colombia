"""Etapas del pipeline: features -> cluster -> stability -> profile -> temporal -> export."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from tipologia.config import Config
from tipologia.data.validate import Reporte, validar_cobertura, validar_matriz_final, validar_rangos
from tipologia.features.build import patron_faltantes
from tipologia.features.impute import resumen_imputacion
from tipologia.features.prepare import cargar_datos_base, preparar
from tipologia.features.transform import varianza_por_dominio

log = logging.getLogger(__name__)


def dir_trabajo(cfg: Config) -> Path:
    d = cfg.ruta("processed") / "trabajo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def guardar_json(obj, ruta: Path) -> None:
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_a_json), encoding="utf-8")


def _a_json(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def etapa_features(cfg: Config, destino: Path) -> dict:
    """Matriz de indicadores, exclusiones, patrón de faltantes, imputación y validaciones."""
    datos = cargar_datos_base(cfg)
    P = preparar(cfg, datos)
    rep = Reporte()
    unidades = {i.id: i.unidad for i in P.matriz.indicadores}
    validar_cobertura(P.matriz.valores, cfg["datos"]["cobertura_min_indicador"], rep)
    validar_rangos(P.matriz.valores, unidades, rep)
    validar_matriz_final(P.X, rep)
    rep.guardar(destino / "validacion_features.json")

    P.valores.to_parquet(destino / "valores.parquet")
    P.Z.to_parquet(destino / "z.parquet")
    P.X.to_parquet(destino / "x.parquet")
    P.imputado.to_parquet(destino / "imputado.parquet")
    P.matriz.anios.to_parquet(destino / "anios_dato.parquet")
    P.municipios.to_parquet(destino / "municipios.parquet")
    P.matriz.excluidos.to_csv(destino / "excluidos.csv", index=False)

    pob = datos.poblacion(cfg, P.anio)
    patron = patron_faltantes(P.matriz.faltante_ref, P.municipios, pob)
    faltantes = {k: {"tabla": v.reset_index().to_dict(orient="records"), **v.attrs} for k, v in patron.items()}
    guardar_json(faltantes, destino / "faltantes.json")

    ind = pd.DataFrame([{
        "id": i.id, "nombre": i.nombre, "dominio": i.dominio, "dominio_nombre": cfg.dominios[i.dominio],
        "codigo_terridata": i.codigo or "+".join(i.numerador) + (f"/{i.denominador}" if i.denominador else ""),
        "regla_anio": i.anio, "anios_usados": ",".join(sorted({str(int(a)) for a in P.matriz.anios[i.id].dropna()})),
        "transformacion": i.transformacion or "", "unidad": i.unidad,
        "cobertura_ref": round(float(P.matriz.cobertura[i.id]), 4),
        "imputados": int(P.imputado[i.id].sum()),
        "imputados_anio_vecino": int(P.matriz.imputado_vecino[i.id].sum()),
        "winsor_recortados": P.winsor.recortados.get(i.id, 0),
        "peso": round(float(P.ponderacion.pesos.get(i.id, 1.0)), 4),
    } for i in P.matriz.indicadores])
    ind.to_csv(destino / "indicadores.csv", index=False)
    resumen = {
        "anio_referencia": P.anio,
        "municipios_incluidos": int(len(P.Z)),
        "municipios_excluidos": int(len(P.matriz.excluidos)),
        "indicadores": int(P.Z.shape[1]),
        "indicadores_descartados": P.matriz.indicadores_descartados,
        "celdas_imputadas": int(P.imputado.sum().sum()),
        "municipios_con_imputacion": int((P.n_imputados > 0).sum()),
        "casos_completos_ref": int((~P.matriz.faltante_ref.any(axis=1)).sum()),
        "varianza_por_dominio": varianza_por_dominio(P.X, P.ponderacion.dominio_columna).round(4).to_dict(),
        "imputacion": resumen_imputacion(P.imputado).query("imputados > 0").to_dict(orient="index"),
        "parametros": P.parametros,
    }
    guardar_json(resumen, destino / "features_resumen.json")
    log.info("features: %s", {k: resumen[k] for k in ("municipios_incluidos", "municipios_excluidos", "indicadores")})
    return resumen


def ejecutar_etapa(etapa: str, cfg: Config, forzar: bool = False, offline: bool = False) -> int:
    destino = dir_trabajo(cfg)
    if etapa == "features":
        etapa_features(cfg, destino)
        return 0
    raise NotImplementedError(f"Etapa {etapa} todavía no implementada")

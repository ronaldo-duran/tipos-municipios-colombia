"""Etapas del pipeline: features -> cluster -> stability -> profile -> temporal -> export."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from tipologia.clustering.models import acuerdo, ajustar, barrido_hdbscan, gap_statistic, seleccion_k
from tipologia.clustering.reduce import entrada_clustering, umap_2d
from tipologia.clustering.seleccion import decidir_k
from tipologia.config import Config
from tipologia.data.validate import Reporte, validar_cobertura, validar_matriz_final, validar_rangos
from tipologia.features.build import patron_faltantes
from tipologia.features.impute import resumen_imputacion
from tipologia.features.prepare import cargar_datos_base, preparar
from tipologia.features.transform import varianza_por_dominio
from tipologia.pipeline.artefactos import ETAPAS_EJECUCION, dir_ejecucion, etapas_completadas, marcar_etapa
from tipologia.pipeline.io import guardar_json

log = logging.getLogger(__name__)


def dir_trabajo(cfg: Config) -> Path:
    """Directorio de la ejecución actual: outputs/<fecha de TerriData>_<hash datos+config+código>."""
    return dir_ejecucion(cfg)


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
    P.ponderacion.pesos.rename("peso").to_csv(destino / "pesos_ponderacion.csv")
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


def etapa_cluster(cfg: Config, destino: Path) -> dict:
    """PCA, UMAP (visualización), métricas internas por algoritmo y k, gap, BIC y HDBSCAN."""
    c = cfg["clustering"]
    X = pd.read_parquet(destino / "x.parquet")
    S, pca = entrada_clustering(X, c["entrada"], c["pca_varianza"], cfg.semilla)
    S.to_parquet(destino / "entrada.parquet")
    if pca is not None:
        pca.varianza.to_csv(destino / "pca_varianza.csv", index=False)
        pca.cargas.to_csv(destino / "pca_cargas.csv")
        pca.scores_todos(X, 5).to_parquet(destino / "pca_scores.parquet")
    ks = list(range(c["k_min"], c["k_max"] + 1))
    sel = seleccion_k(S, ks, cfg.semilla, c)
    sel.to_csv(destino / "seleccion_k.csv", index=False)
    gap = gap_statistic(S.to_numpy(), list(range(1, c["k_max"] + 2)), c["gap_referencias"], cfg.semilla)
    gap.to_csv(destino / "gap.csv", index=False)
    hd, hd_etiq = barrido_hdbscan(S, c["hdbscan_min_cluster_size"], c["hdbscan_min_samples"])
    hd.to_csv(destino / "hdbscan.csv", index=False)
    dec = decidir_k(sel, None, modelo_final=c["modelo_final"], n=len(S))
    dec["tabla"].to_csv(destino / "decision_k_interna.csv", index=False)
    emb = umap_2d(S, cfg["umap"]["n_vecinos"], cfg["umap"]["min_dist"], cfg.semilla)
    emb.to_parquet(destino / "umap.parquet")
    k = dec["k"]
    etiq = {a: ajustar(a, S.to_numpy(), k, cfg.semilla, c["kmeans_n_init"], c["gmm_n_init"], c["gmm_covarianza"]).etiquetas
            for a in c["algoritmos"]}
    t_hd = int(hd.loc[hd["n_grupos"] >= 2, "min_cluster_size"].min()) if (hd["n_grupos"] >= 2).any() else None
    if t_hd is not None:
        etiq["hdbscan"] = hd_etiq[t_hd]
    acuerdo(etiq).to_csv(destino / "acuerdo_algoritmos_k_interno.csv", index=False)
    gmm = sel[sel["algoritmo"] == "gmm"]
    resumen = {
        "entrada": c["entrada"],
        "n_componentes_pca": int(S.shape[1]),
        "varianza_pc1": float(pca.varianza["varianza_explicada"].iloc[0]) if pca is not None else None,
        "k_por_criterios_internos": k,
        "k_gap": int(gap.loc[gap["cumple_regla"], "k"].min()) if gap["cumple_regla"].any() else None,
        "k_bic_gmm": int(gmm.loc[gmm["bic"].idxmin(), "k"]) if not gmm.empty else None,
        "silhouette_max": float(sel["silhouette"].max()),
        "hdbscan_ruido_min_pct": float(hd["pct_ruido"].min()),
    }
    guardar_json(resumen, destino / "cluster_resumen.json")
    log.info("cluster: %s", resumen)
    return resumen


def _funcion_etapa(etapa: str):
    if etapa == "features":
        return etapa_features
    if etapa == "cluster":
        return etapa_cluster
    if etapa == "stability":
        from tipologia.pipeline.etapa_estabilidad import etapa_estabilidad
        return etapa_estabilidad
    if etapa == "profile":
        from tipologia.pipeline.etapa_perfil import etapa_perfil
        return etapa_perfil
    if etapa == "temporal":
        from tipologia.pipeline.etapa_temporal import etapa_temporal
        return etapa_temporal
    if etapa == "export":
        from tipologia.pipeline.etapa_export import etapa_export
        return etapa_export
    raise ValueError(f"Etapa desconocida: {etapa}")


def ejecutar_etapa(etapa: str, cfg: Config, forzar: bool = False, offline: bool = False) -> int:
    """Ejecuta una etapa (o todo el pipeline) sobre el directorio de la ejecución actual.

    Idempotente: si la etapa ya se completó para estos mismos datos, configuración y código, no recomputa
    (salvo --forzar). Como el nombre del directorio incluye el hash de todo eso, datos nuevos producen
    una ejecución nueva.
    """
    destino = dir_trabajo(cfg)
    log.info("Ejecución: %s", destino.name)
    # Un solo hilo en BLAS/OpenMP: las reducciones paralelas cambian el orden de las sumas y con él el
    # último bit de algunos resultados. Con un hilo, la reejecución da artefactos idénticos byte a byte.
    with threadpool_limits(limits=1):
        return _ejecutar(etapa, cfg, destino, forzar)


def _ejecutar(etapa: str, cfg: Config, destino: Path, forzar: bool) -> int:
    etapas = [*ETAPAS_EJECUCION, "export"] if etapa == "pipeline" else [etapa]
    hechas = etapas_completadas(destino)
    if etapa == "pipeline" and not forzar and all(e in hechas for e in etapas):
        log.info("Sin datos, configuración ni código nuevos: la ejecución %s ya está completa; no recomputo", destino.name)
        return 0
    for e in etapas:
        if e in hechas and not forzar and etapa == "pipeline":
            log.info("Etapa %s ya completada; la salto", e)
            continue
        _funcion_etapa(e)(cfg, destino)
        marcar_etapa(destino, e)
    return 0

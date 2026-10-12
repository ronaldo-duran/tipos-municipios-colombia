"""Etapa stability: estabilidad por remuestreo, elección final de k, modelo final con asignaciones
blandas, referencias nulas y análisis de sensibilidad."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from tipologia.clustering.models import acuerdo, ajustar, orden_por_tamano
from tipologia.clustering.reduce import entrada_clustering
from tipologia.clustering.seleccion import decidir_k
from tipologia.config import Config
from tipologia.evaluation.sensitivity import etiqueta_tipo, sensibilidad, sesgo_imputacion, tabla_migracion
from tipologia.evaluation.stability import (
    estabilidad,
    estabilidad_por_tamano,
    perturbacion_por_tamano,
    referencia_nula_gaussiana,
    resumen_coasociacion,
    tabla_estabilidad,
)
from tipologia.features.impute import indicadores_imputados_por_municipio
from tipologia.features.prepare import cargar_datos_base
from tipologia.pipeline.io import guardar_json

log = logging.getLogger(__name__)


def _alinear_a(base: np.ndarray, otra: np.ndarray) -> np.ndarray:
    """Reetiqueta `otra` para maximizar la coincidencia con `base` (solo para mostrar alternativas)."""
    t = pd.crosstab(base, otra)
    f, c = linear_sum_assignment(-t.to_numpy())
    mapa = {t.columns[j]: t.index[i] for i, j in zip(f, c, strict=False)}
    nuevo = max(t.index) + 1
    out = []
    for v in otra:
        if v not in mapa:
            mapa[v] = nuevo
            nuevo += 1
        out.append(mapa[v])
    return np.array(out)


def modelo_final(S: pd.DataFrame, k: int, cfg: Config) -> tuple[np.ndarray, np.ndarray, object, list[int]]:
    """Etiquetas (ordenadas por tamaño), probabilidades de pertenencia, modelo y orden
    (orden[i] = etiqueta original del grupo i)."""
    c = cfg["clustering"]
    alg = c["modelo_final"]
    a = ajustar(alg, S.to_numpy(), k, cfg.semilla, c["kmeans_n_init"], c["gmm_n_init"], c["gmm_covarianza"])
    nuevas = orden_por_tamano(a.etiquetas)
    orden = [int(pd.Series(a.etiquetas[nuevas == i]).iloc[0]) for i in range(k)]
    if a.probabilidades is not None:
        proba = a.probabilidades[:, orden]
    else:
        # Si el modelo final no es probabilístico, la pertenencia blanda sale de un clasificador
        # gaussiano (QDA) ajustado sobre la partición final.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(S, nuevas).predict_proba(S)
    return nuevas, proba, a.modelo, orden


def etapa_estabilidad(cfg: Config, destino: Path) -> dict:
    c, e = cfg["clustering"], cfg["estabilidad"]
    S = pd.read_parquet(destino / "entrada.parquet")
    sel = pd.read_csv(destino / "seleccion_k.csv")
    ks = list(range(c["k_min"], c["k_max"] + 1))

    # 1. Estabilidad por algoritmo y k
    estab, jac = tabla_estabilidad(S, c["algoritmos"], ks, c, e["n_remuestreos"], e["fraccion_submuestra"], cfg.semilla)
    estab.to_csv(destino / "estabilidad.csv", index=False)
    jac.to_csv(destino / "estabilidad_jaccard.csv", index=False)

    # 2. Elección de k con criterios internos + estabilidad
    dec = decidir_k(sel, estab, e["umbral_estable"], c["modelo_final"], n=len(S))
    dec["tabla"].to_csv(destino / "decision_k.csv", index=False)
    dec["detalle"].to_csv(destino / "decision_k_detalle.csv", index=False)
    k = dec["k"]
    log.info("k elegido: %d (ninguna estable: %s)", k, dec["ninguna_particion_estable"])

    # 3. Modelo final y asignaciones blandas
    etiquetas, proba, modelo, orden = modelo_final(S, k, cfg)
    tipos = [etiqueta_tipo(i) for i in range(k)]
    asig = pd.DataFrame(proba, index=S.index, columns=[f"prob_{t}" for t in tipos])
    asig.insert(0, "tipo", [etiqueta_tipo(v) for v in etiquetas])
    srt = np.sort(proba, axis=1)
    asig["prob_max"] = srt[:, -1]
    asig["margen"] = srt[:, -1] - srt[:, -2]
    asig["frontera_probabilidad"] = asig["prob_max"] < c["umbral_frontera"]

    # 4. Co-asociación del modelo final: PAC y estabilidad individual
    r = estabilidad(S.to_numpy(), c["modelo_final"], k, c, e["n_remuestreos"], e["fraccion_submuestra"],
                    cfg.semilla, referencia=etiquetas, coasociar=True)
    s = cfg["sensibilidad"]
    pac, indiv = resumen_coasociacion(r.coasociacion, r.pares, etiquetas, s["pac_umbral_inferior"], s["pac_umbral_superior"])
    asig["estabilidad_individual"] = indiv.to_numpy()
    asig["frontera_remuestreo"] = asig["estabilidad_individual"] < c["umbral_estabilidad_individual"]
    asig["frontera"] = asig["frontera_probabilidad"] | asig["frontera_remuestreo"]
    imputado = pd.read_parquet(destino / "imputado.parquet").reindex(S.index)
    asig["n_imputados"] = imputado.sum(axis=1).astype(int)
    asig["indicadores_imputados"] = indicadores_imputados_por_municipio(imputado)
    r.jaccard.assign(tipo=[etiqueta_tipo(g) for g in r.jaccard["grupo"]]).to_csv(destino / "jaccard_modelo_final.csv", index=False)

    # 5. Particiones alternativas con el mismo k (alineadas) y acuerdo entre algoritmos
    alternativas = {c["modelo_final"]: etiquetas}
    for alg in c["algoritmos"]:
        if alg != c["modelo_final"]:
            lab = ajustar(alg, S.to_numpy(), k, cfg.semilla, c["kmeans_n_init"], c["gmm_n_init"], c["gmm_covarianza"]).etiquetas
            alternativas[alg] = _alinear_a(etiquetas, lab)
    hd = pd.read_csv(destino / "hdbscan.csv")
    if (hd["n_grupos"] >= 2).any():
        t_hd = int(hd.loc[hd["n_grupos"] >= 2, "min_cluster_size"].min())
        lab_hd = ajustar("hdbscan", S.to_numpy(), 0, 0, min_cluster_size=t_hd, min_samples=c["hdbscan_min_samples"]).etiquetas
        alternativas[f"hdbscan_{t_hd}"] = lab_hd
    acuerdo(alternativas).to_csv(destino / "acuerdo_algoritmos.csv", index=False)
    alt = pd.DataFrame({a: [etiqueta_tipo(v) for v in lab] for a, lab in alternativas.items()}, index=S.index)
    alt.to_parquet(destino / "alternativas.parquet")

    # 6. Referencias nulas: ¿cuánto silhouette y estabilidad darían datos sin grupos?
    nulas = []
    for alg in dict.fromkeys([c["modelo_final"], "kmeans"]):
        nul = referencia_nula_gaussiana(S, alg, k, c, replicas=e["referencia_nula_replicas"], semilla=cfg.semilla, n_remuestreos=20,
                                        fraccion=e["fraccion_submuestra"])
        nulas.append(nul.assign(algoritmo=alg))
    nulas = pd.concat(nulas, ignore_index=True)
    nulas.to_csv(destino / "referencia_nula.csv", index=False)

    # 7. Efecto del tamaño poblacional en la confianza de la asignación
    datos = cargar_datos_base(cfg)
    pob = datos.poblacion(cfg, cfg["datos"]["anio_referencia"])
    tam = estabilidad_por_tamano(asig["estabilidad_individual"], asig["frontera"], pob)
    tam.reset_index().to_csv(destino / "estabilidad_por_tamano.csv", index=False)
    pert = None
    if hasattr(modelo, "predict") and cfg["preparacion"]["ponderacion"] in ("dominio", "indicador"):
        X = pd.read_parquet(destino / "x.parquet")
        Z = pd.read_parquet(destino / "z.parquet")
        pesos = pd.read_csv(destino / "pesos_ponderacion.csv", index_col=0)["peso"]
        pca = entrada_clustering(X, c["entrada"], c["pca_varianza"], cfg.semilla)[1]
        inv = {orig: nuevo for nuevo, orig in enumerate(orden)}

        def a_entrada(Xp: pd.DataFrame) -> np.ndarray:
            if pca is None:
                return Xp.to_numpy()
            return pca.modelo.transform(Xp.to_numpy())[:, :pca.n_componentes] * pca.signos[:pca.n_componentes]

        def predecir(Sp: np.ndarray) -> np.ndarray:
            return np.array([inv[v] for v in modelo.predict(Sp)])

        sensibles = [i.id for i in cfg.indicadores if i.sensible_tamano]
        pert, tasa = perturbacion_por_tamano(Z, pesos, a_entrada, predecir, etiquetas, pob, sensibles,
                                             e["ruido_tamano_nivel"], e["ruido_tamano_replicas"], cfg.semilla)
        pert.reset_index().to_csv(destino / "ruido_por_tamano.csv", index=False)
        asig["tasa_cambio_por_ruido"] = tasa.reindex(asig.index).to_numpy()

    # 8. Sensibilidad
    base = pd.Series(etiquetas, index=S.index)
    sens, etq = sensibilidad(cfg, datos, base, k, c["modelo_final"])
    sens.to_csv(destino / "sensibilidad.csv", index=False)
    sesgo = sesgo_imputacion(base.map(etiqueta_tipo), imputado, {k_: v.map(etiqueta_tipo) for k_, v in etq.items()})
    if "sin_conflicto" in etq:
        mig = tabla_migracion(base.map(etiqueta_tipo), etq["sin_conflicto"].map(etiqueta_tipo))
        mig.to_csv(destino / "migracion_sin_conflicto.csv")
    pd.DataFrame({k_: v.map(etiqueta_tipo) for k_, v in etq.items()}).to_parquet(destino / "sensibilidad_etiquetas.parquet")

    asig.to_parquet(destino / "asignaciones_base.parquet")
    obs = {
        "silhouette": float(sel.loc[(sel["algoritmo"] == c["modelo_final"]) & (sel["k"] == k), "silhouette"].iloc[0]),
        "ari_medio": float(estab.loc[(estab["algoritmo"] == c["modelo_final"]) & (estab["k"] == k), "ari_medio"].iloc[0]),
    }
    resumen = {
        "k": k,
        "modelo_final": c["modelo_final"],
        "ninguna_particion_estable": dec["ninguna_particion_estable"],
        "observado_modelo_final": obs,
        "referencia_nula_media": nulas.groupby("algoritmo")[["silhouette", "ari_medio"]].mean().round(4).to_dict(orient="index"),
        "referencia_nula_p95": nulas.groupby("algoritmo")[["silhouette", "ari_medio"]].quantile(0.95).round(4).to_dict(orient="index"),
        "pac": pac["pac"],
        "coasociacion_media_mismo_grupo": pac["coasociacion_media_mismo_grupo"],
        "tamanos": asig["tipo"].value_counts().sort_index().to_dict(),
        "pct_frontera": float(asig["frontera"].mean() * 100),
        "pct_frontera_probabilidad": float(asig["frontera_probabilidad"].mean() * 100),
        "pct_frontera_remuestreo": float(asig["frontera_remuestreo"].mean() * 100),
        "estabilidad_por_tamano_kruskal_p": tam.attrs.get("kruskal_p"),
        "ruido_por_tamano": pert.reset_index().to_dict(orient="records") if pert is not None else None,
        "sesgo_imputacion": sesgo,
        "sensibilidad_ari_mediana": float(sens["ari"].median()),
        "sensibilidad_ari_min": {"escenario": sens.loc[sens["ari"].idxmin(), "escenario"], "ari": float(sens["ari"].min())},
    }
    guardar_json(resumen, destino / "estabilidad_resumen.json")
    log.info("stability: k=%d PAC=%.3f frontera=%.1f%%", k, pac["pac"], resumen["pct_frontera"])
    return resumen

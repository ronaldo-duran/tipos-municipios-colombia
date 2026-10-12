"""Análisis de sensibilidad: cuánto cambian los grupos al variar decisiones metodológicas."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from tipologia.clustering.models import ajustar, metricas_internas, orden_por_tamano, seleccion_k
from tipologia.clustering.reduce import entrada_clustering
from tipologia.clustering.seleccion import decidir_k
from tipologia.config import Config
from tipologia.features.prepare import DatosBase, preparar

log = logging.getLogger(__name__)


@dataclass
class Escenario:
    id: str
    descripcion: str
    grupo: str
    cambios: dict = field(default_factory=dict)
    quitar_dominios: tuple[str, ...] = ()
    anio: int | None = None
    excluir: list[str] | None = None
    indicadores_ids: list[str] | None = None


def escenarios(cfg: Config, datos: DatosBase) -> list[Escenario]:
    """Batería de escenarios definida a partir de la configuración."""
    out: list[Escenario] = []
    for dom, nombre in cfg.dominios.items():
        out.append(Escenario(f"sin_{dom}", f"Sin el dominio {nombre}", "indicadores", quitar_dominios=(dom,)))
    out += [
        Escenario("escalado_robusto", "Escalado robusto (mediana e IQR)", "escalado", {"preparacion.escalado": "robusto"}),
        Escenario("sin_winsorizar", "Sin winsorización", "escalado", {"preparacion.winsorizar.activo": False}),
        Escenario("ponderacion_indicador", "Peso igual por indicador (no por dominio)", "ponderacion",
                  {"preparacion.ponderacion": "indicador"}),
        Escenario("ponderacion_pca_bloque", "PCA por bloque de dominio", "ponderacion",
                  {"preparacion.ponderacion": "pca_bloque"}),
        Escenario("imputacion_mediana_departamento", "Imputación por mediana del departamento", "imputacion",
                  {"preparacion.imputacion": "mediana_departamento"}),
        Escenario("imputacion_knn", "Imputación KNN", "imputacion", {"preparacion.imputacion": "knn"}),
        Escenario("casos_completos", "Solo casos completos (sin ninguna imputación)", "imputacion",
                  {"preparacion.imputacion": "casos_completos"}),
        Escenario("sin_bajo_peso_fec_adol", "Sin bajo peso al nacer ni fecundidad adolescente (los más imputados)",
                  "imputacion", indicadores_ids=[i.id for i in cfg.indicadores_por_uso("entrada")
                                                 if i.id not in ("bajo_peso", "fec_adol")]),
        Escenario("entrada_variables", "Clustering sobre las variables ponderadas (sin PCA)", "entrada",
                  {"clustering.entrada": "variables"}),
    ]
    for anio in cfg["datos"]["anios_sensibilidad"]:
        out.append(Escenario(f"anio_{anio}", f"Año de referencia {anio}", "anio", anio=anio))
    pob = datos.poblacion(cfg, cfg["datos"]["anio_referencia"])
    capitales = datos.municipios.index[datos.municipios["capital"] == 1]
    grandes = pob.index[pob >= cfg["sensibilidad"]["pob_min_ciudad_grande"]]
    excl = sorted(set(capitales) | set(grandes))
    out.append(Escenario("sin_capitales_y_grandes",
                         f"Sin capitales ni municipios de {cfg['sensibilidad']['pob_min_ciudad_grande']:,}+ habitantes "
                         f"({len(excl)} excluidos)", "muestra", excluir=excl))
    return out


def alinear(base: pd.Series, otra: pd.Series) -> tuple[pd.Series, float]:
    """Alinea las etiquetas de `otra` con `base` (asignación húngara) y devuelve la coincidencia."""
    comunes = base.index.intersection(otra.index)
    t = pd.crosstab(base.loc[comunes], otra.loc[comunes])
    filas, cols = linear_sum_assignment(-t.to_numpy())
    mapa = {t.columns[c]: t.index[f] for f, c in zip(filas, cols, strict=False)}
    alineada = otra.map(lambda v: mapa.get(v, f"nuevo_{v}"))
    coincidencia = float((alineada.loc[comunes] == base.loc[comunes]).mean())
    return alineada, coincidencia


def correr_escenario(cfg: Config, datos: DatosBase, esc: Escenario, k: int, modelo: str,
                     muestra_base: pd.Index | None = None) -> tuple[pd.Series, dict]:
    """Prepara, reduce y clusteriza un escenario con el k y el modelo de la línea base.

    Si el escenario solo cambia indicadores o el tratamiento de los datos, se fija la muestra de la línea
    base para que la comparación no se mezcle con un cambio de municipios. Los escenarios de año, de
    casos completos y de exclusión de ciudades sí cambian la muestra a propósito."""
    cfg_e = cfg.con_cambios(esc.cambios) if esc.cambios else cfg
    fija = muestra_base if (esc.anio is None and esc.excluir is None and esc.id != "casos_completos") else None
    P = preparar(cfg_e, datos, anio=esc.anio, quitar_dominios=esc.quitar_dominios, excluir_codigos=esc.excluir,
                 motivo_extra=esc.descripcion, indicadores_ids=esc.indicadores_ids, municipios_fijos=fija)
    c = cfg_e["clustering"]
    S, _ = entrada_clustering(P.X, c["entrada"], c["pca_varianza"], cfg.semilla)
    a = ajustar(modelo, S.to_numpy(), k, cfg.semilla, c["kmeans_n_init"], c["gmm_n_init"], c["gmm_covarianza"])
    etiquetas = pd.Series(orden_por_tamano(a.etiquetas), index=S.index)
    sel = seleccion_k(S, list(range(c["k_min"], c["k_max"] + 1)), cfg.semilla, {**c, "algoritmos": [modelo]})
    k_int = decidir_k(sel, modelo_final=modelo, n=len(S))["k"]
    info = {"n_municipios": int(len(S)), "n_indicadores": int(P.Z.shape[1]), "dimensiones_entrada": int(S.shape[1]),
            **{f"{m}_k_base": v for m, v in metricas_internas(S.to_numpy(), a.etiquetas).items()},
            "k_preferido_criterios_internos": int(k_int)}
    return etiquetas, info


def sensibilidad(cfg: Config, datos: DatosBase, base: pd.Series, k: int, modelo: str) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Corre todos los escenarios y compara con la partición base sobre los municipios comunes."""
    filas, etiquetas = [], {}
    for esc in escenarios(cfg, datos):
        lab, info = correr_escenario(cfg, datos, esc, k, modelo, muestra_base=base.index)
        comunes = base.index.intersection(lab.index)
        _, coinc = alinear(base, lab)
        filas.append({"escenario": esc.id, "descripcion": esc.descripcion, "grupo": esc.grupo,
                      "municipios_comunes": int(len(comunes)),
                      "ari": float(adjusted_rand_score(base.loc[comunes], lab.loc[comunes])),
                      "ami": float(adjusted_mutual_info_score(base.loc[comunes], lab.loc[comunes])),
                      "coincidencia_alineada": coinc, **info})
        etiquetas[esc.id] = lab
        log.info("Sensibilidad %s: ARI=%.3f", esc.id, filas[-1]["ari"])
    return pd.DataFrame(filas), etiquetas


def sesgo_imputacion(base: pd.Series, imputado: pd.DataFrame, etiquetas: dict[str, pd.Series],
                     variables: tuple[str, ...] = ("bajo_peso", "fec_adol")) -> dict:
    """¿La imputación de bajo peso y fecundidad adolescente (concentrada en municipios pequeños) mueve
    asignaciones? Compara (1) casos completos vs base y (2) tasa de cambio de tipo al quitar esas variables,
    entre municipios con esas variables imputadas y el resto."""
    out: dict = {}
    cc = etiquetas.get("casos_completos")
    if cc is not None:
        comunes = base.index.intersection(cc.index)
        out["casos_completos_n"] = int(len(cc))
        out["casos_completos_ari_vs_base"] = float(adjusted_rand_score(base.loc[comunes], cc.loc[comunes]))
    sin = etiquetas.get("sin_bajo_peso_fec_adol")
    if sin is not None:
        alineada, coinc = alinear(base, sin)
        cols = [v for v in variables if v in imputado.columns]
        imp = imputado[cols].any(axis=1).reindex(base.index, fill_value=False)
        cambio = alineada.reindex(base.index) != base
        out["municipios_con_imputacion_en_esas_variables"] = int(imp.sum())
        out["pct_cambio_tipo_imputados"] = float(cambio[imp].mean() * 100) if imp.any() else None
        out["pct_cambio_tipo_resto"] = float(cambio[~imp].mean() * 100)
        out["coincidencia_global"] = coinc
        out["distribucion_tipos_imputados"] = base[imp].value_counts(normalize=True).round(3).to_dict()
        out["distribucion_tipos_resto"] = base[~imp].value_counts(normalize=True).round(3).to_dict()
    return out


def tabla_migracion(base: pd.Series, otra: pd.Series) -> pd.DataFrame:
    """Tabla cruzada base vs escenario con etiquetas alineadas."""
    alineada, _ = alinear(base, otra)
    comunes = base.index.intersection(otra.index)
    return pd.crosstab(base.loc[comunes].rename("base"), alineada.loc[comunes].rename("escenario"))


def etiqueta_tipo(e: int) -> str:
    """0 -> 'A', 1 -> 'B', ... (orden por tamaño, no por mérito)."""
    return chr(ord("A") + int(e)) if e >= 0 else "sin grupo"


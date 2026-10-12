"""Etapa profile: perfiles de los tipos, modelo sustituto, comparación externa y análisis espacial."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tipologia.config import CONFIG_DIR, Config
from tipologia.data.ingest import leer_geometrias
from tipologia.evaluation.external import comparar_ordinal, comparar_particion, validacion_externa
from tipologia.evaluation.profile import (
    descriptores,
    distribucion,
    nombres_tipos,
    perfiles_valores,
    perfiles_z,
    representativos,
    similares,
    tamanos,
)
from tipologia.evaluation.surrogate import arbol_sustituto, bosque_sustituto
from tipologia.features.build import calcular_indicador
from tipologia.features.prepare import cargar_datos_base
from tipologia.pipeline.io import guardar_json, leer_json
from tipologia.spatial.autocorr import lisa, moran_por_tipo, moran_variable, vecinos_mismo_tipo
from tipologia.spatial.constrained import comparar_con_base, ward_restringido
from tipologia.spatial.weights import pesos_desde_lista, vecindad_reina

log = logging.getLogger(__name__)


def obtener_vecindad(cfg: Config, destino: Path, codigos: pd.Index) -> pd.DataFrame | None:
    """Vecindad desde el MGN 2025 local; si no está, reutiliza la lista de vecinos versionada."""
    man = cfg.ruta("raw") / "mgn" / "manifest.json"
    if man.exists():
        m = leer_json(man)
        c = cfg["fuentes"]["mgn"]
        geo = leer_geometrias(cfg.ruta("raw") / "mgn" / m["version"] / m["archivo"], c["capa"], c["columna_codigo"])
        ady = vecindad_reina(geo, codigos)
        ady.attrs["fuente"] = f"MGN 2025 (DANE), versión {m['version']}"
        return ady
    for cand in [destino / "vecinos.csv", *sorted(cfg.ruta("outputs").glob("*/vecinos.csv"))]:
        if cand.exists():
            log.info("Uso la vecindad versionada %s", cand)
            ady = pd.read_csv(cand, dtype=str)
            ady.attrs["fuente"] = f"lista de vecinos versionada ({cand.parent.name})"
            return ady
    log.warning("Sin MGN ni lista de vecinos: se omite el análisis espacial")
    return None


def etapa_perfil(cfg: Config, destino: Path) -> dict:
    it = cfg["interpretacion"]
    asig = pd.read_parquet(destino / "asignaciones_base.parquet")
    Z = pd.read_parquet(destino / "z.parquet")
    valores = pd.read_parquet(destino / "valores.parquet")
    imputado = pd.read_parquet(destino / "imputado.parquet")
    mun = pd.read_parquet(destino / "municipios.parquet")
    S = pd.read_parquet(destino / "entrada.parquet")
    pcs = pd.read_parquet(destino / "pca_scores.parquet") if (destino / "pca_scores.parquet").exists() else None
    tipos = asig["tipo"]
    nombres_ind = {i.id: i.nombre for i in cfg.indicadores}
    dominios = {i.id: i.dominio for i in cfg.indicadores}
    datos = cargar_datos_base(cfg)
    anio = cfg["datos"]["anio_referencia"]
    pob = datos.poblacion(cfg, anio)

    # Perfiles
    pz = perfiles_z(Z, imputado, tipos)
    pz.to_csv(destino / "perfiles_z.csv")
    perfiles_valores(valores, tipos).to_csv(destino / "perfiles_valores.csv")
    tam = tamanos(tipos, pob)
    tam.to_csv(destino / "tamanos.csv")
    distribucion(tipos, mun["dpto"]).to_csv(destino / "distribucion_departamentos.csv", index=False)
    distribucion(tipos, mun["region"]).to_csv(destino / "distribucion_regiones.csv", index=False)
    representativos(S, tipos, asig["prob_max"], mun["nombre"], it["n_representativos"]).to_csv(destino / "representativos.csv", index=False)
    similares(S, it["n_similares"]).to_csv(destino / "similares.csv", index=False)
    desc = descriptores(pz, nombres_ind)
    nom = nombres_tipos(CONFIG_DIR / "nombres_tipos.yaml", desc)
    nom.join(desc).to_csv(destino / "nombres_tipos.csv")

    # Modelo sustituto (explica la partición; no es causal)
    arbol = arbol_sustituto(Z, tipos, it["arbol_profundidad"], it["cv_folds"], cfg.semilla, nombres_ind)
    (destino / "sustituto_arbol.txt").write_text(arbol["reglas"], encoding="utf-8", newline="\n")
    bosque, imp = bosque_sustituto(Z, tipos, dominios, it["rf_arboles"], it["cv_folds"], cfg.semilla)
    imp["nombre"] = imp["indicador"].map(nombres_ind)
    imp.to_csv(destino / "sustituto_importancias.csv", index=False)
    imp_dom = imp.groupby("dominio")["importancia_permutacion"].sum().sort_values(ascending=False)

    # Referencias externas (nunca entraron al clustering)
    ref = cfg.catalogo["referencias"]
    if pcs is not None:
        orden_tipos = pcs["PC1"].groupby(tipos).median().sort_values().index.tolist()
    else:
        orden_tipos = sorted(tipos.unique())
    dnp = comparar_ordinal(tipos, mun["tipologia_dnp_2026"], ref["tipologia_dnp"]["orden"],
                           ref["tipologia_dnp"]["especiales"], orden_tipos)
    dnp["tabla"].to_csv(destino / "externo_dnp_tabla.csv")
    rur = comparar_ordinal(tipos, mun["cat_ruralidad"], ref["ruralidad"]["orden"], [], orden_tipos)
    rur.update(comparar_particion(tipos, mun["cat_ruralidad"]))
    rur["tabla"].to_csv(destino / "externo_ruralidad_tabla.csv")
    validaciones = {}
    for ind in cfg.indicadores_por_uso("validacion"):
        r = calcular_indicador(datos.largo, ind, ind.anio, anio, cfg.catalogo["poblacion_codigo"], cfg.catalogo["area_codigo"])
        v = validacion_externa(tipos, r["valor"])
        v["resumen"].to_csv(destino / f"validacion_{ind.id}.csv")
        validaciones[ind.id] = {"nombre": ind.nombre, "anio": sorted({int(a) for a in r["anio"].dropna()}),
                                **{k: v[k] for k in ("eta2", "epsilon2", "p", "n")}}

    # Análisis espacial
    espacial = {}
    ady = obtener_vecindad(cfg, destino, S.index)
    if ady is not None:
        ady.to_csv(destino / "vecinos.csv", index=False)
        w = pesos_desde_lista(ady, S.index)
        e = cfg["espacial"]
        mpt = moran_por_tipo(w, tipos, e["permutaciones"], cfg.semilla)
        mpt.to_csv(destino / "espacial_moran_tipos.csv", index=False)
        prop, resumen_v = vecinos_mismo_tipo(w, tipos)
        local = pd.DataFrame({"prop_vecinos_mismo_tipo": prop})
        if pcs is not None:
            l_pc1 = lisa(w, pcs["PC1"], e["permutaciones"], cfg.semilla, e["alfa"]).add_prefix("pc1_")
            local = local.join(l_pc1)
            espacial["moran_pc1"] = moran_variable(w, pcs["PC1"], e["permutaciones"], cfg.semilla)
        propio = []
        for t in sorted(tipos.unique()):
            lt = lisa(w, (tipos == t).astype(float), e["permutaciones"], cfg.semilla, e["alfa"])
            propio.append(lt.loc[tipos.index[tipos == t], "lisa_categoria"])
        local["lisa_tipo_propio"] = pd.concat(propio).reindex(local.index).replace(
            {"alto-alto": "núcleo (rodeado de su tipo)", "alto-bajo": "aislado (rodeado de otros tipos)"})
        local["lisa_tipo_propio"] = local["lisa_tipo_propio"].fillna("no significativo")
        local.to_parquet(destino / "espacial_local.parquet")
        restr = ward_restringido(S, w, int(tipos.nunique()))
        restr.map(lambda v: f"R{v + 1}").rename("region_contigua").to_frame().to_parquet(destino / "espacial_restringido.parquet")
        espacial.update({
            "fuente_vecindad": ady.attrs.get("fuente"),
            "pares_vecinos": int(len(ady) // 2),
            "moran_por_tipo": mpt.to_dict(orient="records"),
            "vecinos_mismo_tipo": resumen_v,
            "lisa_tipo_propio": local["lisa_tipo_propio"].value_counts().to_dict(),
            "ward_restringido_vs_base": comparar_con_base(tipos.map(lambda t: ord(t) - 65), restr),
            "moran_prob_max": moran_variable(w, asig["prob_max"], e["permutaciones"], cfg.semilla),
        })

    resumen = {
        "tamanos": tam.reset_index().to_dict(orient="records"),
        "nombres": nom.reset_index().to_dict(orient="records"),
        "descriptores": desc.reset_index().to_dict(orient="records"),
        "sustituto_arbol": {"exactitud_cv_media": arbol["exactitud_cv_media"], "exactitud_cv_sd": arbol["exactitud_cv_sd"],
                            "profundidad": it["arbol_profundidad"]},
        "sustituto_bosque": bosque,
        "importancia_por_dominio": imp_dom.round(4).to_dict(),
        "externo_dnp": {k: v for k, v in dnp.items() if k != "tabla"},
        "externo_ruralidad": {k: v for k, v in rur.items() if k != "tabla"},
        "validacion_externa": validaciones,
        "espacial": espacial,
    }
    guardar_json(resumen, destino / "perfil_resumen.json")
    log.info("profile: árbol %.2f, bosque %.2f, V Cramér DNP %.2f, ruralidad %.2f", arbol["exactitud_cv_media"],
             bosque["exactitud_cv_media"], dnp["cramer_v"], rur["cramer_v"])
    return resumen

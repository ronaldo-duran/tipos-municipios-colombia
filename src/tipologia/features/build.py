"""Construcción de la matriz ancha (municipio x indicador) a partir del formato largo de TerriData."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from tipologia.config import Config, Indicador
from tipologia.data.divipola import es_municipio

log = logging.getLogger(__name__)

COD_REGION = "010010002"


@dataclass
class Matriz:
    """Resultado de construir la matriz para un año de referencia."""

    anio: int
    valores: pd.DataFrame                 # valores originales (sin transformar), solo municipios incluidos
    anios: pd.DataFrame                   # año del dato usado por celda
    imputado_vecino: pd.DataFrame         # True si el dato viene de un año vecino
    faltante_ref: pd.DataFrame            # True si no había dato en el año que pide la regla
    indicadores: list[Indicador]
    cobertura: pd.Series                  # cobertura en el año de referencia (antes de exclusiones)
    excluidos: pd.DataFrame               # municipios excluidos y motivo
    indicadores_descartados: dict[str, str] = field(default_factory=dict)


def cargar_largo(parquet: Path, codigos: set[str]) -> pd.DataFrame:
    """Valores anuales por municipio e indicador. Si hay varios meses, toma el último mes con dato."""
    lista = ",".join(f"'{c}'" for c in sorted(codigos))
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT cod_ent, cod_ind, anio, arg_max(valor, mes) AS valor
        FROM '{parquet.as_posix()}'
        WHERE cod_ind IN ({lista}) AND valor IS NOT NULL
        GROUP BY cod_ent, cod_ind, anio
    """).df()
    return df[df["cod_ent"].map(es_municipio)].reset_index(drop=True)


def municipios(parquet: Path, dnp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de municipios con nombre, departamento, región y referencias externas."""
    con = duckdb.connect()
    p = parquet.as_posix()
    m = con.execute(f"SELECT DISTINCT cod_ent AS cod, entidad AS nombre, cod_dpto, dpto FROM '{p}'").df()
    m = m[m["cod"].map(es_municipio)].set_index("cod").sort_index()
    reg = con.execute(f"SELECT cod_ent AS cod, any_value(dato_cual) AS region FROM '{p}' WHERE cod_ind = '{COD_REGION}' GROUP BY 1").df()
    m = m.join(reg.set_index("cod"))
    if dnp is not None:
        m = m.join(dnp.set_index("cod")[["tipologia_dnp_2026", "cat_ruralidad", "capital", "pdet"]])
    return m


def _por_regla(largo: pd.DataFrame, codigo: str, regla: str, ref: int) -> pd.DataFrame:
    """Valor y año del dato de un código según la regla de año. Columnas: valor, anio."""
    d = largo[largo["cod_ind"] == codigo]
    if regla == "ref":
        d = d[d["anio"] == ref]
    elif regla.startswith("fijo:"):
        d = d[d["anio"] == int(regla.split(":")[1])]
    elif regla.startswith("rezago:"):
        n = int(regla.split(":")[1])
        d = d[(d["anio"] <= ref) & (d["anio"] >= ref - n)].sort_values("anio").groupby("cod_ent").tail(1)
    elif regla == "ultimo":
        d = d.sort_values("anio").groupby("cod_ent").tail(1)
    else:
        raise ValueError(f"Regla de año desconocida: {regla}")
    return d.set_index("cod_ent")[["valor", "anio"]]


def _en_anios(largo: pd.DataFrame, codigo: str, anios: pd.Series) -> pd.Series:
    """Valor de un código en el año indicado para cada municipio."""
    d = largo[largo["cod_ind"] == codigo].set_index(["cod_ent", "anio"])["valor"]
    idx = pd.MultiIndex.from_arrays([anios.index, anios.astype(int)])
    return pd.Series(d.reindex(idx).to_numpy(), index=anios.index)


def calcular_indicador(largo: pd.DataFrame, ind: Indicador, regla: str, ref: int,
                       cod_pob: str, cod_area: str) -> pd.DataFrame:
    """Valor (columna 'valor') y año ('anio') de un indicador para cada municipio."""
    if ind.codigo:
        return _por_regla(largo, ind.codigo, regla, ref)
    partes = [_por_regla(largo, c, regla, ref) for c in ind.numerador]
    base = partes[0].copy()
    for p in partes[1:]:
        comunes = base.index.intersection(p.index)
        base = base.loc[comunes]
        base["valor"] = base["valor"] + p.loc[comunes, "valor"]
    if ind.denominador == "poblacion":
        den = _en_anios(largo, cod_pob, base["anio"])
    elif ind.denominador == "area":
        den = _por_regla(largo, cod_area, "ultimo", ref)["valor"].reindex(base.index)
    else:
        den = _en_anios(largo, ind.denominador, base["anio"])
    den = den.where(den > 0)
    base["valor"] = base["valor"] / den * ind.factor
    return base.dropna(subset=["valor"])


def construir(largo: pd.DataFrame, indicadores: list[Indicador], ref: int, universo: pd.Index,
              cod_pob: str, cod_area: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Matrices de valores, años, imputación por año vecino y faltantes en el año de la regla."""
    valores, anios, vecino, faltante = {}, {}, {}, {}
    for ind in indicadores:
        r = calcular_indicador(largo, ind, ind.anio, ref, cod_pob, cod_area).reindex(universo)
        falta = r["valor"].isna()
        imput = pd.Series(False, index=universo)
        if ind.vecino and falta.any():
            for d in range(1, ind.vecino + 1):
                for anio_v in (ref - d, ref + d):
                    pendientes = r["valor"].isna()
                    if not pendientes.any():
                        break
                    rv = calcular_indicador(largo, ind, "ref", anio_v, cod_pob, cod_area).reindex(universo)
                    usar = pendientes & rv["valor"].notna()
                    r.loc[usar, ["valor", "anio"]] = rv.loc[usar, ["valor", "anio"]]
                    imput |= usar
        valores[ind.id], anios[ind.id] = r["valor"], r["anio"]
        vecino[ind.id], faltante[ind.id] = imput, falta
    return (pd.DataFrame(valores), pd.DataFrame(anios), pd.DataFrame(vecino), pd.DataFrame(faltante))


def regla_exclusion(faltante: pd.DataFrame, max_frac: float, nombres: pd.Series | None = None) -> tuple[pd.Index, pd.DataFrame]:
    """Excluye municipios con más de `max_frac` de indicadores sin dato en el año de referencia."""
    frac = faltante.mean(axis=1)
    excl = frac[frac > max_frac].index
    filas = []
    for cod in excl:
        cols = faltante.columns[faltante.loc[cod]].tolist()
        filas.append({
            "cod": cod,
            "nombre": nombres.get(cod) if nombres is not None else None,
            "n_faltantes": len(cols),
            "n_indicadores": faltante.shape[1],
            "fraccion_faltante": round(float(frac[cod]), 3),
            "motivo": f"faltan {len(cols)} de {faltante.shape[1]} indicadores en el año de referencia "
                      f"(máximo permitido {max_frac:.0%})",
            "indicadores_faltantes": ", ".join(cols),
        })
    excluidos = pd.DataFrame(filas, columns=["cod", "nombre", "n_faltantes", "n_indicadores", "fraccion_faltante",
                                             "motivo", "indicadores_faltantes"])
    return faltante.index.difference(excl), excluidos


def construir_matriz(cfg: Config, anio: int, indicadores: list[Indicador], largo: pd.DataFrame,
                     mun: pd.DataFrame, excluir_codigos: list[str] | None = None,
                     motivo_extra: str = "") -> Matriz:
    """Matriz completa para un año: valores, cobertura, exclusiones por regla y descartes por cobertura."""
    universo = mun.index
    if excluir_codigos:
        universo = universo.difference(pd.Index(excluir_codigos))
    cat = cfg.catalogo
    valores, anios, vecino, faltante = construir(largo, indicadores, anio, universo,
                                                 cat["poblacion_codigo"], cat["area_codigo"])
    cobertura = (~faltante).mean()
    minimo = cfg["datos"]["cobertura_min_indicador"]
    descartados = {c: f"cobertura {cobertura[c]:.1%} < {minimo:.0%} en {anio}" for c in cobertura.index[cobertura < minimo]}
    usados = [i for i in indicadores if i.id not in descartados]
    cols = [i.id for i in usados]
    for c, motivo in descartados.items():
        log.warning("Indicador %s descartado: %s", c, motivo)
    incluidos, excluidos = regla_exclusion(faltante[cols], cfg["datos"]["max_faltantes_municipio"], mun["nombre"])
    if excluir_codigos:
        extra = pd.DataFrame({"cod": list(excluir_codigos), "nombre": mun["nombre"].reindex(excluir_codigos).tolist(),
                              "motivo": motivo_extra or "excluido por el escenario"})
        excluidos = pd.concat([excluidos, extra], ignore_index=True)
    log.info("Año %d: %d municipios incluidos, %d excluidos, %d indicadores", anio, len(incluidos), len(excluidos), len(cols))
    return Matriz(
        anio=anio,
        valores=valores.loc[incluidos, cols],
        anios=anios.loc[incluidos, cols],
        imputado_vecino=vecino.loc[incluidos, cols],
        faltante_ref=faltante.loc[incluidos, cols],
        indicadores=usados,
        cobertura=cobertura,
        excluidos=excluidos,
        indicadores_descartados=descartados,
    )


def patron_faltantes(faltante: pd.DataFrame, mun: pd.DataFrame, poblacion: pd.Series) -> dict[str, pd.DataFrame]:
    """Describe el patrón de faltantes por tamaño poblacional, ruralidad y región (¿son aleatorios?)."""
    from scipy.stats import chi2_contingency

    df = pd.DataFrame({"n_faltantes": faltante.sum(axis=1), "alguno": faltante.any(axis=1)})
    df["quintil_poblacion"] = pd.qcut(poblacion.reindex(df.index), 5, labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (mayor)"])
    df = df.join(mun[["cat_ruralidad", "region"]])
    out = {}
    for g in ("quintil_poblacion", "cat_ruralidad", "region"):
        t = df.groupby(g, observed=True).agg(municipios=("alguno", "size"), con_faltantes=("alguno", "sum"),
                                             faltantes_medios=("n_faltantes", "mean"))
        t["pct_con_faltantes"] = (t["con_faltantes"] / t["municipios"] * 100).round(1)
        tabla = pd.crosstab(df[g], df["alguno"])
        if tabla.shape[1] == 2 and tabla.shape[0] > 1:
            chi2, p, _, _ = chi2_contingency(tabla)
            t.attrs["chi2_p"] = float(p)
        out[g] = t
    por_ind = pd.DataFrame({"faltantes": faltante.sum()})
    q = df["quintil_poblacion"]
    por_ind["pct_en_Q1"] = [round(float((q[faltante[c]] == "Q1 (menor)").mean() * 100), 1) if faltante[c].any() else np.nan
                            for c in faltante.columns]
    out["por_indicador"] = por_ind[por_ind["faltantes"] > 0].sort_values("faltantes", ascending=False)
    return out

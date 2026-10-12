"""Perfiles de los tipos: puntajes z, tamaños, distribución territorial y municipios representativos."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


def perfiles_z(Z: pd.DataFrame, imputado: pd.DataFrame, tipos: pd.Series) -> pd.DataFrame:
    """Mediana del puntaje z por tipo e indicador, usando solo valores observados (no imputados)."""
    obs = Z.mask(imputado.reindex_like(Z).fillna(False).astype(bool))
    return obs.groupby(tipos.reindex(Z.index)).median()


def perfiles_valores(valores: pd.DataFrame, tipos: pd.Series) -> pd.DataFrame:
    """Mediana de cada indicador en sus unidades originales, por tipo."""
    return valores.groupby(tipos.reindex(valores.index)).median()


def tamanos(tipos: pd.Series, poblacion: pd.Series) -> pd.DataFrame:
    pob = poblacion.reindex(tipos.index)
    t = pd.DataFrame({"municipios": tipos.value_counts(), "poblacion": pob.groupby(tipos).sum(),
                      "poblacion_mediana": pob.groupby(tipos).median()})
    t["pct_municipios"] = (t["municipios"] / t["municipios"].sum() * 100).round(1)
    t["pct_poblacion"] = (t["poblacion"] / t["poblacion"].sum() * 100).round(1)
    return t.sort_index()


def distribucion(tipos: pd.Series, grupo: pd.Series) -> pd.DataFrame:
    """Municipios por tipo y grupo territorial (departamento o región), en formato largo con porcentajes."""
    t = pd.crosstab(grupo.reindex(tipos.index), tipos)
    largo = t.stack().rename("municipios").reset_index()
    largo.columns = ["grupo", "tipo", "municipios"]
    largo["pct_del_grupo"] = largo["municipios"] / largo.groupby("grupo")["municipios"].transform("sum") * 100
    largo["pct_del_tipo"] = largo["municipios"] / largo.groupby("tipo")["municipios"].transform("sum") * 100
    return largo[largo["municipios"] > 0].round(2)


def representativos(S: pd.DataFrame, tipos: pd.Series, prob_max: pd.Series, nombres: pd.Series, n: int) -> pd.DataFrame:
    """Medoide de cada tipo (mínima distancia total a los demás miembros) y los n municipios más cercanos a él."""
    filas = []
    for t in sorted(tipos.unique()):
        idx = tipos.index[tipos == t]
        P = S.loc[idx].to_numpy()
        D = cdist(P, P)
        medoide = idx[int(D.sum(axis=1).argmin())]
        dist = pd.Series(cdist(S.loc[[medoide]].to_numpy(), P)[0], index=idx).sort_values()
        for rango, (cod, d) in enumerate(dist.head(n).items(), start=1):
            filas.append({"tipo": t, "rango": rango, "cod": cod, "nombre": nombres.get(cod), "es_medoide": cod == medoide,
                          "distancia_al_medoide": round(float(d), 4), "prob_max": round(float(prob_max.get(cod, np.nan)), 4)})
    return pd.DataFrame(filas)


def similares(S: pd.DataFrame, n: int) -> pd.DataFrame:
    """Los n municipios más parecidos a cada uno en el espacio del clustering (distancia euclidiana)."""
    D = cdist(S.to_numpy(), S.to_numpy())
    np.fill_diagonal(D, np.inf)
    orden = np.argsort(D, axis=1)[:, :n]
    filas = []
    for i, cod in enumerate(S.index):
        for r, j in enumerate(orden[i], start=1):
            filas.append({"cod": cod, "rango": r, "cod_similar": S.index[j], "distancia": round(float(D[i, j]), 4)})
    return pd.DataFrame(filas)


def descriptores(perfil: pd.DataFrame, nombres: dict[str, str], top: int = 4, umbral: float = 0.3) -> pd.DataFrame:
    """Rasgos que más distinguen cada tipo (|mediana z| más alta), en lenguaje descriptivo."""
    filas = []
    for t, fila in perfil.iterrows():
        orden = fila.abs().sort_values(ascending=False)
        rasgos = []
        for ind in orden.index[:top]:
            if abs(fila[ind]) < umbral:
                continue
            rasgos.append(f"{'más' if fila[ind] > 0 else 'menos'} {nombres.get(ind, ind).lower()}")
        filas.append({"tipo": t, "rasgos": "; ".join(rasgos),
                      "firma": ",".join(f"{ind}{'+' if fila[ind] > 0 else '-'}" for ind in orden.index[:3])})
    return pd.DataFrame(filas).set_index("tipo")


def nombres_tipos(ruta: Path, desc: pd.DataFrame) -> pd.DataFrame:
    """Nombres neutros escritos a mano en config/nombres_tipos.yaml. Si la firma del tipo ya no coincide
    (por ejemplo, porque cambiaron los datos), se usa el descriptor automático y se avisa."""
    manual = {}
    if ruta.exists():
        manual = (yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}).get("tipos", {})
    filas = []
    for t, d in desc.iterrows():
        m = manual.get(t)
        firma_ok = bool(m) and set(m.get("firma", "").split(",")) <= set(d["firma"].split(","))
        if m and firma_ok:
            filas.append({"tipo": t, "nombre": m["nombre"], "descripcion": m.get("descripcion", ""), "nombre_manual": True})
        else:
            if m:
                log.warning("Tipo %s: la firma cambió (%s); uso el descriptor automático", t, d["firma"])
            filas.append({"tipo": t, "nombre": f"Tipo {t}", "descripcion": d["rasgos"], "nombre_manual": False})
    return pd.DataFrame(filas).set_index("tipo")


"""Regla explícita para elegir k combinando varios criterios (ninguno basta por sí solo)."""

from __future__ import annotations

import pandas as pd

# Sentido de cada criterio: True si más alto es mejor.
CRITERIOS = {"silhouette": True, "davies_bouldin": False, "calinski_harabasz": True, "ari_medio": True}


def rangos_compuestos(sel: pd.DataFrame, estabilidad: pd.DataFrame | None = None,
                      algoritmos: tuple[str, ...] = ("kmeans", "gmm", "ward")) -> pd.DataFrame:
    """Rango promedio de cada k por algoritmo (1 = mejor) y promedio entre algoritmos.

    Criterios: silhouette, Davies-Bouldin, Calinski-Harabasz y, si se entrega, el ARI medio bajo
    remuestreo. El codo, la gap statistic y el BIC se reportan aparte como apoyo, porque en datos con
    estructura débil tienden a crecer o decrecer de forma monótona.
    """
    df = sel[sel["algoritmo"].isin(algoritmos)].copy()
    if estabilidad is not None:
        df = df.merge(estabilidad[["algoritmo", "k", "ari_medio"]], on=["algoritmo", "k"], how="left")
    criterios = [c for c in CRITERIOS if c in df.columns]
    for c in criterios:
        df[f"rango_{c}"] = df.groupby("algoritmo")[c].rank(ascending=not CRITERIOS[c], method="average")
    df["rango_medio"] = df[[f"rango_{c}" for c in criterios]].mean(axis=1)
    return df


def decidir_k(sel: pd.DataFrame, estabilidad: pd.DataFrame | None = None, umbral_estable: float = 0.75,
              modelo_final: str = "gmm", tamano_min_frac: float = 0.02, n: int | None = None) -> dict:
    """Elige k y documenta por qué.

    1. Rango compuesto promedio entre algoritmos (menor es mejor).
    2. Se descartan particiones con algún grupo de menos de `tamano_min_frac` de los municipios.
    3. Si hay estabilidad: se prefieren los k cuyo modelo final tiene ARI medio >= umbral. Si ninguno lo
       cumple, se elige el mejor rango igualmente y se declara que la estructura es débil.
    Empates: gana el k menor (parsimonia).
    """
    r = rangos_compuestos(sel, estabilidad)
    pork = r.groupby("k")["rango_medio"].mean().rename("rango_compuesto").to_frame()
    if n:
        minimos = sel[sel["algoritmo"] == modelo_final].set_index("k")["tamano_min"]
        pork["grupo_minimo_ok"] = minimos.reindex(pork.index) >= tamano_min_frac * n
    else:
        pork["grupo_minimo_ok"] = True
    if estabilidad is not None:
        est = estabilidad[estabilidad["algoritmo"] == modelo_final].set_index("k")["ari_medio"]
        pork["ari_modelo_final"] = est.reindex(pork.index)
        pork["estable"] = pork["ari_modelo_final"] >= umbral_estable
    else:
        pork["estable"] = True
    pork = pork.reset_index().sort_values(["rango_compuesto", "k"])
    candidatos = pork[pork["grupo_minimo_ok"] & pork["estable"]]
    estructura_debil = candidatos.empty
    elegido = (candidatos if not candidatos.empty else pork[pork["grupo_minimo_ok"]]).iloc[0]
    return {
        "k": int(elegido["k"]),
        "rango_compuesto": float(elegido["rango_compuesto"]),
        "ninguna_particion_estable": bool(estructura_debil),
        "tabla": pork,
        "detalle": r,
    }

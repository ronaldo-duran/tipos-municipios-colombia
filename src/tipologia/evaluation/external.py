"""Comparación con clasificaciones externas y validación con variables que no entraron al clustering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kendalltau, kruskal
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

log = logging.getLogger(__name__)


def cramer_v(tabla: pd.DataFrame) -> float:
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else float("nan")


def gamma_goodman_kruskal(x: np.ndarray, y: np.ndarray) -> float:
    """Gamma de Goodman-Kruskal entre dos variables ordinales (pares concordantes vs discordantes)."""
    t = pd.crosstab(x, y).to_numpy()
    conc = disc = 0.0
    r, c = t.shape
    for i in range(r):
        for j in range(c):
            conc += t[i, j] * t[i + 1:, j + 1:].sum()
            disc += t[i, j] * t[i + 1:, :j].sum()
    return float((conc - disc) / (conc + disc)) if conc + disc > 0 else float("nan")


def epsilon2_kruskal(valores: pd.Series, grupos: pd.Series) -> dict:
    """Kruskal-Wallis y su tamaño de efecto épsilon² = H / (n - 1)."""
    df = pd.DataFrame({"v": valores, "g": grupos}).dropna()
    muestras = [g["v"].to_numpy() for _, g in df.groupby("g")]
    if len(muestras) < 2:
        return {"H": float("nan"), "p": float("nan"), "epsilon2": float("nan"), "n": int(len(df))}
    H, p = kruskal(*muestras)
    return {"H": float(H), "p": float(p), "epsilon2": float(H / (len(df) - 1)), "n": int(len(df))}


def comparar_ordinal(tipos: pd.Series, referencia: pd.Series, orden: list[str], especiales: list[str],
                     orden_tipos: list[str]) -> dict:
    """Tipos (nominales) frente a una clasificación ordinal externa.

    No usa ARI: la referencia es ordinal y mide otra cosa. Reporta la tabla cruzada, la V de Cramér y,
    sin las categorías especiales, Kruskal-Wallis (¿difiere la posición ordinal entre tipos?) y Kendall
    tau-b / gamma con los tipos ordenados por su mediana en el primer componente principal. Ese orden es
    solo un recurso de cálculo, no una jerarquía de los tipos.
    """
    ref = referencia.reindex(tipos.index).astype("string")
    tabla = pd.crosstab(tipos.rename("tipo"), ref.rename("referencia"))
    cols = [c for c in orden + especiales if c in tabla.columns] + [c for c in tabla.columns if c not in orden + especiales]
    tabla = tabla[cols]
    mask = ref.isin(orden)
    codigo_ref = ref[mask].map({v: i for i, v in enumerate(orden)}).astype(float)
    codigo_tipo = tipos[mask].map({t: i for i, t in enumerate(orden_tipos)}).astype(float)
    tau = kendalltau(codigo_tipo, codigo_ref, variant="b")
    out = {
        "tabla": tabla,
        "n": int(tabla.to_numpy().sum()),
        "n_ordinal": int(mask.sum()),
        "cramer_v": cramer_v(tabla),
        "kruskal_referencia_por_tipo": epsilon2_kruskal(codigo_ref, tipos[mask]),
        "kendall_tau_b": float(tau.statistic),
        "kendall_p": float(tau.pvalue),
        "gamma": gamma_goodman_kruskal(codigo_tipo.to_numpy(), codigo_ref.to_numpy()),
        "orden_tipos_usado": orden_tipos,
        "mediana_referencia_por_tipo": codigo_ref.groupby(tipos[mask]).median().map(lambda i: orden[int(round(i))]).to_dict(),
    }
    return out


def comparar_particion(tipos: pd.Series, referencia: pd.Series) -> dict:
    """Para una clasificación nominal (o tratada como tal): ARI, AMI y V de Cramér."""
    ref = referencia.reindex(tipos.index)
    m = ref.notna()
    tabla = pd.crosstab(tipos[m].rename("tipo"), ref[m].rename("referencia"))
    return {"ari": float(adjusted_rand_score(ref[m], tipos[m])), "ami": float(adjusted_mutual_info_score(ref[m], tipos[m])),
            "cramer_v": cramer_v(tabla)}


def validacion_externa(tipos: pd.Series, variable: pd.Series) -> dict:
    """¿Separa la tipología una variable que no entró al clustering? (p. ej. Saber 11)."""
    v = variable.reindex(tipos.index)
    desc = v.groupby(tipos).describe()[["count", "25%", "50%", "75%"]].rename(columns={"50%": "mediana"})
    df = pd.DataFrame({"v": v, "g": tipos}).dropna()
    media = df["v"].mean()
    ss_tot = ((df["v"] - media) ** 2).sum()
    ss_entre = df.groupby("g")["v"].apply(lambda s: len(s) * (s.mean() - media) ** 2).sum()
    return {"resumen": desc, "eta2": float(ss_entre / ss_tot) if ss_tot > 0 else float("nan"),
            **epsilon2_kruskal(df["v"], df["g"])}

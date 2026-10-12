"""Estabilidad de las particiones bajo remuestreo y contraste contra referencias sin grupos."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from tipologia.clustering.models import ajustar, metricas_internas

log = logging.getLogger(__name__)


@dataclass
class ResultadoEstabilidad:
    algoritmo: str
    k: int
    ari: np.ndarray                 # ARI de cada submuestra contra la partición de referencia
    jaccard: pd.DataFrame           # Jaccard medio por grupo de referencia (Hennig, clusterboot)
    coasociacion: np.ndarray | None = None
    pares: np.ndarray | None = None

    def resumen(self) -> dict:
        return {"algoritmo": self.algoritmo, "k": self.k, "ari_medio": float(self.ari.mean()),
                "ari_sd": float(self.ari.std(ddof=0)), "ari_p05": float(np.quantile(self.ari, 0.05)),
                "jaccard_min": float(self.jaccard["jaccard_medio"].min()),
                "jaccard_medio": float(self.jaccard["jaccard_medio"].mean())}


def _ajustar_silencioso(alg, X, k, semilla, cfg_cluster):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ajustar(alg, X, k, semilla, cfg_cluster["kmeans_n_init"], cfg_cluster["gmm_n_init"],
                       cfg_cluster["gmm_covarianza"]).etiquetas


def estabilidad(S: np.ndarray, algoritmo: str, k: int, cfg_cluster: dict, n_remuestreos: int,
                fraccion: float, semilla: int, referencia: np.ndarray | None = None,
                coasociar: bool = False) -> ResultadoEstabilidad:
    """Submuestreo sin reemplazo: se reclusteriza cada submuestra (con otra semilla) y se compara
    con la partición de referencia sobre los municipios comunes."""
    S = np.asarray(S)
    n = len(S)
    m = int(round(fraccion * n))
    rng = np.random.default_rng(semilla)
    ref = referencia if referencia is not None else _ajustar_silencioso(algoritmo, S, k, semilla, cfg_cluster)
    aris = []
    jacc = {c: [] for c in np.unique(ref)}
    co = np.zeros((n, n), dtype=np.float32) if coasociar else None
    pares = np.zeros((n, n), dtype=np.float32) if coasociar else None
    for b in range(n_remuestreos):
        idx = np.sort(rng.choice(n, m, replace=False))
        lab = _ajustar_silencioso(algoritmo, S[idx], k, semilla + 1 + b, cfg_cluster)
        aris.append(adjusted_rand_score(ref[idx], lab))
        for c in jacc:
            A = ref[idx] == c
            if A.sum() == 0:
                continue
            mejor = max((np.sum(A & (lab == d)) / np.sum(A | (lab == d))) for d in np.unique(lab))
            jacc[c].append(mejor)
        if coasociar:
            mismo = (lab[:, None] == lab[None, :]).astype(np.float32)
            co[np.ix_(idx, idx)] += mismo
            pares[np.ix_(idx, idx)] += 1
    jac = pd.DataFrame({"grupo": list(jacc), "jaccard_medio": [float(np.mean(v)) for v in jacc.values()],
                        "tamano": [int((ref == c).sum()) for c in jacc]})
    return ResultadoEstabilidad(algoritmo, k, np.array(aris), jac, co, pares)


def tabla_estabilidad(S: pd.DataFrame, algoritmos: list[str], ks: list[int], cfg_cluster: dict,
                      n_remuestreos: int, fraccion: float, semilla: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ARI medio y Jaccard por algoritmo y k. Devuelve (resumen, jaccard por grupo)."""
    filas, jac = [], []
    for alg in algoritmos:
        for k in ks:
            r = estabilidad(S.to_numpy(), alg, k, cfg_cluster, n_remuestreos, fraccion, semilla)
            filas.append(r.resumen())
            jac.append(r.jaccard.assign(algoritmo=alg, k=k))
        log.info("Estabilidad: %s listo", alg)
    return pd.DataFrame(filas), pd.concat(jac, ignore_index=True)


def resumen_coasociacion(co: np.ndarray, pares: np.ndarray, etiquetas: np.ndarray,
                         inferior: float = 0.1, superior: float = 0.9) -> tuple[dict, pd.Series]:
    """PAC (proporción de pares ambiguos) y estabilidad individual de cada municipio.

    La estabilidad individual es la co-asociación media con los demás miembros de su grupo:
    1 = siempre queda con ellos; valores bajos = municipio frontera bajo remuestreo.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(pares > 0, co / pares, np.nan)
    iu = np.triu_indices_from(M, k=1)
    vals = M[iu]
    vals = vals[~np.isnan(vals)]
    pac = float(np.mean((vals > inferior) & (vals < superior)))
    indiv = np.empty(len(etiquetas))
    for i, c in enumerate(etiquetas):
        miembros = np.where(etiquetas == c)[0]
        miembros = miembros[miembros != i]
        indiv[i] = np.nanmean(M[i, miembros]) if len(miembros) else np.nan
    return {"pac": pac, "coasociacion_media_mismo_grupo": float(np.nanmean(indiv))}, pd.Series(indiv)


def referencia_nula_gaussiana(S: pd.DataFrame, algoritmo: str, k: int, cfg_cluster: dict, replicas: int,
                              semilla: int, n_remuestreos: int = 20, fraccion: float = 0.8) -> pd.DataFrame:
    """Simula datos sin grupos (una sola gaussiana con la misma covarianza que los datos) y mide el
    silhouette y la estabilidad que se obtendrían igual. Si los datos reales no superan claramente
    esta referencia, la partición describe un continuo, no grupos nítidos."""
    X = S.to_numpy()
    rng = np.random.default_rng(semilla)
    cov = np.cov(X, rowvar=False)
    filas = []
    for r in range(replicas):
        Y = rng.multivariate_normal(np.zeros(X.shape[1]), cov, size=len(X))
        lab = _ajustar_silencioso(algoritmo, Y, k, semilla + r, cfg_cluster)
        est = estabilidad(Y, algoritmo, k, cfg_cluster, n_remuestreos, fraccion, semilla + 100 + r, referencia=lab)
        filas.append({"replica": r, **metricas_internas(Y, lab), "ari_medio": float(est.ari.mean())})
    return pd.DataFrame(filas)


def estabilidad_por_tamano(indiv: pd.Series, frontera: pd.Series, poblacion: pd.Series) -> pd.DataFrame:
    """Estabilidad individual y % de municipios frontera por quintil de población."""
    from scipy.stats import kruskal

    q = pd.qcut(poblacion.reindex(indiv.index), 5, labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (mayor)"])
    df = pd.DataFrame({"estabilidad_individual": indiv, "frontera": frontera.reindex(indiv.index), "quintil": q})
    t = df.groupby("quintil", observed=True).agg(municipios=("frontera", "size"),
                                                 estabilidad_media=("estabilidad_individual", "mean"),
                                                 pct_frontera=("frontera", "mean"))
    t["pct_frontera"] = (t["pct_frontera"] * 100).round(1)
    grupos = [g["estabilidad_individual"].dropna().to_numpy() for _, g in df.groupby("quintil", observed=True)]
    t.attrs["kruskal_p"] = float(kruskal(*grupos).pvalue) if len(grupos) > 1 else float("nan")
    return t


def perturbacion_por_tamano(Z: pd.DataFrame, pesos: pd.Series, transformar_a_entrada, predecir,
                            base: np.ndarray, poblacion: pd.Series, columnas: list[str], nivel: float,
                            replicas: int, semilla: int) -> tuple[pd.DataFrame, pd.Series]:
    """Ruido de medición dependiente del tamaño: a las tasas con denominador pequeño se les suma
    ruido N(0, (nivel * sqrt(pob_mediana / pob))^2) en unidades estandarizadas y se reasigna con el
    modelo final ya ajustado (sin reentrenar). Devuelve la tasa de cambio de tipo por quintil y por
    municipio. Es un supuesto explícito, no una estimación del error real de cada indicador."""
    rng = np.random.default_rng(semilla)
    pob = poblacion.reindex(Z.index)
    escala = nivel * np.sqrt(pob.median() / pob.to_numpy())
    cols = [c for c in columnas if c in Z.columns]
    cambios = np.zeros(len(Z))
    for _ in range(replicas):
        Zp = Z.copy()
        ruido = rng.normal(size=(len(Z), len(cols))) * escala[:, None]
        Zp[cols] = Zp[cols].to_numpy() + ruido
        etiquetas = predecir(transformar_a_entrada(Zp * pesos.reindex(Zp.columns).fillna(1.0)))
        cambios += etiquetas != base
    tasa = pd.Series(cambios / replicas, index=Z.index, name="tasa_cambio_por_ruido")
    q = pd.qcut(pob, 5, labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (mayor)"]).rename("quintil")
    t = tasa.groupby(q, observed=True).agg(["mean", "max"]).rename(columns={"mean": "tasa_cambio_media", "max": "tasa_cambio_max"})
    t["pct_municipios_cambian_mas_10pct"] = tasa.groupby(q, observed=True).apply(lambda s: float((s > 0.10).mean() * 100))
    return t, tasa

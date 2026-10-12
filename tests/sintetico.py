"""Generador de datos sintéticos con el formato largo de TerriData y los códigos del catálogo real.

Cada indicador del catálogo recibe una señal latente: con `k` > 0 la señal depende de un grupo plantado;
con `k` = 0 es ruido puro. Los valores crudos se construyen para que, tras las fórmulas y transformaciones
del pipeline, el indicador conserve esa señal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tipologia.config import Config
from tipologia.features.prepare import DatosBase

ANIOS = list(range(2017, 2025))


def datos_sinteticos(cfg: Config, n: int = 240, k: int = 3, separacion: float = 3.0, semilla: int = 11) -> tuple[DatosBase, np.ndarray]:
    rng = np.random.default_rng(semilla)
    cods = [f"{d:02d}{i:03d}" for d in (5, 8, 13, 15) for i in range(1, n // 4 + 1)]
    grupos = np.repeat(np.arange(max(k, 1)), int(np.ceil(n / max(k, 1))))[:n]
    inds = [i for i in cfg.indicadores if "entrada" in i.uso or "temporal" in i.uso or "validacion" in i.uso]
    centros = rng.normal(scale=separacion, size=(max(k, 1), len(inds))) if k > 0 else np.zeros((1, len(inds)))
    latente = centros[grupos] + rng.normal(size=(n, len(inds)))
    pob_cod, area_cod = cfg.catalogo["poblacion_codigo"], cfg.catalogo["area_codigo"]
    j_pob = [i.id for i in inds].index("pob_log")
    poblacion = np.exp(9.5 + 0.8 * latente[:, j_pob])
    filas = []

    def agregar(codigo: str, valores: np.ndarray) -> None:
        for anio in ANIOS:
            v = valores * (1 + rng.normal(scale=0.01, size=n))
            filas.extend(zip(cods, [codigo] * n, [anio] * n, v, strict=False))

    agregar(pob_cod, poblacion)
    agregar(area_cod, np.full(n, 500.0))
    for j, ind in enumerate(inds):
        z = latente[:, j]
        objetivo = np.exp(1.0 + 0.5 * z) if ind.transformacion in ("log", "log1p") else 50 + 8 * z
        if ind.id == "pob_log":
            continue
        if ind.codigo:
            agregar(ind.codigo, objetivo)
            continue
        if ind.denominador == "poblacion":
            den = poblacion
        elif ind.denominador == "area":
            den = np.full(n, 500.0)
        else:
            den = np.full(n, 1000.0)
            agregar(ind.denominador, den)
        num_total = objetivo * den / ind.factor
        for c in ind.numerador:
            agregar(c, num_total / len(ind.numerador))
    largo = pd.DataFrame(filas, columns=["cod_ent", "cod_ind", "anio", "valor"])
    largo = largo.groupby(["cod_ent", "cod_ind", "anio"], as_index=False)["valor"].sum()
    mun = pd.DataFrame({"nombre": [f"Municipio {c}" for c in cods], "cod_dpto": [c[:2] for c in cods],
                        "dpto": [f"Depto {c[:2]}" for c in cods], "region": "Región", "tipologia_dnp_2026": "3",
                        "cat_ruralidad": "Rural", "capital": 0, "pdet": 0}, index=pd.Index(cods, name="cod"))
    return DatosBase(largo=largo, municipios=mun), grupos

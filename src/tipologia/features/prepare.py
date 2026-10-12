"""Orquestación de la preparación: matriz -> transformaciones -> imputación -> escalado -> ponderación."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from tipologia.config import Config
from tipologia.features.build import Matriz, cargar_largo, construir_matriz, municipios
from tipologia.features.impute import imputar, recortar_a_rango_observado
from tipologia.features.transform import (
    Escalado,
    Ponderacion,
    Winsorizacion,
    ajustar_escalado,
    ponderar,
    transformar,
    winsorizar,
)

log = logging.getLogger(__name__)


@dataclass
class DatosBase:
    """Datos de entrada cargados una sola vez (formato largo filtrado y tabla de municipios)."""

    largo: pd.DataFrame
    municipios: pd.DataFrame

    def poblacion(self, cfg: Config, anio: int) -> pd.Series:
        cod = cfg.catalogo["poblacion_codigo"]
        d = self.largo[(self.largo["cod_ind"] == cod) & (self.largo["anio"] == anio)]
        return d.set_index("cod_ent")["valor"]


def cargar_datos_base(cfg: Config) -> DatosBase:
    proc = cfg.ruta("processed")
    codigos = set().union(*(i.codigos_fuente for i in cfg.indicadores))
    codigos |= {cfg.catalogo["poblacion_codigo"], cfg.catalogo["area_codigo"]}
    largo = cargar_largo(proc / "terridata_long.parquet", codigos)
    dnp_path = proc / "dnp_tipologias.parquet"
    dnp = pd.read_parquet(dnp_path) if dnp_path.exists() else None
    mun = municipios(proc / "terridata_long.parquet", dnp)
    return DatosBase(largo=largo, municipios=mun)


@dataclass
class Preparado:
    """Todo lo necesario para clusterizar e interpretar un escenario."""

    anio: int
    municipios: pd.DataFrame
    valores: pd.DataFrame          # unidades originales, con faltantes (lo observado)
    Z: pd.DataFrame                # transformado, winsorizado, imputado y escalado (sin ponderar)
    X: pd.DataFrame                # matriz ponderada: entrada del PCA / clustering
    imputado: pd.DataFrame         # celdas imputadas (año vecino o imputación estadística)
    dominios: dict[str, str]       # dominio de cada columna de Z
    ponderacion: Ponderacion
    matriz: Matriz
    winsor: Winsorizacion
    escalado: Escalado
    parametros: dict = field(default_factory=dict)

    @property
    def n_imputados(self) -> pd.Series:
        return self.imputado.sum(axis=1)


def preparar(cfg: Config, datos: DatosBase, anio: int | None = None, uso: str = "entrada",
             quitar_dominios: tuple[str, ...] = (), excluir_codigos: list[str] | None = None,
             motivo_extra: str = "", indicadores_ids: list[str] | None = None,
             municipios_fijos: pd.Index | None = None) -> Preparado:
    """Prepara la matriz de un escenario. Los parámetros metodológicos salen de `cfg`."""
    anio = anio or cfg["datos"]["anio_referencia"]
    p = cfg["preparacion"]
    inds = [i for i in cfg.indicadores_por_uso(uso) if i.dominio not in quitar_dominios]
    if indicadores_ids is not None:
        inds = [i for i in inds if i.id in indicadores_ids]
    matriz = construir_matriz(cfg, anio, inds, datos.largo, datos.municipios, excluir_codigos, motivo_extra,
                              municipios_fijos)
    usados = matriz.indicadores
    valores = matriz.valores
    if p["imputacion"] == "casos_completos":
        # Caso completo = sin ningún faltante en el año de referencia (tampoco rellenos con año vecino).
        completos = ~matriz.faltante_ref.any(axis=1)
        valores = valores.loc[completos]
        log.info("Casos completos: %d de %d municipios", int(completos.sum()), len(completos))
    T = transformar(valores, usados)
    if p["winsorizar"]["activo"]:
        W, winsor = winsorizar(T, p["winsorizar"]["inferior"], p["winsorizar"]["superior"])
    else:
        W, winsor = T, Winsorizacion()
    imp, mascara = imputar(W, p["imputacion"], grupos=datos.municipios["cod_dpto"], semilla=cfg.semilla,
                         knn_vecinos=p["knn_vecinos"], max_iter=p["iterativa_max_iter"])
    imp = recortar_a_rango_observado(imp, W)
    escalado = ajustar_escalado(imp, p["escalado"])
    Z = escalado.aplicar(imp)
    dominios = {i.id: i.dominio for i in usados}
    X, pond = ponderar(Z, dominios, p["ponderacion"], p["pca_bloque_varianza"], cfg.semilla)
    idx = Z.index
    imputado = (matriz.imputado_vecino.loc[idx] | mascara.loc[idx]).astype(bool)
    return Preparado(
        anio=anio,
        municipios=datos.municipios.loc[idx],
        valores=matriz.valores.loc[idx],
        Z=Z,
        X=X,
        imputado=imputado,
        dominios=dominios,
        ponderacion=pond,
        matriz=matriz,
        winsor=winsor,
        escalado=escalado,
        parametros={"anio": anio, "uso": uso, "quitar_dominios": list(quitar_dominios),
                    "excluir_codigos": len(excluir_codigos or []), **{k: v for k, v in p.items()}},
    )

"""Carga de la configuración versionada (config/*.yaml)."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_DIR = RAIZ / "config"


@dataclass(frozen=True)
class Indicador:
    """Definición de un indicador del catálogo."""

    id: str
    nombre: str
    dominio: str
    anio: str
    uso: tuple[str, ...]
    unidad: str = ""
    codigo: str | None = None
    formula: str | None = None
    numerador: tuple[str, ...] = ()
    denominador: str | None = None
    factor: float = 1.0
    transformacion: str | None = None
    vecino: int = 0
    sensible_tamano: bool = False

    @property
    def codigos_fuente(self) -> set[str]:
        """Códigos de TerriData que necesita el indicador."""
        if self.codigo:
            return {self.codigo}
        cods = set(self.numerador)
        if self.denominador and self.denominador not in ("poblacion", "area"):
            cods.add(self.denominador)
        return cods


@dataclass
class Config:
    """Configuración completa: parámetros generales + catálogo de indicadores."""

    general: dict[str, Any]
    catalogo: dict[str, Any]
    indicadores: list[Indicador] = field(default_factory=list)

    def __getitem__(self, clave: str) -> Any:
        return self.general[clave]

    @property
    def semilla(self) -> int:
        return int(self.general["proyecto"]["semilla"])

    def ruta(self, nombre: str) -> Path:
        return RAIZ / self.general["rutas"][nombre]

    def indicadores_por_uso(self, uso: str) -> list[Indicador]:
        return [i for i in self.indicadores if uso in i.uso]

    @property
    def dominios(self) -> dict[str, str]:
        return dict(self.catalogo["dominios"])

    def con_cambios(self, cambios: dict[str, Any]) -> Config:
        """Copia de la configuración con claves anidadas sobrescritas ("seccion.clave": valor)."""
        nuevo = copy.deepcopy(self.general)
        for ruta, valor in cambios.items():
            nodo = nuevo
            partes = ruta.split(".")
            for p in partes[:-1]:
                nodo = nodo[p]
            nodo[partes[-1]] = valor
        return Config(general=nuevo, catalogo=self.catalogo, indicadores=list(self.indicadores))

    def huella(self) -> str:
        """Hash estable de la configuración (general + catálogo)."""
        texto = json.dumps({"g": self.general, "c": self.catalogo}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _a_indicador(d: dict[str, Any]) -> Indicador:
    return Indicador(
        id=d["id"],
        nombre=d["nombre"],
        dominio=d["dominio"],
        anio=str(d["anio"]),
        uso=tuple(d.get("uso", ["entrada"])),
        unidad=str(d.get("unidad", "")),
        codigo=d.get("codigo"),
        formula=d.get("formula"),
        numerador=tuple(d.get("numerador", ())),
        denominador=d.get("denominador"),
        factor=float(d.get("factor", 1.0)),
        transformacion=d.get("transformacion"),
        vecino=int(d.get("vecino", 0)),
        sensible_tamano=bool(d.get("sensible_tamano", False)),
    )


def cargar_config(general: Path | None = None, catalogo: Path | None = None) -> Config:
    """Lee config.yaml e indicadores.yaml y valida el catálogo."""
    general = general or CONFIG_DIR / "config.yaml"
    catalogo = catalogo or CONFIG_DIR / "indicadores.yaml"
    with open(general, encoding="utf-8") as f:
        g = yaml.safe_load(f)
    with open(catalogo, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    indicadores = [_a_indicador(d) for d in c["indicadores"]]
    ids = [i.id for i in indicadores]
    if len(ids) != len(set(ids)):
        raise ValueError("Hay ids de indicador repetidos en el catálogo")
    for ind in indicadores:
        if ind.dominio not in c["dominios"]:
            raise ValueError(f"{ind.id}: dominio desconocido {ind.dominio}")
        if not ind.codigo and ind.formula != "razon":
            raise ValueError(f"{ind.id}: necesita 'codigo' o formula 'razon'")
    return Config(general=g, catalogo=c, indicadores=indicadores)

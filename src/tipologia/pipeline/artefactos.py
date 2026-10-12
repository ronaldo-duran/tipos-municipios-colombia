"""Directorios de ejecución versionados (outputs/<fecha>_<hash>/), metadata e idempotencia."""

from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from tipologia import __version__
from tipologia.config import RAIZ, Config
from tipologia.pipeline.io import guardar_json, leer_json

log = logging.getLogger(__name__)

ETAPAS_EJECUCION = ["features", "cluster", "stability", "profile"]
NO_DETERMINISTAS = {"metadata.json", "ejecucion.json", "etapas.json"}
# Fuentes que definen el resultado. El MGN solo aporta la vecindad (que se guarda en vecinos.csv),
# así que no entra a la huella: la ejecución es reproducible aunque el MGN no esté disponible.
FUENTES_HUELLA = ("terridata", "divipola", "dnp_tipologias")


def huella_codigo() -> str:
    """SHA-256 del código fuente del paquete (normalizando saltos de línea)."""
    h = hashlib.sha256()
    for p in sorted((RAIZ / "src" / "tipologia").rglob("*.py")):
        h.update(p.relative_to(RAIZ).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def commit_git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifiestos_fuentes(cfg: Config) -> dict[str, dict]:
    out = {}
    for nombre in ("terridata", "divipola", "dnp_tipologias", "geo_app", "mgn"):
        p = cfg.ruta("raw") / nombre / "manifest.json"
        if p.exists():
            out[nombre] = leer_json(p)
    return out


def huella_ejecucion(cfg: Config) -> str:
    """Hash de datos de entrada + configuración + código: define el directorio de la ejecución."""
    man = manifiestos_fuentes(cfg)
    faltan = [f for f in FUENTES_HUELLA if f not in man]
    if faltan:
        raise FileNotFoundError(f"Faltan fuentes ingeridas: {faltan}. Corre primero la etapa ingest.")
    partes = [f"{f}:{man[f]['sha256']}" for f in FUENTES_HUELLA] + [f"config:{cfg.huella()}", f"codigo:{huella_codigo()}"]
    return hashlib.sha256("|".join(partes).encode()).hexdigest()


def nombre_ejecucion(cfg: Config) -> str:
    """<fecha de la versión de TerriData>_<hash de 8 caracteres>."""
    man = manifiestos_fuentes(cfg)
    return f"{man['terridata']['version']}_{huella_ejecucion(cfg)[:8]}"


def dir_ejecucion(cfg: Config) -> Path:
    d = cfg.ruta("outputs") / nombre_ejecucion(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def etapas_completadas(destino: Path) -> dict:
    p = destino / "etapas.json"
    return leer_json(p) if p.exists() else {}


def marcar_etapa(destino: Path, etapa: str) -> None:
    reg = etapas_completadas(destino)
    reg[etapa] = {"completada": True, "fecha": datetime.now(UTC).isoformat(timespec="seconds")}
    guardar_json(reg, destino / "etapas.json")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checksums(destino: Path) -> dict[str, str]:
    """Checksums de los artefactos deterministas (permiten verificar que una reejecución da lo mismo)."""
    return {p.relative_to(destino).as_posix(): sha256(p) for p in sorted(destino.rglob("*"))
            if p.is_file() and p.name not in NO_DETERMINISTAS}


def escribir_metadata(cfg: Config, destino: Path) -> dict:
    """metadata.json: año, indicadores, parámetros, semillas, versiones de datos y de código, checksums."""
    meta = {
        "proyecto": cfg["proyecto"]["nombre"],
        "ejecucion": destino.name,
        "anio_referencia": cfg["datos"]["anio_referencia"],
        "semilla": cfg.semilla,
        "indicadores": [{"id": i.id, "nombre": i.nombre, "dominio": i.dominio, "uso": list(i.uso), "anio": i.anio}
                        for i in cfg.indicadores],
        "parametros": {k: cfg[k] for k in ("datos", "preparacion", "clustering", "estabilidad", "sensibilidad",
                                           "interpretacion", "espacial", "temporal", "umap")},
        "fuentes": manifiestos_fuentes(cfg),
        "codigo": {"version_paquete": __version__, "huella_codigo": huella_codigo(), "commit_git": commit_git()},
        "huella_ejecucion": huella_ejecucion(cfg),
        "entorno": {"python": platform.python_version(), "plataforma": platform.platform()},
        "artefactos_sha256": checksums(destino),
    }
    guardar_json(meta, destino / "metadata.json")
    return meta


def actualizar_latest(cfg: Config, nombre: str) -> None:
    (cfg.ruta("outputs") / "LATEST").write_text(nombre + "\n", encoding="utf-8", newline="\n")


def ejecucion_actual(outputs: Path) -> Path | None:
    p = outputs / "LATEST"
    if not p.exists():
        return None
    d = outputs / p.read_text(encoding="utf-8").strip()
    return d if d.exists() else None


def podar(cfg: Config, conservar: str) -> list[str]:
    """Elimina ejecuciones anteriores (solo directorios con formato <fecha>_<hash>)."""
    borradas = []
    for d in cfg.ruta("outputs").iterdir():
        if d.is_dir() and d.name != conservar and len(d.name.split("_")) == 2 and len(d.name.split("_")[1]) == 8:
            shutil.rmtree(d)
            borradas.append(d.name)
    return borradas

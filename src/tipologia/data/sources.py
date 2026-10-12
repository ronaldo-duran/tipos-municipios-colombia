"""Fuentes de datos con caché local, versión y checksum.

Cada fuente implementa la interfaz `DataSource`:
- `version_remota()` consulta la versión publicada sin descargar (cabeceras HTTP o metadatos de la API).
- `obtener()` devuelve la ruta local del archivo, descargándolo solo si no hay caché o hay versión nueva.
- `manifiesto()` devuelve fecha, tamaño y checksum de lo que hay en caché.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import truststore

# Usa el almacén de certificados del sistema: algunos servidores públicos (p. ej. colaboracion.dnp.gov.co)
# no envían el certificado intermedio y la verificación con certifi falla. Nunca se desactiva la verificación.
truststore.inject_into_ssl()

log = logging.getLogger(__name__)

TIMEOUT = 60
USER_AGENT = "tipologia-municipios/0.1 (proyecto academico)"


def sha256_archivo(ruta: Path, bloque: int = 8 * 1024 * 1024) -> str:
    """Checksum SHA-256 de un archivo, leído por bloques."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class VersionRemota:
    """Lo que la fuente publica sobre su versión actual."""

    identificador: str            # fecha de última modificación (ISO) o equivalente
    tamano: int | None = None
    etag: str | None = None


@dataclass
class Manifiesto:
    """Registro de lo descargado en caché."""

    fuente: str
    url: str
    archivo: str
    version: str
    tamano: int
    sha256: str
    etag: str | None
    descargado_en: str

    def guardar(self, ruta: Path) -> None:
        ruta.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def leer(cls, ruta: Path) -> Manifiesto | None:
        if not ruta.exists():
            return None
        return cls(**json.loads(ruta.read_text(encoding="utf-8")))


class DataSource(ABC):
    """Interfaz común para todas las fuentes."""

    nombre: str

    @abstractmethod
    def version_remota(self) -> VersionRemota | None:
        """Versión publicada (None si no se puede consultar)."""

    @abstractmethod
    def obtener(self, forzar: bool = False, offline: bool = False) -> Path:
        """Ruta local del archivo de la fuente."""

    @abstractmethod
    def manifiesto(self) -> Manifiesto | None:
        """Manifiesto de la copia en caché."""

    def hay_version_nueva(self) -> bool | None:
        """True si la versión remota difiere de la caché; None si no se pudo consultar."""
        local = self.manifiesto()
        remota = self.version_remota()
        if remota is None:
            return None
        if local is None:
            return True
        if remota.etag and local.etag:
            return remota.etag != local.etag
        cambio_tamano = remota.tamano is not None and remota.tamano != local.tamano
        return remota.identificador != local.version or cambio_tamano


class HttpFileSource(DataSource):
    """Archivo publicado por HTTP, cacheado en `<raw>/<nombre>/<version>/<archivo>`."""

    def __init__(self, nombre: str, url: str, raw_dir: Path, archivo: str | None = None):
        self.nombre = nombre
        self.url = url
        self.dir = Path(raw_dir) / nombre
        self.archivo = archivo or url.rsplit("/", 1)[-1].split("?")[0]
        self.ruta_manifiesto = self.dir / "manifest.json"

    def manifiesto(self) -> Manifiesto | None:
        return Manifiesto.leer(self.ruta_manifiesto)

    def version_remota(self) -> VersionRemota | None:
        try:
            r = requests.head(self.url, allow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("%s: no se pudo consultar la versión remota (%s)", self.nombre, e)
            return None
        lm = r.headers.get("Last-Modified")
        ident = parsedate_to_datetime(lm).astimezone(UTC).strftime("%Y-%m-%d") if lm else "sin_fecha"
        tam = int(r.headers["Content-Length"]) if r.headers.get("Content-Length") else None
        return VersionRemota(identificador=ident, tamano=tam, etag=r.headers.get("ETag"))

    def _adoptar_existente(self, destino: Path, tamano: int) -> None:
        """Si ya hay una copia del mismo tamaño en el directorio de la fuente (descarga manual), la reubica."""
        for cand in self.dir.rglob(self.archivo):
            if cand != destino and cand.is_file() and cand.stat().st_size == tamano:
                destino.parent.mkdir(parents=True, exist_ok=True)
                cand.replace(destino)
                log.info("%s: reutilizo copia existente %s", self.nombre, cand)
                return

    def _descargar(self, destino: Path) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(destino.suffix + ".part")
        log.info("%s: descargando %s", self.nombre, self.url)
        with requests.get(self.url, stream=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
        tmp.replace(destino)

    def obtener(self, forzar: bool = False, offline: bool = False) -> Path:
        local = self.manifiesto()
        if local and (self.dir / local.version / local.archivo).exists() and not forzar:
            if offline:
                return self.dir / local.version / local.archivo
            nueva = self.hay_version_nueva()
            if not nueva:
                if nueva is None:
                    log.warning("%s: sin conexión, uso la caché %s", self.nombre, local.version)
                else:
                    log.info("%s: sin cambios (versión %s), uso la caché", self.nombre, local.version)
                return self.dir / local.version / local.archivo
            log.info("%s: hay versión nueva publicada", self.nombre)
        if offline:
            raise FileNotFoundError(f"{self.nombre}: no hay caché y se pidió modo offline")
        remota = self.version_remota()
        if remota is None and not forzar:
            existente = next((c for c in self.dir.rglob(self.archivo) if c.is_file()), None)
            if existente is not None:
                log.warning("%s: fuente remota no disponible, adopto la copia local %s", self.nombre, existente)
                remota = VersionRemota(
                    identificador=datetime.fromtimestamp(existente.stat().st_mtime, UTC).strftime("%Y-%m-%d"),
                    tamano=existente.stat().st_size)
        version = remota.identificador if remota else datetime.now(UTC).strftime("%Y-%m-%d")
        destino = self.dir / version / self.archivo
        if not destino.exists() and not forzar and remota and remota.tamano:
            self._adoptar_existente(destino, remota.tamano)
        if not destino.exists() or forzar or (remota and remota.tamano and destino.stat().st_size != remota.tamano):
            self._descargar(destino)
        else:
            log.info("%s: el archivo ya estaba en %s, solo registro el manifiesto", self.nombre, destino)
        man = Manifiesto(
            fuente=self.nombre,
            url=self.url,
            archivo=self.archivo,
            version=version,
            tamano=destino.stat().st_size,
            sha256=sha256_archivo(destino),
            etag=remota.etag if remota else None,
            descargado_en=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        man.guardar(self.ruta_manifiesto)
        return destino


class TerriDataPortalSource(HttpFileSource):
    """Descarga completa de TerriData (DNP) desde el portal terridata.dnp.gov.co."""

    def __init__(self, url: str, raw_dir: Path):
        super().__init__("terridata", url, raw_dir)


class SocrataSource(DataSource):
    """Dataset de la API Socrata de datos.gov.co, con paginación y caché en CSV."""

    def __init__(self, nombre: str, dominio: str, dataset: str, raw_dir: Path, pagina: int = 50000):
        self.nombre = nombre
        self.dominio = dominio
        self.dataset = dataset
        self.pagina = pagina
        self.dir = Path(raw_dir) / nombre
        self.ruta_manifiesto = self.dir / "manifest.json"

    @property
    def url(self) -> str:
        return f"https://{self.dominio}/resource/{self.dataset}.json"

    def manifiesto(self) -> Manifiesto | None:
        return Manifiesto.leer(self.ruta_manifiesto)

    def version_remota(self) -> VersionRemota | None:
        try:
            r = requests.get(f"https://{self.dominio}/api/views/{self.dataset}.json", timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            meta = r.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("%s: no se pudo consultar metadatos (%s)", self.nombre, e)
            return None
        ts = meta.get("rowsUpdatedAt") or meta.get("viewLastModified")
        ident = datetime.fromtimestamp(int(ts), UTC).strftime("%Y-%m-%d") if ts else "sin_fecha"
        return VersionRemota(identificador=ident)

    def descargar_registros(self) -> list[dict[str, Any]]:
        """Descarga todos los registros paginando con $limit/$offset ordenados por :id."""
        registros: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = {"$limit": self.pagina, "$offset": offset, "$order": ":id"}
            r = requests.get(self.url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            lote = r.json()
            registros.extend(lote)
            log.info("%s: %d registros (offset %d)", self.nombre, len(lote), offset)
            if len(lote) < self.pagina:
                break
            offset += self.pagina
            time.sleep(0.2)
        return registros

    def obtener(self, forzar: bool = False, offline: bool = False) -> Path:
        local = self.manifiesto()
        if local and (self.dir / local.version / local.archivo).exists() and not forzar:
            if offline or not self.hay_version_nueva():
                return self.dir / local.version / local.archivo
        if offline:
            raise FileNotFoundError(f"{self.nombre}: no hay caché y se pidió modo offline")
        remota = self.version_remota()
        version = remota.identificador if remota else datetime.now(UTC).strftime("%Y-%m-%d")
        destino = self.dir / version / f"{self.dataset}.csv"
        destino.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.descargar_registros()).to_csv(destino, index=False)
        man = Manifiesto(
            fuente=self.nombre, url=self.url, archivo=destino.name, version=version,
            tamano=destino.stat().st_size, sha256=sha256_archivo(destino), etag=None,
            descargado_en=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        man.guardar(self.ruta_manifiesto)
        return destino


class LocalFileSource(DataSource):
    """Archivo descargado a mano y dejado en data/incoming/ (por ejemplo, si el portal falla)."""

    def __init__(self, nombre: str, ruta: Path):
        self.nombre = nombre
        self.ruta = Path(ruta)

    def version_remota(self) -> VersionRemota | None:
        if not self.ruta.exists():
            return None
        st = self.ruta.stat()
        return VersionRemota(identificador=datetime.fromtimestamp(st.st_mtime, UTC).strftime("%Y-%m-%d"),
                             tamano=st.st_size)

    def manifiesto(self) -> Manifiesto | None:
        v = self.version_remota()
        if v is None:
            return None
        return Manifiesto(fuente=self.nombre, url=self.ruta.as_uri(), archivo=self.ruta.name, version=v.identificador,
                          tamano=v.tamano or 0, sha256=sha256_archivo(self.ruta), etag=None,
                          descargado_en=v.identificador)

    def obtener(self, forzar: bool = False, offline: bool = False) -> Path:
        if not self.ruta.exists():
            raise FileNotFoundError(self.ruta)
        return self.ruta

    def copiar_a(self, destino: Path) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.ruta, destino)
        return destino


class DANEMicrodatosSource(DataSource):
    """Esqueleto: indicadores adicionales del DANE (por ejemplo, CNPV 2018 a nivel municipal). Sin implementar."""

    nombre = "dane_microdatos"

    def version_remota(self) -> VersionRemota | None:
        raise NotImplementedError("Fuente pendiente: requiere aprobación antes de incorporarse")

    def obtener(self, forzar: bool = False, offline: bool = False) -> Path:
        raise NotImplementedError("Fuente pendiente: requiere aprobación antes de incorporarse")

    def manifiesto(self) -> Manifiesto | None:
        return None


class ICFESSource(DataSource):
    """Esqueleto: resultados Saber 11 posteriores a 2022 desde el ICFES. Sin implementar."""

    nombre = "icfes"

    def version_remota(self) -> VersionRemota | None:
        raise NotImplementedError("Fuente pendiente")

    def obtener(self, forzar: bool = False, offline: bool = False) -> Path:
        raise NotImplementedError("Fuente pendiente")

    def manifiesto(self) -> Manifiesto | None:
        return None

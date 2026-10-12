from tipologia.data.sources import HttpFileSource, Manifiesto, VersionRemota, sha256_archivo


class FuenteFalsa(HttpFileSource):
    """HttpFileSource sin red: la versión remota y la descarga se simulan."""

    def __init__(self, raw_dir, version, contenido=b"datos"):
        super().__init__("falsa", "https://ejemplo.invalid/archivo.zip", raw_dir)
        self._version = version
        self._contenido = contenido
        self.descargas = 0

    def version_remota(self):
        if self._version is None:
            return None
        return VersionRemota(identificador=self._version, tamano=len(self._contenido))

    def _descargar(self, destino):
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(self._contenido)
        self.descargas += 1


def test_descarga_una_vez_y_registra_checksum(tmp_path):
    f = FuenteFalsa(tmp_path, "2026-08-07")
    ruta = f.obtener()
    assert ruta.read_bytes() == b"datos"
    man = f.manifiesto()
    assert man.version == "2026-08-07" and man.sha256 == sha256_archivo(ruta)
    f.obtener()
    assert f.descargas == 1  # sin versión nueva no vuelve a descargar


def test_detecta_version_nueva(tmp_path):
    f = FuenteFalsa(tmp_path, "2026-08-07")
    f.obtener()
    assert f.hay_version_nueva() is False
    f._version, f._contenido = "2026-11-01", b"datos nuevos"
    assert f.hay_version_nueva() is True
    ruta = f.obtener()
    assert "2026-11-01" in ruta.as_posix() and f.descargas == 2


def test_sin_conexion_usa_cache(tmp_path):
    f = FuenteFalsa(tmp_path, "2026-08-07")
    ruta = f.obtener()
    f._version = None
    assert f.hay_version_nueva() is None
    assert f.obtener() == ruta and f.descargas == 1


def test_manifiesto_ida_y_vuelta(tmp_path):
    m = Manifiesto("x", "u", "a.zip", "2026-01-01", 3, "abc", None, "2026-01-01T00:00:00")
    p = tmp_path / "m.json"
    m.guardar(p)
    assert Manifiesto.leer(p) == m

import json

import pytest

from tipologia.config import cargar_config
from tipologia.pipeline import artefactos


@pytest.fixture
def cfg(tmp_path):
    raw = tmp_path / "raw"
    for nombre, sha in [("terridata", "a" * 64), ("divipola", "b" * 64), ("dnp_tipologias", "c" * 64)]:
        (raw / nombre).mkdir(parents=True)
        (raw / nombre / "manifest.json").write_text(json.dumps({"version": "2026-08-07", "sha256": sha}), encoding="utf-8")
    return cargar_config().con_cambios({"rutas.raw": str(raw), "rutas.outputs": str(tmp_path / "outputs")})


def test_nombre_de_ejecucion_estable_y_sensible_a_cambios(cfg):
    n1 = artefactos.nombre_ejecucion(cfg)
    assert n1 == artefactos.nombre_ejecucion(cfg)
    assert n1.startswith("2026-08-07_") and len(n1.split("_")[1]) == 8
    otro = cfg.con_cambios({"datos.max_faltantes_municipio": 0.2})
    assert artefactos.nombre_ejecucion(otro) != n1


def test_cambio_de_datos_cambia_la_ejecucion(cfg):
    n1 = artefactos.nombre_ejecucion(cfg)
    man = cfg.ruta("raw") / "terridata" / "manifest.json"
    man.write_text(json.dumps({"version": "2026-11-01", "sha256": "d" * 64}), encoding="utf-8")
    assert artefactos.nombre_ejecucion(cfg) != n1


def test_faltan_fuentes(tmp_path):
    c = cargar_config().con_cambios({"rutas.raw": str(tmp_path)})
    with pytest.raises(FileNotFoundError):
        artefactos.huella_ejecucion(c)


def test_checksums_excluyen_archivos_no_deterministas_y_poda(cfg):
    d = artefactos.dir_ejecucion(cfg)
    (d / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (d / "etapas.json").write_text("{}", encoding="utf-8")
    sums = artefactos.checksums(d)
    assert list(sums) == ["a.csv"]
    viejo = cfg.ruta("outputs") / "2026-01-01_abcdef12"
    viejo.mkdir()
    otro = cfg.ruta("outputs") / "notas"
    otro.mkdir()
    assert artefactos.podar(cfg, d.name) == ["2026-01-01_abcdef12"]
    assert otro.exists() and d.exists()
    artefactos.actualizar_latest(cfg, d.name)
    assert artefactos.ejecucion_actual(cfg.ruta("outputs")) == d

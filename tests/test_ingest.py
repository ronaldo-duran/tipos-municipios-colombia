import duckdb

from tipologia.data.ingest import consulta_limpieza, leer_csv_sql, resumen_terridata

CABECERA = ("Código Departamento|Departamento|Código Entidad|Entidad|Dimensión|Subcategoría|Indicador|"
            "Código Indicador|Dato Numérico|Dato Cualitativo|Año|Mes|Fuente|Unidad de Medida")


def _fila(ent, ind, valor, anio="2023", cual=""):
    return f"05|Antioquia|{ent}|Mun|Dim|Sub|Ind {ind}|{ind}|{valor}|{cual}|{anio}|12|DANE|Porcentaje"


def _escribir(tmp_path, filas):
    p = tmp_path / "TerriData.txt"
    p.write_text("\r\n".join([CABECERA, *filas]) + "\r\n", encoding="utf-8", newline="")
    return p


def test_deduplica_filas_identicas_y_cuenta(tmp_path):
    filas = [
        _fila("05001", "030010009", "2,02", "2018"),
        _fila("05001", "030010009", "2,02", "2018"),   # repetida idéntica (como en déficit de vivienda)
        _fila("05002", "030010009", "1,41", "2018"),
        _fila("05002", "030010009", "1,41", "2018"),
        _fila("05002", "030010009", "1,41", "2018"),
    ]
    txt = _escribir(tmp_path, filas)
    con = duckdb.connect()
    df = con.execute(consulta_limpieza(leer_csv_sql(txt))).df()
    assert len(df) == 2
    assert sorted(df["valor"].tolist()) == [1.41, 2.02]


def test_conserva_valores_distintos_y_los_reporta_como_conflicto(tmp_path):
    filas = [_fila("05001", "X1", "10,0"), _fila("05001", "X1", "11,0"), _fila("05002", "X1", "5,0")]
    txt = _escribir(tmp_path, filas)
    out = tmp_path / "out.parquet"
    con = duckdb.connect()
    con.execute(f"COPY ({consulta_limpieza(leer_csv_sql(txt))}) TO '{out.as_posix()}' (FORMAT parquet)")
    res = resumen_terridata(con, out)
    assert res["filas"] == 3
    assert res["grupos_en_conflicto"] == 1


def test_interpreta_miles_con_punto_y_decimal_con_coma(tmp_path):
    filas = [
        _fila("05001", "A", "1.140.972,25"),
        _fila("05002", "A", "-62.808,63"),
        _fila("05004", "A", "0,01"),
        _fila("05021", "A", "", cual="preliminar"),
        _fila("5030", "A", "7"),
    ]
    txt = _escribir(tmp_path, filas)
    df = duckdb.connect().execute(consulta_limpieza(leer_csv_sql(txt))).df().set_index("cod_ent")
    assert df.loc["05001", "valor"] == 1140972.25
    assert df.loc["05002", "valor"] == -62808.63
    assert df.loc["05004", "valor"] == 0.01
    assert df["valor"].isna().sum() == 1 and df.loc["05021", "dato_cual"] == "preliminar"
    assert "05030" in df.index  # completa ceros a la izquierda

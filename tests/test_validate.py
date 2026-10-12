import numpy as np
import pandas as pd

from tipologia.data.validate import Reporte, validar_llaves, validar_matriz_final, validar_rangos


def test_validar_llaves_detecta_formato_y_repetidos():
    rep = Reporte()
    validar_llaves(pd.Series(["05001", "05001", "5002", None]), "prueba", rep)
    estados = {c.nombre: c.estado for c in rep.chequeos}
    assert estados["prueba: formato DIVIPOLA"] == "error"
    assert estados["prueba: códigos únicos"] == "error"


def test_validar_rangos_reporta_sin_modificar():
    wide = pd.DataFrame({"ipm": [10.0, 120.0, np.nan], "otra": [1.0, 2.0, 3.0]}, index=["05001", "05002", "05004"])
    rep = Reporte()
    fuera = validar_rangos(wide, {"ipm": "%", "otra": "puntos"}, rep)
    assert fuera == {"ipm": ["05002"]}
    assert wide.loc["05002", "ipm"] == 120.0


def test_matriz_final_sin_nan_ni_inf():
    rep = Reporte()
    validar_matriz_final(pd.DataFrame({"a": [1.0, np.inf], "b": [np.nan, 1.0]}), rep)
    assert len(rep.errores) == 2

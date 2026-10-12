"""Modelo sustituto para explicar el clustering (explicativo del agrupamiento, no causal)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

log = logging.getLogger(__name__)


def arbol_sustituto(Z: pd.DataFrame, tipos: pd.Series, profundidad: int, folds: int, semilla: int,
                    nombres: dict[str, str]) -> dict:
    """Árbol de decisión poco profundo: reglas legibles que aproximan la partición."""
    y = tipos.reindex(Z.index).to_numpy()
    cv = StratifiedKFold(folds, shuffle=True, random_state=semilla)
    arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=semilla)
    acc = cross_val_score(arbol, Z, y, cv=cv, scoring="accuracy")
    arbol.fit(Z, y)
    reglas = export_text(arbol, feature_names=[nombres.get(c, c) for c in Z.columns], decimals=2)
    imp = pd.Series(arbol.feature_importances_, index=Z.columns).sort_values(ascending=False)
    return {"exactitud_cv_media": float(acc.mean()), "exactitud_cv_sd": float(acc.std(ddof=0)),
            "reglas": reglas, "importancias": imp}


def bosque_sustituto(Z: pd.DataFrame, tipos: pd.Series, dominios: dict[str, str], n_arboles: int, folds: int,
                     semilla: int) -> tuple[dict, pd.DataFrame]:
    """Bosque aleatorio + importancia por permutación medida en los pliegues de validación."""
    y = tipos.reindex(Z.index).to_numpy()
    cv = StratifiedKFold(folds, shuffle=True, random_state=semilla)
    importancias, accs = [], []
    for f, (tr, te) in enumerate(cv.split(Z, y)):
        rf = RandomForestClassifier(n_estimators=n_arboles, random_state=semilla, n_jobs=1, min_samples_leaf=2)
        rf.fit(Z.iloc[tr], y[tr])
        accs.append(rf.score(Z.iloc[te], y[te]))
        pi = permutation_importance(rf, Z.iloc[te], y[te], n_repeats=10, random_state=semilla + f, n_jobs=1)
        importancias.append(pi.importances_mean)
    imp = pd.DataFrame({"indicador": Z.columns, "importancia_permutacion": np.mean(importancias, axis=0),
                        "sd_entre_pliegues": np.std(importancias, axis=0, ddof=0)})
    imp["dominio"] = imp["indicador"].map(dominios)
    imp = imp.sort_values("importancia_permutacion", ascending=False).reset_index(drop=True)
    return {"exactitud_cv_media": float(np.mean(accs)), "exactitud_cv_sd": float(np.std(accs, ddof=0))}, imp

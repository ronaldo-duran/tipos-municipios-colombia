"""Portada de la app. Sirve artefactos ya calculados en outputs/ (no recalcula nada al cargar)."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import lib  # noqa: E402

lib.encabezado("¿Cuántos tipos de municipio hay en Colombia?")
est = lib.js("estabilidad_resumen.json")
clu = lib.js("cluster_resumen.json")
feat = lib.js("features_resumen.json")
nom = lib.nombres_tipos()
tam = lib.csv("tamanos.csv").set_index("tipo")
colores = lib.colores_tipos()

k = est["k"]
alg = est["modelo_final"]
obs = est["observado_modelo_final"]
p95 = est["referencia_nula_p95"][alg]
sil = obs["silhouette"]
debil = sil < 0.25
supera_nula = obs["ari_medio"] > p95["ari_medio"] and sil > p95["silhouette"]
n_mun = f"{feat['municipios_incluidos']:,}".replace(",", ".")
frase_nula = ("Se reproduce bajo remuestreo y supera lo que darían datos sin grupos con la misma correlación"
              if supera_nula else "Se reproduce bajo remuestreo, pero no supera con claridad a datos sin grupos")
frase_sep = "la separación entre tipos es débil" if debil else "la separación entre tipos es moderada"
continuo = ("Colombia no se divide en grupos nítidos: hay un gradiente principal (urbano, con servicios y economía "
            "formal, frente a rural, con más pobreza y conflicto) y los tipos son zonas de ese continuo.") if debil else ""

st.markdown(f"""
### Respuesta corta

Con {feat['indicadores']} indicadores socioeconómicos de {n_mun} municipios (año {lib.meta()['anio_referencia']}),
la partición más defendible tiene **{k} tipos**. {frase_nula}, pero **{frase_sep}** (silhouette
{str(round(sil, 2)).replace('.', ',')}). {continuo}
""")

cols = st.columns(len(nom))
for c, (t, fila) in zip(cols, nom.iterrows(), strict=True):
    with c:
        st.markdown(f"<div style='border-left:6px solid {colores[t]};padding:4px 0 4px 10px'><b>Tipo {t}</b><br>{fila['nombre']}</div>",
                    unsafe_allow_html=True)
        st.caption(f"{int(tam.loc[t, 'municipios'])} municipios · {tam.loc[t, 'pct_poblacion']:.0f} % de la población")

asig = lib.asignaciones()
base = asig[["municipio"]].assign(tipo=asig["tipo"].map(lib.etiqueta))
fig = lib.mapa(base, "tipo", discreto={lib.etiqueta(t): colores[t] for t in nom.index},
               orden=[lib.etiqueta(t) for t in sorted(nom.index)], hover_name="municipio", alto=600)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Cómo leer esta app

- **Mapa**: el tipo de cada municipio, la probabilidad de pertenencia, su perfil frente a la mediana de su tipo y los
  municipios más parecidos.
- **Los tipos**: qué caracteriza a cada tipo, dónde están y qué indicadores los separan.
- **Robustez**: por qué hay {k} tipos, qué tan estables son y qué decisiones los cambian. Aquí se dice qué es sólido y qué no.
- **Comparación y método**: contraste con la tipología oficial del DNP y las categorías de ruralidad, análisis
  espacial, cambio entre 2019 y 2023, método y limitaciones.

### Límites que conviene tener presentes

- Describe **territorios, no personas** (falacia ecológica) y **no es causal**: no explica por qué un municipio es
  como es ni recomienda políticas.
- **No es un ranking.** Las letras siguen el tamaño del grupo y los nombres describen rasgos, no méritos.
- Pobreza, NBI, vivienda y servicios básicos vienen del **censo 2018**; el resto, de registros administrativos de
  {anio}. Los datos de municipios pequeños o de difícil acceso pueden ser menos confiables.
""".format(k=k, anio=lib.meta()["anio_referencia"]))
lib.descarga_asignaciones("descarga_inicio")
lib.pie()

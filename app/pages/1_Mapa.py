import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

lib.encabezado("Mapa de tipos de municipio")
asig = lib.asignaciones()
nom = lib.nombres_tipos()
ind = lib.indicadores()
colores = lib.colores_tipos()
geo = lib.geojson()

# Todos los municipios con geometría; los que no tienen asignación se muestran como "Sin dato".
cods_geo = [f["properties"]["cod"] for f in geo["features"]]
nombres_geo = {f["properties"]["cod"]: f"{f['properties']['nombre']} ({f['properties']['dpto']})" for f in geo["features"]}
base = pd.DataFrame(index=pd.Index(cods_geo, name="cod"))
base = base.join(asig[["municipio", "departamento", "tipo", "prob_max", "frontera", "poblacion"]])
base["etiqueta"] = base["tipo"].map(lib.etiqueta).fillna(lib.SIN_DATO)
base["nombre_mapa"] = base["municipio"].fillna(pd.Series(nombres_geo))

col_mapa, col_info = st.columns([3, 2], gap="large")
with col_mapa:
    modo = st.radio("Colorear por", ["Tipo", "Confianza de la asignación"], horizontal=True)
    if modo == "Tipo":
        orden = [lib.etiqueta(t) for t in sorted(nom.index)] + [lib.SIN_DATO]
        fig = lib.mapa(base, "etiqueta", discreto={lib.etiqueta(t): colores[t] for t in nom.index} | {lib.SIN_DATO: lib.GRIS},
                       orden=orden, hover_name="nombre_mapa",
                       hover={"cod": True, "etiqueta": True, "prob_max": ":.2f", "frontera": True}, alto=700)
    else:
        fig = lib.mapa(base, "prob_max", continuo="Blues", rango=(0.5, 1.0), hover_name="nombre_mapa",
                       hover={"cod": True, "etiqueta": True, "prob_max": ":.2f"}, alto=700,
                       etiquetas={"prob_max": "Prob. de pertenencia"})
    evento = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="mapa")
    st.caption("Gris: municipios sin asignación (áreas no municipalizadas, la isla de San Andrés y los municipios "
               "excluidos por exceso de datos faltantes; ver la página de método).")

opciones = asig.sort_values(["departamento", "municipio"])
etiquetas = {c: f"{r.municipio} ({r.departamento})" for c, r in opciones.iterrows()}
seleccion_mapa = None
try:
    puntos = evento.selection.points if evento else []
    if puntos:
        seleccion_mapa = puntos[0].get("location")
except AttributeError:
    pass
lista = list(etiquetas)
indice = lista.index(seleccion_mapa) if seleccion_mapa in etiquetas else lista.index("05001") if "05001" in etiquetas else 0

with col_info:
    cod = st.selectbox("Municipio", lista, index=indice, format_func=lambda c: etiquetas[c])
    r = asig.loc[cod]
    t = r["tipo"]
    st.subheader(f"{r['municipio']} · {r['departamento']}")
    st.markdown(f"<span style='background:{colores[t]};color:white;padding:3px 10px;border-radius:12px'>"
                f"Tipo {t}</span> **{nom.loc[t, 'nombre']}**", unsafe_allow_html=True)
    st.caption(nom.loc[t, "descripcion"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Pertenencia", f"{r['prob_max']:.0%}", help="Probabilidad de pertenencia a su tipo según el GMM")
    c2.metric("Estabilidad", f"{r['estabilidad_individual']:.0%}",
              help="Co-asociación media con los demás municipios de su tipo en 50 submuestras")
    c3.metric("Población", lib.formato_poblacion(r["poblacion"]))
    if r["frontera"]:
        st.warning("Municipio **frontera**: su asignación es incierta (probabilidad < 70 % o inestable bajo "
                   "remuestreo). Comparte rasgos con más de un tipo.")
    if isinstance(r.get("indicadores_imputados"), str):
        st.caption(f"Indicadores imputados para este municipio: {r['indicadores_imputados']}.")
    probs = pd.DataFrame({"tipo": [f"Tipo {x}" for x in sorted(nom.index)],
                          "prob": [r[f"prob_{x}"] for x in sorted(nom.index)]})
    figp = go.Figure(go.Bar(x=probs["prob"], y=probs["tipo"], orientation="h",
                            marker_color=[colores[x] for x in sorted(nom.index)], text=probs["prob"].map("{:.0%}".format)))
    figp.update_layout(height=60 + 40 * len(probs), margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[0, 1], tickformat=".0%"),
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(figp, use_container_width=True)

st.divider()
st.subheader("Perfil del municipio frente a la mediana de su tipo")
z = lib.parquet("z.parquet")
imp = lib.parquet("imputado.parquet")
valores = lib.parquet("valores.parquet")
pz = lib.csv("perfiles_z.csv").set_index("tipo")
pv = lib.csv("perfiles_valores.csv").set_index("tipo")
dominios = ind["dominio_nombre"]
perfil = pd.DataFrame({
    "indicador": ind.loc[z.columns, "nombre"], "dominio": dominios.loc[z.columns],
    "municipio": z.loc[cod], "mediana_tipo": pz.loc[t, z.columns],
    "imputado": imp.loc[cod, z.columns].astype(bool),
}).reset_index(drop=True)
perfil = perfil.sort_values(["dominio", "indicador"])
figz = go.Figure()
figz.add_bar(y=perfil["indicador"], x=perfil["mediana_tipo"], orientation="h", name=f"Mediana del tipo {t}",
             marker_color=colores[t], opacity=0.35)
figz.add_scatter(y=perfil["indicador"], x=perfil["municipio"], mode="markers", name=r["municipio"],
                 marker=dict(color="black", size=8, symbol=["x" if i else "circle" for i in perfil["imputado"]]))
figz.update_layout(height=26 * len(perfil) + 80, margin=dict(l=0, r=0, t=30, b=0), barmode="overlay",
                   xaxis_title="Puntaje z (0 = promedio nacional de municipios)", legend=dict(orientation="h", y=1.02))
st.plotly_chart(figz, use_container_width=True)
st.caption("Puntajes estandarizados tras transformar (log), winsorizar e imputar. Una X marca un valor imputado. "
           "Valores altos o bajos no son mejores ni peores: describen diferencias.")
with st.expander("Ver los valores en unidades originales"):
    tabla = pd.DataFrame({"Indicador": ind.loc[valores.columns, "nombre"], "Unidad": ind.loc[valores.columns, "unidad"],
                          "Años del dato": ind.loc[valores.columns, "anios_usados"],
                          r["municipio"]: valores.loc[cod].to_numpy(), f"Mediana tipo {t}": pv.loc[t, valores.columns].to_numpy()})
    st.dataframe(tabla, hide_index=True, use_container_width=True)

st.subheader("Municipios más parecidos")
sim = lib.csv("similares.csv", dtype={"cod": str, "cod_similar": str})
sim = sim[sim["cod"] == cod].copy()
sim["Municipio"] = sim["cod_similar"].map(lambda c: etiquetas.get(c, c))
sim["Tipo"] = sim["cod_similar"].map(asig["tipo"]).map(lib.etiqueta)
st.dataframe(sim[["rango", "Municipio", "Tipo", "distancia"]].rename(columns={"rango": "#", "distancia": "Distancia"}),
             hide_index=True, use_container_width=True)
st.caption("Distancia euclidiana en el espacio de componentes principales usado para el clustering. "
           "Parecido estadístico en estos indicadores, no cercanía geográfica.")
lib.descarga_asignaciones("descarga_mapa")
lib.pie()

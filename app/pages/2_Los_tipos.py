import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

lib.encabezado("Los tipos de municipio")
nom = lib.nombres_tipos()
ind = lib.indicadores()
colores = lib.colores_tipos()
tam = lib.csv("tamanos.csv").set_index("tipo")
pz = lib.csv("perfiles_z.csv").set_index("tipo")
rep = lib.csv("representativos.csv", dtype={"cod": str})
reg = lib.csv("distribucion_regiones.csv")
dep = lib.csv("distribucion_departamentos.csv")
perfil = lib.js("perfil_resumen.json")

st.markdown("Cada tipo resume un patrón de indicadores compartido por un conjunto de municipios. Los nombres son "
            "**descriptivos**: no ordenan los tipos de mejor a peor. Las letras siguen el tamaño del grupo "
            "(A es el más numeroso).")

cols = st.columns(len(nom))
for c, (t, fila) in zip(cols, nom.iterrows(), strict=True):
    with c:
        st.markdown(f"<div style='border-left:6px solid {colores[t]};padding-left:10px'><b>Tipo {t}</b><br>"
                    f"<span style='font-size:1.1em'>{fila['nombre']}</span></div>", unsafe_allow_html=True)
        st.metric("Municipios", f"{int(tam.loc[t, 'municipios'])} ({tam.loc[t, 'pct_municipios']:.0f} %)")
        st.metric("Población", f"{tam.loc[t, 'pct_poblacion']:.0f} % del total")
        st.caption(f"Población mediana: {int(tam.loc[t, 'poblacion_mediana']):,} hab.".replace(",", "."))
        st.write(fila["descripcion"])

st.subheader("Perfil de los tipos: mediana del puntaje z por indicador")
orden = ind.sort_values(["dominio_nombre", "nombre"]).index
etiquetas_y = [f"{ind.loc[i, 'dominio_nombre']} · {ind.loc[i, 'nombre']}" for i in orden]
fig = go.Figure(go.Heatmap(
    z=pz[orden].T.to_numpy(), x=[lib.etiqueta(t) for t in pz.index], y=etiquetas_y, colorscale="RdBu_r",
    zmid=0, zmin=-2, zmax=2, colorbar=dict(title="z"), text=pz[orden].T.round(2).to_numpy(), texttemplate="%{text}",
))
fig.update_layout(height=24 * len(orden) + 120, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Medianas calculadas solo con valores observados (sin imputados). 0 = promedio de los municipios; "
           "rojo = por encima, azul = por debajo. Colores divergentes para mostrar dirección, no mérito.")

st.subheader("Dónde están")
c1, c2 = st.columns(2)
with c1:
    reg["Tipo"] = reg["tipo"].map(lib.etiqueta)
    f = px.bar(reg, x="grupo", y="municipios", color="Tipo", color_discrete_map={lib.etiqueta(t): colores[t] for t in nom.index},
               labels={"grupo": "Región", "municipios": "Municipios"}, height=400)
    f.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(f, use_container_width=True)
with c2:
    t_sel = st.selectbox("Departamentos con más municipios del tipo", list(nom.index), format_func=lib.etiqueta)
    d = dep[dep["tipo"] == t_sel].sort_values("municipios", ascending=False).head(12)
    f2 = px.bar(d, x="municipios", y="grupo", orientation="h", color_discrete_sequence=[colores[t_sel]],
                hover_data={"pct_del_grupo": ":.0f"}, labels={"grupo": "", "municipios": "Municipios",
                                                           "pct_del_grupo": "% de los municipios del depto."}, height=400)
    f2.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(f2, use_container_width=True)

st.subheader("Municipios representativos")
st.caption("El medoide es el municipio con menor distancia total al resto de su tipo; se listan también los más "
           "cercanos a él. Son ejemplos del patrón, no 'los mejores' ni 'los peores'.")
cols = st.columns(len(nom))
for c, t in zip(cols, nom.index, strict=True):
    with c:
        r = rep[rep["tipo"] == t]
        st.markdown(f"**Tipo {t}**")
        for _, x in r.iterrows():
            st.write(f"{'★ ' if x['es_medoide'] else ''}{x['nombre']}")

st.subheader("¿Qué indicadores separan más a los tipos?")
imp = lib.csv("sustituto_importancias.csv")
imp["Dominio"] = imp["dominio"].map(ind.drop_duplicates("dominio").set_index("dominio")["dominio_nombre"])
f3 = px.bar(imp.head(15), x="importancia_permutacion", y="nombre", color="Dominio", orientation="h",
            error_x="sd_entre_pliegues", labels={"importancia_permutacion": "Pérdida de exactitud al permutar", "nombre": ""},
            height=480)
f3.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
st.plotly_chart(f3, use_container_width=True)
arbol, bosque = perfil["sustituto_arbol"], perfil["sustituto_bosque"]
st.caption(f"Bosque aleatorio sustituto que predice el tipo a partir de los indicadores (exactitud en validación "
           f"cruzada {bosque['exactitud_cv_media']:.0%}). Explica **cómo se formó la partición**, no causas: que un "
           "indicador separe a los tipos no significa que los produzca.")
with st.expander(f"Reglas de un árbol de decisión de profundidad {arbol['profundidad']} "
                 f"(exactitud en validación cruzada {arbol['exactitud_cv_media']:.0%})"):
    st.code(lib.texto("sustituto_arbol.txt"), language="text")
    st.caption("Umbrales en puntaje z. El árbol es una aproximación legible de la partición, no la regla que la generó.")
lib.pie()

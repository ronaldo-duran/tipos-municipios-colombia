import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

lib.encabezado("Robustez: ¿cuánto confiar en estos tipos?")
est = lib.js("estabilidad_resumen.json")
clu = lib.js("cluster_resumen.json")
sel = lib.csv("seleccion_k.csv")
estab = lib.csv("estabilidad.csv")
dec = lib.csv("decision_k.csv")
nulas = lib.csv("referencia_nula.csv")
k = est["k"]
modelo = est["modelo_final"].upper()
obs = est["observado_modelo_final"]
nula_p95 = est["referencia_nula_p95"]
alg_final = est["modelo_final"]

# --- Veredicto con franqueza, calculado a partir de las métricas -------------------------------------------
sil = obs["silhouette"]
supera_nula = obs["ari_medio"] > nula_p95[alg_final]["ari_medio"] and sil > nula_p95[alg_final]["silhouette"]
km_nula = nula_p95.get("kmeans", {})
acu = lib.csv("acuerdo_algoritmos.csv")
acu_k = acu[~acu["b"].str.startswith("hdbscan") & ~acu["a"].str.startswith("hdbscan")]
sens_todos = lib.csv("sensibilidad.csv")
otro_k = sens_todos["k_preferido_criterios_internos"].value_counts()
k_mas_comun = int(otro_k.idxmax())

st.subheader("Lo que es sólido y lo que no")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Sólido**")
    st.markdown(
        f"- Con {modelo} y k = {k}, la partición se reproduce bajo remuestreo: ARI medio **{obs['ari_medio']:.2f}** "
        f"en {lib.meta()['parametros']['estabilidad']['n_remuestreos']} submuestras del 80 %.\n"
        f"- Esa estabilidad **supera claramente** la de datos sin grupos con la misma correlación "
        f"(referencia nula gaussiana, percentil 95: ARI {nula_p95[alg_final]['ari_medio']:.2f}, silhouette "
        f"{nula_p95[alg_final]['silhouette']:.2f}): hay estructura más allá de una nube continua.\n"
        f"- Solo el {est['pct_frontera']:.1f} % de los municipios queda en la frontera entre tipos "
        f"(PAC = {est['pac']:.2f}).\n"
        f"- Las decisiones de imputación casi no mueven los grupos (casos completos: ARI "
        f"{est['sesgo_imputacion'].get('casos_completos_ari_vs_base', float('nan')):.2f})."
        if supera_nula else
        f"- La partición con {modelo} y k = {k} se reproduce bajo remuestreo (ARI {obs['ari_medio']:.2f}), pero "
        "no supera con claridad la referencia nula."
    )
with c2:
    st.markdown("**Débil o discutible**")
    st.markdown(
        f"- La separación entre tipos es **baja**: silhouette {sil:.2f} (por debajo de 0,25 suele leerse como "
        "estructura débil). Los tipos son regiones de un continuo más que grupos nítidos.\n"
        f"- Los algoritmos no coinciden mucho entre sí con k = {k}: ARI entre {acu_k['ari'].min():.2f} y "
        f"{acu_k['ari'].max():.2f}. La forma de los tipos depende de cómo define 'grupo' cada método.\n"
        f"- HDBSCAN (basado en densidad) deja al menos el {clu['hdbscan_ruido_min_pct']:.0f} % de los municipios como "
        "ruido: no hay grupos densos separados por zonas vacías.\n"
        f"- La estabilidad de K-means no prueba que haya grupos: datos sin grupos dan una estabilidad parecida "
        f"(percentil 95 nulo: {km_nula.get('ari_medio', float('nan')):.2f}).\n"
        f"- El número de tipos no es inamovible: en {int((sens_todos['k_preferido_criterios_internos'] != k).sum())} de "
        f"{len(sens_todos)} escenarios de sensibilidad los criterios internos preferirían otro k "
        f"(el más frecuente: k = {k_mas_comun}).\n"
        f"- Hay decisiones que sí cambian los grupos (ver sensibilidad): la más fuerte es "
        f"**{est['sensibilidad_ari_min']['escenario'].replace('_', ' ')}** (ARI {est['sensibilidad_ari_min']['ari']:.2f})."
    )

st.subheader(f"¿Por qué {k} tipos?")
st.markdown("Se combinan varios criterios porque ninguno basta solo: silhouette (↑), Davies-Bouldin (↓), "
            "Calinski-Harabasz (↑) y estabilidad bajo remuestreo (↑), para K-means, GMM y Ward. Se elige el k con "
            "mejor rango promedio entre los que tienen estabilidad ≥ 0,75 en el modelo final y ningún grupo menor al "
            "2 % de los municipios. La gap statistic, el BIC y el codo se muestran como apoyo.")
metricas = sel.merge(estab[["algoritmo", "k", "ari_medio"]], on=["algoritmo", "k"], how="left")
nombres_m = {"silhouette": "Silhouette (↑)", "davies_bouldin": "Davies-Bouldin (↓)",
             "calinski_harabasz": "Calinski-Harabasz (↑)", "ari_medio": "Estabilidad ARI (↑)"}
largo = metricas.melt(id_vars=["algoritmo", "k"], value_vars=list(nombres_m), var_name="metrica", value_name="valor")
largo["metrica"] = largo["metrica"].map(nombres_m)
f = px.line(largo, x="k", y="valor", color="algoritmo", facet_col="metrica", facet_col_wrap=4, markers=True, height=330)
f.update_yaxes(matches=None, title="")
f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
f.add_vline(x=k, line_dash="dot", line_color="gray")
f.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(f, use_container_width=True)
c1, c2, c3 = st.columns(3)
with c1:
    gap = lib.csv("gap.csv")
    fg = go.Figure(go.Scatter(x=gap["k"], y=gap["gap"], error_y=dict(array=gap["s"]), mode="lines+markers"))
    fg.update_layout(title="Gap statistic", height=280, margin=dict(l=0, r=0, t=40, b=0), xaxis_title="k")
    st.plotly_chart(fg, use_container_width=True)
    st.caption(f"Crece sin un codo claro (la regla de Tibshirani marca k = {clu['k_gap']}): típico de un continuo.")
with c2:
    g = sel[sel["algoritmo"] == "gmm"]
    fb = go.Figure(go.Scatter(x=g["k"], y=g["bic"], mode="lines+markers"))
    fb.update_layout(title="BIC del GMM (↓)", height=280, margin=dict(l=0, r=0, t=40, b=0), xaxis_title="k")
    st.plotly_chart(fb, use_container_width=True)
    st.caption(f"Mínimo en k = {clu['k_bic_gmm']}.")
with c3:
    st.markdown("**Rango compuesto por k**")
    st.dataframe(dec.rename(columns={"rango_compuesto": "rango", "ari_modelo_final": f"ARI {modelo}"})
                 [["k", "rango", f"ARI {modelo}", "estable"]].round(3), hide_index=True, use_container_width=True, height=280)

st.subheader("Contraste con datos sin grupos")
obs_df = pd.DataFrame([{"algoritmo": alg_final, "silhouette": obs["silhouette"], "ari_medio": obs["ari_medio"], "serie": "Datos reales"}])
km_obs = metricas[(metricas["algoritmo"] == "kmeans") & (metricas["k"] == k)]
if not km_obs.empty and alg_final != "kmeans":
    obs_df.loc[len(obs_df)] = {"algoritmo": "kmeans", "silhouette": float(km_obs["silhouette"].iloc[0]),
                               "ari_medio": float(km_obs["ari_medio"].iloc[0]), "serie": "Datos reales"}
nul = nulas.assign(serie="Referencia nula (gaussiana)")[["algoritmo", "silhouette", "ari_medio", "serie"]]
fn = px.scatter(pd.concat([nul, obs_df]), x="silhouette", y="ari_medio", color="serie", facet_col="algoritmo",
                symbol="serie", height=330, labels={"ari_medio": "Estabilidad (ARI medio)"})
fn.update_traces(marker=dict(size=11))
fn.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fn, use_container_width=True)
st.caption("Cada punto gris es un conjunto simulado con la misma covarianza que los datos pero sin grupos. Si los "
           "datos reales cayeran dentro de esa nube, la partición no aportaría más que cortar un continuo.")

st.subheader("Estabilidad por tipo y acuerdo entre algoritmos")
c1, c2 = st.columns(2)
with c1:
    jac = lib.csv("jaccard_modelo_final.csv")
    fj = px.bar(jac, x="tipo", y="jaccard_medio", color="tipo", color_discrete_map=lib.colores_tipos(), range_y=[0, 1],
                labels={"jaccard_medio": "Jaccard medio (remuestreo)"}, height=300)
    fj.add_hline(y=0.75, line_dash="dot")
    fj.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fj, use_container_width=True)
    st.caption("Hennig: < 0,6 el grupo se disuelve; 0,6–0,75 patrón débil; > 0,85 muy estable.")
with c2:
    st.dataframe(acu.rename(columns={"a": "Partición A", "b": "Partición B", "ari": "ARI", "ami": "AMI"}).round(3),
                 hide_index=True, use_container_width=True)

st.subheader("Análisis de sensibilidad")
sens = lib.csv("sensibilidad.csv")
sens = sens.sort_values("ari")
fs = px.bar(sens, x="ari", y="descripcion", color="grupo", orientation="h", height=34 * len(sens) + 80,
            hover_data={"municipios_comunes": True, "k_preferido_criterios_internos": True},
            labels={"ari": "ARI frente a la tipología base", "descripcion": ""})
fs.add_vline(x=0.75, line_dash="dot")
fs.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[0, 1]))
st.plotly_chart(fs, use_container_width=True)
st.caption(f"Cada escenario cambia una decisión y rehace la tipología con k = {k}. ARI 1 = misma partición; "
           "0 = coincidencia al azar. La línea punteada marca 0,75.")
if lib.existe("migracion_sin_conflicto.csv"):
    with st.expander("¿Qué cambia sin el dominio de conflicto?"):
        st.dataframe(lib.csv("migracion_sin_conflicto.csv").rename(columns={"base": "Tipo base"}), hide_index=True)
        st.caption("Filas: tipo en la tipología base; columnas: tipo sin el dominio de conflicto (etiquetas alineadas).")
sesgo = est["sesgo_imputacion"]
st.markdown(
    f"**Sesgo por imputación.** Bajo peso al nacer y fecundidad adolescente faltan sobre todo en municipios "
    f"pequeños. Solo con casos completos ({sesgo.get('casos_completos_n')} municipios) la partición coincide con "
    f"la base con ARI {sesgo.get('casos_completos_ari_vs_base', float('nan')):.2f}. Al quitar esas dos variables, "
    f"cambia de tipo el {sesgo.get('pct_cambio_tipo_imputados', float('nan')):.1f} % de los municipios donde se "
    f"imputaron, frente al {sesgo.get('pct_cambio_tipo_resto', float('nan')):.1f} % del resto."
)

st.subheader("Tamaño poblacional y confianza")
c1, c2 = st.columns(2)
with c1:
    tam = lib.csv("estabilidad_por_tamano.csv")
    ft = px.bar(tam, x="quintil", y="pct_frontera", labels={"pct_frontera": "% municipios frontera", "quintil": "Quintil de población"},
                height=300)
    ft.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(ft, use_container_width=True)
    q1, q5 = tam["pct_frontera"].iloc[0], tam["pct_frontera"].iloc[-1]
    st.caption(f"Bajo remuestreo, el {q1:.1f} % de los municipios más pequeños es frontera, frente al {q5:.1f} % de los "
               "más grandes. El remuestreo no perturba los valores de cada municipio; por eso se simula aparte el "
               "ruido de medición (gráfico de la derecha).")
with c2:
    if lib.existe("ruido_por_tamano.csv"):
        rt = lib.csv("ruido_por_tamano.csv")
        fr = px.bar(rt, x="quintil", y="tasa_cambio_media", height=300,
                    labels={"tasa_cambio_media": "Tasa media de cambio de tipo", "quintil": "Quintil de población"})
        fr.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fr, use_container_width=True)
        nivel = lib.meta()["parametros"]["estabilidad"]["ruido_tamano_nivel"]
        st.caption(f"Ruido de medición simulado en las tasas con denominador pequeño (desviación {nivel} z en un "
                   "municipio de tamaño mediano, creciente al achicarse la población). Es un supuesto explícito, no "
                   "una medición del error real.")

st.subheader("Visualizaciones exploratorias")
c1, c2 = st.columns(2)
asig = lib.asignaciones()
with c1:
    pcs = lib.parquet("pca_scores.parquet").join(asig[["tipo", "municipio"]])
    var = lib.csv("pca_varianza.csv")
    fp = px.scatter(pcs, x="PC1", y="PC2", color="tipo", color_discrete_map=lib.colores_tipos(), hover_name="municipio",
                    height=420, opacity=0.7,
                    labels={"PC1": f"PC1 ({var['varianza_explicada'][0]:.0%})", "PC2": f"PC2 ({var['varianza_explicada'][1]:.0%})"})
    fp.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fp, use_container_width=True)
    cargas = lib.csv("pca_cargas.csv", index_col=0)
    ind = lib.indicadores()
    with st.expander("Cargas de los primeros componentes"):
        top = cargas.iloc[:3].T
        top.index = ind.loc[top.index, "nombre"]
        st.dataframe(top.round(2).sort_values("PC1"), use_container_width=True)
with c2:
    um = lib.parquet("umap.parquet").join(asig[["tipo", "municipio"]])
    fu = px.scatter(um, x="umap_1", y="umap_2", color="tipo", color_discrete_map=lib.colores_tipos(), hover_name="municipio",
                    height=420, opacity=0.7)
    fu.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(showticklabels=False, title=""),
                     yaxis=dict(showticklabels=False, title=""))
    st.plotly_chart(fu, use_container_width=True)
    st.caption("UMAP solo para explorar: sus distancias y tamaños de grupo **no son interpretables** y no se usó "
               "para decidir los tipos.")
lib.pie()

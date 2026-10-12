import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

lib.encabezado("Comparación externa, espacio, tiempo y método")
perfil = lib.js("perfil_resumen.json")
asig = lib.asignaciones()
colores = lib.colores_tipos()
geo = lib.geojson()
meta = lib.meta()


def heat(tabla: pd.DataFrame, titulo_x: str) -> go.Figure:
    pct = tabla.div(tabla.sum(axis=1), axis=0) * 100
    f = go.Figure(go.Heatmap(z=pct.to_numpy(), x=[str(c) for c in tabla.columns], y=[lib.etiqueta(t) for t in tabla.index],
                             colorscale="Greys", text=tabla.to_numpy(), texttemplate="%{text}",
                             colorbar=dict(title="% del tipo")))
    f.update_layout(height=90 + 70 * len(tabla), margin=dict(l=0, r=0, t=10, b=0), xaxis_title=titulo_x)
    return f


tab1, tab2, tab3, tab4 = st.tabs(["Clasificaciones oficiales", "Análisis espacial", "Cambio en el tiempo", "Método y limitaciones"])

with tab1:
    st.subheader("Tipologías del DNP 2026")
    dnp = perfil["externo_dnp"]
    st.markdown(
        "La tipología oficial del DNP (Resolución 3910 de 2025) **mide otra cosa**: ordena a los municipios por "
        "capacidad institucional y fiscal (promedio del IDI y el IDF 2021–2024, ponderado por distancias y densidad) "
        "y la corta en 5 grupos más dos categorías especiales. Es ordinal y unidimensional; esta tipología es "
        "multidimensional y descriptiva. Por eso no se usa ARI: se reporta la tabla cruzada y medidas de asociación "
        "ordinal. Nunca entró al clustering.")
    st.plotly_chart(heat(lib.csv("externo_dnp_tabla.csv", index_col=0), "Tipología DNP 2026 (1 = mayor capacidad)"),
                    use_container_width=True)
    kr = dnp["kruskal_referencia_por_tipo"]
    c = st.columns(4)
    c[0].metric("V de Cramér", f"{dnp['cramer_v']:.2f}")
    c[1].metric("ε² de Kruskal-Wallis", f"{kr['epsilon2']:.2f}")
    c[2].metric("Kendall τ-b", f"{dnp['kendall_tau_b']:.2f}")
    c[3].metric("γ de Goodman-Kruskal", f"{dnp['gamma']:.2f}")
    st.caption(f"Medidas ordinales sin las categorías especiales (n = {dnp['n_ordinal']}). Para τ-b y γ los tipos se "
               f"ordenan por su mediana en el primer componente principal ({' < '.join(dnp['orden_tipos_usado'])}); ese "
               "orden es un recurso de cálculo, no una jerarquía. Mediana de la categoría DNP por tipo: "
               + ", ".join(f"{t}: {v}" for t, v in dnp["mediana_referencia_por_tipo"].items()) + ".")

    st.subheader("Categorías de ruralidad (validación externa)")
    rur = perfil["externo_ruralidad"]
    st.plotly_chart(heat(lib.csv("externo_ruralidad_tabla.csv", index_col=0), "Categoría de ruralidad (DNP, 2024)"),
                    use_container_width=True)
    c = st.columns(4)
    c[0].metric("ARI", f"{rur['ari']:.2f}")
    c[1].metric("AMI", f"{rur['ami']:.2f}")
    c[2].metric("V de Cramér", f"{rur['cramer_v']:.2f}")
    c[3].metric("Kendall τ-b", f"{rur['kendall_tau_b']:.2f}")
    st.caption("Las categorías de ruralidad se basan en tamaño, densidad y peso de la población urbana. La tipología "
               "usa densidad y % rural entre otros 38 indicadores, así que algo de coincidencia es esperable.")

    st.subheader("Saber 11 (validación con una variable que no entró al clustering)")
    for vid, v in perfil["validacion_externa"].items():
        res = lib.csv(f"validacion_{vid}.csv", index_col=0)
        f = go.Figure()
        for t, fila in res.iterrows():
            f.add_trace(go.Box(name=lib.etiqueta(t), q1=[fila["25%"]], median=[fila["mediana"]], q3=[fila["75%"]],
                               lowerfence=[fila["25%"]], upperfence=[fila["75%"]], marker_color=colores.get(t)))
        f.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=v["nombre"])
        st.plotly_chart(f, use_container_width=True)
        st.caption(f"{v['nombre']} ({', '.join(map(str, v['anio']))}). Rango intercuartílico por tipo. η² = {v['eta2']:.2f}, "
                   f"ε² = {v['epsilon2']:.2f} (n = {v['n']}). Se sacó de los indicadores de entrada para poder usarla como "
                   "validación externa.")

with tab2:
    esp = perfil.get("espacial", {})
    if not esp:
        st.info("El análisis espacial no está disponible en esta ejecución.")
    else:
        st.markdown(f"Vecindad por contigüidad tipo reina a partir del **{esp['fuente_vecindad']}** "
                    f"({esp['pares_vecinos']:,} pares de municipios vecinos).".replace(",", "."))
        mt = pd.DataFrame(esp["moran_por_tipo"])
        mt["Tipo"] = mt["tipo"].map(lib.etiqueta)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(mt[["Tipo", "moran_I", "moran_p", "pares_mismo_tipo_obs", "pares_mismo_tipo_esperados", "join_count_p"]]
                         .rename(columns={"moran_I": "I de Moran", "moran_p": "p (perm.)", "pares_mismo_tipo_obs": "Pares vecinos mismo tipo",
                                          "pares_mismo_tipo_esperados": "Esperados al azar", "join_count_p": "p join count"}).round(3),
                         hide_index=True, use_container_width=True)
        with c2:
            v = esp["vecinos_mismo_tipo"]
            st.metric("Vecinos del mismo tipo (promedio)", f"{v['observado']:.0%}", f"{(v['observado'] - v['esperado_al_azar']) * 100:+.0f} pp vs. al azar")
            st.metric("I de Moran del 1er componente", f"{esp['moran_pc1']['moran_I']:.2f}")
        st.caption("I de Moran sobre el indicador binario de cada tipo (999 permutaciones). Valores positivos y "
                   "significativos: los municipios de ese tipo tienden a estar juntos.")
        local = asig[["lisa_tipo_propio", "region_contigua"]].copy() if "region_contigua" in asig else asig[["lisa_tipo_propio"]].copy()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**¿Núcleo o aislado?** (Moran local del tipo de cada municipio)")
            cats = {"núcleo (rodeado de su tipo)": "#2b6a99", "aislado (rodeado de otros tipos)": "#d9822b", "no significativo": lib.GRIS}
            f = lib.mapa(local, "lisa_tipo_propio", discreto=cats, orden=list(cats), alto=520)
            st.plotly_chart(f, use_container_width=True)
        if "region_contigua" in local:
            with c2:
                rest = esp["ward_restringido_vs_base"]
                st.markdown(f"**Contraste: Ward con restricción de contigüidad** (ARI con la tipología: {rest['ari']:.2f})")
                regiones = sorted(local["region_contigua"].dropna().unique())
                f2 = lib.mapa(local, "region_contigua", discreto=dict(zip(regiones, px.colors.qualitative.Set2, strict=False)),
                              orden=regiones, alto=520)
                st.plotly_chart(f2, use_container_width=True)
                st.caption("Obliga a que cada grupo sea una región continua. Sirve para ver cuánto se parece la "
                           "tipología a una regionalización; no la reemplaza.")

with tab3:
    if lib.existe("temporal_resumen.json"):
        tr = lib.js("temporal_resumen.json")
        st.markdown(
            f"Se repite la tipología con {tr['anio_base']} y {tr['anio_final']} usando los "
            f"{len(tr['indicadores'])} indicadores comparables entre ambos años (sin Sisbén IV, que empieza en 2022, ni "
            f"el área protegida; la dependencia de transferencias se calcula igual en los dos años). El modelo se ajusta "
            f"con {tr['anio_final']} y {tr['anio_base']} se proyecta sobre el mismo marco.")
        c = st.columns(3)
        c[0].metric("Municipios comparados", tr["municipios_comparados"])
        c[1].metric("Cambian de tipo", f"{tr['pct_cambia_de_tipo']:.1f} %")
        c[2].metric(f"ARI con la tipología principal ({tr['anio_final']})", f"{tr['ari_modelo_temporal_vs_tipologia_principal']:.2f}")
        mat = lib.csv("temporal_transicion.csv", index_col=0)
        st.plotly_chart(heat(mat, f"Tipo en {tr['anio_final']}"), use_container_width=True)
        st.warning(f"Cuidado con la comparabilidad: {len(tr['indicadores_constantes_censo_2018'])} indicadores vienen del "
                   "censo 2018 y son idénticos en ambos años, así que los cambios reflejan solo los indicadores anuales "
                   "(fiscales, salud, educación, conectividad, conflicto y economía). Entre 2019 y 2023 está además la "
                   "pandemia. Un cambio de tipo es un cambio de patrón estadístico, no una mejora ni un deterioro.")
    else:
        st.info("El análisis temporal no está disponible en esta ejecución.")

with tab4:
    feat = lib.js("features_resumen.json")
    excl = lib.csv("excluidos.csv", dtype={"cod": str})
    st.markdown(f"""
**Pregunta.** ¿Cuántos tipos distintos de municipio hay en Colombia según sus indicadores socioeconómicos?

**Datos.** TerriData (DNP), versión {meta['fuentes']['terridata']['version']}, año de referencia
{meta['anio_referencia']}. {feat['indicadores']} indicadores en 8 dominios. Pobreza multidimensional, NBI, déficit
de vivienda y coberturas de acueducto y alcantarillado son del **censo 2018** (no hay datos municipales más recientes).
El valor agregado usa el último año con dato (≤ 1 año de rezago) y Sisbén IV el año de referencia.

**Preparación.** Conteos convertidos a tasas o per cápita; logaritmo en variables muy sesgadas; winsorización al
percentil 1 y 99; imputación iterativa marcada celda por celda (nunca con información del clustering); escalado
estándar; **cada dominio pesa lo mismo** (cada indicador se multiplica por 1/√k del dominio), para que un dominio con
más indicadores no domine.

**Clustering.** PCA con los componentes que explican el 80 % de la varianza; K-means, mezcla gaussiana (GMM), Ward y
HDBSCAN; elección de k con varios criterios y estabilidad por submuestreo; asignaciones blandas con el GMM.

**Exclusiones.** {feat['municipios_excluidos']} municipios excluidos por tener más del
{meta['parametros']['datos']['max_faltantes_municipio']:.0%} de los indicadores sin dato en el año de referencia. No
tienen datos en TerriData: Nuevo Belén de Bajirá (municipio nuevo), la isla de San Andrés y las 18 áreas no municipalizadas.
""")
    st.dataframe(excl[["cod", "nombre", "motivo", "indicadores_faltantes"]], hide_index=True, use_container_width=True)
    st.markdown("""
**Ética y límites**

- Una tipología municipal describe **territorios, no a las personas** que viven en ellos (falacia ecológica): un
  municipio de un tipo puede tener realidades internas muy distintas.
- Es **descriptiva, no causal**: no explica por qué un municipio es como es ni sugiere políticas.
- **No hay rankings**: los tipos son perfiles, no puntajes de mérito; las letras siguen el tamaño del grupo.
- Los datos de municipios pequeños o de difícil acceso pueden ser **menos confiables**: la app marca los municipios
  frontera, los valores imputados y muestra cómo cambia la asignación con ruido de medición.
- La estructura encontrada es **débil**: los tipos se leen mejor como zonas de un continuo que como grupos separados.
""")
    ind = lib.indicadores()
    with st.expander("Indicadores usados"):
        st.dataframe(ind[["nombre", "dominio_nombre", "unidad", "regla_anio", "anios_usados", "transformacion",
                          "cobertura_ref", "imputados", "peso"]], use_container_width=True)
    lib.descarga_asignaciones("descarga_metodo")
lib.pie()

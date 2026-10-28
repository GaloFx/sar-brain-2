"""
Streamlit app: SAR espacial + grafo cerebral dinámico 3D
Adaptado desde notebook Colab/Gradio → GitHub + Streamlit.

Ejecutar localmente:
    streamlit run app.py

Despliegue en Streamlit Community Cloud: apuntar a este repositorio y a app.py.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from brain_sar import run_full_pipeline, SEED
from viz import build_brain_figure

st.set_page_config(
    page_title="SAR + Grafo Cerebral 3D (HCP DD)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 SAR Espacial + Grafo Cerebral Dinámico 3D")
st.caption(
    "Geometría real (fsaverage5 + Destrieux) · Fenotipo DD calibrado a distribución empírica HCP · "
    "Grosor y FC sintéticos (bloque sustituible)"
)


@st.cache_resource(show_spinner="Construyendo grafo cortical y estimando modelos SAR… (primera vez puede tardar ~30-60 s)")
def load_state(seed: int = SEED):
    return run_full_pipeline(seed=seed)


# Sidebar controls
with st.sidebar:
    st.header("Controles de visualización")
    seed = st.number_input("Semilla reproducible", value=SEED, step=1)
    state = load_state(int(seed))

    n_frames = len(state["frame_times"])
    frame_idx = st.slider("Frame temporal", 0, n_frames - 1, 0, 1)
    color_by = st.radio(
        "Colorear nodos por",
        options=["activity", "DD", "thickness"],
        format_func=lambda x: {
            "activity": "Actividad sintética",
            "DD": "Delay Discounting (HCP-scaled)",
            "thickness": "Grosor cortical",
        }[x],
        index=1,
    )
    n_edges = st.slider("Nº aristas funcionales", 20, 150, 80, 5)
    show_surface = st.checkbox("Mostrar superficie cortical", True)
    show_spatial = st.checkbox("Mostrar capa espacial", True)
    show_functional = st.checkbox("Mostrar capa funcional", True)

    st.divider()
    st.markdown("**Mejor W (por AIC)**")
    st.success(f"{state['best_w']}  ·  ρ ≈ {state['summary_df'].loc[state['best_w'], 'rho']:.3f}")
    st.caption(f"ρ verdadero (generador) = {state['rho_true']}")

# Main layout
tab_viz, tab_sar, tab_data, tab_audit = st.tabs(
    ["Visualización 3D", "Resultados SAR", "Datos & distribución", "Auditoría"]
)

with tab_viz:
    fig = build_brain_figure(
        state,
        frame_idx=frame_idx,
        color_by=color_by,
        n_functional_edges=n_edges,
        show_surface=show_surface,
        show_spatial=show_spatial,
        show_functional=show_functional,
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Nodos: {state['N']}  ·  Aristas espaciales: {len(state['sp_i'])}  ·  "
        f"Frames: {n_frames}  ·  Fuente DD: `{state['dd_source']}`"
    )

with tab_sar:
    st.subheader("Diagnóstico espacial – Moran's I sobre DD")
    moran = state["moran_df"].copy()
    moran["sig"] = moran["p_sim"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    st.dataframe(
        moran.style.format(
            {"Moran_I": "{:.4f}", "p_sim": "{:.4f}", "sparsity_pct": "{:.1f}"}
        ),
        use_container_width=True,
    )

    st.subheader("Estimación SAR (ML_Lag)")
    summary = state["summary_df"].copy()
    st.dataframe(
        summary.style.format(
            {
                "rho": "{:.4f}",
                "p_rho": "{:.4f}",
                "beta_thickness": "{:.4f}",
                "beta_funcconn": "{:.4f}",
                "LogLik": "{:.2f}",
                "AIC": "{:.2f}",
                "PseudoR2": "{:.4f}",
            }
        ).highlight_min(subset=["AIC"], color="#c6efce"),
        use_container_width=True,
    )
    st.markdown(
        f"""
        **Interpretación (diseño sintético actual)**  
        - Mejor especificación: **{state['best_w']}** (mínimo AIC).  
        - ρ estimado ≈ **{summary.loc[state['best_w'], 'rho']:.3f}** (verdadero = {state['rho_true']}).  
        - β_grosor negativo → mayor grosor se asocia a menor DD (tras controlar dependencia espacial).  
        - β_conectividad positivo → mayor FC local se asocia a mayor DD en este diseño.
        """
    )

with tab_data:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución DD (HCP real / fallback)")
        st.write(
            pd.Series(state["dd_real"], name="DDisc_SV_1mo_40K")
            .describe()
            .round(1)
            .to_frame()
        )
        st.caption(f"Fuente: `{state['dd_source']}` · N sujetos = {len(state['dd_real'])}")
    with col2:
        st.subheader("Variables a nivel de nodo")
        st.write(
            state["df"][["cortical_thickness", "functional_conn", "DD"]]
            .describe()
            .round(3)
        )
        st.caption(f"N nodos corticales = {state['N']}")

    st.subheader("Nodos corticales (primeras filas)")
    st.dataframe(state["nodes_df"].head(15), use_container_width=True)

    st.download_button(
        "Descargar variables_SAR.csv",
        data=state["df"].to_csv(index=False).encode("utf-8"),
        file_name="variables_SAR.csv",
        mime="text/csv",
    )
    st.download_button(
        "Descargar resultados_SAR.csv",
        data=state["summary_df"].to_csv().encode("utf-8"),
        file_name="resultados_SAR.csv",
        mime="text/csv",
    )

with tab_audit:
    st.markdown(
        """
### Auditoría del pipeline (v1.1 – Streamlit / GitHub)

#### ✅ Cumple
| Aspecto | Estado |
|---------|--------|
| Geometría cortical real (nilearn fsaverage5 + Destrieux) | ✅ |
| Fenotipo DD usa distribución empírica real del CSV HCP | ✅ |
| Matrices W row-estandarizadas; estimación ML_Lag | ✅ |
| Proceso generador con ρ verdadero conocido (validación) | ✅ |
| Visualización desacoplada de la estimación SAR | ✅ |
| Submuestreo de caras de malla para rendimiento | ✅ |
| Manejo de valores faltantes en CSV de DD | ✅ |
| Semilla fija para reproducibilidad | ✅ |
| Código libre de dependencias Colab / Drive | ✅ |
| Interfaz Streamlit (reemplaza Gradio) | ✅ |

#### ⚠️ Limitaciones declaradas
- **Grosor cortical y matriz de FC son sintéticos** (aunque realistas). El bloque está claramente marcado en `brain_sar.py` → `generate_node_features`.
- El SAR se estima a **nivel de nodos corticales (regiones)**, no a nivel de sujetos. El CSV HCP es sujeto-nivel; solo se usa su distribución para calibrar `y`.
- Algunos nodos pueden quedar como *islands* en W4 (DistanceBand); se silenciaron los warnings y se elevó ligeramente el umbral de distancia.
- Primera ejecución descarga datos de nilearn (~decenas de MB) y puede tardar 30–60 s; luego se cachea con `@st.cache_resource`.

#### Cómo sustituir por datos reales (HCP / BALSA)
1. Descargar matrices de grosor cortical medio por región (Destrieux) desde BALSA o HCP.
2. Descargar matrices de conectividad funcional (resting-state, Fisher-z).
3. Sustituir los vectores `cortical_thickness` y `functional_conn` (y la matriz `sim`) dentro de `generate_node_features`.
4. Para análisis multi-sujeto: pasar a panel espacial o estimar SAR por sujeto y meta-analizar ρ y β.
5. Para W3 usar directamente la matriz de correlación funcional real; para W4 la adyacencia anatómica del atlas.

#### Extensiones posibles
- Spatial Durbin (efectos directos + spillovers).
- Panel espacial multi-sujeto.
- Animación nativa Plotly con slider de frames.
- Covariables demográficas del CSV (Gender, Age) si se cambia a modelo sujeto-nivel.

#### Datos y licencia
- El CSV de Delay Discounting proviene del Human Connectome Project (HCP). Respetar los términos de uso de HCP al redistribuir o publicar resultados.
- El código de este repositorio (pipeline, visualización, app) puede usarse libremente con atribución.
        """
    )

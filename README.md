# SAR Espacial + Grafo Cerebral Dinámico 3D (HCP Delay Discounting)

Pipeline educativo que combina:

1. **Grafo cortical realista** (fsaverage5 + atlas Destrieux, ~150 nodos).
2. **Modelo SAR** (Spatial Autoregressive) para explicar Delayed Discounting a partir de grosor cortical y conectividad funcional, comparando 4 matrices de pesos espaciales \(W\).
3. **Visualización 3D interactiva** (Plotly) + interfaz **Streamlit**.

$$
y = \rho W y + X\beta + \varepsilon
$$

## Estructura del repositorio

```
sar_brain_streamlit/
├── app.py                 # Aplicación Streamlit (punto de entrada)
├── brain_sar.py           # Lógica: geometría, features, W, SAR, dinámica
├── viz.py                 # Helpers de visualización Plotly 3D
├── requirements.txt
├── data/
│   └── Delay_Discounting_40K_1_Month.csv   # Fenotipo HCP (DDisc_SV_1mo_40K)
├── .streamlit/
│   └── config.toml
└── README.md
```

## Instalación y ejecución local

```bash
git clone <tu-repo>
cd sar_brain_streamlit
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La primera ejecución descarga las mallas de nilearn (fsaverage5 + Destrieux) y puede tardar 30–60 s. Las ejecuciones posteriores usan caché de Streamlit (`@st.cache_resource`).

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) crea una app apuntando a `app.py`.
3. Asegúrate de que el archivo CSV esté en `data/` (ya incluido).

**Nota:** Streamlit Cloud tiene límites de memoria/tiempo; la descarga de nilearn se realiza en el arranque. Si falla por timeout, considera pre-empaquetar los datos de nilearn o usar un servicio con más recursos.

## Datos

| Componente | Origen | Notas |
|------------|--------|-------|
| Geometría cortical | nilearn (fsaverage5 + Destrieux) | **Real** |
| Distribución DD | CSV HCP `DDisc_SV_1mo_40K` | **Real** (media/SD/rango) |
| Grosor cortical por región | Sintético | Bloque marcado en `brain_sar.py` |
| Conectividad funcional | Sintética | Bloque marcado en `brain_sar.py` |

El SAR se estima a **nivel de nodos (regiones)**, no de sujetos. El CSV HCP solo calibra la distribución de \(y\).

## Auditoría (resumen)

- ✅ Sin dependencias de Google Colab / Drive.
- ✅ CSV resuelto de forma robusta (`data/` relativo al paquete).
- ✅ Semilla fija (`SEED = 20261011`).
- ✅ Matrices \(W\) row-estandarizadas; estimación por máxima verosimilitud (`ML_Lag`).
- ✅ ρ verdadero conocido → validación de recuperación del parámetro.
- ✅ Warnings de *islands* en DistanceBand silenciados; umbral ligeramente aumentado.
- ⚠️ Grosor y FC siguen siendo sintéticos hasta que se sustituyan por datos HCP/BALSA.

Ver pestaña **Auditoría** dentro de la app para el checklist completo y las instrucciones de sustitución por datos reales.

## Licencia y atribución

- Datos de fenotipo: Human Connectome Project (respetar términos de uso HCP).
- Código del pipeline y la app: uso libre con atribución.

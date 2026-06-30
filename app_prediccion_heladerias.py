"""
app_prediccion_heladerias.py
----------------------------
Interfaz de usuario Streamlit para la app de predicción de heladerías.

Este archivo SOLO maneja la UI: layout, inputs, outputs visuales.
Toda la lógica de negocio vive en src/.

Para ejecutar:
    streamlit run app_prediccion_heladerias.py
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src import (
    cargar_dataframe,
    corregir_pandemia,
    ejecutar_pipeline,
    formato_numero,
    generar_pdf,
    generar_plantilla_excel,
    mes_anio_es,
    etiquetas_eje_fecha_es,
    transformar_a_serie,
    validar_dataframe,
)

_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_agp_blema.png")

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AGP Blema | Estimación de la Demanda",
    page_icon=_LOGO_PATH,
    layout="wide",
)

col_logo, col_titulo = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    st.image(_LOGO_PATH, width=110)
with col_titulo:
    st.markdown("## AGP Blema")
    st.markdown("### Estimación de la Demanda para Heladerías")
st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar: configuración
# ---------------------------------------------------------------------------

st.sidebar.image(_LOGO_PATH, width=140)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración")

# --- Plantilla descargable ---
st.sidebar.subheader("1. Cargar datos")
st.sidebar.download_button(
    label="📥 Descargar plantilla Excel (.xlsx)",
    data=generar_plantilla_excel(),
    file_name="plantilla_ventas_heladeria.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
    help="Modelo con columnas AÑO y ENERO…DICIEMBRE para completar y subir.",
)

uploaded_file = st.sidebar.file_uploader(
    "Subí tu archivo CSV o Excel",
    type=["csv", "xlsx", "xlsm"],
)

# --- Parámetros ---
st.sidebar.subheader("2. Parámetros")

es_excel = uploaded_file is not None and uploaded_file.name.lower().endswith((".xlsx", ".xlsm"))

if es_excel:
    st.sidebar.caption("Archivo Excel: separador y codificación no aplican.")
    separador = ";"
    encoding = "utf-8"
else:
    separador = st.sidebar.selectbox(
        "Separador del CSV", [";", ",", "\t"], index=0
    )
    encoding = st.sidebar.selectbox(
        "Codificación", ["latin1", "utf-8", "utf-8-sig", "cp1252"], index=0
    )

meses_validacion = st.sidebar.number_input(
    "Meses de validación",
    min_value=3,
    max_value=24,
    value=12,
    help="Meses recientes reservados para validar el modelo.",
)

corregir_2020 = st.sidebar.checkbox(
    "Corregir datos de pandemia (2020)",
    value=False,
    help="Ajusta marzo y abril 2020 con promedios históricos.",
)


# ---------------------------------------------------------------------------
# Procesamiento principal
# ---------------------------------------------------------------------------

if uploaded_file is None:
    st.info("👆 Subí un archivo CSV o Excel para comenzar.")
    st.stop()

# 1. Carga
try:
    df_raw = cargar_dataframe(uploaded_file, separador, encoding)
except ValueError as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

# 2. Validación
resultado_val = validar_dataframe(df_raw)
if not resultado_val.valido:
    st.error("El archivo no tiene el formato esperado:\n\n" + resultado_val.resumen())
    st.stop()
if resultado_val.advertencias:
    st.warning(resultado_val.resumen())

# 3. Vista previa de datos
st.subheader("📊 Datos Cargados")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df_raw, use_container_width=True)
with col2:
    st.metric("Filas", df_raw.shape[0])
    st.metric("Columnas", df_raw.shape[1])

# 4. Transformación
df_serie = transformar_a_serie(df_raw)

if corregir_2020:
    df_serie = corregir_pandemia(df_serie)
    st.info("✅ Corrección de pandemia 2020 aplicada (marzo y abril).")

# 5. Serie temporal
st.subheader("📈 Serie Temporal de Ventas")
fig_serie = px.line(
    df_serie, x="fecha", y="ventas",
    title="Ventas Históricas",
    labels={"fecha": "Fecha", "ventas": "Ventas"},
)
fig_serie.update_traces(line_color="#3FBF9F", line_width=2)
tickvals, ticktext = etiquetas_eje_fecha_es(df_serie["fecha"])
fig_serie.update_xaxes(tickvals=tickvals, ticktext=ticktext)
fig_serie.update_layout(hovermode="x unified")
st.plotly_chart(fig_serie, use_container_width=True)

st.markdown("---")

# 6. Botón para ejecutar el modelo
if not st.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True):
    st.stop()


# ---------------------------------------------------------------------------
# Ejecución del modelo
# ---------------------------------------------------------------------------

with st.spinner("Evaluando combinaciones de parámetros..."):
    try:
        resultado = ejecutar_pipeline(df_serie, n_meses_validacion=int(meses_validacion))
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

train = resultado.train
test = resultado.test
metricas = resultado.metricas
predicciones_val = resultado.predicciones_validacion
df_futuro = resultado.df_futuro

periodo_inicio = mes_anio_es(test["fecha"].min(), abreviado=True)
periodo_fin = mes_anio_es(test["fecha"].max(), abreviado=True)

st.info(
    f"📊 Validación: últimos **{len(test)} meses** ({periodo_inicio} - {periodo_fin}) | "
    f"Modelo: trend=**{resultado.trend or 'ninguna'}**, seasonal=**{resultado.seasonal}**"
)

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📊 Resultados del Modelo")

total_real = test["ventas"].sum()
total_predicho = predicciones_val.sum()
error_total_pct = (metricas.error_absoluto_total / total_real * 100) if total_real else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MAE", formato_numero(metricas.mae))
col2.metric("RMSE", formato_numero(metricas.rmse))
col3.metric("R²", f"{metricas.r2:.4f}")
col4.metric("Error Abs. Total", formato_numero(metricas.error_absoluto_total))
col5.metric("Error Abs. %", f"{error_total_pct:.2f}%".replace(".", ","))

# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

st.subheader(f"🔍 Validación: Predicción vs Real ({periodo_inicio} - {periodo_fin})")

comparativa = test[["fecha", "ventas"]].copy()
comparativa["prediccion"] = predicciones_val.values
comparativa["error"] = abs(comparativa["ventas"] - comparativa["prediccion"])
comparativa["error_pct"] = comparativa["error"] / comparativa["ventas"] * 100
comparativa["mes"] = comparativa["fecha"].apply(lambda f: mes_anio_es(f, abreviado=True))

diferencia = total_real - total_predicho
diferencia_pct = (diferencia / total_real * 100) if total_real else 0

st.markdown("**Totales del período de validación:**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Real", formato_numero(total_real))
c2.metric("Total Predicción", formato_numero(total_predicho))
c3.metric("Diferencia", formato_numero(diferencia))
c4.metric("Diferencia %", f"{diferencia_pct:.2f}%".replace(".", ","))

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(name="Real", x=comparativa["mes"], y=comparativa["ventas"],
                          marker_color="#6B4730"))
fig_comp.add_trace(go.Bar(name="Predicción", x=comparativa["mes"], y=comparativa["prediccion"],
                          marker_color="#3FBF9F"))
fig_comp.update_layout(
    title="Comparación: Ventas Reales vs Predicción",
    barmode="group",
    xaxis_title="Mes",
    yaxis_title="Ventas",
)
st.plotly_chart(fig_comp, use_container_width=True)

# Tabla de validación con formato
comp_display = comparativa[["mes", "ventas", "prediccion", "error", "error_pct"]].copy()
comp_display["ventas"] = comp_display["ventas"].apply(formato_numero)
comp_display["prediccion"] = comp_display["prediccion"].apply(formato_numero)
comp_display["error"] = comp_display["error"].apply(formato_numero)
comp_display["error_pct"] = comp_display["error_pct"].apply(
    lambda x: f"{x:.2f}%".replace(".", ",")
)
st.dataframe(
    comp_display.rename(columns={
        "mes": "Mes", "ventas": "Venta Real",
        "prediccion": "Predicción", "error": "Error", "error_pct": "Error %"
    }),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Predicciones futuras
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔮 Predicciones Futuras (Próximos 12 meses)")

fig_fut = make_subplots(rows=1, cols=2,
                        subplot_titles=("Serie Completa", "Predicciones Futuras"))
df_historico = df_serie.dropna(subset=["ventas"])

fig_fut.add_trace(
    go.Scatter(x=df_historico["fecha"], y=df_historico["ventas"],
               name="Histórico", line=dict(color="#6B4730")),
    row=1, col=1,
)
fig_fut.add_trace(
    go.Scatter(x=df_futuro["fecha"], y=df_futuro["prediccion"],
               name="Predicción", line=dict(color="#D97757", dash="dash")),
    row=1, col=1,
)
fig_fut.add_trace(
    go.Bar(x=df_futuro["mes"], y=df_futuro["prediccion"],
           name="Predicción Mensual", marker_color="#3FBF9F"),
    row=1, col=2,
)
fig_fut.update_layout(height=400, showlegend=True)
tickvals_hist, ticktext_hist = etiquetas_eje_fecha_es(
    pd.concat([df_historico["fecha"], df_futuro["fecha"]])
)
fig_fut.update_xaxes(tickvals=tickvals_hist, ticktext=ticktext_hist, row=1, col=1)
st.plotly_chart(fig_fut, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total Anual Predicho", formato_numero(df_futuro["prediccion"].sum()))
c2.metric("Promedio Mensual", formato_numero(df_futuro["prediccion"].mean()))
c3.metric("Mes Pico", df_futuro.loc[df_futuro["prediccion"].idxmax(), "mes"])

fut_display = df_futuro[["mes", "prediccion"]].copy()
fut_display["prediccion"] = fut_display["prediccion"].apply(formato_numero)
st.dataframe(
    fut_display.rename(columns={"mes": "Mes", "prediccion": "Predicción"}),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Exportar
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📥 Exportar Resultados")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.download_button(
        label="📥 Descargar Predicciones (CSV)",
        data=df_futuro.to_csv(index=False),
        file_name="predicciones_futuras.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    with st.spinner("Generando informe PDF..."):
        pdf_bytes = generar_pdf(
            df_raw=df_raw,
            df_serie=df_serie,
            comparativa=comparativa,
            df_futuro=df_futuro,
            metricas=metricas,
            trend=resultado.trend,
            seasonal=resultado.seasonal,
            total_real=total_real,
            total_predicho=total_predicho,
            correccion_pandemia=corregir_2020,
        )
    st.download_button(
        label="📄 Descargar Informe PDF",
        data=pdf_bytes,
        file_name="informe_prediccion_heladerias.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

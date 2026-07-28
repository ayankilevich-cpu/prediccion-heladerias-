"""
app_prediccion_heladerias.py
----------------------------
Interfaz de usuario Streamlit para la app de predicción de heladerías.

Este archivo SOLO maneja la UI: layout, inputs, outputs visuales.
Toda la lógica de negocio vive en src/.

Para ejecutar:
    streamlit run app_prediccion_heladerias.py
"""

import hashlib
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
# Paleta de marca (AGP Blema)
# ---------------------------------------------------------------------------

TEAL = "#3FBF9F"
TEAL_DARK = "#2E9B80"
TEAL_SOFT = "#EAF8F4"
BROWN = "#6B4730"
ORANGE = "#D97757"
BG = "#F6F8F7"
CARD = "#FFFFFF"
BORDER = "#E3EFEA"
TEXT = "#3A2B22"
MUTED = "#8C7B6E"
GREEN_OK = "#1E9E6D"
RED_BAD = "#C0392B"

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AGP Blema | Estimación de la Demanda",
    page_icon=_LOGO_PATH,
    layout="wide",
)


# ---------------------------------------------------------------------------
# CSS global: look de dashboard (tarjetas redondeadas, tabs tipo pill, etc.)
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
    }}

    /* --- Topbar --- */
    .agp-topbar {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 18px 28px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(58, 43, 34, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }}
    .agp-topbar-title h1 {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {TEXT};
        margin: 0;
    }}
    .agp-topbar-title p {{
        font-size: 0.92rem;
        color: {MUTED};
        margin: 2px 0 0 0;
    }}
    .agp-badge {{
        background: {TEAL_SOFT};
        color: {TEAL_DARK};
        font-weight: 600;
        font-size: 0.82rem;
        padding: 6px 16px;
        border-radius: 999px;
        border: 1px solid {BORDER};
        white-space: nowrap;
    }}

    /* --- Tarjetas KPI --- */
    .kpi-grid {{
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        margin-bottom: 18px;
    }}
    .kpi-card {{
        flex: 1;
        min-width: 160px;
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(58, 43, 34, 0.04);
    }}
    .kpi-card .kpi-label {{
        font-size: 0.78rem;
        color: {MUTED};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }}
    .kpi-card .kpi-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.2;
    }}
    .kpi-delta {{
        display: inline-block;
        margin-top: 8px;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }}
    .kpi-delta.pos {{ background: #E4F8EF; color: {GREEN_OK}; }}
    .kpi-delta.neg {{ background: #FDEDEA; color: {RED_BAD}; }}
    .kpi-delta.neutral {{ background: {TEAL_SOFT}; color: {TEAL_DARK}; }}

    /* --- Secciones en tarjeta --- */
    .agp-section {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px rgba(58, 43, 34, 0.04);
    }}
    .agp-section h3 {{
        margin-top: 0;
        color: {TEXT};
        font-size: 1.05rem;
    }}

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {BORDER};
    }}

    /* --- Botones --- */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 999px !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {TEAL} !important;
        border-color: {TEAL} !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {TEAL_DARK} !important;
        border-color: {TEAL_DARK} !important;
    }}

    /* --- Tabs tipo pill --- */
    div[data-baseweb="tab-list"] {{
        gap: 6px;
        background: {TEAL_SOFT};
        padding: 6px;
        border-radius: 999px;
        margin-bottom: 18px;
    }}
    button[data-baseweb="tab"] {{
        border-radius: 999px !important;
        padding: 8px 20px !important;
        color: {MUTED} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: {TEAL} !important;
        color: white !important;
    }}
    div[data-baseweb="tab-highlight"] {{ display: none; }}
    div[data-baseweb="tab-border"] {{ display: none; }}

    /* --- Tablas y dataframes --- */
    [data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid {BORDER};
    }}

    /* --- Hero (estado sin archivo) --- */
    .av-hero {{
        background: linear-gradient(135deg, {TEAL_SOFT} 0%, #FEFEFC 100%);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 28px;
        border: 1px solid {BORDER};
    }}
    .av-hero h2 {{
        color: {TEXT};
        font-size: 1.6rem;
        margin: 0 0 8px 0;
        font-weight: 700;
    }}
    .av-steps {{
        display: flex;
        gap: 16px;
        margin: 28px 0 8px 0;
        flex-wrap: wrap;
    }}
    .av-step {{
        flex: 1;
        min-width: 180px;
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
    }}
    .av-step .av-num {{
        display: inline-block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        border-radius: 50%;
        background: {TEAL};
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }}
    .av-step .av-icon {{
        font-size: 1.8rem;
        display: block;
        margin-bottom: 6px;
    }}
    .av-step .av-title {{
        color: {TEXT};
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .av-benefits {{
        margin-top: 24px;
        padding-top: 20px;
        border-top: 1px solid {BORDER};
    }}
    .av-benefits .av-benefits-title {{
        color: {TEXT};
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }}
    .av-benefit-item {{
        color: {MUTED};
        font-size: 0.92rem;
        margin-bottom: 6px;
    }}
    .av-benefit-item b {{ color: {TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def kpi_row(items: list[dict]) -> None:
    """Renderiza una fila de tarjetas KPI estilo dashboard.

    items: lista de dicts con keys: label, value, delta (opcional), tono (pos/neg/neutral).
    """
    cards_html = []
    for item in items:
        delta_html = ""
        if item.get("delta"):
            tono = item.get("tono", "neutral")
            delta_html = f'<span class="kpi-delta {tono}">{item["delta"]}</span><br/>'
        cards_html.append(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{item['label']}</div>
                <div class="kpi-value">{item['value']}</div>
                {delta_html}
            </div>
            """
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def aplicar_tema_plotly(fig, height: int | None = None):
    """Aplica un tema visual consistente (fuente, colores, grillas) a un gráfico Plotly."""
    fig.update_layout(
        font=dict(family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif", color=TEXT, size=13),
        plot_bgcolor=CARD,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=55, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        title_font=dict(size=15, color=TEXT),
        colorway=[TEAL, BROWN, ORANGE],
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT, bordercolor=BORDER),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(color=MUTED))
    if height:
        fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Topbar
# ---------------------------------------------------------------------------

col_logo, col_titulo, col_badge = st.columns([0.6, 4, 1.4], vertical_alignment="center")
with col_logo:
    st.image(_LOGO_PATH, width=64)
with col_titulo:
    st.markdown(
        """
        <div class="agp-topbar-title">
            <h1>AGP Blema</h1>
            <p>Estimación de la Demanda para Heladerías</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown('<div class="agp-badge">🍦 Holt-Winters</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: configuración
# ---------------------------------------------------------------------------

st.sidebar.image(_LOGO_PATH, width=110)
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
# Estado vacío: sin archivo cargado
# ---------------------------------------------------------------------------

if uploaded_file is None:
    st.markdown(
        """
        <div class="av-hero">
            <h2>Predecí, planificá y maximizá la rentabilidad de tu heladería.</h2>
            <div class="av-steps">
                <div class="av-step">
                    <span class="av-num">1</span>
                    <span class="av-icon">📥</span>
                    <div class="av-title">Descargá la plantilla</div>
                </div>
                <div class="av-step">
                    <span class="av-num">2</span>
                    <span class="av-icon">📝</span>
                    <div class="av-title">Completá tus ventas mensuales</div>
                </div>
                <div class="av-step">
                    <span class="av-num">3</span>
                    <span class="av-icon">🔮</span>
                    <div class="av-title">Subí el archivo y mirá la predicción</div>
                </div>
            </div>
            <div class="av-benefits">
                <div class="av-benefits-title">¿Qué vas a obtener?</div>
                <div class="av-benefit-item">✓ <b>Predicción</b> de los próximos 12 meses de demanda</div>
                <div class="av-benefit-item">✓ El <b>margen de error</b> del modelo, para saber qué tan confiable es</div>
                <div class="av-benefit-item">✓ Un <b>informe en PDF</b> listo para compartir con tu equipo</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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

# 3. Transformación (se necesita en todas las pestañas)
df_serie = transformar_a_serie(df_raw)
if corregir_2020:
    df_serie = corregir_pandemia(df_serie)

# ---------------------------------------------------------------------------
# Firma de la corrida actual: si cambia el archivo o los parámetros,
# se invalida el resultado del modelo guardado en session_state.
# ---------------------------------------------------------------------------

firma_actual = hashlib.md5(
    f"{uploaded_file.name}-{uploaded_file.size}-{meses_validacion}-{corregir_2020}".encode()
).hexdigest()

if st.session_state.get("agp_firma") != firma_actual:
    st.session_state["agp_firma"] = firma_actual
    st.session_state["agp_resultado"] = None

resultado = st.session_state.get("agp_resultado")


# ---------------------------------------------------------------------------
# Tabs de navegación (estilo dashboard)
# ---------------------------------------------------------------------------

tab_datos, tab_modelo, tab_prediccion, tab_export = st.tabs(
    ["📊 Datos & Serie", "🚀 Modelo & Validación", "🔮 Predicción", "📥 Exportar"]
)

# ---------------------------------------------------------------------------
# Tab 1: Datos y serie temporal
# ---------------------------------------------------------------------------

with tab_datos:
    kpi_row([
        {"label": "Filas cargadas", "value": df_raw.shape[0]},
        {"label": "Columnas", "value": df_raw.shape[1]},
        {"label": "Meses en la serie", "value": len(df_serie)},
        {
            "label": "Corrección pandemia",
            "value": "Aplicada" if corregir_2020 else "No aplicada",
            "delta": "✓ 2020 ajustado" if corregir_2020 else None,
            "tono": "neutral",
        },
    ])

    st.markdown('<div class="agp-section">', unsafe_allow_html=True)
    st.markdown("### 📋 Vista previa de los datos")
    st.dataframe(df_raw, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="agp-section">', unsafe_allow_html=True)
    st.markdown("### 📈 Serie temporal de ventas")
    fig_serie = px.line(
        df_serie, x="fecha", y="ventas",
        labels={"fecha": "Fecha", "ventas": "Ventas"},
    )
    fig_serie.update_traces(line_color=TEAL, line_width=2.5)
    tickvals, ticktext = etiquetas_eje_fecha_es(df_serie["fecha"])
    fig_serie.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    fig_serie.update_layout(hovermode="x unified")
    aplicar_tema_plotly(fig_serie, height=380)
    st.plotly_chart(fig_serie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True):
        with st.spinner("Evaluando combinaciones de parámetros..."):
            try:
                resultado = ejecutar_pipeline(df_serie, n_meses_validacion=int(meses_validacion))
                st.session_state["agp_resultado"] = resultado
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
        st.rerun()


# ---------------------------------------------------------------------------
# A partir de acá, todo depende de que el modelo ya se haya ejecutado.
# ---------------------------------------------------------------------------

if resultado is None:
    for tab in (tab_modelo, tab_prediccion, tab_export):
        with tab:
            st.info("👈 Cargá tus parámetros y tocá **Ejecutar Modelo** en la pestaña “Datos & Serie” para ver esta sección.")
    st.stop()

train = resultado.train
test = resultado.test
metricas = resultado.metricas
predicciones_val = resultado.predicciones_validacion
df_futuro = resultado.df_futuro

periodo_inicio = mes_anio_es(test["fecha"].min(), abreviado=True)
periodo_fin = mes_anio_es(test["fecha"].max(), abreviado=True)

total_real = test["ventas"].sum()
total_predicho = predicciones_val.sum()
error_total_pct = (metricas.error_absoluto_total / total_real * 100) if total_real else 0

comparativa = test[["fecha", "ventas"]].copy()
comparativa["prediccion"] = predicciones_val.values
comparativa["error"] = abs(comparativa["ventas"] - comparativa["prediccion"])
comparativa["error_pct"] = comparativa["error"] / comparativa["ventas"] * 100
comparativa["mes"] = comparativa["fecha"].apply(lambda f: mes_anio_es(f, abreviado=True))

diferencia = total_real - total_predicho
diferencia_pct = (diferencia / total_real * 100) if total_real else 0


# ---------------------------------------------------------------------------
# Tab 2: Modelo y validación
# ---------------------------------------------------------------------------

with tab_modelo:
    st.markdown(
        f'<div class="agp-badge">📊 Validación: últimos {len(test)} meses '
        f'({periodo_inicio} – {periodo_fin}) · trend={resultado.trend or "ninguna"} · '
        f'seasonal={resultado.seasonal}</div><br/><br/>',
        unsafe_allow_html=True,
    )

    kpi_row([
        {"label": "MAE", "value": formato_numero(metricas.mae)},
        {"label": "RMSE", "value": formato_numero(metricas.rmse)},
        {"label": "R²", "value": f"{metricas.r2:.4f}"},
        {"label": "Error Abs. Total", "value": formato_numero(metricas.error_absoluto_total)},
        {
            "label": "Error Abs. %",
            "value": f"{error_total_pct:.2f}%".replace(".", ","),
            "delta": "Bajo error" if error_total_pct < 10 else "Revisar ajuste",
            "tono": "pos" if error_total_pct < 10 else "neg",
        },
    ])

    st.markdown('<div class="agp-section">', unsafe_allow_html=True)
    st.markdown(f"### 🔍 Predicción vs Real ({periodo_inicio} – {periodo_fin})")

    kpi_row([
        {"label": "Total Real", "value": formato_numero(total_real)},
        {"label": "Total Predicción", "value": formato_numero(total_predicho)},
        {"label": "Diferencia", "value": formato_numero(diferencia)},
        {
            "label": "Diferencia %",
            "value": f"{diferencia_pct:.2f}%".replace(".", ","),
            "delta": "A favor" if diferencia >= 0 else "En contra",
            "tono": "pos" if diferencia >= 0 else "neg",
        },
    ])

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="Real", x=comparativa["mes"], y=comparativa["ventas"],
                              marker_color=BROWN))
    fig_comp.add_trace(go.Bar(name="Predicción", x=comparativa["mes"], y=comparativa["prediccion"],
                              marker_color=TEAL))
    fig_comp.update_layout(barmode="group", xaxis_title="Mes", yaxis_title="Ventas")
    aplicar_tema_plotly(fig_comp, height=380)
    st.plotly_chart(fig_comp, use_container_width=True)

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
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 3: Predicciones futuras
# ---------------------------------------------------------------------------

with tab_prediccion:
    st.markdown('<div class="agp-section">', unsafe_allow_html=True)
    st.markdown("### 🔮 Predicciones futuras (próximos 12 meses)")

    kpi_row([
        {"label": "Total Anual Predicho", "value": formato_numero(df_futuro["prediccion"].sum())},
        {"label": "Promedio Mensual", "value": formato_numero(df_futuro["prediccion"].mean())},
        {"label": "Mes Pico", "value": df_futuro.loc[df_futuro["prediccion"].idxmax(), "mes"]},
    ])

    fig_fut = make_subplots(rows=1, cols=2,
                            subplot_titles=("Serie Completa", "Predicciones Futuras"))
    df_historico = df_serie.dropna(subset=["ventas"])

    fig_fut.add_trace(
        go.Scatter(x=df_historico["fecha"], y=df_historico["ventas"],
                   name="Histórico", line=dict(color=BROWN)),
        row=1, col=1,
    )
    fig_fut.add_trace(
        go.Scatter(x=df_futuro["fecha"], y=df_futuro["prediccion"],
                   name="Predicción", line=dict(color=ORANGE, dash="dash")),
        row=1, col=1,
    )
    fig_fut.add_trace(
        go.Bar(x=df_futuro["mes"], y=df_futuro["prediccion"],
               name="Predicción Mensual", marker_color=TEAL),
        row=1, col=2,
    )
    tickvals_hist, ticktext_hist = etiquetas_eje_fecha_es(
        pd.concat([df_historico["fecha"], df_futuro["fecha"]])
    )
    fig_fut.update_xaxes(tickvals=tickvals_hist, ticktext=ticktext_hist, row=1, col=1)
    aplicar_tema_plotly(fig_fut, height=420)
    st.plotly_chart(fig_fut, use_container_width=True)

    fut_display = df_futuro[["mes", "prediccion"]].copy()
    fut_display["prediccion"] = fut_display["prediccion"].apply(formato_numero)
    st.dataframe(
        fut_display.rename(columns={"mes": "Mes", "prediccion": "Predicción"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4: Exportar
# ---------------------------------------------------------------------------

with tab_export:
    st.markdown('<div class="agp-section">', unsafe_allow_html=True)
    st.markdown("### 📥 Exportar resultados")

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
    st.markdown('</div>', unsafe_allow_html=True)

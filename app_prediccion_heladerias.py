"""
App de Predicción de Demanda para Heladerías
=============================================
Aplicación Streamlit para analizar y predecir ventas usando Holt-Winters.

Para ejecutar:
    streamlit run app_prediccion_heladerias.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import io
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import locale

def formato_numero(numero, decimales=2):
    """Formatea número con punto como separador de miles y coma como decimal."""
    formato = f"{{:,.{decimales}f}}"
    texto = formato.format(numero)
    # Convertir de formato US (1,234.56) a formato AR (1.234,56)
    texto = texto.replace(',', 'X').replace('.', ',').replace('X', '.')
    return texto


_MESES_ES = (
    'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
)

_ABREV_MES_ENCABEZADO = {
    'ENERO': 'Ene', 'FEBRERO': 'Feb', 'MARZO': 'Mar', 'ABRIL': 'Abr',
    'MAYO': 'May', 'JUNIO': 'Jun', 'JULIO': 'Jul', 'AGOSTO': 'Ago',
    'SEPTIEMBRE': 'Sep', 'OCTUBRE': 'Oct', 'NOVIEMBRE': 'Nov', 'DICIEMBRE': 'Dic',
}


def _fecha_texto_es(fecha):
    """Mes y año en español (evita depender del locale del sistema)."""
    if fecha is None or pd.isna(fecha):
        return ''
    t = pd.Timestamp(fecha)
    return f'{_MESES_ES[t.month - 1]} {t.year}'


def _limpiar_texto_encabezado_pdf(s):
    """Quita BOM y corrige texto típico mal decodificado (p. ej. AÃ±o -> Año)."""
    if s is None:
        return ''
    t = str(s).strip()
    if t.startswith('\ufeff'):
        t = t[1:].strip()
    if 'Ã' in t:
        try:
            t = t.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return t.strip()


def _encabezado_columna_datos_raw(nombre_columna):
    """Encabezado corto para la tabla del CSV en el PDF (meses abreviados)."""
    t = _limpiar_texto_encabezado_pdf(nombre_columna)
    if t.lower() in ('año', 'ano'):
        return 'Año'
    u = t.upper().strip()
    return _ABREV_MES_ENCABEZADO.get(u, t[:8])


def _celda_preview_csv(valor, max_len=11):
    """Texto legible para celdas del CSV: sin nan, enteros sin .0."""
    if pd.isna(valor):
        return '-'
    if isinstance(valor, (np.integer, int)):
        return str(int(valor))
    if isinstance(valor, (np.floating, float)):
        if np.isnan(valor):
            return '-'
        if abs(valor - round(valor)) < 1e-9:
            return str(int(round(valor)))
        return formato_numero(float(valor), 1)
    s = str(valor).strip()
    if s.lower() in ('nan', 'none', ''):
        return '-'
    if len(s) > max_len:
        return s[: max_len - 1] + '…'
    return s


# Configuración de la página
st.set_page_config(
    page_title="Predicción de Demanda - Heladerías",
    page_icon="🍦",
    layout="wide"
)

# Título principal
st.title("🍦 Predicción de Demanda para Heladerías")
st.markdown("### Modelo Holt-Winters para Series Temporales")
st.markdown("---")

# ============================================================================
# FUNCIONES DEL MODELO
# ============================================================================

def cargar_y_transformar_datos(df):
    """Transforma el DataFrame del formato ancho al formato fecha-ventas."""
    # Obtener el nombre de la primera columna (AÑO)
    id_column = df.columns[0]
    
    # Melt para transformar de ancho a largo
    df_melted = df.melt(id_vars=[id_column], var_name='mes', value_name='ventas')
    df_melted = df_melted.rename(columns={id_column: 'AÑO'})
    
    # Limpiar datos vacíos (strings vacíos)
    df_melted = df_melted[df_melted['ventas'].astype(str).str.strip() != '']
    
    # Detectar formato de números
    muestra = df_melted['ventas'].dropna().iloc[0] if len(df_melted['ventas'].dropna()) > 0 else None
    
    if muestra is not None and isinstance(muestra, str):
        if ',' in str(muestra) and '.' in str(muestra):
            df_melted['ventas'] = df_melted['ventas'].astype(str).str.replace('.', '', regex=False)
            df_melted['ventas'] = df_melted['ventas'].str.replace(',', '.', regex=False)
        elif ',' in str(muestra) and '.' not in str(muestra):
            df_melted['ventas'] = df_melted['ventas'].astype(str).str.replace(',', '.', regex=False)
    
    df_melted['ventas'] = pd.to_numeric(df_melted['ventas'], errors='coerce')
    
    # Mapeo de meses
    meses = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    df_melted['mes_num'] = df_melted['mes'].str.upper().map(meses)
    df_melted['fecha'] = pd.to_datetime(dict(year=df_melted['AÑO'], month=df_melted['mes_num'], day=1))
    
    df_final = df_melted[['fecha', 'ventas']].sort_values('fecha').reset_index(drop=True)
    
    # Encontrar la última fecha con dato real (no NaN)
    ultima_fecha_real = df_final[df_final['ventas'].notna()]['fecha'].max()
    
    # Filtrar solo hasta la última fecha con dato real (eliminar meses futuros sin datos)
    df_final = df_final[df_final['fecha'] <= ultima_fecha_real].copy()
    
    # Imputar valores NaN SOLO para meses históricos (dentro del rango de datos)
    if df_final['ventas'].isna().sum() > 0:
        df_final['mes'] = df_final['fecha'].dt.month
        promedios = df_final.groupby('mes')['ventas'].mean()
        
        for idx in df_final[df_final['ventas'].isna()].index:
            mes = df_final.loc[idx, 'mes']
            df_final.loc[idx, 'ventas'] = promedios[mes]
        
        df_final = df_final.drop(columns=['mes'])
    
    return df_final


def corregir_pandemia(df, meses_afectados):
    """Corrige valores atípicos de la pandemia."""
    df = df.copy()
    df_sin_2020 = df[df['fecha'].dt.year != 2020]
    
    for mes, tipo in meses_afectados.items():
        fecha_str = f'2020-{mes:02d}-01'
        promedio = df_sin_2020[df_sin_2020['fecha'].dt.month == mes]['ventas'].mean()
        
        if tipo == 'promedio':
            df.loc[df['fecha'] == fecha_str, 'ventas'] = promedio
        elif tipo == 'mixto':
            valor_real = df.loc[df['fecha'] == fecha_str, 'ventas'].values
            if len(valor_real) > 0:
                df.loc[df['fecha'] == fecha_str, 'ventas'] = (promedio + valor_real[0]) / 2
    
    return df


def entrenar_modelo(train, test, trend, seasonal):
    """
    Entrena el modelo usando la optimización automática de statsmodels.
    Esta es la forma más precisa de ajustar los parámetros del modelo.
    """
    # Usar optimización automática de statsmodels (sin parámetros manuales)
    modelo = ExponentialSmoothing(
        train['ventas'],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12
    ).fit()  # Sin argumentos = optimización automática
    
    predicciones = modelo.forecast(len(test))
    
    mae = mean_absolute_error(test['ventas'], predicciones)
    rmse = np.sqrt(mean_squared_error(test['ventas'], predicciones))
    r2 = r2_score(test['ventas'], predicciones)
    error_total = np.abs(test['ventas'].values - predicciones.values).sum()
    
    return modelo, predicciones, {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Error Absoluto Total': error_total
    }


# ============================================================================
# GENERACIÓN DE INFORME PDF (usa matplotlib para los gráficos, sin Chrome)
# ============================================================================

def _fig_a_png_bytes(fig_mpl):
    """Convierte una figura matplotlib a bytes PNG en memoria."""
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig_mpl)
    buf.seek(0)
    return buf


def _crear_grafico_serie(df):
    """Crea gráfico de serie temporal con matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['fecha'], df['ventas'], color='#1f77b4', linewidth=1.8)
    ax.set_title('Ventas Históricas', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Fecha', fontsize=10)
    ax.set_ylabel('Ventas', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: formato_numero(x, 0)))
    fig.tight_layout()
    return fig


def _crear_grafico_comparacion(comparativa):
    """Crea gráfico de barras comparativo con matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(comparativa))
    ancho = 0.35
    ax.bar(x - ancho/2, comparativa['ventas'], ancho, label='Real', color='#2ecc71')
    ax.bar(x + ancho/2, comparativa['prediccion'], ancho, label='Predicción', color='#3498db')
    ax.set_title('Comparación: Ventas Reales vs Predicción', fontsize=13,
                 fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Mes', fontsize=10)
    ax.set_ylabel('Ventas', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(comparativa['mes'], rotation=45, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_numero(v, 0)))
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def _crear_grafico_futuro(df_historico, df_futuro):
    """Crea gráfico de predicciones futuras con matplotlib (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(df_historico['fecha'], df_historico['ventas'], color='#2ecc71',
             linewidth=1.5, label='Histórico')
    ax1.plot(df_futuro['fecha'], df_futuro['prediccion'], color='#e74c3c',
             linewidth=1.5, linestyle='--', label='Predicción')
    ax1.set_title('Serie Completa', fontsize=11, fontweight='bold', color='#2c3e50')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_numero(v, 0)))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(df_futuro)), df_futuro['prediccion'], color='#3498db')
    ax2.set_title('Predicciones Futuras', fontsize=11, fontweight='bold', color='#2c3e50')
    ax2.set_xticks(range(len(df_futuro)))
    ax2.set_xticklabels(df_futuro['mes'], rotation=45, ha='right', fontsize=7)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_numero(v, 0)))
    ax2.grid(True, axis='y', alpha=0.3)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


class InformePDF(FPDF):
    """PDF personalizado con encabezado y pie de página."""

    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Informe de Predicción de Demanda - Heladerías', 0, 1, 'R')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def titulo_seccion(self, texto):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(44, 62, 80)
        self.set_fill_color(236, 240, 241)
        self.cell(0, 10, texto, 0, 1, 'L', fill=True)
        self.ln(3)

    def subtitulo(self, texto):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, texto, 0, 1, 'L')
        self.ln(1)

    def texto_normal(self, texto):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, texto)
        self.ln(2)

    def agregar_metrica(self, nombre, valor, x, y, ancho=45):
        self.set_xy(x, y)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(100, 100, 100)
        self.cell(ancho, 5, nombre, 0, 2, 'C')
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(44, 62, 80)
        self.cell(ancho, 7, str(valor), 0, 2, 'C')

    def agregar_tabla(self, encabezados, filas, anchos=None, *,
                      tam_encabezado=9, tam_celdas=8, alto_encabezado=8, alto_fila=7):
        if anchos is None:
            ancho_total = self.w - 20
            anchos = [ancho_total / len(encabezados)] * len(encabezados)

        self.set_font('Helvetica', 'B', tam_encabezado)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        for i, enc in enumerate(encabezados):
            self.cell(anchos[i], alto_encabezado, str(enc), 1, 0, 'C', fill=True)
        self.ln()

        self.set_font('Helvetica', '', tam_celdas)
        self.set_text_color(0, 0, 0)
        for idx, fila in enumerate(filas):
            if idx % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for i, celda in enumerate(fila):
                self.cell(anchos[i], alto_fila, str(celda), 1, 0, 'C', fill=True)
            self.ln()

    def agregar_imagen_de_bytes(self, img_buf, ancho=190):
        """Inserta una imagen desde un buffer BytesIO en el PDF."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(img_buf.read())
        try:
            alto_img = ancho * 0.45
            espacio = self.h - self.get_y() - 20
            if espacio < alto_img:
                self.add_page()
            self.image(tmp_path, x=10, w=ancho)
            self.ln(5)
        finally:
            os.unlink(tmp_path)


def generar_informe_pdf(df_raw, df, comparativa, df_futuro, df_historico, metricas,
                        mejor_trend, mejor_seasonal,
                        periodo_inicio, periodo_fin,
                        total_real, total_predicho, diferencia_total, diferencia_pct,
                        correccion_pandemia):
    """Genera el informe PDF completo con gráficos matplotlib (sin Chrome)."""
    pdf = InformePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Portada ---
    pdf.ln(20)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'Informe de Predicción de Demanda', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Modelo Holt-Winters para Heladerías', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, f'Fecha del informe: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')

    rango_datos = (
        f"{_fecha_texto_es(df['fecha'].min())} a {_fecha_texto_es(df['fecha'].max())}"
    )
    pdf.cell(0, 8, f'Rango de datos: {rango_datos}', 0, 1, 'C')
    pdf.cell(0, 8, f'Total de observaciones: {len(df)}', 0, 1, 'C')

    config_trend = mejor_trend if mejor_trend else 'Sin tendencia'
    pdf.cell(0, 8, f'Configuración: Tendencia={config_trend} | Estacionalidad={mejor_seasonal}', 0, 1, 'C')
    if correccion_pandemia:
        pdf.cell(0, 8, 'Corrección de pandemia 2020 aplicada', 0, 1, 'C')

    # --- Datos cargados ---
    pdf.add_page()
    pdf.titulo_seccion('1. Datos Cargados')
    pdf.texto_normal(
        f'Vista del archivo en formato ancho: {df_raw.shape[0]} filas y '
        f'{df_raw.shape[1]} columnas (típicamente una fila por año y una columna por mes). '
        f'La serie mensual transformada que usa el modelo abarca desde '
        f'{_fecha_texto_es(df["fecha"].min())} hasta {_fecha_texto_es(df["fecha"].max())} '
        f'({len(df)} meses con dato).'
    )

    cols_preview = list(df_raw.columns[: min(len(df_raw.columns), 13)])
    enc_raw = [_encabezado_columna_datos_raw(c) for c in cols_preview]
    max_cols = len(enc_raw)
    ancho_disponible = pdf.w - 20
    if max_cols <= 1:
        anchos_raw = [ancho_disponible]
    else:
        ancho_ano = 16.0
        resto = ancho_disponible - ancho_ano
        anchos_raw = [ancho_ano] + [resto / (max_cols - 1)] * (max_cols - 1)

    filas_raw = []
    for _, row in df_raw.head(10).iterrows():
        fila = [_celda_preview_csv(row[c]) for c in cols_preview]
        filas_raw.append(fila)
    if len(df_raw) > 10:
        filas_raw.append(['...' if i == 0 else '-' for i in range(max_cols)])

    pdf.agregar_tabla(
        enc_raw,
        filas_raw,
        anchos_raw,
        tam_encabezado=6,
        tam_celdas=6,
        alto_encabezado=6,
        alto_fila=5,
    )
    pdf.ln(5)

    # --- Serie temporal ---
    pdf.titulo_seccion('2. Serie Temporal de Ventas')
    fig_serie_mpl = _crear_grafico_serie(df)
    pdf.agregar_imagen_de_bytes(_fig_a_png_bytes(fig_serie_mpl))

    # --- Resultados del modelo ---
    pdf.add_page()
    pdf.titulo_seccion('3. Resultados del Modelo')

    pdf.subtitulo('Métricas de rendimiento')
    y_metricas = pdf.get_y()
    pdf.agregar_metrica('MAE', formato_numero(metricas['MAE']), 10, y_metricas, 38)
    pdf.agregar_metrica('RMSE', formato_numero(metricas['RMSE']), 48, y_metricas, 38)
    pdf.agregar_metrica('R²', f"{metricas['R²']:.4f}", 86, y_metricas, 38)
    pdf.agregar_metrica('Error Abs. Total (kg)', formato_numero(metricas['Error Absoluto Total']), 124, y_metricas, 38)
    error_pct = (metricas['Error Absoluto Total'] / total_real * 100) if total_real else 0
    pdf.agregar_metrica('Error Abs. Total %', f"{error_pct:.2f}%".replace('.', ','), 162, y_metricas, 38)
    pdf.set_y(y_metricas + 20)
    pdf.ln(5)

    # --- Validación ---
    pdf.titulo_seccion(f'4. Validación: Real vs Predicho ({periodo_inicio} - {periodo_fin})')

    pdf.subtitulo('Totales del período de validación')
    y_totales = pdf.get_y()
    pdf.agregar_metrica('Total Venta Real (kg)', formato_numero(total_real), 10, y_totales, 47)
    pdf.agregar_metrica('Total Predicción (kg)', formato_numero(total_predicho), 57, y_totales, 47)
    pdf.agregar_metrica('Diferencia (kg)', formato_numero(diferencia_total), 104, y_totales, 47)
    pdf.agregar_metrica('Diferencia %', f"{diferencia_pct:.2f}%".replace('.', ','), 151, y_totales, 47)
    pdf.set_y(y_totales + 20)
    pdf.ln(5)

    pdf.subtitulo('Gráfico comparativo')
    fig_comp_mpl = _crear_grafico_comparacion(comparativa)
    pdf.agregar_imagen_de_bytes(_fig_a_png_bytes(fig_comp_mpl))

    pdf.subtitulo('Detalle mensual')
    enc_comp = ['Mes', 'Venta Real', 'Predicción', 'Error', 'Error %']
    anchos_comp = [35, 40, 40, 40, 35]
    filas_comp = []
    for _, row in comparativa.iterrows():
        filas_comp.append([
            row['mes'],
            formato_numero(row['ventas']),
            formato_numero(row['prediccion']),
            formato_numero(row['error']),
            f"{row['error_pct']:.2f}%".replace('.', ',')
        ])
    pdf.agregar_tabla(enc_comp, filas_comp, anchos_comp)
    pdf.ln(5)

    # --- Predicciones futuras ---
    pdf.add_page()
    pdf.titulo_seccion('5. Predicciones Futuras (Próximos 12 meses)')

    pdf.subtitulo('Resumen')
    y_res = pdf.get_y()
    pdf.agregar_metrica('Total Anual Predicho (kg)', formato_numero(df_futuro['prediccion'].sum()), 10, y_res, 63)
    pdf.agregar_metrica('Promedio Mensual (kg)', formato_numero(df_futuro['prediccion'].mean()), 73, y_res, 63)
    mes_pico = df_futuro.loc[df_futuro['prediccion'].idxmax(), 'mes']
    pdf.agregar_metrica('Mes Pico', mes_pico, 136, y_res, 63)
    pdf.set_y(y_res + 20)
    pdf.ln(5)

    pdf.subtitulo('Gráfico de predicciones')
    fig_fut_mpl = _crear_grafico_futuro(df_historico, df_futuro)
    pdf.agregar_imagen_de_bytes(_fig_a_png_bytes(fig_fut_mpl))

    pdf.subtitulo('Detalle mensual de predicciones')
    enc_fut = ['Mes', 'Predicción (kg)']
    anchos_fut = [100, 90]
    filas_fut = []
    for _, row in df_futuro.iterrows():
        filas_fut.append([row['mes'], formato_numero(row['prediccion'])])
    pdf.agregar_tabla(enc_fut, filas_fut, anchos_fut)

    # fpdf2 devuelve bytearray con output(); Streamlit solo acepta bytes/str/archivo.
    # Escribir en BytesIO garantiza tipo bytes vía getvalue().
    buf_pdf = io.BytesIO()
    pdf.output(buf_pdf)
    return buf_pdf.getvalue()


# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Cargar archivo
st.sidebar.subheader("1. Cargar datos")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona el archivo CSV",
    type=['csv'],
    help="El archivo debe tener columnas: AÑO, ENERO, FEBRERO, ..., DICIEMBRE"
)

# Parámetros del modelo
st.sidebar.subheader("2. Parámetros")

separador = st.sidebar.selectbox(
    "Separador del CSV",
    [';', ',', '\t'],
    index=0
)

encoding = st.sidebar.selectbox(
    "Codificación",
    ['latin1', 'utf-8', 'cp1252'],
    index=0
)

meses_validacion = st.sidebar.number_input(
    "Meses de validación",
    min_value=3,
    max_value=24,
    value=12,
    help="Cantidad de meses recientes con datos reales para validar el modelo (ventana deslizante)"
)

corregir_2020 = st.sidebar.checkbox(
    "Corregir datos de pandemia (2020)",
    value=False,
    help="Ajusta marzo y abril 2020 usando promedios históricos"
)

# Nota: La optimización automática de statsmodels se usa siempre (es más precisa)

# ============================================================================
# PROCESAMIENTO Y VISUALIZACIÓN
# ============================================================================

if uploaded_file is not None:
    # Cargar datos
    try:
        df_raw = pd.read_csv(uploaded_file, sep=separador, encoding=encoding)
        
        # Mostrar datos cargados
        st.subheader("📊 Datos Cargados")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_raw, use_container_width=True)
        
        with col2:
            st.metric("Filas", df_raw.shape[0])
            st.metric("Columnas", df_raw.shape[1])
        
        # Transformar datos
        df = cargar_y_transformar_datos(df_raw)
        
        # Corregir pandemia si está activado
        if corregir_2020:
            df = corregir_pandemia(df, {3: 'mixto', 4: 'promedio'})
            st.info("✅ Se aplicó corrección de datos de pandemia (marzo y abril 2020)")
        
        # Mostrar serie temporal
        st.subheader("📈 Serie Temporal de Ventas")
        
        fig_serie = px.line(
            df, x='fecha', y='ventas',
            title='Ventas Históricas',
            labels={'fecha': 'Fecha', 'ventas': 'Ventas'}
        )
        fig_serie.update_traces(line_color='#1f77b4', line_width=2)
        fig_serie.update_layout(hovermode='x unified')
        st.plotly_chart(fig_serie, use_container_width=True)
        
        # División train/test: siempre usar los últimos N meses con datos reales
        df = df.dropna(subset=['ventas']).sort_values('fecha').reset_index(drop=True)
        
        n_val = min(meses_validacion, len(df) - 12)
        if n_val <= 0:
            n_val = len(df) // 2
        
        train = df.iloc[:-n_val].copy()
        test = df.iloc[-n_val:].copy()
        
        periodo_inicio = test['fecha'].min().strftime('%b %Y')
        periodo_fin = test['fecha'].max().strftime('%b %Y')
        st.info(f"📊 Validación: últimos **{len(test)} meses** con datos reales ({periodo_inicio} - {periodo_fin})")
        
        st.markdown("---")
        
        # Botón para ejecutar el modelo
        if st.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True):
            
            with st.spinner("Evaluando combinaciones de modelo..."):
                # Evaluar combinaciones
                combinaciones = [
                    ('additive', 'additive'),
                    ('additive', 'multiplicative'),
                    (None, 'additive'),
                    (None, 'multiplicative'),
                ]
                
                resultados = []
                for trend, seasonal in combinaciones:
                    try:
                        _, _, metricas = entrenar_modelo(train, test, trend, seasonal)
                        resultados.append({
                            'Tendencia': str(trend),
                            'Estacionalidad': seasonal,
                            'MAE': metricas['MAE'],
                            'R²': metricas['R²']
                        })
                    except:
                        pass
                
                if resultados:
                    mejor = min(resultados, key=lambda x: x['MAE'])
                    mejor_trend = None if mejor['Tendencia'] == 'None' else mejor['Tendencia']
                    mejor_seasonal = mejor['Estacionalidad']
            
            # Entrenar modelo final con optimización automática de statsmodels
            with st.spinner("Entrenando modelo con optimización automática..."):
                modelo_final, predicciones, _ = entrenar_modelo(train, test, mejor_trend, mejor_seasonal)
            
            # Calcular métricas finales
            mae = mean_absolute_error(test['ventas'], predicciones)
            rmse = np.sqrt(mean_squared_error(test['ventas'], predicciones))
            r2 = r2_score(test['ventas'], predicciones)
            error_total = np.abs(test['ventas'].values - predicciones.values).sum()
            error_total_pct = (error_total / test['ventas'].sum()) * 100
            
            st.markdown("---")
            st.subheader("📊 Resultados del Modelo")
            
            # Métricas en cards
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", formato_numero(mae))
            col2.metric("RMSE", formato_numero(rmse))
            col3.metric("R²", f"{r2:.4f}")
            col4.metric("Error Absoluto Total (kg)", formato_numero(error_total))
            col5.metric("Error Absoluto Total (%)", f"{error_total_pct:.2f}%".replace('.', ','))
            
            # Comparativa de validación
            st.subheader(f"🔍 Validación: Predicción vs Real ({periodo_inicio} - {periodo_fin})")
            
            comparativa = test[['fecha', 'ventas']].copy()
            comparativa['prediccion'] = predicciones.values
            comparativa['error'] = abs(comparativa['ventas'] - comparativa['prediccion'])
            comparativa['error_pct'] = (comparativa['error'] / comparativa['ventas'] * 100)
            comparativa['mes'] = comparativa['fecha'].dt.strftime('%b %Y')
            
            # Calcular totales
            total_real = comparativa['ventas'].sum()
            total_predicho = comparativa['prediccion'].sum()
            diferencia_total = total_real - total_predicho
            diferencia_pct = (diferencia_total / total_real * 100)
            
            # Mostrar totales en cards
            st.markdown("**Totales del período de validación:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Venta Real (kg)", formato_numero(total_real))
            col2.metric("Total Predicción (kg)", formato_numero(total_predicho))
            col3.metric("Diferencia (kg)", formato_numero(diferencia_total))
            col4.metric("Diferencia %", f"{diferencia_pct:.2f}%".replace('.', ','))
            
            # Gráfico de comparación
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='Real',
                x=comparativa['mes'],
                y=comparativa['ventas'],
                marker_color='#2ecc71'
            ))
            fig_comp.add_trace(go.Bar(
                name='Predicción',
                x=comparativa['mes'],
                y=comparativa['prediccion'],
                marker_color='#3498db'
            ))
            fig_comp.update_layout(
                title='Comparación: Ventas Reales vs Predicción',
                barmode='group',
                xaxis_title='Mes',
                yaxis_title='Ventas'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Preparar tabla con formato argentino
            comparativa_display = comparativa[['mes', 'ventas', 'prediccion', 'error', 'error_pct']].copy()
            comparativa_display['ventas'] = comparativa_display['ventas'].apply(lambda x: formato_numero(x))
            comparativa_display['prediccion'] = comparativa_display['prediccion'].apply(lambda x: formato_numero(x))
            comparativa_display['error'] = comparativa_display['error'].apply(lambda x: formato_numero(x))
            comparativa_display['error_pct'] = comparativa_display['error_pct'].apply(lambda x: f"{x:.2f}%".replace('.', ','))
            
            # Tabla de comparación
            st.dataframe(
                comparativa_display.rename(columns={
                    'mes': 'Mes',
                    'ventas': 'Venta Real',
                    'prediccion': 'Predicción',
                    'error': 'Error',
                    'error_pct': 'Error %'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # Predicciones futuras
            st.subheader("🔮 Predicciones Futuras (Próximos 12 meses)")
            
            # Filtrar solo datos con ventas reales (sin NaN)
            df_con_datos = df[df['ventas'].notna()].copy()
            
            # Re-entrenar con todos los datos reales
            modelo_futuro = ExponentialSmoothing(
                df_con_datos['ventas'],
                trend=mejor_trend,
                seasonal=mejor_seasonal,
                seasonal_periods=12
            ).fit()
            
            # Última fecha con datos reales
            ultima_fecha = df_con_datos['fecha'].max()
            fechas_futuras = pd.date_range(
                start=ultima_fecha + pd.DateOffset(months=1),
                periods=12,
                freq='MS'
            )
            
            pred_futuras = modelo_futuro.forecast(steps=12)
            
            df_futuro = pd.DataFrame({
                'fecha': fechas_futuras,
                'prediccion': pred_futuras.round(1)
            })
            df_futuro['mes'] = df_futuro['fecha'].dt.strftime('%B %Y')
            
            # Gráfico de predicciones futuras
            fig_futuro = make_subplots(rows=1, cols=2, subplot_titles=('Serie Completa', 'Predicciones Futuras'))
            
            # Serie histórica + predicción (solo datos reales)
            fig_futuro.add_trace(
                go.Scatter(x=df_con_datos['fecha'], y=df_con_datos['ventas'], name='Histórico', line=dict(color='#2ecc71')),
                row=1, col=1
            )
            fig_futuro.add_trace(
                go.Scatter(x=df_futuro['fecha'], y=df_futuro['prediccion'], name='Predicción', 
                          line=dict(color='#e74c3c', dash='dash')),
                row=1, col=1
            )
            
            # Solo predicciones
            fig_futuro.add_trace(
                go.Bar(x=df_futuro['mes'], y=df_futuro['prediccion'], name='Predicción Mensual',
                      marker_color='#3498db'),
                row=1, col=2
            )
            
            fig_futuro.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_futuro, use_container_width=True)
            
            # Resumen de predicciones
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Anual Predicho (kg)", formato_numero(df_futuro['prediccion'].sum()))
            col2.metric("Promedio Mensual (kg)", formato_numero(df_futuro['prediccion'].mean()))
            col3.metric("Mes Pico", f"{df_futuro.loc[df_futuro['prediccion'].idxmax(), 'mes']}")
            
            # Preparar tabla con formato argentino
            df_futuro_display = df_futuro[['mes', 'prediccion']].copy()
            df_futuro_display['prediccion'] = df_futuro_display['prediccion'].apply(lambda x: formato_numero(x))
            
            # Tabla de predicciones
            st.dataframe(
                df_futuro_display.rename(columns={
                    'mes': 'Mes',
                    'prediccion': 'Predicción (kg)'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Botones de descarga
            st.markdown("---")
            st.subheader("📥 Exportar Resultados")
            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                csv = df_futuro.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Predicciones (CSV)",
                    data=csv,
                    file_name="predicciones_futuras.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_dl2:
                with st.spinner("Generando informe PDF..."):
                    metricas_pdf = {
                        'MAE': mae,
                        'RMSE': rmse,
                        'R²': r2,
                        'Error Absoluto Total': error_total
                    }
                    pdf_bytes = generar_informe_pdf(
                        df_raw=df_raw,
                        df=df,
                        comparativa=comparativa,
                        df_futuro=df_futuro,
                        df_historico=df_con_datos,
                        metricas=metricas_pdf,
                        mejor_trend=mejor_trend,
                        mejor_seasonal=mejor_seasonal,
                        periodo_inicio=periodo_inicio,
                        periodo_fin=periodo_fin,
                        total_real=total_real,
                        total_predicho=total_predicho,
                        diferencia_total=diferencia_total,
                        diferencia_pct=diferencia_pct,
                        correccion_pandemia=corregir_2020
                    )
                st.download_button(
                    label="📄 Descargar Informe Completo (PDF)",
                    data=pdf_bytes,
                    file_name=f"informe_prediccion_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.success("✅ Análisis completado exitosamente!")
            
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        st.info("Verifica que el formato del CSV sea correcto (AÑO, ENERO, FEBRERO, ..., DICIEMBRE)")

else:
    # Mostrar instrucciones cuando no hay archivo
    st.info("👈 Carga un archivo CSV desde el panel lateral para comenzar")
    
    st.markdown("""
    ### 📋 Formato esperado del archivo CSV
    
    | AÑO | ENERO | FEBRERO | MARZO | ... | DICIEMBRE |
    |-----|-------|---------|-------|-----|-----------|
    | 2020 | 12345 | 11234 | 10123 | ... | 15678 |
    | 2021 | 13456 | 12345 | 11234 | ... | 16789 |
    
    ### 🔧 Parámetros configurables
    
    - **Separador CSV**: Caracter que separa las columnas (`;`, `,`, etc.)
    - **Meses de validación**: Cantidad de meses recientes con datos reales para validar el modelo
    - **Corrección pandemia**: Ajusta valores atípicos de marzo/abril 2020
    - **Optimización**: Busca automáticamente los mejores hiperparámetros
    """)

# Footer
st.markdown("---")
st.markdown("*Desarrollado con Streamlit y Holt-Winters | Modelo de Series Temporales para Heladerías*")

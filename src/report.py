"""
report.py
---------
Generación del informe PDF completo.
Usa matplotlib para los gráficos (sin dependencias de browser).
Sin dependencias de Streamlit.
"""

import io
import os
import tempfile
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla, debe ir antes de importar pyplot
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from fpdf import FPDF

from src.formatting import (
    celda_csv_preview,
    encabezado_columna_pdf,
    fecha_a_texto_es,
    formato_numero,
    mes_anio_es,
    texto_seguro_pdf,
    COLUMNAS_MES,
)


# ---------------------------------------------------------------------------
# Gráficos matplotlib -> bytes PNG
# ---------------------------------------------------------------------------

def _fig_a_png(fig) -> io.BytesIO:
    """Convierte una figura matplotlib a buffer PNG en memoria."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _formateador_eje_mes_anio(num, pos=None):
    """Formatea números de fecha de matplotlib como 'ene 2024' en español."""
    return mes_anio_es(mdates.num2date(num), abreviado=True)


def grafico_serie(df: pd.DataFrame) -> io.BytesIO:
    """Gráfico de línea de la serie temporal histórica."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["fecha"], df["ventas"], color="#1f77b4", linewidth=1.8)
    ax.set_title("Ventas Históricas", fontsize=13, fontweight="bold", color="#2c3e50")
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel("Ventas", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_formateador_eje_mes_anio))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: formato_numero(x, 0)))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return _fig_a_png(fig)


def grafico_comparacion(comparativa: pd.DataFrame) -> io.BytesIO:
    """Gráfico de barras agrupadas: real vs predicción."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(comparativa))
    ancho = 0.35
    ax.bar(x - ancho / 2, comparativa["ventas"], ancho, label="Real", color="#2ecc71")
    ax.bar(x + ancho / 2, comparativa["prediccion"], ancho, label="Predicción", color="#3498db")
    ax.set_title("Comparación: Ventas Reales vs Predicción", fontsize=13,
                 fontweight="bold", color="#2c3e50")
    ax.set_xticks(x)
    ax.set_xticklabels(comparativa["mes"], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_numero(v, 0)))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_a_png(fig)


def grafico_futuro(df_historico: pd.DataFrame, df_futuro: pd.DataFrame) -> io.BytesIO:
    """Dos subplots: serie completa + barras de predicciones futuras."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(df_historico["fecha"], df_historico["ventas"],
             color="#2ecc71", linewidth=1.5, label="Histórico")
    ax1.plot(df_futuro["fecha"], df_futuro["prediccion"],
             color="#e74c3c", linewidth=1.5, linestyle="--", label="Predicción")
    ax1.set_title("Serie Completa", fontsize=11, fontweight="bold", color="#2c3e50")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_formateador_eje_mes_anio))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_numero(v, 0)))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(df_futuro)), df_futuro["prediccion"], color="#3498db")
    ax2.set_title("Predicciones Futuras", fontsize=11, fontweight="bold", color="#2c3e50")
    ax2.set_xticks(range(len(df_futuro)))
    ax2.set_xticklabels(df_futuro["mes"], rotation=45, ha="right", fontsize=7)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: formato_numero(v, 0)))
    ax2.grid(True, axis="y", alpha=0.3)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return _fig_a_png(fig)


# ---------------------------------------------------------------------------
# Clase PDF
# ---------------------------------------------------------------------------

class _InformePDF(FPDF):
    """Subclase FPDF con encabezado, pie de página y helpers de estilo."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Informe de Predicción de Demanda - Heladerías", 0, 1, "R")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", 0, 0, "C")

    def titulo_seccion(self, texto: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(44, 62, 80)
        self.set_fill_color(236, 240, 241)
        self.cell(0, 10, texto_seguro_pdf(texto), 0, 1, "L", fill=True)
        self.ln(3)

    def subtitulo(self, texto: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, texto_seguro_pdf(texto), 0, 1, "L")
        self.ln(1)

    def texto_normal(self, texto: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, texto_seguro_pdf(texto))
        self.ln(2)

    def metrica(self, nombre: str, valor: str, x: float, y: float, ancho: float = 45):
        self.set_xy(x, y)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(ancho, 5, texto_seguro_pdf(nombre), 0, 2, "C")
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(44, 62, 80)
        self.cell(ancho, 7, texto_seguro_pdf(str(valor)), 0, 2, "C")

    def tabla(self, encabezados: list, filas: list, anchos: Optional[list] = None,
              tam_enc: int = 9, tam_cel: int = 8, alto_enc: int = 8, alto_fila: int = 7):
        if anchos is None:
            ancho_total = self.w - 20
            anchos = [ancho_total / len(encabezados)] * len(encabezados)

        self.set_font("Helvetica", "B", tam_enc)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        for i, enc in enumerate(encabezados):
            self.cell(anchos[i], alto_enc, texto_seguro_pdf(str(enc)), 1, 0, "C", fill=True)
        self.ln()

        self.set_font("Helvetica", "", tam_cel)
        self.set_text_color(0, 0, 0)
        for idx, fila in enumerate(filas):
            self.set_fill_color(245, 245, 245) if idx % 2 == 0 else self.set_fill_color(255, 255, 255)
            for i, celda in enumerate(fila):
                self.cell(anchos[i], alto_fila, texto_seguro_pdf(str(celda)), 1, 0, "C", fill=True)
            self.ln()

    def imagen_desde_buf(self, buf: io.BytesIO, ancho: float = 190):
        """Inserta una imagen desde un buffer PNG en el PDF."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(buf.read())
        try:
            alto_img = ancho * 0.45
            if (self.h - self.get_y() - 20) < alto_img:
                self.add_page()
            self.image(tmp_path, x=10, w=ancho)
            self.ln(5)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def generar_pdf(
    df_raw: pd.DataFrame,
    df_serie: pd.DataFrame,
    comparativa: pd.DataFrame,
    df_futuro: pd.DataFrame,
    metricas,
    trend: Optional[str],
    seasonal: str,
    total_real: float,
    total_predicho: float,
    correccion_pandemia: bool = False,
) -> bytes:
    """
    Genera el informe PDF completo.

    Args:
        df_raw: DataFrame original del archivo cargado (para preview).
        df_serie: serie temporal transformada.
        comparativa: DataFrame con columnas fecha, ventas, prediccion, error, error_pct, mes.
        df_futuro: DataFrame con columnas fecha, prediccion, mes.
        metricas: objeto Metricas con mae, rmse, r2, error_absoluto_total.
        trend: tipo de tendencia usado.
        seasonal: tipo de estacionalidad usado.
        total_real: suma de ventas reales en el período de validación.
        total_predicho: suma de predicciones en el período de validación.
        correccion_pandemia: si se aplicó corrección 2020.

    Returns:
        Bytes del PDF generado.
    """
    periodo_inicio = mes_anio_es(comparativa["fecha"].min(), abreviado=True)
    periodo_fin = mes_anio_es(comparativa["fecha"].max(), abreviado=True)
    diferencia_total = total_real - total_predicho
    diferencia_pct = (diferencia_total / total_real * 100) if total_real else 0
    error_pct = (metricas.error_absoluto_total / total_real * 100) if total_real else 0

    pdf = _InformePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Portada ----
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "Informe de Predicción de Demanda", 0, 1, "C")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Modelo Holt-Winters para Heladerías", 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Fecha del informe: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, "C")
    rango = f"{fecha_a_texto_es(df_serie['fecha'].min())} a {fecha_a_texto_es(df_serie['fecha'].max())}"
    pdf.cell(0, 8, f"Rango de datos: {rango}", 0, 1, "C")
    pdf.cell(0, 8, f"Total de observaciones: {len(df_serie)}", 0, 1, "C")
    config = f"Tendencia={trend or 'Sin tendencia'} | Estacionalidad={seasonal}"
    pdf.cell(0, 8, f"Configuración: {config}", 0, 1, "C")
    if correccion_pandemia:
        pdf.cell(0, 8, "Corrección de pandemia 2020 aplicada", 0, 1, "C")

    # ---- Datos cargados ----
    pdf.add_page()
    pdf.titulo_seccion("1. Datos Cargados")
    pdf.texto_normal(
        f"Vista del archivo: {df_raw.shape[0]} filas y {df_raw.shape[1]} columnas. "
        f"Serie mensual: {fecha_a_texto_es(df_serie['fecha'].min())} a "
        f"{fecha_a_texto_es(df_serie['fecha'].max())} ({len(df_serie)} meses)."
    )

    cols_preview = list(df_raw.columns[:13])
    enc_raw = [encabezado_columna_pdf(c) for c in cols_preview]
    n_cols = len(enc_raw)
    ancho_disp = pdf.w - 20
    anchos_raw = (
        [ancho_disp] if n_cols <= 1
        else [16.0] + [(ancho_disp - 16.0) / (n_cols - 1)] * (n_cols - 1)
    )
    filas_raw = [
        [celda_csv_preview(row[c]) for c in cols_preview]
        for _, row in df_raw.head(10).iterrows()
    ]
    if len(df_raw) > 10:
        filas_raw.append(["..." if i == 0 else "-" for i in range(n_cols)])

    pdf.tabla(enc_raw, filas_raw, anchos_raw,
              tam_enc=6, tam_cel=6, alto_enc=6, alto_fila=5)
    pdf.ln(5)

    # ---- Serie temporal ----
    pdf.titulo_seccion("2. Serie Temporal de Ventas")
    pdf.imagen_desde_buf(grafico_serie(df_serie))

    # ---- Resultados del modelo ----
    pdf.add_page()
    pdf.titulo_seccion("3. Resultados del Modelo")
    pdf.subtitulo("Métricas de rendimiento")
    y0 = pdf.get_y()
    pdf.metrica("MAE", formato_numero(metricas.mae), 10, y0, 38)
    pdf.metrica("RMSE", formato_numero(metricas.rmse), 48, y0, 38)
    pdf.metrica("R²", f"{metricas.r2:.4f}", 86, y0, 38)
    pdf.metrica("Error Abs. Total", formato_numero(metricas.error_absoluto_total), 124, y0, 38)
    pdf.metrica("Error Abs. %", f"{error_pct:.2f}%".replace(".", ","), 162, y0, 38)
    pdf.set_y(y0 + 20)
    pdf.ln(5)

    # ---- Validación ----
    pdf.titulo_seccion(f"4. Validación: Real vs Predicho ({periodo_inicio} - {periodo_fin})")
    pdf.subtitulo("Totales del período")
    y1 = pdf.get_y()
    pdf.metrica("Total Real", formato_numero(total_real), 10, y1, 47)
    pdf.metrica("Total Predicción", formato_numero(total_predicho), 57, y1, 47)
    pdf.metrica("Diferencia", formato_numero(diferencia_total), 104, y1, 47)
    pdf.metrica("Diferencia %", f"{diferencia_pct:.2f}%".replace(".", ","), 151, y1, 47)
    pdf.set_y(y1 + 20)
    pdf.ln(5)

    pdf.subtitulo("Gráfico comparativo")
    pdf.imagen_desde_buf(grafico_comparacion(comparativa))

    pdf.subtitulo("Detalle mensual")
    filas_comp = [
        [
            row["mes"],
            formato_numero(row["ventas"]),
            formato_numero(row["prediccion"]),
            formato_numero(row["error"]),
            f"{row['error_pct']:.2f}%".replace(".", ","),
        ]
        for _, row in comparativa.iterrows()
    ]
    pdf.tabla(
        ["Mes", "Venta Real", "Predicción", "Error", "Error %"],
        filas_comp,
        [35, 40, 40, 40, 35],
    )
    pdf.ln(5)

    # ---- Predicciones futuras ----
    pdf.add_page()
    pdf.titulo_seccion("5. Predicciones Futuras (Próximos 12 meses)")
    y2 = pdf.get_y()
    pdf.metrica("Total Anual", formato_numero(df_futuro["prediccion"].sum()), 10, y2, 63)
    pdf.metrica("Promedio Mensual", formato_numero(df_futuro["prediccion"].mean()), 73, y2, 63)
    mes_pico = df_futuro.loc[df_futuro["prediccion"].idxmax(), "mes"]
    pdf.metrica("Mes Pico", mes_pico, 136, y2, 63)
    pdf.set_y(y2 + 20)
    pdf.ln(5)

    pdf.subtitulo("Gráfico de predicciones")
    pdf.imagen_desde_buf(grafico_futuro(df_serie, df_futuro))

    pdf.subtitulo("Detalle mensual")
    filas_fut = [[row["mes"], formato_numero(row["prediccion"])] for _, row in df_futuro.iterrows()]
    pdf.tabla(["Mes", "Predicción"], filas_fut, [100, 90])

    buf_pdf = io.BytesIO()
    pdf.output(buf_pdf)
    return buf_pdf.getvalue()

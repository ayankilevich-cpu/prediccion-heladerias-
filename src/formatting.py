"""
formatting.py
-------------
Funciones de formato de números y texto para la app.
Centraliza toda la lógica de presentación reutilizable.
"""

import pandas as pd

# Nombres completos de los meses en español, en orden (índice 0 = enero)
MESES_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)

# Abreviaciones en español para tablas y ejes (ene 2024, feb 2024, …)
MESES_ES_ABREV = (
    "ene", "feb", "mar", "abr", "may", "jun",
    "jul", "ago", "sep", "oct", "nov", "dic",
)

# Abreviaciones para encabezados de tablas PDF
ABREV_MES_ENCABEZADO = {
    "ENERO": "Ene", "FEBRERO": "Feb", "MARZO": "Mar", "ABRIL": "Abr",
    "MAYO": "May", "JUNIO": "Jun", "JULIO": "Jul", "AGOSTO": "Ago",
    "SEPTIEMBRE": "Sep", "OCTUBRE": "Oct", "NOVIEMBRE": "Nov", "DICIEMBRE": "Dic",
}

# Columnas de meses en el orden esperado del archivo de entrada
COLUMNAS_MES = [
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
    "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE",
]


def formato_numero(numero: float, decimales: int = 2) -> str:
    """
    Formatea un número al estilo argentino:
    punto como separador de miles, coma como decimal.

    Ejemplo: 1234567.89 -> '1.234.567,89'
    """
    fmt = f"{{:,.{decimales}f}}"
    texto = fmt.format(numero)
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def quitar_bom(s: str) -> str:
    """
    Elimina el BOM UTF-8 (U+FEFF) del inicio de un string.
    str.strip() no lo elimina; FPDF falla con ese carácter.
    """
    if s is None:
        return ""
    t = str(s)
    while t and t[0] == "\ufeff":
        t = t[1:]
    return t.strip()


def texto_seguro_pdf(s: str) -> str:
    """Texto listo para celdas FPDF con fuentes estándar (sin BOM)."""
    return quitar_bom(s)


def mes_anio_es(fecha, abreviado: bool = False) -> str:
    """
    Convierte una fecha a texto 'mes año' en español sin depender del locale.

    Args:
        fecha: fecha a formatear.
        abreviado: si True usa abreviatura (ene 2024); si False nombre completo (enero 2024).
    """
    if fecha is None or pd.isna(fecha):
        return ""
    t = pd.Timestamp(fecha)
    meses = MESES_ES_ABREV if abreviado else MESES_ES
    return f"{meses[t.month - 1]} {t.year}"


def etiquetas_eje_fecha_es(fechas, max_etiquetas: int = 12) -> tuple[list, list]:
    """
    Genera tickvals y ticktext en español para ejes de gráficos (Plotly, etc.).
    """
    serie = pd.Series(fechas).dropna().drop_duplicates().sort_values()
    if len(serie) > max_etiquetas:
        paso = max(1, len(serie) // max_etiquetas)
        serie = serie.iloc[::paso]
    tickvals = serie.tolist()
    ticktext = [mes_anio_es(f, abreviado=True) for f in tickvals]
    return tickvals, ticktext


def fecha_a_texto_es(fecha) -> str:
    """
    Convierte una fecha a texto 'mes año' en español sin depender del locale.
    Ejemplo: 2024-01-01 -> 'enero 2024'
    """
    return mes_anio_es(fecha, abreviado=False)


def limpiar_encabezado_pdf(s: str) -> str:
    """
    Quita BOM y corrige texto mal decodificado típico de Excel en Windows
    (p. ej. 'AÃ±o' -> 'Año').
    """
    if s is None:
        return ""
    t = quitar_bom(s)
    if "Ã" in t:
        try:
            t = t.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return t.strip()


def encabezado_columna_pdf(nombre_columna: str) -> str:
    """Encabezado corto para columnas de tabla en el PDF."""
    t = limpiar_encabezado_pdf(nombre_columna)
    if t.lower() in ("año", "ano"):
        return "Año"
    return ABREV_MES_ENCABEZADO.get(t.upper().strip(), t[:8])


def celda_csv_preview(valor, max_len: int = 11) -> str:
    """
    Texto legible para celdas de preview del CSV:
    - Reemplaza NaN por '-'
    - Enteros sin '.0'
    - Trunca strings largos
    """
    import numpy as np

    if pd.isna(valor):
        return "-"
    if isinstance(valor, (np.integer, int)):
        return str(int(valor))
    if isinstance(valor, (np.floating, float)):
        if np.isnan(valor):
            return "-"
        if abs(valor - round(valor)) < 1e-9:
            return str(int(round(valor)))
        return formato_numero(float(valor), 1)
    s = quitar_bom(str(valor))
    if s.lower() in ("nan", "none", ""):
        return "-"
    return s[: max_len - 3] + "..." if len(s) > max_len else s

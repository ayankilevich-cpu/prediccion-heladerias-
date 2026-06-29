"""
loader.py
---------
Carga, transforma y limpia los datos de ventas.
Responsabilidades:
  - Leer CSV o Excel
  - Transformar formato ancho (año x meses) a serie temporal larga
  - Imputar NaN históricos
  - Corregir meses afectados por pandemia
  - Generar plantilla Excel descargable
"""

import io
from typing import Optional
import numpy as np
import pandas as pd

from src.formatting import COLUMNAS_MES, quitar_bom


# ---------------------------------------------------------------------------
# Carga de archivos
# ---------------------------------------------------------------------------

def es_excel(nombre_archivo: str) -> bool:
    """Devuelve True si el archivo es .xlsx o .xlsm."""
    if not nombre_archivo:
        return False
    return nombre_archivo.lower().endswith((".xlsx", ".xlsm"))


def cargar_dataframe(archivo, separador: str = ";", encoding: str = "latin1") -> pd.DataFrame:
    """
    Lee un archivo CSV o Excel y devuelve un DataFrame crudo.

    Args:
        archivo: objeto file-like (UploadedFile de Streamlit u otro).
        separador: separador de columnas para CSV.
        encoding: codificación para CSV.

    Returns:
        DataFrame con los datos tal como vienen del archivo,
        con BOM eliminado de los nombres de columnas.

    Raises:
        ValueError: si el formato no es reconocido o el archivo está corrupto.
    """
    archivo.seek(0)
    nombre = getattr(archivo, "name", "") or ""

    try:
        if es_excel(nombre):
            df = pd.read_excel(archivo, engine="openpyxl", sheet_name=0)
        else:
            df = pd.read_csv(archivo, sep=separador, encoding=encoding)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo: {e}") from e

    # Limpiar BOM de nombres de columnas
    df.columns = [quitar_bom(str(c)) for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Transformación
# ---------------------------------------------------------------------------

def transformar_a_serie(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el DataFrame de formato ancho (AÑO x MESES) a serie temporal.

    Input esperado:
        AÑO  | ENERO | FEBRERO | ... | DICIEMBRE
        2022 |  1200 |    1150 | ... |      1300

    Output:
        fecha       | ventas
        2022-01-01  |   1200
        2022-02-01  |   1150
        ...

    - Descarta meses futuros (sin dato real).
    - Imputa NaN históricos por el promedio del mismo mes en otros años.
    """
    col_anio = df.columns[0]

    # Conservar solo columnas de meses reconocidas + la de año
    cols_meses = [c for c in df.columns if str(c).upper().strip() in COLUMNAS_MES]

    df_trabajo = df[[col_anio] + cols_meses].copy()
    df_trabajo.columns = ["AÑO"] + [str(c).upper().strip() for c in cols_meses]

    # Formato largo
    df_largo = df_trabajo.melt(id_vars=["AÑO"], var_name="mes", value_name="ventas")

    # Convertir ventas a numérico (maneja formatos con coma decimal)
    df_largo["ventas"] = _convertir_a_numerico(df_largo["ventas"])

    # Construir fechas
    meses_num = {m: i + 1 for i, m in enumerate(COLUMNAS_MES)}
    df_largo["mes_num"] = df_largo["mes"].map(meses_num)
    df_largo["AÑO"] = pd.to_numeric(df_largo["AÑO"], errors="coerce")
    df_largo = df_largo.dropna(subset=["AÑO", "mes_num"])
    df_largo["fecha"] = pd.to_datetime(
        dict(year=df_largo["AÑO"].astype(int), month=df_largo["mes_num"].astype(int), day=1)
    )

    df_serie = df_largo[["fecha", "ventas"]].sort_values("fecha").reset_index(drop=True)

    # Descartar meses futuros (sin dato real)
    ultima_fecha_real = df_serie.loc[df_serie["ventas"].notna(), "fecha"].max()
    df_serie = df_serie[df_serie["fecha"] <= ultima_fecha_real].copy()

    # Imputar NaN históricos por promedio del mes
    df_serie = _imputar_por_promedio_mes(df_serie)

    return df_serie


def _convertir_a_numerico(serie: pd.Series) -> pd.Series:
    """
    Convierte una serie a numérico manejando formatos con coma decimal
    y punto como separador de miles (formato argentino/europeo).
    """
    # Si ya es numérica, retornar directamente
    if pd.api.types.is_numeric_dtype(serie):
        return serie

    s = serie.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Detectar formato: si tiene coma Y punto, asumir punto=miles, coma=decimal
    muestra = s.dropna().iloc[0] if s.dropna().shape[0] > 0 else ""
    if "," in muestra and "." in muestra:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif "," in muestra:
        s = s.str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def _imputar_por_promedio_mes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores NaN en 'ventas' usando el promedio histórico
    del mismo mes calendario. Solo afecta meses ya pasados.
    """
    df = df.copy()
    if df["ventas"].isna().sum() == 0:
        return df

    df["_mes"] = df["fecha"].dt.month
    promedios = df.groupby("_mes")["ventas"].mean()

    mask_nan = df["ventas"].isna()
    df.loc[mask_nan, "ventas"] = df.loc[mask_nan, "_mes"].map(promedios)
    df = df.drop(columns=["_mes"])
    return df


# ---------------------------------------------------------------------------
# Corrección de pandemia
# ---------------------------------------------------------------------------

def corregir_pandemia(
    df: pd.DataFrame,
    anio: int = 2020,
    meses_afectados: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Corrige meses atípicos por pandemia reemplazando o suavizando
    los valores usando el promedio histórico del mismo mes.

    Args:
        df: DataFrame con columnas 'fecha' y 'ventas'.
        anio: año afectado por la pandemia.
        meses_afectados: dict {num_mes: tipo} donde tipo es:
            - 'promedio': reemplaza por el promedio histórico
            - 'mixto': promedia entre el valor real y el histórico

    Returns:
        DataFrame corregido.
    """
    if meses_afectados is None:
        meses_afectados = {3: "mixto", 4: "promedio"}

    df = df.copy()
    df_sin_anio = df[df["fecha"].dt.year != anio]

    for mes_num, tipo in meses_afectados.items():
        promedio_historico = (
            df_sin_anio[df_sin_anio["fecha"].dt.month == mes_num]["ventas"].mean()
        )
        mask = (df["fecha"].dt.year == anio) & (df["fecha"].dt.month == mes_num)

        if tipo == "promedio":
            df.loc[mask, "ventas"] = promedio_historico
        elif tipo == "mixto":
            valor_real = df.loc[mask, "ventas"].values
            if len(valor_real) > 0:
                df.loc[mask, "ventas"] = (promedio_historico + valor_real[0]) / 2

    return df


# ---------------------------------------------------------------------------
# Plantilla Excel
# ---------------------------------------------------------------------------

def generar_plantilla_excel() -> bytes:
    """
    Genera un archivo .xlsx modelo con columnas AÑO + 12 meses
    y filas vacías para los últimos 4 años.

    Returns:
        Bytes del archivo Excel listo para descargar.
    """
    from datetime import datetime

    anio_actual = datetime.now().year
    filas = [
        {"AÑO": anio, **{mes: None for mes in COLUMNAS_MES}}
        for anio in range(anio_actual - 3, anio_actual + 1)
    ]

    df_plantilla = pd.DataFrame(filas)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_plantilla.to_excel(writer, index=False, sheet_name="Ventas")
    buf.seek(0)
    return buf.getvalue()

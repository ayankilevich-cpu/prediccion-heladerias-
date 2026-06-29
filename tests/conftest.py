"""
conftest.py
-----------
Fixtures compartidos entre todos los módulos de test.

Un fixture es básicamente un "dato de prueba" que pytest
prepara automáticamente antes de cada test que lo necesite.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# DataFrames en formato ANCHO (como vienen del archivo del usuario)
# ---------------------------------------------------------------------------

@pytest.fixture
def df_ancho_ok():
    """
    DataFrame válido en formato ancho: 4 años completos, 12 meses.
    Este es el caso feliz — todo en orden.
    """
    meses = [
        "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
        "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE",
    ]
    np.random.seed(42)
    filas = []
    for anio in range(2021, 2025):
        base = 1000 + 300 * np.sin(np.arange(12) * 2 * np.pi / 12)
        ventas = (base + np.random.normal(0, 30, 12)).round(0).astype(int).tolist()
        filas.append({"AÑO": anio, **dict(zip(meses, ventas))})
    return pd.DataFrame(filas)


@pytest.fixture
def df_ancho_con_nan(df_ancho_ok):
    """
    Mismo DataFrame válido pero con algunos NaN en el medio
    (simula meses sin dato en el historial).
    """
    df = df_ancho_ok.copy()
    df.loc[1, "MARZO"] = None   # 2022-03
    df.loc[2, "JULIO"] = None   # 2023-07
    return df


@pytest.fixture
def df_ancho_dos_anios():
    """DataFrame mínimo: solo 2 años (caso límite para el modelo)."""
    meses = [
        "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
        "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE",
    ]
    filas = [
        {"AÑO": 2023, **{m: 1000 for m in meses}},
        {"AÑO": 2024, **{m: 1100 for m in meses}},
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def df_ancho_columnas_faltantes():
    """DataFrame con solo 4 meses — no alcanza para el modelo."""
    return pd.DataFrame([
        {"AÑO": 2022, "ENERO": 100, "FEBRERO": 110, "MARZO": 90, "ABRIL": 95},
        {"AÑO": 2023, "ENERO": 120, "FEBRERO": 115, "MARZO": 105, "ABRIL": 100},
    ])


@pytest.fixture
def df_ancho_sin_columna_anio():
    """DataFrame sin columna de año — solo meses."""
    return pd.DataFrame([
        {"ENERO": 100, "FEBRERO": 110, "MARZO": 90},
    ])


@pytest.fixture
def df_ancho_formato_coma():
    """
    DataFrame donde los números de ventas usan coma como decimal
    y punto como separador de miles (formato argentino/europeo en string).
    """
    meses = [
        "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
        "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE",
    ]
    filas = [
        {"AÑO": 2022, **{m: "1.200,50" for m in meses}},
        {"AÑO": 2023, **{m: "1.350,75" for m in meses}},
        {"AÑO": 2024, **{m: "1.100,00" for m in meses}},
    ]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Series temporales (formato largo, output de transformar_a_serie)
# ---------------------------------------------------------------------------

@pytest.fixture
def serie_temporal_ok():
    """
    Serie temporal limpia en formato largo: fecha + ventas.
    48 meses con estacionalidad sintética.
    """
    np.random.seed(0)
    fechas = pd.date_range("2021-01-01", periods=48, freq="MS")
    ventas = 1000 + 300 * np.sin(np.arange(48) * 2 * np.pi / 12) + np.random.normal(0, 40, 48)
    return pd.DataFrame({"fecha": fechas, "ventas": ventas.round(1)})


@pytest.fixture
def serie_temporal_con_pandemia(serie_temporal_ok):
    """
    Serie temporal con valores bajos en marzo y abril 2020
    (no aplica si la serie empieza en 2021, pero sirve para
    testear que la función de corrección no rompe nada fuera del año).
    """
    return serie_temporal_ok.copy()

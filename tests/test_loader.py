"""
test_loader.py
--------------
Tests para src/loader.py

El loader es el módulo más crítico para la integridad de los datos.
Testeamos la transformación, la imputación y la corrección de pandemia.
"""

import io
import numpy as np
import pandas as pd
import pytest
from src.loader import (
    corregir_pandemia,
    es_excel,
    generar_plantilla_excel,
    transformar_a_serie,
)


class TestEsExcel:
    """Detección del tipo de archivo por extensión."""

    def test_xlsx_es_excel(self):
        assert es_excel("ventas.xlsx") is True

    def test_xlsm_es_excel(self):
        assert es_excel("ventas.xlsm") is True

    def test_csv_no_es_excel(self):
        assert es_excel("ventas.csv") is False

    def test_nombre_vacio_no_es_excel(self):
        assert es_excel("") is False

    def test_mayusculas_funciona(self):
        assert es_excel("VENTAS.XLSX") is True


class TestTransformarASerie:
    """transformar_a_serie convierte formato ancho a serie temporal."""

    def test_output_tiene_columnas_correctas(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert "fecha" in df.columns
        assert "ventas" in df.columns

    def test_output_tiene_48_filas_para_4_anios(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert len(df) == 48  # 4 años × 12 meses

    def test_fechas_son_primer_dia_del_mes(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert (df["fecha"].dt.day == 1).all()

    def test_serie_ordenada_por_fecha(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert df["fecha"].is_monotonic_increasing

    def test_sin_nan_en_ventas_cuando_datos_completos(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert df["ventas"].isna().sum() == 0

    def test_nan_historicos_se_imputan(self, df_ancho_con_nan):
        df = transformar_a_serie(df_ancho_con_nan)
        # Después de imputar no debe haber NaN en fechas históricas
        assert df["ventas"].isna().sum() == 0

    def test_imputacion_usa_promedio_del_mes(self, df_ancho_ok):
        """
        Si marzo 2022 es NaN, debe imputarse con el promedio de marzo
        de los otros años — no con cero ni con el promedio global.
        """
        df_nan = df_ancho_ok.copy()
        # Calcular el promedio real de MARZO antes de poner NaN
        promedio_marzo = df_nan.loc[df_nan["AÑO"] != 2022, "MARZO"].mean()
        df_nan.loc[df_nan["AÑO"] == 2022, "MARZO"] = None

        df_serie = transformar_a_serie(df_nan)
        valor_imputado = df_serie.loc[
            df_serie["fecha"] == "2022-03-01", "ventas"
        ].values[0]

        # El valor imputado debe estar cerca del promedio del mes
        assert abs(valor_imputado - promedio_marzo) < 1.0

    def test_formato_numerico_con_coma(self, df_ancho_formato_coma):
        """Ventas con '1.200,50' deben convertirse a 1200.50."""
        df = transformar_a_serie(df_ancho_formato_coma)
        assert df["ventas"].isna().sum() == 0
        assert df["ventas"].iloc[0] == pytest.approx(1200.50, abs=0.1)

    def test_primera_fecha_es_enero_del_primer_anio(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert df["fecha"].min() == pd.Timestamp("2021-01-01")

    def test_ultima_fecha_es_diciembre_del_ultimo_anio(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert df["fecha"].max() == pd.Timestamp("2024-12-01")

    def test_ventas_son_numericas(self, df_ancho_ok):
        df = transformar_a_serie(df_ancho_ok)
        assert pd.api.types.is_numeric_dtype(df["ventas"])


class TestCorregirPandemia:
    """corregir_pandemia ajusta meses atípicos del año de pandemia."""

    @pytest.fixture
    def serie_con_2020(self):
        """Serie temporal que incluye 2020 con valores bajos en marzo y abril."""
        np.random.seed(1)
        fechas = pd.date_range("2018-01-01", periods=60, freq="MS")
        ventas = 1000 + 200 * np.sin(np.arange(60) * 2 * np.pi / 12)
        ventas = ventas.round(0)
        df = pd.DataFrame({"fecha": fechas, "ventas": ventas})
        # Simular caída por pandemia
        df.loc[df["fecha"] == "2020-03-01", "ventas"] = 200
        df.loc[df["fecha"] == "2020-04-01", "ventas"] = 50
        return df

    def test_marzo_2020_se_corrige(self, serie_con_2020):
        df_corregido = corregir_pandemia(serie_con_2020)
        valor_antes = 200
        valor_despues = df_corregido.loc[
            df_corregido["fecha"] == "2020-03-01", "ventas"
        ].values[0]
        # El valor corregido debe ser mayor que el original anómalo
        assert valor_despues > valor_antes

    def test_abril_2020_se_corrige_a_promedio(self, serie_con_2020):
        """Abril usa tipo 'promedio' — debe quedar igual al promedio histórico de abril."""
        df_sin_2020 = serie_con_2020[serie_con_2020["fecha"].dt.year != 2020]
        promedio_abril = df_sin_2020[df_sin_2020["fecha"].dt.month == 4]["ventas"].mean()

        df_corregido = corregir_pandemia(serie_con_2020, meses_afectados={4: "promedio"})
        valor_abril = df_corregido.loc[
            df_corregido["fecha"] == "2020-04-01", "ventas"
        ].values[0]

        assert abs(valor_abril - promedio_abril) < 0.1

    def test_meses_no_afectados_no_cambian(self, serie_con_2020):
        """Los meses que no están en meses_afectados no deben cambiar."""
        df_corregido = corregir_pandemia(serie_con_2020)
        valor_original = serie_con_2020.loc[
            serie_con_2020["fecha"] == "2020-01-01", "ventas"
        ].values[0]
        valor_corregido = df_corregido.loc[
            df_corregido["fecha"] == "2020-01-01", "ventas"
        ].values[0]
        assert valor_original == valor_corregido

    def test_anios_distintos_a_2020_no_cambian(self, serie_con_2020):
        """Ningún valor de 2019 o 2021 debe modificarse."""
        df_corregido = corregir_pandemia(serie_con_2020)
        for anio in [2018, 2019]:
            mask = serie_con_2020["fecha"].dt.year == anio
            originales = serie_con_2020.loc[mask, "ventas"].values
            corregidos = df_corregido.loc[mask, "ventas"].values
            np.testing.assert_array_equal(originales, corregidos)

    def test_serie_sin_2020_no_falla(self, serie_temporal_ok):
        """Si no hay datos de 2020, la función no debe fallar."""
        df_corregido = corregir_pandemia(serie_temporal_ok)
        assert len(df_corregido) == len(serie_temporal_ok)


class TestGenerarPlantillaExcel:
    """generar_plantilla_excel produce un archivo Excel válido."""

    def test_devuelve_bytes(self):
        resultado = generar_plantilla_excel()
        assert isinstance(resultado, bytes)

    def test_bytes_no_vacios(self):
        resultado = generar_plantilla_excel()
        assert len(resultado) > 0

    def test_excel_es_legible(self):
        resultado = generar_plantilla_excel()
        df = pd.read_excel(io.BytesIO(resultado), engine="openpyxl")
        assert not df.empty

    def test_plantilla_tiene_columna_anio(self):
        resultado = generar_plantilla_excel()
        df = pd.read_excel(io.BytesIO(resultado), engine="openpyxl")
        assert "AÑO" in df.columns

    def test_plantilla_tiene_12_columnas_de_meses(self):
        from src.formatting import COLUMNAS_MES
        resultado = generar_plantilla_excel()
        df = pd.read_excel(io.BytesIO(resultado), engine="openpyxl")
        meses_en_plantilla = [c for c in df.columns if c in COLUMNAS_MES]
        assert len(meses_en_plantilla) == 12

    def test_plantilla_tiene_4_filas(self):
        resultado = generar_plantilla_excel()
        df = pd.read_excel(io.BytesIO(resultado), engine="openpyxl")
        assert len(df) == 4

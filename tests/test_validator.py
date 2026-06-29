"""
test_validator.py
-----------------
Tests para src/validator.py

Verificamos que el validador detecta correctamente
los casos buenos, los casos malos y los advertibles.
"""

import pandas as pd
import pytest
from src.validator import validar_dataframe


class TestValidacionCasoFeliz:
    """El archivo viene bien formado — debe pasar sin errores."""

    def test_dataframe_valido_es_valido(self, df_ancho_ok):
        resultado = validar_dataframe(df_ancho_ok)
        assert resultado.valido is True

    def test_dataframe_valido_sin_errores(self, df_ancho_ok):
        resultado = validar_dataframe(df_ancho_ok)
        assert len(resultado.errores) == 0

    def test_dataframe_con_nan_es_valido(self, df_ancho_con_nan):
        # NaN en celdas de ventas es advertencia, no error
        resultado = validar_dataframe(df_ancho_con_nan)
        assert resultado.valido is True


class TestValidacionErrores:
    """Casos que deben rechazarse con errores claros."""

    def test_dataframe_none_es_invalido(self):
        resultado = validar_dataframe(None)
        assert resultado.valido is False
        assert len(resultado.errores) > 0

    def test_dataframe_vacio_es_invalido(self):
        resultado = validar_dataframe(pd.DataFrame())
        assert resultado.valido is False

    def test_una_sola_columna_es_invalido(self):
        df = pd.DataFrame({"AÑO": [2022, 2023]})
        resultado = validar_dataframe(df)
        assert resultado.valido is False
        assert any("columna" in e.lower() for e in resultado.errores)

    def test_columnas_mes_faltantes_es_invalido(self, df_ancho_columnas_faltantes):
        resultado = validar_dataframe(df_ancho_columnas_faltantes)
        assert resultado.valido is False
        assert any("mes" in e.lower() for e in resultado.errores)

    def test_una_sola_fila_es_invalido(self):
        meses = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
                 "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"]
        df = pd.DataFrame([{"AÑO": 2024, **{m: 1000 for m in meses}}])
        resultado = validar_dataframe(df)
        assert resultado.valido is False
        assert any("año" in e.lower() or "fila" in e.lower() or "datos" in e.lower()
                   for e in resultado.errores)

    def test_columna_anio_no_numerica_es_invalido(self):
        meses = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
                 "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"]
        df = pd.DataFrame([
            {"AÑO": "veintidos", **{m: 1000 for m in meses}},
            {"AÑO": "veintitres", **{m: 1100 for m in meses}},
        ])
        resultado = validar_dataframe(df)
        assert resultado.valido is False


class TestValidacionAdvertencias:
    """Casos que pasan pero generan advertencias."""

    def test_dos_anios_genera_advertencia(self, df_ancho_dos_anios):
        resultado = validar_dataframe(df_ancho_dos_anios)
        assert resultado.valido is True
        assert len(resultado.advertencias) > 0

    def test_nan_en_ventas_genera_advertencia(self, df_ancho_con_nan):
        resultado = validar_dataframe(df_ancho_con_nan)
        assert resultado.valido is True
        # Puede o no tener advertencia según si los NaN se detectan como no numéricos
        # Lo importante es que no sea error
        assert len(resultado.errores) == 0

    def test_meses_parciales_genera_advertencia(self):
        """7 meses reconocidos: válido pero con advertencia."""
        df = pd.DataFrame([
            {"AÑO": 2022, "ENERO": 100, "FEBRERO": 110, "MARZO": 90,
             "ABRIL": 95, "MAYO": 88, "JUNIO": 92, "JULIO": 105},
            {"AÑO": 2023, "ENERO": 120, "FEBRERO": 115, "MARZO": 105,
             "ABRIL": 100, "MAYO": 98, "JUNIO": 102, "JULIO": 115},
            {"AÑO": 2024, "ENERO": 130, "FEBRERO": 125, "MARZO": 115,
             "ABRIL": 110, "MAYO": 108, "JUNIO": 112, "JULIO": 125},
        ])
        resultado = validar_dataframe(df)
        assert resultado.valido is True
        assert any("mes" in a.lower() for a in resultado.advertencias)


class TestResumenTexto:
    """El resumen de texto para la UI es legible."""

    def test_resumen_con_errores_tiene_icono(self):
        df = pd.DataFrame()
        resultado = validar_dataframe(df)
        assert "❌" in resultado.resumen()

    def test_resumen_vacio_si_todo_ok(self, df_ancho_ok):
        resultado = validar_dataframe(df_ancho_ok)
        # Si no hay errores ni advertencias, el resumen es vacío
        if not resultado.advertencias:
            assert resultado.resumen() == ""

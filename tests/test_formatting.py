"""
test_formatting.py
------------------
Tests para src/formatting.py

Son los más simples: funciones puras de texto,
sin dependencias externas. Buenos para arrancar.
"""

import pytest
from src.formatting import (
    celda_csv_preview,
    encabezado_columna_pdf,
    fecha_a_texto_es,
    formato_numero,
    quitar_bom,
    texto_seguro_pdf,
)
import pandas as pd
import numpy as np


class TestFormatoNumero:
    """formato_numero convierte floats al estilo argentino."""

    def test_numero_basico(self):
        assert formato_numero(1234.56) == "1.234,56"

    def test_numero_sin_decimales(self):
        assert formato_numero(1234567, decimales=0) == "1.234.567"

    def test_numero_con_un_decimal(self):
        assert formato_numero(999.5, decimales=1) == "999,5"

    def test_numero_pequeño(self):
        assert formato_numero(0.99) == "0,99"

    def test_numero_negativo(self):
        # Los negativos deben mantener el signo
        assert formato_numero(-1500.0) == "-1.500,00"

    def test_numero_cero(self):
        assert formato_numero(0) == "0,00"

    def test_millones(self):
        assert formato_numero(1_000_000.0, decimales=0) == "1.000.000"


class TestQuitarBom:
    """quitar_bom elimina el carácter BOM al inicio del string."""

    def test_sin_bom(self):
        assert quitar_bom("hola") == "hola"

    def test_con_bom(self):
        assert quitar_bom("\ufeffAÑO") == "AÑO"

    def test_con_varios_bom(self):
        assert quitar_bom("\ufeff\ufeffENERO") == "ENERO"

    def test_none_devuelve_string_vacio(self):
        assert quitar_bom(None) == ""

    def test_string_vacio(self):
        assert quitar_bom("") == ""

    def test_solo_espacios_se_stripean(self):
        assert quitar_bom("  columna  ") == "columna"


class TestTextoSeguoPDF:
    """texto_seguro_pdf es un alias de quitar_bom — mismo comportamiento."""

    def test_quita_bom(self):
        assert texto_seguro_pdf("\ufeffValor") == "Valor"

    def test_none(self):
        assert texto_seguro_pdf(None) == ""


class TestFechaATextoEs:
    """fecha_a_texto_es convierte fechas a texto en español."""

    def test_enero(self):
        assert fecha_a_texto_es(pd.Timestamp("2024-01-01")) == "enero 2024"

    def test_diciembre(self):
        assert fecha_a_texto_es(pd.Timestamp("2023-12-01")) == "diciembre 2023"

    def test_todos_los_meses(self):
        meses_esperados = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ]
        for i, nombre in enumerate(meses_esperados, start=1):
            fecha = pd.Timestamp(f"2024-{i:02d}-01")
            assert fecha_a_texto_es(fecha).startswith(nombre)

    def test_none_devuelve_vacio(self):
        assert fecha_a_texto_es(None) == ""

    def test_nan_devuelve_vacio(self):
        assert fecha_a_texto_es(float("nan")) == ""


class TestEncabezadoColumnaPDF:
    """encabezado_columna_pdf abrevia nombres de columnas para el PDF."""

    def test_enero_abrevia(self):
        assert encabezado_columna_pdf("ENERO") == "Ene"

    def test_diciembre_abrevia(self):
        assert encabezado_columna_pdf("DICIEMBRE") == "Dic"

    def test_columna_anio(self):
        assert encabezado_columna_pdf("AÑO") == "Año"

    def test_columna_ano_sin_tilde(self):
        # Por si el BOM decodificó mal el año
        assert encabezado_columna_pdf("ANO") == "Año"

    def test_columna_desconocida_devuelve_truncada(self):
        resultado = encabezado_columna_pdf("COLUMNADESCONOCIDA")
        assert len(resultado) <= 8


class TestCeldaCSVPreview:
    """celda_csv_preview formatea valores para mostrar en tablas del PDF."""

    def test_nan_devuelve_guion(self):
        assert celda_csv_preview(float("nan")) == "-"

    def test_none_devuelve_guion(self):
        assert celda_csv_preview(None) == "-"

    def test_entero_sin_punto(self):
        assert celda_csv_preview(1200) == "1200"

    def test_float_entero_sin_punto(self):
        # 1200.0 debe mostrarse como "1200", no "1200.0"
        assert celda_csv_preview(1200.0) == "1200"

    def test_numpy_integer(self):
        assert celda_csv_preview(np.int64(500)) == "500"

    def test_string_largo_se_trunca(self):
        resultado = celda_csv_preview("texto muy largo que supera el límite", max_len=11)
        assert len(resultado) <= 11
        assert resultado.endswith("...")

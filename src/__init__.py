# src/__init__.py
# Expone la API pública del paquete para imports limpios desde app.py

from src.loader import cargar_dataframe, transformar_a_serie, corregir_pandemia, generar_plantilla_excel
from src.validator import validar_dataframe
from src.model import ejecutar_pipeline
from src.report import generar_pdf
from src.formatting import formato_numero, fecha_a_texto_es

__all__ = [
    "cargar_dataframe",
    "transformar_a_serie",
    "corregir_pandemia",
    "generar_plantilla_excel",
    "validar_dataframe",
    "ejecutar_pipeline",
    "generar_pdf",
    "formato_numero",
    "fecha_a_texto_es",
]

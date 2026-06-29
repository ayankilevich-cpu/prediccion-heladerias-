# Predicción de Demanda para Heladerías

Aplicación en Python + Streamlit para analizar ventas históricas de heladerías y generar predicciones mensuales de demanda usando modelos de series temporales.

## Problema de negocio

Las heladerías tienen una demanda fuertemente estacional. Una mala estimación puede generar quiebres de stock o sobre-stock. Este proyecto busca anticipar la demanda para mejorar decisiones de compra, producción y planificación.

## Qué hace la aplicación

- Carga archivos CSV o Excel.
- Transforma ventas mensuales en serie temporal.
- Permite seleccionar período de entrenamiento y validación.
- Corrige meses atípicos afectados por pandemia.
- Entrena un modelo Holt-Winters.
- Compara ventas reales vs predicción.
- Calcula MAE, RMSE, R² y error absoluto total.
- Genera predicción para los próximos 12 meses.
- Exporta un informe PDF.

## Modelo utilizado

El modelo principal es Holt-Winters / Exponential Smoothing, adecuado para series con tendencia y estacionalidad mensual.

## Estructura esperada del archivo

El archivo debe tener una columna de año y una columna por mes:

| AÑO | ENERO | FEBRERO | MARZO | ... | DICIEMBRE |
|---|---:|---:|---:|---:|---:|
| 2022 | 1200 | 1150 | 980 | ... | 1300 |
| 2023 | 1280 | 1190 | 1010 | ... | 1380 |

## Tecnologías

- Python
- Streamlit
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- Plotly
- Matplotlib
- FPDF
- OpenPyXL

## Cómo ejecutar

```bash
git clone https://github.com/ayankilevich-cpu/prediccion-heladerias-.git
cd prediccion-heladerias-
pip install streamlit pandas numpy statsmodels scikit-learn plotly matplotlib fpdf openpyxl
streamlit run app_prediccion_heladerias.py

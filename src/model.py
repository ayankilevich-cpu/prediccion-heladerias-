"""
model.py
--------
Lógica del modelo de predicción Holt-Winters.
Responsabilidades:
  - Dividir datos en train/test
  - Evaluar combinaciones de parámetros
  - Entrenar el modelo final
  - Calcular métricas
  - Generar predicciones futuras

Sin ninguna dependencia de Streamlit: puede usarse
y testearse de forma completamente independiente.
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults

from src.formatting import mes_anio_es

logger = logging.getLogger(__name__)

# Combinaciones de parámetros a evaluar en la selección automática
COMBINACIONES_PARAMETROS = [
    ("additive", "additive"),
    ("additive", "multiplicative"),
    (None, "additive"),
    (None, "multiplicative"),
]


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class Metricas:
    """Métricas de evaluación del modelo."""
    mae: float
    rmse: float
    r2: float
    error_absoluto_total: float

    @property
    def error_absoluto_total_pct(self) -> float:
        """Porcentaje del error absoluto sobre el total real (se calcula afuera)."""
        return 0.0  # se completa en ResultadoValidacion


@dataclass
class ResultadoModelo:
    """Todo lo necesario para mostrar resultados en la UI."""
    modelo: HoltWintersResults
    trend: Optional[str]
    seasonal: str
    metricas: Metricas
    predicciones_validacion: pd.Series
    train: pd.DataFrame
    test: pd.DataFrame
    df_futuro: pd.DataFrame


# ---------------------------------------------------------------------------
# División de datos
# ---------------------------------------------------------------------------

def dividir_train_test(df: pd.DataFrame, n_meses_validacion: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide la serie en train y test usando los últimos N meses como test.

    Args:
        df: DataFrame con columnas 'fecha' y 'ventas', sin NaN.
        n_meses_validacion: cantidad de meses a reservar para validación.

    Returns:
        Tupla (train, test).
    """
    df_limpio = df.dropna(subset=["ventas"]).sort_values("fecha").reset_index(drop=True)

    n_val = min(n_meses_validacion, len(df_limpio) - 12)
    if n_val <= 0:
        n_val = len(df_limpio) // 2

    train = df_limpio.iloc[:-n_val].copy()
    test = df_limpio.iloc[-n_val:].copy()
    return train, test


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar_modelo(
    train: pd.DataFrame,
    n_forecast: int,
    trend: Optional[str],
    seasonal: str,
) -> tuple[HoltWintersResults, pd.Series]:
    """
    Entrena un modelo Holt-Winters con optimización automática de parámetros.

    Args:
        train: DataFrame con columnas 'fecha' y 'ventas'.
        n_forecast: cantidad de períodos a predecir.
        trend: tipo de tendencia ('additive', None).
        seasonal: tipo de estacionalidad ('additive', 'multiplicative').

    Returns:
        Tupla (modelo_entrenado, predicciones).

    Raises:
        ValueError: si el modelo no converge o los datos son insuficientes.
    """
    modelo = ExponentialSmoothing(
        train["ventas"],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12,
    ).fit()

    predicciones = modelo.forecast(n_forecast)
    return modelo, predicciones


def calcular_metricas(reales: pd.Series, predichos: pd.Series) -> Metricas:
    """Calcula MAE, RMSE, R² y error absoluto total."""
    return Metricas(
        mae=mean_absolute_error(reales, predichos),
        rmse=float(np.sqrt(mean_squared_error(reales, predichos))),
        r2=r2_score(reales, predichos),
        error_absoluto_total=float(np.abs(reales.values - predichos.values).sum()),
    )


# ---------------------------------------------------------------------------
# Selección automática de parámetros
# ---------------------------------------------------------------------------

def seleccionar_mejor_modelo(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Optional[str], str, Metricas]:
    """
    Evalúa COMBINACIONES_PARAMETROS y devuelve la combinación con menor MAE.

    Args:
        train: datos de entrenamiento.
        test: datos de validación.

    Returns:
        Tupla (mejor_trend, mejor_seasonal, metricas_del_mejor).

    Raises:
        RuntimeError: si ninguna combinación converge.
    """
    resultados = []

    for trend, seasonal in COMBINACIONES_PARAMETROS:
        try:
            _, predicciones = entrenar_modelo(train, len(test), trend, seasonal)
            metricas = calcular_metricas(test["ventas"], predicciones)
            resultados.append((trend, seasonal, metricas))
            logger.debug(
                "Combinación trend=%s seasonal=%s -> MAE=%.2f",
                trend, seasonal, metricas.mae
            )
        except Exception as e:
            logger.warning("Combinación trend=%s seasonal=%s falló: %s", trend, seasonal, e)

    if not resultados:
        raise RuntimeError(
            "Ninguna combinación de parámetros Holt-Winters convergió. "
            "Revisá que tengas al menos 24 meses de datos sin NaN."
        )

    mejor = min(resultados, key=lambda x: x[2].mae)
    return mejor[0], mejor[1], mejor[2]


# ---------------------------------------------------------------------------
# Predicciones futuras
# ---------------------------------------------------------------------------

def generar_predicciones_futuras(
    df_completo: pd.DataFrame,
    trend: Optional[str],
    seasonal: str,
    n_meses: int = 12,
) -> pd.DataFrame:
    """
    Re-entrena el modelo con todos los datos disponibles y genera
    predicciones para los próximos N meses.

    Nota: se re-entrena intencionalmente con el dataset completo
    (no solo el train) para maximizar la información disponible
    en las predicciones hacia adelante.

    Args:
        df_completo: serie temporal completa, sin NaN.
        trend: tipo de tendencia del mejor modelo.
        seasonal: tipo de estacionalidad del mejor modelo.
        n_meses: cantidad de meses a predecir.

    Returns:
        DataFrame con columnas 'fecha', 'mes', 'prediccion'.
    """
    df_limpio = df_completo.dropna(subset=["ventas"]).sort_values("fecha")

    modelo_final, pred = entrenar_modelo(df_limpio, n_meses, trend, seasonal)

    ultima_fecha = df_limpio["fecha"].max()
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=1),
        periods=n_meses,
        freq="MS",
    )

    df_futuro = pd.DataFrame({
        "fecha": fechas_futuras,
        "prediccion": pred.round(1).values,
    })
    df_futuro["mes"] = df_futuro["fecha"].apply(lambda f: mes_anio_es(f, abreviado=True))

    return df_futuro


# ---------------------------------------------------------------------------
# Función principal: orquesta todo el flujo del modelo
# ---------------------------------------------------------------------------

def ejecutar_pipeline(
    df: pd.DataFrame,
    n_meses_validacion: int = 12,
) -> ResultadoModelo:
    """
    Punto de entrada principal. Orquesta:
    1. División train/test
    2. Selección del mejor modelo
    3. Entrenamiento final con validación
    4. Predicciones futuras

    Args:
        df: serie temporal con columnas 'fecha' y 'ventas'.
        n_meses_validacion: meses a usar como período de validación.

    Returns:
        ResultadoModelo con todo lo necesario para la UI.
    """
    train, test = dividir_train_test(df, n_meses_validacion)

    mejor_trend, mejor_seasonal, _ = seleccionar_mejor_modelo(train, test)

    # Entrenamiento final sobre el mismo train para métricas de validación
    modelo, predicciones_val = entrenar_modelo(train, len(test), mejor_trend, mejor_seasonal)
    metricas = calcular_metricas(test["ventas"], predicciones_val)

    # Predicciones futuras (re-entrenando con todos los datos)
    df_futuro = generar_predicciones_futuras(df, mejor_trend, mejor_seasonal)

    return ResultadoModelo(
        modelo=modelo,
        trend=mejor_trend,
        seasonal=mejor_seasonal,
        metricas=metricas,
        predicciones_validacion=predicciones_val,
        train=train,
        test=test,
        df_futuro=df_futuro,
    )

"""
validator.py
------------
Valida que el archivo de entrada tenga el schema correcto
antes de procesarlo. Devuelve errores claros en lugar de
que pandas explote con mensajes crípticos.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from src.formatting import COLUMNAS_MES


@dataclass
class ResultadoValidacion:
    """Resultado de validar un DataFrame de entrada."""
    valido: bool
    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)

    def resumen(self) -> str:
        """Texto para mostrar al usuario en la UI."""
        lineas = []
        for e in self.errores:
            lineas.append(f"❌ {e}")
        for a in self.advertencias:
            lineas.append(f"⚠️ {a}")
        return "\n".join(lineas)


def validar_dataframe(df: pd.DataFrame) -> ResultadoValidacion:
    """
    Valida el DataFrame recién cargado.

    Reglas:
    - Debe tener al menos una columna (la de año).
    - La primera columna debe ser interpretable como año (valores 4 dígitos).
    - Debe tener al menos 6 columnas de meses reconocidas.
    - Debe tener al menos 2 filas de datos (mínimo para Holt-Winters).
    - Los valores de ventas deben ser numéricos o convertibles.

    Returns:
        ResultadoValidacion con valido=True/False y lista de errores/advertencias.
    """
    errores = []
    advertencias = []

    # 1. Estructura mínima
    if df is None or df.empty:
        return ResultadoValidacion(valido=False, errores=["El archivo está vacío."])

    if df.shape[1] < 2:
        errores.append(
            "El archivo debe tener al menos 2 columnas: una de año y al menos una de mes."
        )
        return ResultadoValidacion(valido=False, errores=errores)

    # 2. Primera columna = año
    col_anio = df.columns[0]
    try:
        anios = pd.to_numeric(df[col_anio], errors="coerce")
        if anios.isna().all():
            errores.append(
                f"La primera columna '{col_anio}' no contiene valores numéricos de año."
            )
        else:
            anios_validos = anios.dropna()
            if not ((anios_validos >= 1900) & (anios_validos <= 2100)).all():
                advertencias.append(
                    f"La columna '{col_anio}' tiene valores fuera del rango 1900-2100. "
                    "Verificá que sea la columna de año."
                )
    except Exception:
        errores.append(f"No se pudo leer la columna de año '{col_anio}'.")

    # 3. Columnas de meses reconocidas
    columnas_upper = [str(c).upper().strip() for c in df.columns[1:]]
    meses_encontrados = [c for c in columnas_upper if c in COLUMNAS_MES]

    if len(meses_encontrados) < 6:
        errores.append(
            f"Solo se reconocieron {len(meses_encontrados)} columnas de meses "
            f"(mínimo 6). Asegurate de usar nombres en español en mayúsculas: "
            f"ENERO, FEBRERO, ... DICIEMBRE."
        )
    elif len(meses_encontrados) < 12:
        advertencias.append(
            f"Se encontraron {len(meses_encontrados)} de 12 meses. "
            "El modelo funcionará pero con menor precisión estacional."
        )

    # 4. Filas mínimas
    if df.shape[0] < 2:
        errores.append(
            "El archivo debe tener al menos 2 años de datos para entrenar el modelo."
        )
    elif df.shape[0] < 3:
        advertencias.append(
            "Con solo 2 años de datos el modelo tendrá poca capacidad predictiva. "
            "Se recomiendan al menos 3 años."
        )

    # 5. Valores numéricos en columnas de meses
    cols_meses_df = [
        c for c in df.columns[1:]
        if str(c).upper().strip() in COLUMNAS_MES
    ]
    if cols_meses_df:
        muestra = df[cols_meses_df].copy()
        # Intentar convertir a numérico para detectar valores no convertibles
        no_numericos = 0
        for col in cols_meses_df:
            converted = pd.to_numeric(
                muestra[col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                errors="coerce"
            )
            # Contar solo los que no son NaN originales y no se pudieron convertir
            originales_no_nan = muestra[col].notna()
            no_numericos += (converted.isna() & originales_no_nan).sum()

        if no_numericos > 0:
            advertencias.append(
                f"Se encontraron {no_numericos} celdas con valores no numéricos "
                "en las columnas de meses. Serán ignoradas."
            )

    valido = len(errores) == 0
    return ResultadoValidacion(valido=valido, errores=errores, advertencias=advertencias)

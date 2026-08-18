"""
test_model.py
-------------
Tests para src/model.py

El modelo es el módulo más complejo. Lo testeamos en capas:
1. División de datos (determinista, fácil de verificar)
2. Métricas (matemática pura)
3. Pipeline completo (integración)
"""

import numpy as np
import pandas as pd
import pytest
from src.model import (
    Metricas,
    calcular_metricas,
    dividir_train_test,
    dividir_train_val_test,
    ejecutar_pipeline,
    generar_predicciones_futuras,
    prediccion_naive_estacional,
    seleccionar_mejor_modelo,
)


class TestDividirTrainTest:
    """dividir_train_test separa la serie en entrenamiento y validación."""

    def test_suma_de_filas_igual_al_total(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, n_meses_validacion=12)
        assert len(train) + len(test) == len(serie_temporal_ok)

    def test_test_tiene_n_meses(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, n_meses_validacion=12)
        assert len(test) == 12

    def test_train_viene_antes_que_test(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, n_meses_validacion=12)
        assert train["fecha"].max() < test["fecha"].min()

    def test_con_6_meses_validacion(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, n_meses_validacion=6)
        assert len(test) == 6

    def test_n_meses_mayor_que_disponible_usa_minimo(self, serie_temporal_ok):
        """Si se piden más meses de los disponibles, no debe fallar."""
        train, test = dividir_train_test(serie_temporal_ok, n_meses_validacion=100)
        # Debe usar un valor razonable, no crashear
        assert len(test) > 0
        assert len(train) > 0

    def test_sin_nan_en_output(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, n_meses_validacion=12)
        assert train["ventas"].isna().sum() == 0
        assert test["ventas"].isna().sum() == 0


class TestDividirTrainValTest:
    """dividir_train_val_test separa en tres partes cronológicas, o cae a dos
    si no hay historial suficiente (ver hallazgo 14.1 de la auditoría)."""

    def test_con_historial_suficiente_hace_split_de_tres_partes(self, serie_temporal_ok):
        # 48 meses, 12 de validación + 6 de test -> train de 30, alcanza el mínimo de 24.
        train, validacion, test, degradado = dividir_train_val_test(
            serie_temporal_ok, n_meses_validacion=12, n_meses_test=6
        )
        assert degradado is False
        assert len(train) == 30
        assert len(validacion) == 12
        assert len(test) == 6
        assert len(train) + len(validacion) + len(test) == len(serie_temporal_ok)

    def test_orden_cronologico_train_validacion_test(self, serie_temporal_ok):
        train, validacion, test, _ = dividir_train_val_test(
            serie_temporal_ok, n_meses_validacion=12, n_meses_test=6
        )
        assert train["fecha"].max() < validacion["fecha"].min()
        assert validacion["fecha"].max() < test["fecha"].min()

    def test_test_nunca_participa_en_la_seleccion(self, serie_temporal_ok):
        """El punto central del fix: seleccionar_mejor_modelo solo debe ver
        train+validación, nunca las fechas de `test`."""
        train, validacion, test, degradado = dividir_train_val_test(
            serie_temporal_ok, n_meses_validacion=12, n_meses_test=6
        )
        assert not degradado
        fechas_disponibles_para_elegir = set(train["fecha"]) | set(validacion["fecha"])
        assert fechas_disponibles_para_elegir.isdisjoint(set(test["fecha"]))

    def test_historial_corto_cae_a_dos_partes(self):
        """Con poco historial, no debe inventar un test que rompa el mínimo
        de convergencia de Holt-Winters — cae al split viejo de dos partes."""
        fechas = pd.date_range("2023-01-01", periods=24, freq="MS")
        df_corto = pd.DataFrame({"fecha": fechas, "ventas": [1000.0] * 24})
        train, validacion, test, degradado = dividir_train_val_test(
            df_corto, n_meses_validacion=6, n_meses_test=6
        )
        assert degradado is True
        # test == validacion en el caso degradado (mismo rol, documentado)
        assert test["fecha"].equals(validacion["fecha"])
        assert len(train) + len(validacion) == len(df_corto)


class TestCalcularMetricas:
    """calcular_metricas calcula MAE, RMSE y R² correctamente."""

    def test_prediccion_perfecta_da_mae_cero(self):
        reales = pd.Series([100.0, 200.0, 300.0])
        predichos = pd.Series([100.0, 200.0, 300.0])
        m = calcular_metricas(reales, predichos)
        assert m.mae == pytest.approx(0.0)

    def test_prediccion_perfecta_da_rmse_cero(self):
        reales = pd.Series([100.0, 200.0, 300.0])
        predichos = pd.Series([100.0, 200.0, 300.0])
        m = calcular_metricas(reales, predichos)
        assert m.rmse == pytest.approx(0.0)

    def test_prediccion_perfecta_da_r2_uno(self):
        reales = pd.Series([100.0, 200.0, 300.0])
        predichos = pd.Series([100.0, 200.0, 300.0])
        m = calcular_metricas(reales, predichos)
        assert m.r2 == pytest.approx(1.0)

    def test_mae_es_correcto(self):
        reales = pd.Series([100.0, 200.0])
        predichos = pd.Series([110.0, 190.0])
        m = calcular_metricas(reales, predichos)
        assert m.mae == pytest.approx(10.0)

    def test_error_absoluto_total_es_correcto(self):
        reales = pd.Series([100.0, 200.0, 300.0])
        predichos = pd.Series([90.0, 210.0, 280.0])
        m = calcular_metricas(reales, predichos)
        # |10| + |10| + |20| = 40
        assert m.error_absoluto_total == pytest.approx(40.0)

    def test_devuelve_instancia_metricas(self):
        reales = pd.Series([100.0, 200.0])
        predichos = pd.Series([100.0, 200.0])
        m = calcular_metricas(reales, predichos)
        assert isinstance(m, Metricas)

    def test_rmse_mayor_o_igual_que_mae(self):
        """Por definición, RMSE >= MAE siempre."""
        reales = pd.Series([100.0, 200.0, 150.0, 300.0])
        predichos = pd.Series([110.0, 180.0, 160.0, 280.0])
        m = calcular_metricas(reales, predichos)
        assert m.rmse >= m.mae


class TestPrediccionNaiveEstacional:
    """prediccion_naive_estacional repite el mismo mes del año anterior."""

    def test_repite_el_mismo_mes_del_anio_anterior(self):
        fechas_train = pd.date_range("2022-01-01", periods=24, freq="MS")
        train = pd.DataFrame({"fecha": fechas_train, "ventas": list(range(100, 124))})
        periodo = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=3, freq="MS")})

        pred = prediccion_naive_estacional(train, periodo)

        # Enero 2024 debe repetir enero 2023 (índice 12 en train, valor 112)
        assert pred.iloc[0] == pytest.approx(112.0)

    def test_usa_promedio_del_mes_si_no_hay_dato_exacto_un_anio_antes(self):
        # Solo un año de train: no existe "el mismo mes, un año antes" para nada del período.
        fechas_train = pd.date_range("2023-01-01", periods=12, freq="MS")
        train = pd.DataFrame({"fecha": fechas_train, "ventas": [100.0] * 11 + [200.0]})
        periodo = pd.DataFrame({"fecha": pd.date_range("2024-12-01", periods=1, freq="MS")})

        pred = prediccion_naive_estacional(train, periodo)

        # Cae al promedio de diciembre en train, que es 200.0 (solo un dato)
        assert pred.iloc[0] == pytest.approx(200.0)

    def test_devuelve_mismo_largo_que_periodo(self, serie_temporal_ok):
        train, _, test, _ = dividir_train_val_test(serie_temporal_ok, 12, 6)
        pred = prediccion_naive_estacional(train, test)
        assert len(pred) == len(test)


class TestSeleccionarMejorModelo:
    """seleccionar_mejor_modelo elige la combinación con menor MAE."""

    def test_devuelve_tres_valores(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, 12)
        trend, seasonal, metricas = seleccionar_mejor_modelo(train, test)
        assert trend is None or trend in ("additive", "multiplicative")
        assert seasonal in ("additive", "multiplicative")
        assert isinstance(metricas, Metricas)

    def test_mae_es_positivo(self, serie_temporal_ok):
        train, test = dividir_train_test(serie_temporal_ok, 12)
        _, _, metricas = seleccionar_mejor_modelo(train, test)
        assert metricas.mae > 0

    def test_falla_con_datos_insuficientes(self):
        """Con menos de 12 meses en train debe lanzar RuntimeError."""
        fechas = pd.date_range("2024-01-01", periods=10, freq="MS")
        df_corto = pd.DataFrame({
            "fecha": fechas,
            "ventas": [100.0] * 10,
        })
        train = df_corto.iloc[:5]
        test = df_corto.iloc[5:]
        # match usa regex — buscamos substring que aparezca en el mensaje real
        with pytest.raises(RuntimeError, match="Ninguna combinación"):
            seleccionar_mejor_modelo(train, test)


class TestGenerarPrediccionesFuturas:
    """generar_predicciones_futuras produce predicciones para N meses hacia adelante."""

    def test_devuelve_12_filas_por_defecto(self, serie_temporal_ok):
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        assert len(df) == 12

    def test_devuelve_n_filas(self, serie_temporal_ok):
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive", n_meses=6)
        assert len(df) == 6

    def test_fechas_empiezan_despues_del_historico(self, serie_temporal_ok):
        ultima_fecha_historico = serie_temporal_ok["fecha"].max()
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        assert df["fecha"].min() > ultima_fecha_historico

    def test_fechas_son_consecutivas_mensualmente(self, serie_temporal_ok):
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        diferencias = df["fecha"].diff().dropna()
        # Todas las diferencias deben ser aproximadamente 1 mes
        assert (diferencias.dt.days >= 28).all()
        assert (diferencias.dt.days <= 31).all()

    def test_predicciones_son_positivas(self, serie_temporal_ok):
        """Para ventas de heladerías, las predicciones deben ser positivas."""
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        assert (df["prediccion"] > 0).all()

    def test_tiene_columna_mes(self, serie_temporal_ok):
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        assert "mes" in df.columns

    def test_columna_mes_no_esta_vacia(self, serie_temporal_ok):
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        assert df["mes"].notna().all()

    def test_columna_mes_en_espanol(self, serie_temporal_ok):
        df = generar_predicciones_futuras(serie_temporal_ok, "additive", "additive")
        assert df["mes"].iloc[0].startswith(("ene ", "feb ", "mar ", "abr ", "may ", "jun ",
                                               "jul ", "ago ", "sep ", "oct ", "nov ", "dic "))
        assert "January" not in df["mes"].str.cat()


class TestEjecutarPipeline:
    """ejecutar_pipeline es el test de integración — orquesta todo el flujo."""

    def test_pipeline_completo_no_falla(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok, n_meses_validacion=12)
        assert resultado is not None

    def test_pipeline_devuelve_modelo(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok)
        assert resultado.modelo is not None

    def test_pipeline_devuelve_metricas(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok)
        assert isinstance(resultado.metricas, Metricas)
        assert resultado.metricas.mae > 0

    def test_pipeline_devuelve_predicciones_futuras(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok)
        assert len(resultado.df_futuro) == 12

    def test_pipeline_train_y_test_no_se_solapan(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok, n_meses_validacion=12)
        assert resultado.train["fecha"].max() < resultado.test["fecha"].min()

    def test_pipeline_predicciones_val_tienen_mismo_largo_que_test(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok, n_meses_validacion=12)
        assert len(resultado.predicciones_validacion) == len(resultado.test)

    def test_pipeline_seasonal_es_valido(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok)
        assert resultado.seasonal in ("additive", "multiplicative")

    def test_pipeline_con_datos_minimos_lanza_error_claro(self, df_ancho_dos_anios):
        """
        Con solo 2 años (24 meses), statsmodels necesita al menos 2 ciclos
        estacionales completos en el train. Si train < 24 meses, el modelo
        no puede arrancar y debe lanzar un RuntimeError con mensaje claro,
        no un traceback críptico de statsmodels.
        """
        from src.loader import transformar_a_serie
        df_serie = transformar_a_serie(df_ancho_dos_anios)
        # Con 24 meses y 6 de validación, train tiene 18 meses < 2 ciclos completos
        with pytest.raises(RuntimeError, match="Ninguna combinación"):
            ejecutar_pipeline(df_serie, n_meses_validacion=6)

    def test_pipeline_con_historial_suficiente_no_esta_degradado(self, serie_temporal_ok):
        """Regresión del hallazgo 14.1: con 48 meses, el pipeline debe poder
        reservar un test genuino, nunca visto en la selección del modelo."""
        resultado = ejecutar_pipeline(serie_temporal_ok, n_meses_validacion=12, n_meses_test=6)
        assert resultado.split_degradado is False
        assert len(resultado.test) == 6

    def test_pipeline_devuelve_metricas_baseline(self, serie_temporal_ok):
        resultado = ejecutar_pipeline(serie_temporal_ok)
        assert isinstance(resultado.metricas_baseline, Metricas)
        assert resultado.metricas_baseline.mae > 0

    def test_pipeline_avisa_split_degradado_con_historial_corto(self):
        """Regresión del hallazgo 14.1 en el caso límite: con historial corto
        para un split de tres partes (pero suficiente para que el modelo
        igual converja con el split de dos partes), el pipeline debe seguir
        funcionando (no romper) pero avisar que la métrica reportada puede
        ser optimista. 30 meses: no alcanza para train>=24 + val(6) + test(6),
        pero sí alcanza para train(24) + validación(6) en el split degradado.
        """
        fechas = pd.date_range("2022-07-01", periods=30, freq="MS")
        np.random.seed(1)
        ventas = 1000 + 300 * np.sin(np.arange(30) * 2 * np.pi / 12) + np.random.normal(0, 30, 30)
        df_corto = pd.DataFrame({"fecha": fechas, "ventas": ventas.round(1)})

        resultado = ejecutar_pipeline(df_corto, n_meses_validacion=6, n_meses_test=6)

        assert resultado.split_degradado is True
        assert len(resultado.train) == 24

    def test_pipeline_modelo_no_es_notablemente_peor_que_baseline_naive(self, serie_temporal_ok):
        """
        Regresión del hallazgo 14.6: nada protegía antes contra que el modelo
        empeorara silenciosamente hasta ser peor que "repetir el año pasado".
        No exigimos que el modelo siempre GANE (con ruido aleatorio puede no
        pasar en todas las semillas), pero sí que no sea dramáticamente peor
        que la referencia — eso sería señal de que algo se rompió.
        """
        resultado = ejecutar_pipeline(serie_temporal_ok, n_meses_validacion=12, n_meses_test=6)
        assert resultado.metricas.mae <= resultado.metricas_baseline.mae * 1.5

"""
App de Predicción de Demanda para Heladerías
=============================================
Aplicación Streamlit para analizar y predecir ventas usando Holt-Winters.

Para ejecutar:
    streamlit run app_prediccion_heladerias.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import locale

def formato_numero(numero, decimales=2):
    """Formatea número con punto como separador de miles y coma como decimal."""
    formato = f"{{:,.{decimales}f}}"
    texto = formato.format(numero)
    # Convertir de formato US (1,234.56) a formato AR (1.234,56)
    texto = texto.replace(',', 'X').replace('.', ',').replace('X', '.')
    return texto

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Demanda - Heladerías",
    page_icon="🍦",
    layout="wide"
)

# Título principal
st.title("🍦 Predicción de Demanda para Heladerías")
st.markdown("### Modelo Holt-Winters para Series Temporales")
st.markdown("---")

# ============================================================================
# FUNCIONES DEL MODELO
# ============================================================================

def cargar_y_transformar_datos(df):
    """Transforma el DataFrame del formato ancho al formato fecha-ventas."""
    # Obtener el nombre de la primera columna (AÑO)
    id_column = df.columns[0]
    
    # Melt para transformar de ancho a largo
    df_melted = df.melt(id_vars=[id_column], var_name='mes', value_name='ventas')
    df_melted = df_melted.rename(columns={id_column: 'AÑO'})
    
    # Limpiar datos vacíos (strings vacíos)
    df_melted = df_melted[df_melted['ventas'].astype(str).str.strip() != '']
    
    # Detectar formato de números
    muestra = df_melted['ventas'].dropna().iloc[0] if len(df_melted['ventas'].dropna()) > 0 else None
    
    if muestra is not None and isinstance(muestra, str):
        if ',' in str(muestra) and '.' in str(muestra):
            df_melted['ventas'] = df_melted['ventas'].astype(str).str.replace('.', '', regex=False)
            df_melted['ventas'] = df_melted['ventas'].str.replace(',', '.', regex=False)
        elif ',' in str(muestra) and '.' not in str(muestra):
            df_melted['ventas'] = df_melted['ventas'].astype(str).str.replace(',', '.', regex=False)
    
    df_melted['ventas'] = pd.to_numeric(df_melted['ventas'], errors='coerce')
    
    # Mapeo de meses
    meses = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    df_melted['mes_num'] = df_melted['mes'].str.upper().map(meses)
    df_melted['fecha'] = pd.to_datetime(dict(year=df_melted['AÑO'], month=df_melted['mes_num'], day=1))
    
    df_final = df_melted[['fecha', 'ventas']].sort_values('fecha').reset_index(drop=True)
    
    # Encontrar la última fecha con dato real (no NaN)
    ultima_fecha_real = df_final[df_final['ventas'].notna()]['fecha'].max()
    
    # Filtrar solo hasta la última fecha con dato real (eliminar meses futuros sin datos)
    df_final = df_final[df_final['fecha'] <= ultima_fecha_real].copy()
    
    # Imputar valores NaN SOLO para meses históricos (dentro del rango de datos)
    if df_final['ventas'].isna().sum() > 0:
        df_final['mes'] = df_final['fecha'].dt.month
        promedios = df_final.groupby('mes')['ventas'].mean()
        
        for idx in df_final[df_final['ventas'].isna()].index:
            mes = df_final.loc[idx, 'mes']
            df_final.loc[idx, 'ventas'] = promedios[mes]
        
        df_final = df_final.drop(columns=['mes'])
    
    return df_final


def corregir_pandemia(df, meses_afectados):
    """Corrige valores atípicos de la pandemia."""
    df = df.copy()
    df_sin_2020 = df[df['fecha'].dt.year != 2020]
    
    for mes, tipo in meses_afectados.items():
        fecha_str = f'2020-{mes:02d}-01'
        promedio = df_sin_2020[df_sin_2020['fecha'].dt.month == mes]['ventas'].mean()
        
        if tipo == 'promedio':
            df.loc[df['fecha'] == fecha_str, 'ventas'] = promedio
        elif tipo == 'mixto':
            valor_real = df.loc[df['fecha'] == fecha_str, 'ventas'].values
            if len(valor_real) > 0:
                df.loc[df['fecha'] == fecha_str, 'ventas'] = (promedio + valor_real[0]) / 2
    
    return df


def entrenar_modelo(train, test, trend, seasonal):
    """
    Entrena el modelo usando la optimización automática de statsmodels.
    Esta es la forma más precisa de ajustar los parámetros del modelo.
    """
    # Usar optimización automática de statsmodels (sin parámetros manuales)
    modelo = ExponentialSmoothing(
        train['ventas'],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12
    ).fit()  # Sin argumentos = optimización automática
    
    predicciones = modelo.forecast(len(test))
    
    mae = mean_absolute_error(test['ventas'], predicciones)
    rmse = np.sqrt(mean_squared_error(test['ventas'], predicciones))
    r2 = r2_score(test['ventas'], predicciones)
    error_total = np.abs(test['ventas'].values - predicciones.values).sum()
    
    return modelo, predicciones, {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Error Absoluto Total': error_total
    }


# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Cargar archivo
st.sidebar.subheader("1. Cargar datos")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona el archivo CSV",
    type=['csv'],
    help="El archivo debe tener columnas: AÑO, ENERO, FEBRERO, ..., DICIEMBRE"
)

# Parámetros del modelo
st.sidebar.subheader("2. Parámetros")

separador = st.sidebar.selectbox(
    "Separador del CSV",
    [';', ',', '\t'],
    index=0
)

encoding = st.sidebar.selectbox(
    "Codificación",
    ['latin1', 'utf-8', 'cp1252'],
    index=0
)

año_corte = st.sidebar.number_input(
    "Año de corte para entrenamiento",
    min_value=2015,
    max_value=2026,
    value=2024,
    help="El modelo entrena hasta este año y valida con el siguiente"
)

corregir_2020 = st.sidebar.checkbox(
    "Corregir datos de pandemia (2020)",
    value=False,
    help="Ajusta marzo y abril 2020 usando promedios históricos"
)

# Nota: La optimización automática de statsmodels se usa siempre (es más precisa)

# ============================================================================
# PROCESAMIENTO Y VISUALIZACIÓN
# ============================================================================

if uploaded_file is not None:
    # Cargar datos
    try:
        df_raw = pd.read_csv(uploaded_file, sep=separador, encoding=encoding)
        
        # Mostrar datos cargados
        st.subheader("📊 Datos Cargados")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_raw, use_container_width=True)
        
        with col2:
            st.metric("Filas", df_raw.shape[0])
            st.metric("Columnas", df_raw.shape[1])
        
        # Transformar datos
        df = cargar_y_transformar_datos(df_raw)
        
        # Corregir pandemia si está activado
        if corregir_2020:
            df = corregir_pandemia(df, {3: 'mixto', 4: 'promedio'})
            st.info("✅ Se aplicó corrección de datos de pandemia (marzo y abril 2020)")
        
        # Mostrar serie temporal
        st.subheader("📈 Serie Temporal de Ventas")
        
        fig_serie = px.line(
            df, x='fecha', y='ventas',
            title='Ventas Históricas',
            labels={'fecha': 'Fecha', 'ventas': 'Ventas'}
        )
        fig_serie.update_traces(line_color='#1f77b4', line_width=2)
        fig_serie.update_layout(hovermode='x unified')
        st.plotly_chart(fig_serie, use_container_width=True)
        
        # División train/test
        df = df.dropna(subset=['ventas']).sort_values('fecha').reset_index(drop=True)
        train = df[df['fecha'].dt.year <= año_corte].copy()
        test = df[df['fecha'].dt.year == año_corte + 1].copy()
        
        if len(test) == 0:
            st.warning("No hay datos para el año de validación. Usando últimos 12 meses.")
            train = df.iloc[:-12].copy()
            test = df.iloc[-12:].copy()
        
        st.markdown("---")
        
        # Botón para ejecutar el modelo
        if st.button("🚀 Ejecutar Modelo", type="primary", use_container_width=True):
            
            with st.spinner("Evaluando combinaciones de modelo..."):
                # Evaluar combinaciones
                combinaciones = [
                    ('additive', 'additive'),
                    ('additive', 'multiplicative'),
                    (None, 'additive'),
                    (None, 'multiplicative'),
                ]
                
                resultados = []
                for trend, seasonal in combinaciones:
                    try:
                        _, _, metricas = entrenar_modelo(train, test, trend, seasonal)
                        resultados.append({
                            'Tendencia': str(trend),
                            'Estacionalidad': seasonal,
                            'MAE': metricas['MAE'],
                            'R²': metricas['R²']
                        })
                    except:
                        pass
                
                if resultados:
                    mejor = min(resultados, key=lambda x: x['MAE'])
                    mejor_trend = None if mejor['Tendencia'] == 'None' else mejor['Tendencia']
                    mejor_seasonal = mejor['Estacionalidad']
            
            # Entrenar modelo final con optimización automática de statsmodels
            with st.spinner("Entrenando modelo con optimización automática..."):
                modelo_final, predicciones, _ = entrenar_modelo(train, test, mejor_trend, mejor_seasonal)
            
            # Calcular métricas finales
            mae = mean_absolute_error(test['ventas'], predicciones)
            rmse = np.sqrt(mean_squared_error(test['ventas'], predicciones))
            r2 = r2_score(test['ventas'], predicciones)
            error_total = np.abs(test['ventas'].values - predicciones.values).sum()
            
            st.markdown("---")
            st.subheader("📊 Resultados del Modelo")
            
            # Métricas en cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", formato_numero(mae))
            col2.metric("RMSE", formato_numero(rmse))
            col3.metric("R²", f"{r2:.4f}")
            col4.metric("Error Absoluto Total", formato_numero(error_total))
            
            # Comparativa de validación
            st.subheader("🔍 Validación: Predicción vs Real")
            
            comparativa = test[['fecha', 'ventas']].copy()
            comparativa['prediccion'] = predicciones.values
            comparativa['error'] = abs(comparativa['ventas'] - comparativa['prediccion'])
            comparativa['error_pct'] = (comparativa['error'] / comparativa['ventas'] * 100)
            comparativa['mes'] = comparativa['fecha'].dt.strftime('%b %Y')
            
            # Calcular totales
            total_real = comparativa['ventas'].sum()
            total_predicho = comparativa['prediccion'].sum()
            diferencia_total = total_real - total_predicho
            diferencia_pct = (diferencia_total / total_real * 100)
            
            # Mostrar totales en cards
            st.markdown("**Totales del período de validación:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Venta Real (kg)", formato_numero(total_real))
            col2.metric("Total Predicción (kg)", formato_numero(total_predicho))
            col3.metric("Diferencia (kg)", formato_numero(diferencia_total))
            col4.metric("Diferencia %", f"{diferencia_pct:.2f}%".replace('.', ','))
            
            # Gráfico de comparación
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='Real',
                x=comparativa['mes'],
                y=comparativa['ventas'],
                marker_color='#2ecc71'
            ))
            fig_comp.add_trace(go.Bar(
                name='Predicción',
                x=comparativa['mes'],
                y=comparativa['prediccion'],
                marker_color='#3498db'
            ))
            fig_comp.update_layout(
                title='Comparación: Ventas Reales vs Predicción',
                barmode='group',
                xaxis_title='Mes',
                yaxis_title='Ventas'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Preparar tabla con formato argentino
            comparativa_display = comparativa[['mes', 'ventas', 'prediccion', 'error', 'error_pct']].copy()
            comparativa_display['ventas'] = comparativa_display['ventas'].apply(lambda x: formato_numero(x))
            comparativa_display['prediccion'] = comparativa_display['prediccion'].apply(lambda x: formato_numero(x))
            comparativa_display['error'] = comparativa_display['error'].apply(lambda x: formato_numero(x))
            comparativa_display['error_pct'] = comparativa_display['error_pct'].apply(lambda x: f"{x:.2f}%".replace('.', ','))
            
            # Tabla de comparación
            st.dataframe(
                comparativa_display.rename(columns={
                    'mes': 'Mes',
                    'ventas': 'Venta Real',
                    'prediccion': 'Predicción',
                    'error': 'Error',
                    'error_pct': 'Error %'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # Predicciones futuras
            st.subheader("🔮 Predicciones Futuras (Próximos 12 meses)")
            
            # Filtrar solo datos con ventas reales (sin NaN)
            df_con_datos = df[df['ventas'].notna()].copy()
            
            # Re-entrenar con todos los datos reales
            modelo_futuro = ExponentialSmoothing(
                df_con_datos['ventas'],
                trend=mejor_trend,
                seasonal=mejor_seasonal,
                seasonal_periods=12
            ).fit()
            
            # Última fecha con datos reales
            ultima_fecha = df_con_datos['fecha'].max()
            fechas_futuras = pd.date_range(
                start=ultima_fecha + pd.DateOffset(months=1),
                periods=12,
                freq='MS'
            )
            
            pred_futuras = modelo_futuro.forecast(steps=12)
            
            df_futuro = pd.DataFrame({
                'fecha': fechas_futuras,
                'prediccion': pred_futuras.round(1)
            })
            df_futuro['mes'] = df_futuro['fecha'].dt.strftime('%B %Y')
            
            # Gráfico de predicciones futuras
            fig_futuro = make_subplots(rows=1, cols=2, subplot_titles=('Serie Completa', 'Predicciones Futuras'))
            
            # Serie histórica + predicción (solo datos reales)
            fig_futuro.add_trace(
                go.Scatter(x=df_con_datos['fecha'], y=df_con_datos['ventas'], name='Histórico', line=dict(color='#2ecc71')),
                row=1, col=1
            )
            fig_futuro.add_trace(
                go.Scatter(x=df_futuro['fecha'], y=df_futuro['prediccion'], name='Predicción', 
                          line=dict(color='#e74c3c', dash='dash')),
                row=1, col=1
            )
            
            # Solo predicciones
            fig_futuro.add_trace(
                go.Bar(x=df_futuro['mes'], y=df_futuro['prediccion'], name='Predicción Mensual',
                      marker_color='#3498db'),
                row=1, col=2
            )
            
            fig_futuro.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_futuro, use_container_width=True)
            
            # Resumen de predicciones
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Anual Predicho (kg)", formato_numero(df_futuro['prediccion'].sum()))
            col2.metric("Promedio Mensual (kg)", formato_numero(df_futuro['prediccion'].mean()))
            col3.metric("Mes Pico", f"{df_futuro.loc[df_futuro['prediccion'].idxmax(), 'mes']}")
            
            # Preparar tabla con formato argentino
            df_futuro_display = df_futuro[['mes', 'prediccion']].copy()
            df_futuro_display['prediccion'] = df_futuro_display['prediccion'].apply(lambda x: formato_numero(x))
            
            # Tabla de predicciones
            st.dataframe(
                df_futuro_display.rename(columns={
                    'mes': 'Mes',
                    'prediccion': 'Predicción (kg)'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Botón de descarga
            csv = df_futuro.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Predicciones (CSV)",
                data=csv,
                file_name="predicciones_futuras.csv",
                mime="text/csv"
            )
            
            st.success("✅ Análisis completado exitosamente!")
            
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        st.info("Verifica que el formato del CSV sea correcto (AÑO, ENERO, FEBRERO, ..., DICIEMBRE)")

else:
    # Mostrar instrucciones cuando no hay archivo
    st.info("👈 Carga un archivo CSV desde el panel lateral para comenzar")
    
    st.markdown("""
    ### 📋 Formato esperado del archivo CSV
    
    | AÑO | ENERO | FEBRERO | MARZO | ... | DICIEMBRE |
    |-----|-------|---------|-------|-----|-----------|
    | 2020 | 12345 | 11234 | 10123 | ... | 15678 |
    | 2021 | 13456 | 12345 | 11234 | ... | 16789 |
    
    ### 🔧 Parámetros configurables
    
    - **Separador CSV**: Caracter que separa las columnas (`;`, `,`, etc.)
    - **Año de corte**: Último año para entrenamiento (el siguiente se usa para validación)
    - **Corrección pandemia**: Ajusta valores atípicos de marzo/abril 2020
    - **Optimización**: Busca automáticamente los mejores hiperparámetros
    """)

# Footer
st.markdown("---")
st.markdown("*Desarrollado con Streamlit y Holt-Winters | Modelo de Series Temporales para Heladerías*")

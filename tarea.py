import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ===== Título de la app =====
st.title("Pronóstico COVID-19 - Casos y Muertes (14 días)")

# ===== Cargar dataset =====
df = pd.read_csv("acti_data.csv")  # Ajusta el nombre de tu CSV

# ===== Selección de país =====
paises = df['Country_Region'].unique()
pais_seleccionado = st.selectbox("Seleccione un país:", paises)

# ===== Filtrar datos por país =====
data_pais = df[df['Country_Region'] == pais_seleccionado]

# ===== Convertir columna 'Date' a datetime y ordenar =====
if 'Date' in data_pais.columns:
    data_pais['Date'] = pd.to_datetime(data_pais['Date'])
    data_pais = data_pais.sort_values('Date')

# ===== Función para modelar y proyectar =====
def pronostico_serie(serie, tipo="Casos"):
    st.subheader(f"Serie histórica de {tipo} en {pais_seleccionado}")
    
    # Gráfico histórico
    fig, ax = plt.subplots()
    ax.plot(serie, label=f"{tipo} históricos", color="blue")
    ax.set_xlabel("Días")
    ax.set_ylabel(tipo)
    ax.legend()
    st.pyplot(fig)
    
    # Intentar SARIMA primero
    try:
        modelo = SARIMAX(
            serie,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        resultado = modelo.fit(disp=False)
        forecast = resultado.get_forecast(steps=14)
        pred = forecast.predicted_mean
        modelo_usado = "SARIMA"
    except:
        # Alternativa ETS
        modelo_ets = ExponentialSmoothing(serie, seasonal='add', seasonal_periods=7)
        resultado = modelo_ets.fit()
        pred = resultado.forecast(14)
        modelo_usado = "ETS"
    
    # Gráfico histórico + pronóstico
    st.subheader(f"Pronóstico a 14 días de {tipo} ({modelo_usado})")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(serie)), serie, label=f"{tipo} históricos", color="blue")
    ax2.plot(range(len(serie), len(serie)+14), pred, label="Pronóstico", color="red", linestyle="--")
    ax2.set_xlabel("Días")
    ax2.set_ylabel(tipo)
    ax2.legend()
    st.pyplot(fig2)

# ===== Serie de Casos =====
serie_casos = data_pais['Confirmed'].reset_index(drop=True)
pronostico_serie(serie_casos, tipo="Casos")

# ===== Serie de Muertes =====
if 'Deaths' in data_pais.columns:
    serie_muertes = data_pais['Deaths'].reset_index(drop=True)
    pronostico_serie(serie_muertes, tipo="Muertes")
else:
    st.warning("El dataset no contiene columna 'Deaths'. No se puede proyectar muertes.")


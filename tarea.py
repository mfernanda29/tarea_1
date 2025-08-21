import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Título
st.title("Pronóstico COVID-19 - Laboratorio 3.2")

# Cargar dataset
df = pd.read_csv("acti_data.csv")  # tu archivo CSV

# Lista de países únicos
paises = df['Country_Region'].unique()
pais_seleccionado = st.selectbox("Seleccione un país:", paises)

# Filtrar por país
data_pais = df[df['Country_Region'] == pais_seleccionado]

# Ordenar por fecha
if 'Date' in data_pais.columns:
    data_pais['Date'] = pd.to_datetime(data_pais['Date'])
    data_pais = data_pais.sort_values('Date')

# Serie temporal de Confirmados
serie = data_pais['Confirmed'].reset_index(drop=True)

st.subheader(f"Serie histórica de casos confirmados en {pais_seleccionado}")
fig, ax = plt.subplots()
ax.plot(serie, label="Confirmados")
ax.set_xlabel("Días")
ax.set_ylabel("Casos")
ax.legend()
st.pyplot(fig)

# Intentar modelo SARIMA
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
except Exception as e:
    st.warning(f"SARIMA falló: {e}\nUsando modelo ETS como alternativa.")
    # Modelo ETS alternativo
    modelo_ets = ExponentialSmoothing(serie, seasonal='add', seasonal_periods=7)
    resultado = modelo_ets.fit()
    pred = resultado.forecast(14)
    modelo_usado = "ETS"

# Mostrar pronóstico
st.subheader(f"Pronóstico a 14 días para {pais_seleccionado} ({modelo_usado})")
fig2, ax2 = plt.subplots()
ax2.plot(range(len(serie)), serie, label="Histórico")
ax2.plot(range(len(serie), len(serie)+14), pred, label="Pronóstico", color="red")
ax2.set_xlabel("Días")
ax2.set_ylabel("Casos")
ax2.legend()
st.pyplot(fig2)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Título
st.title("Pronóstico COVID-19 - Laboratorio 3.2")

# Cargar dataset
df = pd.read_csv("acti_data.csv")  # Asegúrate que tenga columnas: Country_Region, Date, Confirmed, Deaths

# Lista de países únicos
paises = df['Country_Region'].unique()
pais_seleccionado = st.selectbox("Seleccione un país:", paises)

# Filtrar por país y ordenar fechas
data_pais = df[df['Country_Region'] == pais_seleccionado].copy()
data_pais['Date'] = pd.to_datetime(data_pais['Date'])
data_pais = data_pais.sort_values('Date')

# Función para entrenar modelo y predecir
def pronosticar(serie, variable, pasos=14):
    try:
        modelo = SARIMAX(
            serie,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        resultado = modelo.fit(disp=False)
        forecast = resultado.get_forecast(steps=pasos)
        pred = forecast.predicted_mean
        modelo_usado = "SARIMA"
    except Exception as e:
        st.warning(f"SARIMA falló para {variable}: {e}\nUsando modelo ETS como alternativa.")
        modelo_ets = ExponentialSmoothing(serie, seasonal='add', seasonal_periods=7)
        resultado = modelo_ets.fit()
        pred = resultado.forecast(pasos)
        modelo_usado = "ETS"
    return pred, modelo_usado

# ========================
# Pronóstico de Confirmados
# ========================
st.subheader(f"Serie histórica de casos confirmados en {pais_seleccionado}")
fig, ax = plt.subplots()
ax.plot(data_pais['Date'], data_pais['Confirmed'], label="Confirmados")
ax.set_xlabel("Fecha")
ax.set_ylabel("Casos")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

pred_confirmados, modelo_c = pronosticar(data_pais['Confirmed'], "Confirmados")
fechas_futuras = pd.date_range(start=data_pais['Date'].iloc[-1] + pd.Timedelta(days=1), periods=14)

st.subheader(f"Pronóstico de casos confirmados a 14 días ({modelo_c})")
fig2, ax2 = plt.subplots()
ax2.plot(data_pais['Date'], data_pais['Confirmed'], label="Histórico")
ax2.plot(fechas_futuras, pred_confirmados, label="Pronóstico", color="red")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Casos confirmados")
ax2.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)

# ========================
# Pronóstico de Muertes
# ========================
if 'Deaths' in data_pais.columns:
    st.subheader(f"Serie histórica de muertes en {pais_seleccionado}")
    fig3, ax3 = plt.subplots()
    ax3.plot(data_pais['Date'], data_pais['Deaths'], label="Muertes", color="orange")
    ax3.set_xlabel("Fecha")
    ax3.set_ylabel("Muertes")
    ax3.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    pred_muertes, modelo_m = pronosticar(data_pais['Deaths'], "Muertes")
    
    st.subheader(f"Pronóstico de muertes a 14 días ({modelo_m})")
    fig4, ax4 = plt.subplots()
    ax4.plot(data_pais['Date'], data_pais['Deaths'], label="Histórico", color="orange")
    ax4.plot(fechas_futuras, pred_muertes, label="Pronóstico", color="red")
    ax4.set_xlabel("Fecha")
    ax4.set_ylabel("Muertes")
    ax4.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig4)



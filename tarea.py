import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Cargar los datos
df = pd.read_csv("tus_datos.csv")

# Convertir fechas
df["Last_Update"] = pd.to_datetime(df["Last_Update"])

# Selección de país
paises = df["Country_Region"].unique()
pais_seleccionado = st.selectbox("Selecciona un país:", sorted(paises))

# Filtrar datos del país
data_pais = df[df["Country_Region"] == pais_seleccionado].copy()
data_pais = data_pais.groupby("Last_Update")[["Confirmed", "Deaths"]].sum().reset_index()

# --- Serie histórica ---
st.subheader(f"Serie histórica de casos en {pais_seleccionado}")
fig, ax = plt.subplots()
ax.plot(data_pais["Last_Update"], data_pais["Confirmed"], label="Confirmados")
ax.plot(data_pais["Last_Update"], data_pais["Deaths"], label="Muertes")
ax.set_xlabel("Fecha")
ax.set_ylabel("Casos")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Pronóstico con ETS ---
st.subheader("Pronóstico a 14 días (ETS)")

for columna in ["Confirmed", "Deaths"]:
    serie = data_pais.set_index("Last_Update")[columna]

    # Modelo ETS
    modelo = ExponentialSmoothing(serie, trend="add", seasonal=None).fit()
    pronostico = modelo.forecast(14)

    fig, ax = plt.subplots()
    ax.plot(serie.index, serie, label="Histórico")
    ax.plot(pronostico.index, pronostico, label="Pronóstico", linestyle="--")
    ax.set_title(f"{columna} - {pais_seleccionado}")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- Pronóstico con SARIMA ---
st.subheader("Pronóstico a 14 días (SARIMA)")

for columna in ["Confirmed", "Deaths"]:
    serie = data_pais.set_index("Last_Update")[columna]

    # Modelo SARIMA
    modelo = SARIMAX(serie, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    pronostico = modelo.get_forecast(steps=14).predicted_mean

    fig, ax = plt.subplots()
    ax.plot(serie.index, serie, label="Histórico")
    ax.plot(pronostico.index, pronostico, label="Pronóstico", linestyle="--")
    ax.set_title(f"{columna} - {pais_seleccionado}")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)



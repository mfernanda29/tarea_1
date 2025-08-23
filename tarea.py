# app.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# =====================
# Función MAPE manual
# =====================
def mape(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    # Evita división por cero
    return (abs((y_true - y_pred) / y_true.replace(0, 1))).mean() * 100

# =====================
# 1. Título y carga de datos
# =====================
st.title("Modelado y Proyección COVID-19")

try:
    data = pd.read_csv("acti_data.csv")
except FileNotFoundError:
    st.error("El archivo acti_data.csv no se encuentra en el repositorio.")
    st.stop()

# Procesar fecha
data['Last_Update'] = pd.to_datetime(data['Last_Update'], errors='coerce')
data = data.dropna(subset=['Last_Update'])

# Agrupar por fecha
df = data.groupby('Last_Update')[['Confirmed', 'Deaths']].sum()

# Resample diario
df = df.resample('D').sum().fillna(0)

if df.empty or len(df) < 30:
    st.error("No hay suficientes datos para entrenar el modelo. Se necesitan al menos 30 días de datos.")
    st.stop()

# Mostrar datos históricos
st.subheader("Datos Históricos")
st.line_chart(df)

# =====================
# 2. Selección de variable
# =====================
variable = st.selectbox("Selecciona variable a pronosticar:", ['Confirmed', 'Deaths'])
serie = df[variable]

# =====================
# 3. División para backtesting
# =====================
if len(serie) < 30:
    st.error("Datos insuficientes para generar pronóstico.")
    st.stop()

train = serie[:-14]
test = serie[-14:]

# =====================
# 4. Modelo ETS (Holt-Winters)
# =====================
try:
    model = ExponentialSmoothing(train, trend='add', seasonal=None)
    fit = model.fit()
except Exception as e:
    st.error(f"Error al ajustar el modelo: {e}")
    st.stop()

# Pronóstico 14 días (para test)
forecast = fit.forecast(14)

# =====================
# 5. Validación
# =====================
mae = mean_absolute_error(test, forecast)
mape_val = mape(test, forecast)

st.subheader("Validación del Modelo")
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MAPE", f"{mape_val:.2f}%")

# =====================
# 6. Visualización del pronóstico vs real
# =====================
st.subheader("Pronóstico vs Real")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train.index, train, label="Entrenamiento", color="blue")
ax.plot(test.index, test, label="Real", color="green")
ax.plot(forecast.index, forecast, label="Pronóstico", color="red", linestyle="--")
ax.set_title(f"Pronóstico 14 días ({variable})")
ax.legend()
st.pyplot(fig)

# =====================
# 7. Proyección hacia adelante
# =====================
future_dates = pd.date_range(start=serie.index[-1] + pd.Timedelta(days=1), periods=14)
future_forecast = fit.forecast(14)
future_df = pd.DataFrame({f"Pronóstico {variable}": future_forecast}, index=future_dates)

st.subheader("Proyección a 14 días")
st.line_chart(future_df)
st.write(future_df)

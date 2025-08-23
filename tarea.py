# app.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# =====================
# 1. Cargar datos
# =====================
st.title("Modelado y Proyección COVID-19")

# Cargar CSV
data = pd.read_csv("acti_data.csv")

# Filtrar columnas necesarias
data['Last_Update'] = pd.to_datetime(data['Last_Update'])
df = data.groupby('Last_Update')[['Confirmed', 'Deaths']].sum().reset_index()

# Resample diario (asegurar continuidad)
df = df.set_index('Last_Update').resample('D').sum().fillna(0)

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
train = serie[:-14]
test = serie[-14:]

# =====================
# 4. Modelo ETS (Holt-Winters)
# =====================
model = ExponentialSmoothing(train, trend='add', seasonal=None)
fit = model.fit()

# Pronóstico 14 días
forecast = fit.forecast(14)

# =====================
# 5. Validación
# =====================
mae = mean_absolute_error(test, forecast)
mape = mean_absolute_percentage_error(test, forecast) * 100

st.subheader("Validación del Modelo")
st.metric("MAE", f"{mae:.2f}")
st.metric("MAPE", f"{mape:.2f}%")

# =====================
# 6. Visualización
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
future_forecast = fit.forecast(14)
future_dates = pd.date_range(start=serie.index[-1] + pd.Timedelta(days=1), periods=14)
future_df = pd.DataFrame({f"Pronóstico {variable}": future_forecast}, index=future_dates)

st.subheader("Proyección a 14 días")
st.line_chart(future_df)
st.write(future_df)

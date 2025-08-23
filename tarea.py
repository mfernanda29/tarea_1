# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Funciones auxiliares ---
def mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def simulate_history(last_value, days=60, daily_growth=0.03, noise=0.0, last_date=pd.Timestamp("2022-04-18")):
    if last_value < 0:
        last_value = 0
    values_backward = [float(last_value)]
    rng = np.random.default_rng(42)
    for _ in range(days - 1):
        g = daily_growth
        if noise > 0:
            g = max(-0.5, g + rng.normal(0, noise * daily_growth))
        prev = values_backward[-1] / (1.0 + max(g, -0.9))
        prev = max(prev, 0.0)
        values_backward.append(prev)
    values = list(reversed(values_backward))
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1])
    start_date = pd.to_datetime(last_date) - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    return pd.Series(values, index=dates)

# --- Interfaz Streamlit ---
st.title("Modelado y Proyección COVID-19")
st.caption("⚠️ Solo se dispone del reporte diario del 18/04/2022. Se simula un histórico para fines didácticos.")

# Cargar datos
DATA_PATH = "acti_data.csv"
try:
    raw = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("No se encontró 'acti_data.csv'. Súbelo al repositorio junto a app.py.")
    st.stop()

raw["Last_Update"] = pd.to_datetime(raw["Last_Update"], errors="coerce")
raw = raw.dropna(subset=["Last_Update"])
if raw.empty:
    st.error("El archivo no tiene fechas válidas en 'Last_Update'.")
    st.stop()

# Consolidar datos
last_date = raw["Last_Update"].max().normalize()
daily = (raw.loc[raw["Last_Update"].dt.normalize() == last_date,
                 ["Country_Region", "Confirmed", "Deaths"]]
            .groupby("Country_Region", as_index=False)
            .sum())

if daily.empty:
    st.error("No se encontraron registros para la fecha del archivo.")
    st.stop()

# Selección país
st.subheader("Datos del día base (18/04/2022)")
country = st.selectbox("Selecciona un país", options=sorted(daily["Country_Region"].unique()))
row = daily.loc[daily["Country_Region"] == country].iloc[0]
base_confirmed = int(row["Confirmed"])
base_deaths = int(row["Deaths"])

colA, colB = st.columns(2)
with colA:
    st.metric("Confirmados", f"{base_confirmed:,}")
with colB:
    st.metric("Muertes", f"{base_deaths:,}")

# Parámetros de simulación
st.subheader("Parámetros para generar histórico simulado")
sim_days = st.slider("Días histórico", 30, 120, 60)
growth_confirmed = st.slider("Crecimiento Confirmados", 0.0, 0.08, 0.03, 0.005)
growth_deaths = st.slider("Crecimiento Muertes", 0.0, 0.08, 0.02, 0.005)
noise_level = st.slider("Ruido", 0.0, 0.5, 0.1, 0.01)

# Generar histórico
sim_confirmed = simulate_history(base_confirmed, days=sim_days, daily_growth=growth_confirmed, noise=noise_level, last_date=last_date)
sim_deaths = simulate_history(base_deaths, days=sim_days, daily_growth=growth_deaths, noise=noise_level, last_date=last_date)
hist_df = pd.DataFrame({"Confirmed": sim_confirmed, "Deaths": sim_deaths})
st.subheader("Histórico SIMULADO")
st.line_chart(hist_df)

# -------------------------
# 3.1 Modelado y Proyección (ETS)
# -------------------------
st.markdown("### **3.2 Implementación del modelo ETS para pronóstico**")
target = st.selectbox("Variable a pronosticar", ["Confirmed", "Deaths"])
series = hist_df[target]

if len(series) < 30:
    st.error("Se requieren al menos 30 días simulados para entrenar ETS.")
    st.stop()

h = 14
train = series.iloc[:-h]
test = series.iloc[-h:]

try:
    model = ExponentialSmoothing(train, trend='add', seasonal=None)
    fit = model.fit()
    preds = fit.forecast(h)
except Exception as e:
    st.error(f"Error al ajustar ETS: {e}")
    st.stop()

# -------------------------
# 3.2 Validación del modelo (Backtesting con MAE/MAPE)
# -------------------------
st.markdown("### **3.3 Validación con Backtesting (MAE / MAPE)**")
mae_val = mae(test.values, preds.values)
mape_val = mape(test.values, preds.values)

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae_val:,.2f}")
c2.metric("MAPE", f"{mape_val:.2f}%")

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(train.index, train.values, label="Entrenamiento")
ax1.plot(test.index, test.values, label="Real (test)")
ax1.plot(preds.index, preds.values, "--", label="Pronóstico (test)")
ax1.set_title(f"Backtesting ETS - {target}")
ax1.legend()
st.pyplot(fig1)

# Proyección a futuro
st.subheader("Proyección a 14 días hacia adelante")
fit_full = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
future_fc = fit_full.forecast(h)
future_df = pd.DataFrame({f"Forecast_{target}": future_fc})
st.line_chart(future_df)
st.dataframe(future_df.style.format("{:,.0f}"))

st.info("⚠️ Este análisis es ilustrativo, basado en un histórico simulado debido a que solo se cuenta con un día real.")


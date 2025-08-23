# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # Evitar división por cero reemplazando 0 por 1 (convención conservadora)
    denom = np.where(y_true == 0, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def simulate_history(last_value, days=60, daily_growth=0.03, noise=0.0, last_date=pd.Timestamp("2022-04-18")):
    """
    Genera una serie acumulada "creíble" hacia atrás desde un valor final (last_value).
    Trabajamos hacia atrás dividiendo por (1+g), y luego invertimos para ir de pasado a presente.
    noise: nivel de ruido aleatorio (0..0.2 aprox). 0 significa sin ruido.
    """
    if last_value < 0:
        last_value = 0

    # Construir valores del presente hacia el pasado
    values_backward = [float(last_value)]
    rng = np.random.default_rng(42)  # semilla reproducible
    for _ in range(days - 1):
        g = daily_growth
        if noise > 0:
            # ruido pequeño alrededor del crecimiento
            g = max(-0.5, g + rng.normal(0, noise * daily_growth))
        prev = values_backward[-1] / (1.0 + max(g, -0.9))  # evitar dividir por ~0
        prev = max(prev, 0.0)
        values_backward.append(prev)

    # Invertir para tener cronología del pasado al presente
    values = list(reversed(values_backward))

    # Suavizar/forzar no-decreciente (acumulados)
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1])

    # Índice de fechas
    start_date = pd.to_datetime(last_date) - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    return pd.Series(values, index=dates)
st.title("Modelado y Proyección COVID-19")
st.caption("⚠️ Solo se dispone del reporte diario del 18/04/2022. "
           "Para poder entrenar un modelo de series temporales, se simula un histórico a partir de ese dato final. "
           "Resultados con fines didácticos.")
DATA_PATH = "acti_data.csv"   # Debe estar en el repo
try:
    raw = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("No se encontró 'acti_data.csv' en el repositorio. Súbelo a tu GitHub junto al app.py.")
    st.stop()

# Limpieza básica
raw["Last_Update"] = pd.to_datetime(raw["Last_Update"], errors="coerce")
raw = raw.dropna(subset=["Last_Update"])
if raw.empty:
    st.error("El archivo se cargó pero no tiene fechas válidas en 'Last_Update'.")
    st.stop()

# Consolidar por país (un solo día)
cols_expected = ["Country_Region", "Confirmed", "Deaths"]
missing = [c for c in cols_expected if c not in raw.columns]
if missing:
    st.error(f"Faltan columnas en el CSV: {missing}")
    st.stop()

# Tomar la fecha más reciente disponible (debería ser 2022-04-18)
last_date = raw["Last_Update"].max().normalize()
daily = (raw.loc[raw["Last_Update"].dt.normalize() == last_date,
                 ["Country_Region", "Confirmed", "Deaths"]]
            .groupby("Country_Region", as_index=False)
            .sum())

if daily.empty:
    st.error("No se encontraron registros para la fecha del archivo.")
    st.stop()

# UI: selección de país
st.subheader("Datos del día base (18/04/2022)")
country = st.selectbox("Selecciona un país", options=sorted(daily["Country_Region"].unique()))

row = daily.loc[daily["Country_Region"] == country].iloc[0]
base_confirmed = int(row["Confirmed"])
base_deaths = int(row["Deaths"])

colA, colB = st.columns(2)
with colA:
    st.metric("Confirmados (día base)", f"{base_confirmed:,}")
with colB:
    st.metric("Muertes (día base)", f"{base_deaths:,}")

st.write(f"*Fecha base:* **{last_date.date()}** — Fuente: reporte diario JHU (1 día).")

# -------------------------
# 2) Parámetros de simulación
# -------------------------
st.subheader("Parámetros de simulación del histórico (para poder entrenar el modelo)")
sim_days = st.slider("Días de histórico simulado", 30, 120, 60, help="Mínimo 30 para entrenar ETS.")
growth_confirmed = st.slider("Crecimiento diario (Confirmados)", 0.0, 0.08, 0.03, 0.005,
                             help="Proporción aproximada de crecimiento acumulado diario.")
growth_deaths = st.slider("Crecimiento diario (Muertes)", 0.0, 0.08, 0.02, 0.005)
noise_level = st.slider("Ruido relativo (0 = sin ruido)", 0.0, 0.5, 0.1, 0.01,
                        help="Pequeñas variaciones alrededor del crecimiento.")

# -------------------------
# 3) Simular series acumuladas
# -------------------------
sim_confirmed = simulate_history(base_confirmed, days=sim_days,
                                 daily_growth=growth_confirmed, noise=noise_level,
                                 last_date=last_date)
sim_deaths = simulate_history(base_deaths, days=sim_days,
                              daily_growth=growth_deaths, noise=noise_level,
                              last_date=last_date)

hist_df = pd.DataFrame({
    "Confirmed": sim_confirmed,
    "Deaths": sim_deaths
})

st.subheader("Histórico (SIMULADO) acumulado")
st.line_chart(hist_df)

# -------------------------
# 4) Entrenar ETS y Backtesting
# -------------------------
st.subheader("Modelado y validación (Backtesting)")
target = st.selectbox("Variable a pronosticar", ["Confirmed", "Deaths"])
series = hist_df[target]

if len(series) < 30:
    st.error("Se requieren al menos 30 días simulados para entrenar ETS.")
    st.stop()

# División train/test (últimos 14 días para validación)
h = 14
train = series.iloc[:-h]
test = series.iloc[-h:]

try:
    # ETS simple con tendencia aditiva (sin estacionalidad)
    model = ExponentialSmoothing(train, trend='add', seasonal=None)
    fit = model.fit()
    preds = fit.forecast(h)
except Exception as e:
    st.error(f"Error al ajustar ETS: {e}")
    st.stop()

# Métricas
mae_val = mae(test.values, preds.values)
mape_val = mape(test.values, preds.values)

c1, c2 = st.columns(2)
c1.metric("MAE (validación)", f"{mae_val:,.2f}")
c2.metric("MAPE (validación)", f"{mape_val:.2f}%")

# Gráfico Pronóstico vs Real (backtest)
fig1, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(train.index, train.values, label="Entrenamiento")
ax1.plot(test.index, test.values, label="Real (test)")
ax1.plot(preds.index, preds.values, "--", label="Pronóstico (test)")
ax1.set_title(f"Backtesting ETS - {target} (h=14)")
ax1.legend()
st.pyplot(fig1)

# -------------------------
# 5) Proyección 14 días hacia adelante
# -------------------------
st.subheader("Proyección a 14 días hacia adelante")
try:
    fit_full = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    future_fc = fit_full.forecast(h)
except Exception as e:
    st.error(f"Error al proyectar: {e}")
    st.stop()

# Fechas futuras
future_df = pd.DataFrame({f"Forecast_{target}": future_fc})
st.line_chart(future_df)

st.dataframe(future_df.style.format("{:,.0f}"))

st.info(
    "🔔 **Importante**: Debido a que solo se cuenta con un día de datos reales, "
    "el histórico utilizado para entrenar el modelo es SIMULADO. "
    "Las proyecciones y la validación (MAE/MAPE) son ilustrativas para demostrar "
    "el proceso de modelado (ETS), subdivisión train/test y cálculo de métricas."
)

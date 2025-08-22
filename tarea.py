import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ===== Cargar datos =====
df = pd.read_csv("acti_data.csv")  # tu archivo del 18 de abril de 2022

# ===== Seleccionar país =====
pais_seleccionado = "Peru"  # Cambia por el país que quieras
data_pais = df[df['Country_Region'] == pais_seleccionado]

# ===== Ordenar por fecha =====
data_pais['Date'] = pd.to_datetime(data_pais['Date'])
data_pais = data_pais.sort_values('Date')

# ===== Función de pronóstico =====
def pronostico(serie, tipo="Casos"):
    print(f"\nSerie histórica de {tipo}:")
    print(serie.tail())

    # Intentar SARIMA
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
    
    # Mostrar pronóstico
    print(f"\nPronóstico a 14 días ({modelo_usado}):")
    print(pred)

    # Gráfico
    plt.figure(figsize=(10,5))
    plt.plot(range(len(serie)), serie, label="Histórico")
    plt.plot(range(len(serie), len(serie)+14), pred, label="Pronóstico", color="red", linestyle="--")
    plt.title(f"{tipo} - {pais_seleccionado}")
    plt.xlabel("Días")
    plt.ylabel(tipo)
    plt.legend()
    plt.show()

# ===== Pronóstico de Casos =====
serie_casos = data_pais['Confirmed'].reset_index(drop=True)
pronostico(serie_casos, tipo="Casos")

# ===== Pronóstico de Muertes =====
if 'Deaths' in data_pais.columns:
    serie_muertes = data_pais['Deaths'].reset_index(drop=True)
    pronostico(serie_muertes, tipo="Muertes")
else:
    print("No hay columna 'Deaths' en el dataset.")

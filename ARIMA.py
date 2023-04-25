import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Download VIX data
vix_data = yf.download('SPY', start='2007-01-01', end='2023-02-31')

# Resample VIX data to monthly frequency
vix_monthly = vix_data['Close'].resample('M').last()

# Fit ARIMA model
arima_model = ARIMA(vix_monthly, order=(1, 1, 1))
arima_results = arima_model.fit()

# Forecast the next month's VIX value
forecast_horizon = 1
forecast, stderr, conf_int = arima_results.forecast(steps=forecast_horizon, alpha=0.05)

# Print the forecasted VIX values
print("Forecasted VIX values for the next month:")
print(forecast)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Define the symbol we aim to predict
predicted_symbol = '^VIX'

# Define list of symbols to use as predictors
predictor_symbols = ['^VVIX', 'SPY', 'XLP', 'PKW', 'VYM']

# Define dictionary of relative symbols to create
relative_dict = {'SPY': 'XLP', 'PKW': 'VYM'}

# Download data for training
train_data_list = []
for symbol in predictor_symbols + [predicted_symbol]:
    data = yf.download(symbol, start="2007-01-01", end="2019-12-31", interval="1d")
    train_data_list.append(data['Close'].rename(symbol))

# Merge data into a single DataFrame and rename columns
train_data = pd.concat(train_data_list, axis=1)
train_data.columns = predictor_symbols + [predicted_symbol]

# Calculate relative symbols
for symbol, relative_symbol in relative_dict.items():
    train_data[symbol + '/' + relative_symbol] = train_data[symbol] / train_data[relative_symbol]

# Remove any missing or infinite values
train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

# Fit Gradient Boosting Regression model
model = GradientBoostingRegressor(random_state=42).fit(train_data[predictor_symbols], train_data[predicted_symbol])

# Download data for testing
test_data_list = []
for symbol in predictor_symbols + [predicted_symbol]:
    data = yf.download(symbol, start="2020-01-01", end="2023-03-10", interval="1d")
    test_data_list.append(data['Close'].rename(symbol))

# Merge data into a single DataFrame and rename columns
test_data = pd.concat(test_data_list, axis=1)
test_data.columns = predictor_symbols + [predicted_symbol]

# Calculate relative symbols
for symbol, relative_symbol in relative_dict.items():
    test_data[symbol + '/' + relative_symbol] = test_data[symbol] / test_data[relative_symbol]

# Remove any missing or infinite values
test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna()

# Initialize an empty list to store forecasted VIX values
forecasted_vix = []

# Use a sliding window approach to predict VIX values for the next 5 days
for i in range(len(test_data) - 5):
    # Get the 5-day window of data
    window = test_data.iloc[i:i + 5]

    # Use the model to predict the VIX value for the next day
    prediction = model.predict(window[predictor_symbols].iloc[-1].values.reshape(1, -1))[0]

    # Append the prediction to the list of forecasted VIX values
    forecasted_vix.append(prediction)

# Add the forecasted VIX values to the test_data DataFrame
test_data['forecasted_VIX_1d'] = model.predict(test_data[predictor_symbols])
# Use the forecasted VIX for the next 4 days to predict the VIX movement

test_data['forecasted_VIX_2d'] = model.predict(test_data[predictor_symbols].shift(-1))
test_data['forecasted_VIX_3d'] = model.predict(test_data[predictor_symbols].shift(-2))
test_data['forecasted_VIX_4d'] = model.predict(test_data[predictor_symbols].shift(-3))
test_data['forecasted_VIX_5d'] = model.predict(test_data[predictor_symbols].shift(-4))
# Save forecasted VIX and actual VIX to CSV

test_data[['forecasted_VIX_1d', 'forecasted_VIX_2d', 'forecasted_VIX_3d', 'forecasted_VIX_4d', 'forecasted_VIX_5d', predicted_symbol]].to_csv('vix_forecast.csv')
# Calculate correlation coefficient

corr_coef_1d = np.corrcoef(test_data[predicted_symbol], test_data['forecasted_VIX_1d'])[0, 1]
corr_coef_2d = np.corrcoef(test_data[predicted_symbol].shift(-1).dropna(), test_data['forecasted_VIX_2d'].shift(1).dropna())[0, 1]
corr_coef_3d = np.corrcoef(test_data[predicted_symbol].shift(-2).dropna(), test_data['forecasted_VIX_3d'].shift(2).dropna())[0, 1]
corr_coef_4d = np.corrcoef(test_data[predicted_symbol].shift(-3).dropna(), test_data['forecasted_VIX_4d'].shift(3).dropna())[0, 1]
corr_coef_5d = np.corrcoef(test_data[predicted_symbol].shift(-4).dropna(), test_data['forecasted_VIX_5d'].shift(4).dropna())[0, 1]
# Print correlation coefficients

print('Correlation Coefficient 1-Day Forecast: {:.2f}'.format(corr_coef_1d))
print('Correlation Coefficient 2-Day Forecast: {:.2f}'.format(corr_coef_2d))
print('Correlation Coefficient 3-Day Forecast: {:.2f}'.format(corr_coef_3d))
print('Correlation Coefficient 4-Day Forecast: {:.2f}'.format(corr_coef_4d))
print('Correlation Coefficient 5-Day Forecast: {:.2f}'.format(corr_coef_5d))


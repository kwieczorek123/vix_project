import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

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

# Fit SVM model
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model.fit(train_data[predictor_symbols], train_data[predicted_symbol])

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

# Predict using the model for test data
test_data['forecasted_VIX'] = model.predict(test_data[predictor_symbols])

# Calculate correlation coefficient
corr_coef = np.corrcoef(test_data[predicted_symbol], test_data['forecasted_VIX'])[0, 1]
print('Correlation Coefficient: {:.2f}'.format(corr_coef))

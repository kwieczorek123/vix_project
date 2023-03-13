import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Define the symbol we aim to predict
predicted_symbol = '^VIX'

# Define list of symbols to use as predictors
predictor_symbols = ['^VVIX', 'SPY', 'XLP', 'PKW', 'VYM']

# Define dictionary of relative symbols to create
relative_dict = {'SPY': 'XLP', 'PKW': 'VYM'}

# Download data for symbols
data_list = []
for symbol in predictor_symbols + [predicted_symbol]:
    data = yf.download(symbol, start="2010-01-01", end="2023-02-28", interval="1d")
    data_list.append(data['Close'].rename(symbol))

# Merge data into a single DataFrame
data = pd.concat(data_list, axis=1)

# Calculate relative symbols
for symbol, relative_symbol in relative_dict.items():
    data[symbol+'/'+relative_symbol] = data[symbol] / data[relative_symbol]

# Remove any missing or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Separate the VIX data
vix_data = data.pop(predicted_symbol)

# Initialize arrays to store performance metrics
r2_scores = []
mse_scores = []

# Loop SVM model and evaluation for 10 iterations
for i in range(10):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, vix_data, test_size=0.2)

    # Fit SVM model
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X_train, y_train)

    # Predict using the model
    predictions = model.predict(X_test)

    # Evaluate model performance
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Append performance metrics to arrays
    r2_scores.append(r2)
    mse_scores.append(mse)

# Print average performance metrics
print("Average R-squared: ", np.mean(r2_scores))
print("Average Mean squared error: ", np.mean(mse_scores))

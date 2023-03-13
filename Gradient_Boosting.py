import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

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

# Initialize variables to store evaluation results
r2_scores = []
mse_scores = []

# Fit and evaluate model 100 times
for i in range(100):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[predictor_symbols], data[predicted_symbol], test_size=0.2, random_state=i)

    # Fit Gradient Boosting Regression model
    model = GradientBoostingRegressor(random_state=i).fit(X_train, y_train)

    # Predict using the model
    predictions = model.predict(X_test)

    # Evaluate model performance
    r2_scores.append(r2_score(y_test, predictions))
    mse_scores.append(mean_squared_error(y_test, predictions))

# Print average evaluation results
print("Average R-squared: ", np.mean(r2_scores))
print("Average Mean Squared Error: ", np.mean(mse_scores))

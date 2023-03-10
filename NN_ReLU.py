import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf

# Define the symbol we aim to predict
predicted_symbol = '^VIX'

# Define list of symbols to use as predictors
predictor_symbols = ['^VVIX', 'SPY', 'XLP', 'GLD', 'USO', 'PKW', 'VYM']

# Define dictionary of relative symbols to create
relative_dict = {'SPY': 'XLP', 'GLD': 'USO', 'PKW': 'VYM'}

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

# Split into training and testing sets
train_data = data.iloc[:int(0.8 * len(data)), :]
test_data = data.iloc[int(0.8 * len(data)):, :]

# Initialize variables to store evaluation results
r2_scores = []
mse_scores = []

# Fit and evaluate model 100 times
for i in range(100):
    # Fit Random Forest Regression model
    X_train = train_data[predictor_symbols]
    y_train = train_data[predicted_symbol]
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=i).fit(X_train, y_train)

    # Predict using the model
    X_test = test_data[predictor_symbols]
    y_test = test_data[predicted_symbol]
    predictions = model.predict(X_test)

    # Evaluate model performance
    r2_scores.append(r2_score(y_test, predictions))
    mse_scores.append(mean_squared_error(y_test, predictions))

# Print average evaluation results
print("Average R-squared: ", np.mean(r2_scores))
print("Average Mean Squared Error: ", np.mean(mse_scores))

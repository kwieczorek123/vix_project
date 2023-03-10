import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

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

# Fit linear regression model
X = sm.add_constant(train_data[[symbol for symbol in predictor_symbols]])
model = sm.OLS(train_data[predicted_symbol], X).fit()

# Predict using the model
X_test = sm.add_constant(test_data[[symbol for symbol in predictor_symbols]])
predictions = model.predict(X_test)

# Save summary to CSV file
with open('linear_regression_summary.csv', 'w') as f:
    f.write(model.summary().as_csv())

# Save R-squared and mean squared error to CSV file
results_df = pd.DataFrame({
    'R-squared': [model.rsquared],
    'Mean squared error': [model.mse_model]
})
results_df.to_csv('linear_regression_results.csv', index=False)

# Print summary of the model
print(model.summary())

# Print R-squared and mean squared error
print("R-squared: ", model.rsquared)
print("Mean squared error: ", model.mse_model)

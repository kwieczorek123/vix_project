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

# Predict using the model for test data
test_data['forecasted_VIX'] = model.predict(test_data[predictor_symbols])

# Save forecasted VIX and actual VIX to CSV
test_data[['forecasted_VIX', predicted_symbol]].to_csv('vix_forecast.csv')

# Calculate correlation coefficient
corr_coef = np.corrcoef(test_data[predicted_symbol], test_data['forecasted_VIX'])[0, 1]

# Calculate the percentage of time that the forecasted VIX correctly predicts an upward swing in the actual VIX
upward_actual = np.diff(test_data[predicted_symbol]) > 0
upward_forecast = np.diff(test_data['forecasted_VIX']) > 0
correct_predictions = upward_actual[:-1] == upward_forecast[:-1]
percentage_upward_correct = np.mean(correct_predictions) * 100
print('Percentage of correct upward predictions: {:.2f}%'.format(percentage_upward_correct))

# Calculate the percentage of time that the forecasted VIX correctly predicts a downward move in the actual VIX
downward_actual = np.diff(test_data[predicted_symbol]) < 0
downward_forecast = np.diff(test_data['forecasted_VIX']) < 0
correct_predictions = downward_actual[:-1] == downward_forecast[:-1]
percentage_downward_correct = np.mean(correct_predictions) * 100
print('Percentage of correct downward predictions: {:.2f}%'.format(percentage_downward_correct))

roll_period = 21

# Calculate percentage change of VIX
test_data['VIX_change'] = test_data[predicted_symbol].pct_change()

# Calculate percentage change of forecasted VIX
test_data['forecasted_VIX_change'] = test_data['forecasted_VIX'].pct_change()

# Create a column to indicate upward moves of at least 50% within a maximum of 21 time periods (rows)
test_data['upward_move'] = (test_data[predicted_symbol].rolling(21).max() / test_data[predicted_symbol]) <= 0.5

# Create a column to indicate if the forecast correctly predicted the upward move
test_data['upward_correct'] = np.where(
    (test_data['VIX_change'] > 0.5) & (test_data['forecasted_VIX_change'] > 0.25 * test_data['VIX_change']), 1, 0)

# Calculate the percentage of correct upward predictions
upward_move_sum = test_data['upward_move'].sum()
if upward_move_sum > 0:
    percentage_major_upward_correct = 100 * test_data['upward_correct'].sum() / upward_move_sum
else:
    percentage_major_upward_correct = 0

# Print the percentage of correct upward predictions
print('Percentage of Correct Major Upward Predictions: {:.2f}%'.format(percentage_major_upward_correct))

# Plot forecasted VIX and actual VIX on one chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test_data.index, test_data[predicted_symbol], label='Actual VIX')
ax.plot(test_data.index, test_data['forecasted_VIX'], label='Forecasted VIX')
ax.set_xlabel('Date')
ax.set_ylabel('VIX')
ax.set_title('Forecasted VIX vs Actual VIX\nCorrelation Coefficient: {:.2f}'.format(corr_coef))
ax.legend()

# Add correlation coefficient to chart as text annotation
plt.text(test_data.index[0], np.max(test_data[predicted_symbol]) * 0.95,
         'Correlation Coefficient: {:.2f}'.format(corr_coef), fontsize=8)

# Add percentage of correct upward and downward predictions to chart as text annotations
plt.text(test_data.index[0], np.max(test_data[predicted_symbol]) * 0.85,
         'Upward Correct: {:.2f}%'.format(percentage_upward_correct), fontsize=6)
plt.text(test_data.index[0], np.max(test_data[predicted_symbol]) * 0.8,
         'Downward Correct: {:.2f}%'.format(percentage_downward_correct), fontsize=6)
plt.savefig('gradient_VIX.png')
plt.show()

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def get_monthly_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data = data['Close'].resample('M').last()
    return data.rename(symbol)


# Define the symbol we aim to predict
predicted_symbol = '^VIX'

# Define list of symbols to use as predictors
predictor_symbols = ['^VVIX', 'SPY', 'XLP', 'PKW', 'VYM']

# Define dictionary of relative symbols to create
relative_dict = {'SPY': 'XLP', 'PKW': 'VYM'}

# Download data for training
train_data_list = []
for symbol in predictor_symbols + [predicted_symbol]:
    data = get_monthly_data(symbol, start="2007-01-01", end="2021-12-31")
    train_data_list.append(data)

# Merge data into a single DataFrame and rename columns
train_data = pd.concat(train_data_list, axis=1)
train_data.columns = predictor_symbols + [predicted_symbol]

# Calculate relative symbols
for symbol, relative_symbol in relative_dict.items():
    train_data[symbol + '/' + relative_symbol] = train_data[symbol] / train_data[relative_symbol]

# Remove any missing or infinite values
train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

# MODIFICATION: Calculate percentage change for VIX in the train dataset
train_data['VIX_pct_change'] = train_data[predicted_symbol].pct_change()

# Remove any missing or infinite values after calculating percentage change
train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

# Fit Gradient Boosting Regression model
model = GradientBoostingRegressor(random_state=42).fit(train_data[predictor_symbols], train_data[predicted_symbol])

# Download data for testing
test_data_list = []
for symbol in predictor_symbols + [predicted_symbol]:
    data = get_monthly_data(symbol, start="2022-01-01", end="2023-03-31")
    test_data_list.append(data)

# Merge data into a single DataFrame and rename columns
test_data = pd.concat(test_data_list, axis=1)
test_data.columns = predictor_symbols + [predicted_symbol]

# Calculate relative symbols
for symbol, relative_symbol in relative_dict.items():
    test_data[symbol + '/' + relative_symbol] = test_data[symbol] / test_data[relative_symbol]

# Remove any missing or infinite values
test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna()

# MODIFICATION: Calculate percentage change for VIX in the test dataset
test_data['VIX_pct_change'] = test_data[predicted_symbol].pct_change()

# Remove any missing or infinite values after calculating percentage change
test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna()

# Predict using the model for test data
test_data['forecasted_VIX_pct_change'] = model.predict(test_data[predictor_symbols])

# MODIFICATION: Calculate the direction of the VIX move (1 for up, 0 for down)
test_data['VIX_move_direction'] = np.where(test_data['VIX_pct_change'] > 0, 1, 0)

test_data['forecasted_VIX_move_direction'] = np.where(test_data['forecasted_VIX_pct_change'] > 0, 1, 0)

# Calculate the percentage of time that the forecasted VIX correctly predicts an upward swing in the actual VIX
upward_actual = test_data['VIX_move_direction'] == 1
upward_forecast = test_data['forecasted_VIX_move_direction'] == 1
correct_predictions = upward_actual == upward_forecast
percentage_upward_correct = np.mean(correct_predictions) * 100
print('Percentage of correct upward predictions: {:.2f}%'.format(percentage_upward_correct))

# Calculate the percentage of time that the forecasted VIX correctly predicts a downward move in the actual VIX
downward_actual = test_data['VIX_move_direction'] == 0
downward_forecast = test_data['forecasted_VIX_move_direction'] == 0
correct_predictions = downward_actual == downward_forecast
percentage_downward_correct = np.mean(correct_predictions) * 100
print('Percentage of correct downward predictions: {:.2f}%'.format(percentage_downward_correct))

# Calculate correlation coefficient
corr_coef = np.corrcoef(test_data['VIX_pct_change'], test_data['forecasted_VIX_pct_change'])[0, 1]

# Plot forecasted VIX percentage change and actual VIX percentage change on one chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test_data.index, test_data['VIX_pct_change'], label='Actual VIX Percentage Change')
ax.plot(test_data.index, test_data['forecasted_VIX_pct_change'], label='Forecasted VIX Percentage Change')
ax.set_xlabel('Date')
ax.set_ylabel('VIX Percentage Change')
ax.set_title('Forecasted VIX Percentage Change vs Actual VIX Percentage Change\nCorrelation Coefficient: {:.2f}'.format(corr_coef))
ax.legend()

# Add correlation coefficient to chart as text annotation
plt.text(test_data.index[0], np.max(test_data['VIX_pct_change']) * 0.95,
         'Correlation Coefficient: {:.2f}'.format(corr_coef), fontsize=8)

# Add percentage of correct upward and downward predictions to chart as text annotations
plt.text(test_data.index[0], np.max(test_data['VIX_pct_change']) * 0.85,
         'Upward Correct: {:.2f}%'.format(percentage_upward_correct), fontsize=6)
plt.text(test_data.index[0], np.max(test_data['VIX_pct_change']) * 0.8,
         'Downward Correct: {:.2f}%'.format(percentage_downward_correct), fontsize=6)
plt.savefig('gradient_VIX_pct_change.png')
plt.show()

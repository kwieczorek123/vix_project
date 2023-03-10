import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf
import csv

# Define the symbol we aim to predict
predicted_symbol = '^VIX'

# Define list of symbols to use as predictors
symbols_list = ['^VVIX', 'SPY', 'XLP', 'GLD', 'USO']

# Define dictionary of relative symbols to create
relative_dict = {'SPY': 'XLP', 'GLD': 'USO'}

# Download data for symbols
data_list = []
for symbol in predictor_symbols + [predicted_symbol]:
    data = yf.download(symbol, start="2010-01-01", end="2023-02-28", interval="1d")
    data_list.append(data['Close'].rename(symbol))

# Merge data into a single DataFrame
data = pd.concat(data_list, axis=1)

# Assert that the number of predictor variables matches the number of columns in the training data
assert len(predictor_symbols) + 1 == len(data.columns), "Number of predictor variables does not match number of columns in training data"

# Calculate relative symbols
for symbol, relative_symbol in relative_dict.items():
    data[symbol+'/'+relative_symbol] = data[symbol] / data[relative_symbol]

# Remove any missing or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# set max lag and run Granger causality test
maxlag = 30
test_results = grangercausalitytests(data, maxlag=maxlag, verbose=False)

# create a list to hold the results
results = []

# loop through the test results and save the lag, p-value, and reject null hypothesis to the results list
for lag in range(1, maxlag+1):
    p_value = test_results[lag][0]['ssr_ftest'][1]
    reject_null = p_value < 0.05
    results.append((lag, p_value, reject_null))

# create a dataframe from the results list and save to CSV
results_df = pd.DataFrame(results, columns=['Lag', 'P-Value', 'Reject_Null_Hypothesis'])
results_df.to_csv('granger_causality_results.csv', index=False)

alpha = 0.05

with open('granger_results.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Lag', 'P-Value', 'Reject Null'])

    for lag in range(1, maxlag + 1):
        p_val = test_results[lag][0]['ssr_ftest'][1]
        reject_null = p_val < alpha
        results_writer.writerow([lag, p_val, reject_null])
        print(f"Lag {lag}: P-Value = {p_val}, Reject Null = {reject_null}")

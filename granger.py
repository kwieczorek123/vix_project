import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf
import csv

# download and clean the data
data_vix = yf.download("^VIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_vvix = yf.download("^VVIX", start="2010-01-01", end="2023-02-28", interval="1d")
close_df = pd.concat([data_vix['Close'], data_vvix['Close']], axis=1)
close_df.columns = ['VIX', 'VVIX']
close_df.dropna(inplace=True)

# set max lag and run Granger causality test
maxlag = 30
test_results = grangercausalitytests(close_df, maxlag=maxlag, verbose=False)

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

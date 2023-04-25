import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np
import scipy.stats

# Define symbols and frequency
symbols = ['GC=F', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
freq = ['D', 'W', 'M']

# Define rolling window size
roll = 20

for symbol in symbols:
    # Retrieve data from Yahoo Finance
    data = yf.download(symbol, start='2000-01-01', end='2023-04-25')

    # Calculate log returns
    log_returns = np.log(data['Adj Close']).diff().dropna()

    # Compute GARCH(1,1) volatility based on log returns
    garch_model_lr = arch_model(log_returns, p=1, q=1)
    garch_res_lr = garch_model_lr.fit(update_freq=5)
    garch_vol_lr = garch_res_lr.conditional_volatility

    # Loop over frequencies
    for f in freq:
        # Resample data to specified frequency
        data_f = data['Adj Close'].resample(f).last().dropna()

        # Compute absolute log returns
        abs_log_returns = abs(np.log(data_f).diff().dropna())

        # Compute GARCH(1,1) volatility based on absolute log returns
        garch_model_alr = arch_model(abs_log_returns, p=1, q=1)
        garch_res_alr = garch_model_alr.fit(update_freq=5)
        garch_vol_alr = garch_res_alr.conditional_volatility

        # Combine into DataFrame
        df = pd.DataFrame({
            'Price': data_f,
            'Absolute Log Return': abs_log_returns,
            'GARCH(1,1) Log Returns': garch_vol_lr
        })

        # Save to CSV
        filename = f'{symbol}_{f}.csv'
        df.to_csv(filename)
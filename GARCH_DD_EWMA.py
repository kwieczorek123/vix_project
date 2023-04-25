import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np
import scipy.stats


# DD-EWMA and rho_cal functions
def rho_cal(X):
    rho_hat = scipy.stats.pearsonr(X - np.mean(X), np.sign(
        X - np.mean(X)))  # rho_hat[0]: Pearson correlation , rho_hat[1]: two-tailed p-value
    return rho_hat[0]


def DD_volatility(y, cut_t, alpha):
    t = len(y)
    rho = rho_cal(y)  # calculate sample sign correlation
    vol = abs(y - np.mean(y)) / rho  # calculate observed volatility
    MSE_alpha = np.zeros(len(alpha))
    sn = np.zeros(len(alpha))  # volatility
    for a in range(len(alpha)):
        s = np.mean(vol[0:cut_t])  # initial smoothed statistic
        error = np.zeros(t)
        for i in range(t):
            error[i] = vol[i] - s
            s = alpha[a] * vol[i] + (1 - alpha[a]) * s
        MSE_alpha[a] = np.mean((error[(len(error) - cut_t):(len(error))]) ** 2)  # forecast error sum of squares (FESS)
        sn[a] = s
    vol_forecast = sn[[i for i, j in enumerate(MSE_alpha) if j == min(MSE_alpha)]]  # which min
    RMSE = np.sqrt(min(MSE_alpha))
    return vol_forecast, RMSE


# Define symbols and frequency
symbols = ['GC=F', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
freq = ['D', 'W', 'M']

# Set the cut-off time and list of alpha values
cut_t = 30
alpha = [0.01, 0.02, 0.03, 0.04, 0.05]

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

        # Calculate the forecasted volatility and RMSE for each date using the DD_volatility() function
        forecasted_volatility = [np.nan] * cut_t
        ddewma_rmse = [np.nan] * cut_t
        for i in range(len(abs_log_returns) - cut_t):
            log_returns_window = log_returns[i:cut_t + i]
            vol_forecast, RMSE = DD_volatility(log_returns_window, cut_t, alpha)
            forecasted_volatility.append(vol_forecast[0])
            ddewma_rmse.append(RMSE)

        # Combine into DataFrame
        df = pd.DataFrame({
            'Price': data_f,
            'Absolute Log Return': abs_log_returns,
            'GARCH(1,1) Log Returns': garch_vol_lr.reindex(abs_log_returns.index),
            'GARCH(1,1) Absolute Log Returns': garch_vol_alr,
            'DD-EWMA Forecasted Volatility': pd.Series(forecasted_volatility, index=abs_log_returns.index),
            'DD-EWMA RMSE': pd.Series(ddewma_rmse, index=abs_log_returns.index)
        })

        # Save to CSV
        filename = f'{symbol}_{f}.csv'
        df.to_csv(filename)

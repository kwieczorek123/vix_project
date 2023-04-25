import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


# DD-EWMA and rho_cal functions
def rho_cal(X):
    if len(X) < 2:
        return 0
    rho_hat = scipy.stats.pearsonr(X - np.mean(X), np.sign(
        X - np.mean(X)))  # rho_hat[0]: Pearson correlation , rho_hat[1]: two-tailed p-value
    return rho_hat[0]


def GARCH_EWMA_Volatility(abs_log_returns_f, log_returns_f, alpha):
    # Compute GARCH(1,1) volatility based on absolute log returns for the specified frequency
    garch_model_alr_f = arch_model(abs_log_returns_f, p=1, q=1)
    garch_res_alr_f = garch_model_alr_f.fit(update_freq=5)
    garch_vol_alr_f = garch_res_alr_f.conditional_volatility

    # Compute EWMA volatility for the specified frequency and alpha
    ewma_vol_f = np.zeros(len(log_returns_f))
    ewma_vol_f[0] = log_returns_f[0] ** 2
    for i in range(1, len(log_returns_f)):
        ewma_vol_f[i] = alpha * log_returns_f[i - 1] ** 2 + (1 - alpha) * ewma_vol_f[i - 1]

    ewma_vol_f = np.sqrt(ewma_vol_f)

    return garch_vol_alr_f, ewma_vol_f


def DD_volatility(y, cut_t, alpha):
    abs_log_returns_f = pd.Series(np.abs(y))
    log_returns_f = y

    # Slice the data using cut_t
    abs_log_returns_f = abs_log_returns_f.iloc[:cut_t]
    log_returns_f = log_returns_f.iloc[:cut_t]

    garch_vol_alr_f, ewma_vol_f = GARCH_EWMA_Volatility(abs_log_returns_f, log_returns_f, alpha)
    mse_alpha = mean_squared_error(garch_vol_alr_f, ewma_vol_f)
    return mse_alpha, garch_vol_alr_f



def DD_volatility_mse(y, cut_t, alpha):
    mse_alpha, _ = DD_volatility(y, cut_t, alpha)
    return mse_alpha


# Define symbols and frequency
symbols = ['GC=F', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
freq = ['D', 'W', 'M']

# Set the cut-off time
cut_t = 30

# Create an empty DataFrame to store the correlation coefficients for each symbol and frequency
corr_df = pd.DataFrame(columns=['Symbol', 'Frequency', 'Abs_Log_Return_vs_GARCH_Log_Returns',
                                'Abs Log Return, GARCH Abs Log Returns', 'Abs_Log_Return_vs_DD_EWMA'])

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

        # Compute log returns for the specified frequency
        log_returns_f = np.log(data_f).diff().dropna()

        # Compute absolute log returns for the specified frequency
        abs_log_returns_f = abs(log_returns_f)

        # Compute GARCH(1,1) volatility based on log returns for the specified frequency
        garch_model_lr_f = arch_model(log_returns_f, p=1, q=1)
        garch_res_lr_f = garch_model_lr_f.fit(update_freq=5)
        garch_vol_lr_f = garch_res_lr_f.conditional_volatility

        # Compute GARCH(1,1) volatility based on absolute log returns for the specified frequency
        garch_model_alr_f = arch_model(abs_log_returns_f, p=1, q=1)
        garch_model_alr_f.first_obs = 0
        garch_model_alr_f.last_obs = len(log_returns_f)
        garch_res_alr_f = garch_model_alr_f.fit(update_freq=5)

        garch_vol_alr_f = garch_res_alr_f.conditional_volatility

        # Optimize alpha value for DD-EWMA
        result = minimize(DD_volatility_mse, x0=0.02, args=(log_returns_f, cut_t), bounds=[(0.01, 0.05)])
        best_alpha = result.x[0]

        # Calculate the forecasted volatility and RMSE for each date using the DD_volatility() function with the best
        # alpha value
        forecasted_volatility = [np.nan] * cut_t
        ddewma_rmse = [np.nan] * cut_t
        for i in range(len(abs_log_returns_f) - cut_t):
            log_returns_window = log_returns_f[i:cut_t + i]
            mse, vol_forecast = DD_volatility(log_returns_window, cut_t, [best_alpha])
            forecasted_volatility.append(vol_forecast)
            ddewma_rmse.append(np.sqrt(mse))

        # Combine into DataFrame
        df = pd.DataFrame({
            'Price': data_f,
            'Absolute Log Return': abs_log_returns_f,
            'GARCH(1,1) Log Returns': garch_vol_lr_f,
            'GARCH(1,1) Absolute Log Returns': garch_vol_alr_f,
            'DD-EWMA Forecasted Volatility': pd.Series(forecasted_volatility, index=abs_log_returns_f.index),
            'DD-EWMA RMSE': pd.Series(ddewma_rmse, index=abs_log_returns_f.index)
        })

        # Calculate correlation coefficients
        corr_garch_lr = df['Absolute Log Return'].iloc[cut_t:].corr(df['GARCH(1,1) Log Returns'].iloc[cut_t:])
        corr_garch_alr = df['Absolute Log Return'].iloc[cut_t:].corr(df['GARCH(1,1) Absolute Log Returns'].iloc[cut_t:])
        corr_ddewma_fv = df['Absolute Log Return'].iloc[cut_t:].corr(df['DD-EWMA Forecasted Volatility'].iloc[cut_t:])

        # Add correlation coefficients to the DataFrame
        df['Corr(Abs Log Return, GARCH Log Returns)'] = corr_garch_lr
        df['Corr(Abs Log Return, GARCH Abs Log Returns)'] = corr_garch_alr
        df['Corr(Abs Log Return, DD-EWMA Forecasted Volatility)'] = corr_ddewma_fv

        # Append the correlation coefficients for the current symbol and frequency to the DataFrame
        corr_df = corr_df.append({
            'Symbol': symbol,
            'Frequency': f,
            'Abs_Log_Return_vs_GARCH_Log_Returns': corr_garch_lr,
            'Abs Log Return, GARCH Abs Log Returns': corr_garch_alr,
            'Abs_Log_Return_vs_DD_EWMA': corr_ddewma_fv
        }, ignore_index=True)

        # Save to CSV
        filename = f'{symbol}_{f}.csv'
        df.to_csv(filename)

# Save the correlation coefficients DataFrame to a CSV file
corr_df.to_csv('correlation_coefficients.csv', index=False)

"""
based on correlation coefficient, the best models for each timeframe are:
1D -> DD_EWMA (0.28 on avg)
1W -> GARCH Log Returns (0.26 on avg)
1M -> GARCH Abs Log Returns (0.21 on avg)
"""

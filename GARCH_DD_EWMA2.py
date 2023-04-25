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
        garch_res_alr_f = garch_model_alr_f.fit(update_freq=5)
        garch_vol_alr_f = garch_res_alr_f.conditional_volatility

        # Calculate the forecasted volatility and RMSE for each date using the DD_volatility() function
        forecasted_volatility = [np.nan] * cut_t
        ddewma_rmse = [np.nan] * cut_t
        for i in range(len(abs_log_returns_f) - cut_t):
            log_returns_window = log_returns[i:cut_t + i]
            vol_forecast, RMSE = DD_volatility(log_returns_window, cut_t, alpha)
            forecasted_volatility.append(vol_forecast[0])
            ddewma_rmse.append(RMSE)

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
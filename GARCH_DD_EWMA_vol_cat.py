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

    # check if all values in MSE_alpha are NaN
    if np.isnan(MSE_alpha).all():
        vol_forecast = np.nan
    else:
        vol_forecast = sn[[i for i, j in enumerate(MSE_alpha) if j == min(MSE_alpha)]]  # which min

    RMSE = np.sqrt(np.nanmin(MSE_alpha))

    return vol_forecast, RMSE



def categorize_volatility(value, low_volatility_cap, medium_volatility_cap):
    if pd.isna(value):
        return "N/A"
    elif value < low_volatility_cap:
        return "Low volatility"
    elif low_volatility_cap <= value < medium_volatility_cap:
        return "Medium volatility"
    elif value >= medium_volatility_cap:
        return "High volatility"
    else:
        return "N/A"


# Define symbols and frequency
symbols = ['GC=F', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
freq = ['D', 'W', 'M']

# Set the cut-off time and list of alpha values
cut_t = 30
alpha = [0.01, 0.02, 0.03, 0.04, 0.05]

# Set the volatility tresholds
low_vol_theshold = 68
mid_vol_treshold = 95

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
        forecasted_volatility = [np.nan] * len(abs_log_returns_f)
        ddewma_rmse = [np.nan] * len(abs_log_returns_f)
        for i in range(len(abs_log_returns_f) - cut_t):
            log_returns_window = log_returns[i:cut_t + i]
            vol_forecast, RMSE = DD_volatility(log_returns_window, cut_t, alpha)
            forecasted_volatility[i + cut_t] = vol_forecast[0]
            ddewma_rmse[i + cut_t] = RMSE

        # Assign NaN values to the first cut_t elements
        forecasted_volatility[:cut_t] = [np.nan] * cut_t
        ddewma_rmse[:cut_t] = [np.nan] * cut_t

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

        # Calculate the low_volatility_cap and medium_volatility_cap for each column
        low_volatility_cap_abs_lr = np.percentile(df['Absolute Log Return'].iloc[cut_t:], low_vol_theshold)
        medium_volatility_cap_abs_lr = np.percentile(df['Absolute Log Return'].iloc[cut_t:], mid_vol_treshold)

        low_volatility_cap_garch_lr = np.percentile(df['GARCH(1,1) Log Returns'].iloc[cut_t:], low_vol_theshold)
        medium_volatility_cap_garch_lr = np.percentile(df['GARCH(1,1) Log Returns'].iloc[cut_t:], mid_vol_treshold)

        low_volatility_cap_garch_alr = np.percentile(df['GARCH(1,1) Absolute Log Returns'].iloc[cut_t:],
                                                     low_vol_theshold)
        medium_volatility_cap_garch_alr = np.percentile(df['GARCH(1,1) Absolute Log Returns'].iloc[cut_t:],
                                                        mid_vol_treshold)

        low_volatility_cap_ddewma_fv = np.percentile(df['DD-EWMA Forecasted Volatility'].iloc[cut_t:], low_vol_theshold)
        medium_volatility_cap_ddewma_fv = np.percentile(df['DD-EWMA Forecasted Volatility'].iloc[cut_t:],
                                                        mid_vol_treshold)

        # Create new columns for volatility categories
        df['Volatility_Absolute_Log_Return'] = df['Absolute Log Return'].apply(
            lambda x: categorize_volatility(x, low_volatility_cap_abs_lr, medium_volatility_cap_abs_lr))

        df['Volatility_GARCH_Log_Returns'] = df['GARCH(1,1) Log Returns'].apply(
            lambda x: categorize_volatility(x, low_volatility_cap_garch_lr, medium_volatility_cap_garch_lr))

        df['Volatility_GARCH_Absolute_Log_Returns'] = df['GARCH(1,1) Absolute Log Returns'].apply(
            lambda x: categorize_volatility(x, low_volatility_cap_garch_alr, medium_volatility_cap_garch_alr))

        df['Volatility_DD_EWMA_Forecasted_Volatility'] = df['DD-EWMA Forecasted Volatility'].apply(
            lambda x: categorize_volatility(x, low_volatility_cap_ddewma_fv, medium_volatility_cap_ddewma_fv))

        # Save to CSV
        filename = f'{symbol}_{f}.csv'
        df.to_csv(filename)

# Save the correlation coefficients DataFrame to a CSV file
corr_df.to_csv('correlation_coefficients.csv', index=False)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Symbol', 'Frequency', 'Match_GARCH_Log_Returns', 'Match_GARCH_Absolute_Log_Returns', 'Match_DD_EWMA_Forecasted_Volatility'])

for symbol in symbols:
    for f in freq:
        # Load the data
        filename = f'{symbol}_{f}.csv'
        df = pd.read_csv(filename, index_col=0)

        # Shift the volatility categories by one row
        shifted_df = df.shift(1)

        # Count the number of matching rows between Volatility_Absolute_Log_Return and the other three categories
        match_garch_lr = df['Volatility_Absolute_Log_Return'].eq(shifted_df['Volatility_GARCH_Log_Returns']).mean() * 100
        match_garch_alr = df['Volatility_Absolute_Log_Return'].eq(shifted_df['Volatility_GARCH_Absolute_Log_Returns']).mean() * 100
        match_ddewma_fv = df['Volatility_Absolute_Log_Return'].eq(shifted_df['Volatility_DD_EWMA_Forecasted_Volatility']).mean() * 100

        # Append the results for the current symbol and frequency to the DataFrame
        results_df = results_df.append({
            'Symbol': symbol,
            'Frequency': f,
            'Match_GARCH_Log_Returns': match_garch_lr,
            'Match_GARCH_Absolute_Log_Returns': match_garch_alr,
            'Match_DD_EWMA_Forecasted_Volatility': match_ddewma_fv
        }, ignore_index=True)

# Save the results DataFrame to a CSV file
results_df.to_csv('matching_volatility_categories.csv', index=False)

"""
based on correlation coefficient, the best models for each timeframe are:
1D -> DD_EWMA (0.28 on avg)
1W -> GARCH Log Returns (0.26 on avg)
1M -> GARCH Abs Log Returns (0.21 on avg)

volatility matches for all timeframes:
59% for Match_GARCH_Log_Returns
58% for Match_GARCH_Absolute_Log_Returns


"""

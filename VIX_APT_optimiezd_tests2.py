import yfinance as yf
import pandas as pd
from fredapi import Fred
import statsmodels.api as sm
import numpy as np

# Set FRED API key
fred = Fred(api_key='70e9773ec02c6048488df54a0ada35b2')

# Define start and end dates
start_date = "2007-01-01"
end_date = "2023-03-31"
start_date_returns = "2018-01-01"
end_date_returns = "2023-03-31"

# Download data from Yahoo Finance for VIX
vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")

# Download data on consumer confidence index
umcsent = fred.get_series('UMCSENT', observation_start=start_date, observation_end=end_date)

# Download data on BofA Merrill Lynch US High Yield Master II Option-Adjusted Spread
baml = fred.get_series('BAMLH0A0HYM2EY', observation_start=start_date, observation_end=end_date)

# Download data on Job Openings and Labor Turnover Survey: Job Openings
jtsjol = fred.get_series('JTSJOL', observation_start=start_date, observation_end=end_date)

# Download data for VVIX
vvix = yf.download("^VVIX", start=start_date, end=end_date, interval="1d")['Close']

recprousm156n = fred.get_series('RECPROUSM156N', observation_start=start_date, observation_end=end_date)

# Combine data into a single DataFrame
data = pd.concat([vix['Close'], umcsent, baml, jtsjol, vvix, recprousm156n], axis=1)


# Resample the data to a monthly frequency
data = data.resample('M').last()

# Forward fill missing values
data = data.ffill()

# Calculate the monthly percentage change for each column
data_pct = data.pct_change()

# Filter data for start date of 1/1/2007
data = data.loc[start_date:]
data_pct = data_pct.loc[start_date:]

# Rename columns
data.columns = ['VIX', 'Consumer Confidence', 'High Yield Spread', 'Job Openings', 'VVIX', 'Recession Probabilities']

# Save data to CSV file
data.to_csv('data.csv', index=True)

# Create a new DataFrame with the independent variables
X = sm.add_constant(data[['Consumer Confidence', 'High Yield Spread', 'Job Openings', 'VVIX',
                          'Recession Probabilities']])

# Fit a linear regression model with VIX as the dependent variable
model = sm.OLS(data['VIX'], X).fit()

# Print the model summary
print(model.summary())

# Rename columns for DB with returns
data_pct.columns = ['VIX', 'Consumer Confidence', 'High Yield Spread', 'Job Openings', 'VVIX',
                    'Recession Probabilities']

# Replace any zeros, infinities, and negative infinities in the DataFrame data_pct with NaN values
data_pct.replace([0, np.inf, -np.inf], np.nan, inplace=True)

# Forward-fill any missing values
data_pct = data_pct.ffill()

# Backward-fill any missing values
data_pct = data_pct.bfill()


data_pct.to_csv('data_pct.csv', index=True)

# Define the dependent variable as the historical returns of each asset
returns = data_pct[['VIX']]

# Define the independent variables as the macroeconomic factors
independent_variables = data_pct[['Consumer Confidence', 'High Yield Spread', 'Job Openings', 'VVIX',
                                  'Recession Probabilities']]

# Use a multiple linear regression model to estimate the beta coefficients for each asset
betas = pd.DataFrame(index=returns.columns, columns=independent_variables.columns)
for asset in returns.columns:
    model = sm.OLS(returns[asset], sm.add_constant(independent_variables)).fit()
    betas.loc[asset] = model.params[1:]

# Save the beta coefficients for each asset
betas.to_csv('betas.csv', index=True)

# Load the beta coefficients for each asset
betas = pd.read_csv('betas.csv', index_col=0)

# Download data on U.S. Treasury Securities at 1-Year Constant Maturity
dtb3 = fred.get_series('dtb3', observation_start=start_date, observation_end=end_date)

# Resample the data to a monthly frequency
dtb3_monthly = dtb3.resample('M').last()

# Convert yearly rate to monthly rate
dtb3_monthly = ((1 + dtb3_monthly)**(1/52)) - 1

# Forward fill missing values
dtb3_monthly = dtb3_monthly.ffill()

dtb3_monthly.to_csv('dtb3_monthly.csv')

# Merge data_pct and dtb3_monthly using merge_asof
data_pct_rf = pd.merge_asof(data_pct, dtb3_monthly.to_frame(), left_index=True, right_index=True, direction='forward')

# Fill missing values with the last available value
data_pct_rf.fillna(method='ffill', inplace=True)

# Rename the last column to 'risk-free rate'
data_pct_rf = data_pct_rf.rename(columns={data_pct_rf.columns[-1]: 'risk free rate'})

data_pct_rf.to_csv('data_pct_rf.csv')

# Calculate excess returns for each asset
excess_returns = data_pct_rf.subtract(dtb3_monthly.values[0], axis=0)

excess_returns.to_csv('excess_returns.csv', index=True)

# Calculate average excess returns for each factor
avg_excess_returns = {}
for factor in excess_returns.columns[1:]:
    excess_returns_rf = excess_returns[factor] - dtb3_monthly.values.flatten()
    avg_excess_returns[factor] = excess_returns_rf.mean()

print(f'avg_excess_returns{avg_excess_returns}')

# Calculate historical risk premiums
risk_premiums = {}
for factor in avg_excess_returns.keys():
    risk_premiums[factor] = avg_excess_returns[factor]

# Print the historical risk premiums
print(f'risk_premiums: {risk_premiums}')

# Calculate the expected returns for each asset using the APT model formula
expected_returns = pd.DataFrame(index=betas.index, columns=['Expected Return'])
for asset in betas.index:
    expected_return = dtb3_monthly.iloc[-1]
    for factor in betas.columns:
        premium = risk_premiums[factor]
        beta = betas.loc[asset, factor]
        expected_return += beta * premium
    expected_returns.loc[asset, 'Expected Return'] = expected_return

# Print the expected returns for each asset
print(expected_returns)

# Load historical returns data
data_pct = pd.read_csv('data_pct.csv', index_col=0, parse_dates=True)

# Calculate actual returns for each asset
actual_returns = data_pct.mean()

# Compare actual and expected returns
comparison = pd.concat([actual_returns, expected_returns], axis=1)
comparison.columns = ['Actual Return', 'Expected Return']
comparison['Difference'] = comparison['Actual Return'] - comparison['Expected Return']
print(comparison)

# Create a list containing all month-end dates from start_date_returns to end_date_returns
dates = pd.date_range(start=start_date_returns, end=end_date_returns, freq='M')

# Initialize a DataFrame to store the expected returns for each date
expected_returns_df = pd.DataFrame(columns=['Date', 'Expected Return'])
expected_returns_df['Date'] = dates
expected_returns_df.set_index('Date', inplace=True)

# Loop through each date and recalculate the expected returns
for date in dates:
    # Filter data up to the current date
    current_data = data.loc[start_date:date]
    current_data_pct = data_pct.loc[start_date:date]
    current_dtb3_monthly = dtb3_monthly.loc[start_date:date]

    # Calculate betas for the assets up to the current date
    current_returns = current_data_pct[['VIX']]
    current_independent_variables = current_data_pct[['Consumer Confidence', 'High Yield Spread', 'Job Openings',
                                                      'VVIX', 'Recession Probabilities']]
    current_betas = pd.DataFrame(index=current_returns.columns, columns=current_independent_variables.columns)

    for asset in current_returns.columns:
        model = sm.OLS(current_returns[asset], sm.add_constant(current_independent_variables)).fit()
        current_betas.loc[asset] = model.params[1:]

    # Calculate average excess returns for each factor up to the current date
    current_excess_returns = current_data_pct.subtract(current_dtb3_monthly.values[0], axis=0)
    current_avg_excess_returns = {}

    for factor in current_excess_returns.columns[1:]:
        current_excess_returns_rf = current_excess_returns[factor] - current_dtb3_monthly.values.flatten()
        current_avg_excess_returns[factor] = current_excess_returns_rf.mean()

    # Calculate historical risk premiums up to the current date
    current_risk_premiums = {}
    for factor in current_avg_excess_returns.keys():
        current_risk_premiums[factor] = current_avg_excess_returns[factor]

    # Calculate the expected returns for each asset up to the current date using the APT model formula
    current_expected_returns = pd.DataFrame(index=current_betas.index, columns=['Expected Return'])

    for asset in current_betas.index:
        current_expected_return = current_dtb3_monthly.iloc[-1]
        for factor in current_betas.columns:
            premium = current_risk_premiums[factor]
            beta = current_betas.loc[asset, factor]
            current_expected_return += beta * premium
        current_expected_returns.loc[asset, 'Expected Return'] = current_expected_return

    # Store the expected return for the VIX at the current date
    expected_returns_df.loc[date, 'Expected Return'] = current_expected_returns.loc['VIX', 'Expected Return']

# Calculate the actual VIX returns for 1-month, 3-month, and 6-month windows
data_pct['1M_return'] = data_pct['VIX'].shift(-1)
data_pct['3M_return'] = data_pct['VIX'].rolling(window=3).sum().shift(-3)
data_pct['6M_return'] = data_pct['VIX'].rolling(window=6).sum().shift(-6)

# Add the new columns to the expected_returns_df DataFrame
expected_returns_df['1M_return'] = data_pct.loc[dates, '1M_return']
expected_returns_df['3M_return'] = data_pct.loc[dates, '3M_return']
expected_returns_df['6M_return'] = data_pct.loc[dates, '6M_return']

# Save the expected returns DataFrame to a CSV file
expected_returns_df.to_csv('expected_returns.csv', index=True)

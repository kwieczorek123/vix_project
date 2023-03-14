import yfinance as yf
import pandas as pd
from fredapi import Fred
import statsmodels.api as sm
import numpy as np

# Set FRED API key
fred = Fred(api_key='70e9773ec02c6048488df54a0ada35b2')

# Download data from Yahoo Finance for VIX
vix = yf.download("^VIX", start="2007-01-01", end="2022-12-31", interval="1d")

# Download data on GDP growth (Real Gross Domestic Product)
gdp = fred.get_series('GDPC1', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on inflation (Consumer Price Index) Total All Items for the United States
cpi = fred.get_series('CPALTT01USM657N', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on consumer confidence index
umcsent = fred.get_series('UMCSENT', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on BofA Merrill Lynch US High Yield Master II Option-Adjusted Spread
baml = fred.get_series('BAMLH0A0HYM2EY', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
t10y2y = fred.get_series('T10Y2Y', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on M2 Money Stock
m2sl = fred.get_series('M2SL', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on Delinquency Rate on Commercial and Industrial Loans, All Commercial Banks
drcclacbs = fred.get_series('DRCCLACBS', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on Job Openings and Labor Turnover Survey: Job Openings
jtsjol = fred.get_series('JTSJOL', observation_start='2007-01-01', observation_end='2022-12-31')

# Download data on S&P 500
sp500 = yf.download("^GSPC", start="2007-01-01", end="2022-12-31", interval="1d")

# Download data for SPY and XLP separately
spy = yf.download("SPY", start="2007-01-01", end="2022-12-31", interval="1d")['Close']
xlp = yf.download("XLP", start="2007-01-01", end="2022-12-31", interval="1d")['Close']

# Calculate the ratio of SPY to XLP
spy_xlp = spy / xlp

# Download data for PKW and VYM separately
pkw = yf.download("PKW", start="2007-01-01", end="2022-12-31", interval="1d")['Close']
vym = yf.download("VYM", start="2007-01-01", end="2022-12-31", interval="1d")['Close']

# Calculate the ratio of PKW to VYM
pkw_vym = pkw / vym

# Download data for VVIX
vvix = yf.download("^VVIX", start="2007-01-01", end="2022-12-31", interval="1d")['Close']


# Combine data into a single DataFrame
data = pd.concat([vix['Close'], cpi, gdp, umcsent, baml, t10y2y, m2sl, drcclacbs, jtsjol, sp500['Close'], spy_xlp,
                  pkw_vym, vvix], axis=1)


# Resample the data to a weekly frequency
data = data.resample('W').last()

# Forward fill missing values
data = data.ffill()

# Calculate the weekly percentage change for each column
data_pct = data.pct_change()

# Filter data for start date of 1/1/2007
data = data.loc['2007-01-01':]
data_pct = data_pct.loc['2007-01-01':]

# Rename columns
data.columns = ['VIX', 'Inflation', 'GDP Growth', 'Consumer Confidence', 'High Yield Spread', '10y-2y Yield Spread',
                'M2 Money Stock', 'Delinquency Rate', 'Job Openings', 'S&P 500', 'SPY XLP', 'PKW VYM', 'VVIX']

# Save data to CSV file
data.to_csv('data.csv', index=True)

# Create a new DataFrame with the independent variables
X = sm.add_constant(data[['Inflation', 'GDP Growth', 'Consumer Confidence', 'High Yield Spread', '10y-2y Yield Spread',
                          'M2 Money Stock', 'Delinquency Rate', 'Job Openings', 'S&P 500', 'SPY XLP', 'PKW VYM',
                          'VVIX']])

# Fit a linear regression model with VIX as the dependent variable
model = sm.OLS(data['VIX'], X).fit()

# Print the model summary
print(model.summary())

# Rename columns for DB with returns
data_pct.columns = ['VIX', 'Inflation', 'GDP Growth', 'Consumer Confidence', 'High Yield Spread', '10y-2y Yield Spread',
                    'M2 Money Stock', 'Delinquency Rate', 'Job Openings', 'S&P 500', 'SPY XLP', 'PKW VYM', 'VVIX']

# Replace any zeros, infinities, and negative infinities in the DataFrame data_pct with NaN values
data_pct.replace([0, np.inf, -np.inf], np.nan, inplace=True)

# Resample the DataFrame to a daily frequency
data_pct_daily = data_pct.resample('D').last()

# Forward-fill any missing values
data_pct = data_pct.ffill()

# Backward-fill any missing values
data_pct = data_pct.bfill()


data_pct.to_csv('data_pct.csv', index=True)

# Define the dependent variable as the historical returns of each asset
returns = data_pct[['VIX']]

# Define the independent variables as the macroeconomic factors
independent_variables = data_pct[['Inflation', 'GDP Growth', 'Consumer Confidence', 'High Yield Spread',
                                  '10y-2y Yield Spread', 'M2 Money Stock', 'Delinquency Rate', 'Job Openings',
                                  'S&P 500', 'SPY XLP', 'PKW VYM', 'VVIX']]

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
dgs1 = fred.get_series('DGS1', observation_start='2007-01-01', observation_end='2022-12-31')

# Define the risk-free rate as the last data point in dgs1
rf = dgs1.iloc[-1]

# Load historical returns data
data_pct = pd.read_csv('data_pct.csv', index_col=0, parse_dates=True)

# Calculate excess returns for each asset
excess_returns = data_pct.subtract(rf/52, axis=0)

excess_returns.to_csv('excess_returns.csv', index=True)

# Calculate average excess returns for each factor
avg_excess_returns = {}
for factor in excess_returns.columns[1:]:
    avg_excess_returns[factor] = excess_returns[factor].mean()

print(f'avg_excess_returns{avg_excess_returns}')

# Calculate historical risk premiums
risk_premiums = {}
for factor in avg_excess_returns.keys():
    risk_premiums[factor] = avg_excess_returns[factor]

# Print the historical risk premiums
print(risk_premiums)

# Calculate the expected returns for each asset using the APT model formula
expected_returns = pd.DataFrame(index=betas.index, columns=['Expected Return'])
for asset in betas.index:
    expected_return = rf
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





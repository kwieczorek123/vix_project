import yfinance as yf
import pandas as pd
from fredapi import Fred
import statsmodels.api as sm

# Download data from Yahoo Finance for VIX
vix = yf.download("^VIX", start="2007-01-01", end="2023-03-13", interval="1d")

# Download data on inflation (Consumer Price Index) Total All Items for the United States
fred = Fred(api_key='70e9773ec02c6048488df54a0ada35b2 ')

# Download data on GDP growth (Real Gross Domestic Product)
gdp = fred.get_series('GDPC1')

# Download data on inflation (Consumer Price Index) Total All Items for the United States
cpi = fred.get_series('CPALTT01USM657N')

# Download data on unemployment rate
unrate = fred.get_series('UNRATE')

# Download data on consumer confidence index
umcsent = fred.get_series('UMCSENT')

# Download data on industrial production
indpro = fred.get_series('INDPRO')

# Download data on housing starts
houst = fred.get_series('HOUST')

# Download data on 10-year Treasury rate
tenyr = fred.get_series('GS10')

# Download data on 3-month Treasury rate
threem = fred.get_series('DTB3')

# Compute the difference between 10-year and 3-month yields
yield_diff = tenyr - threem

# Download data on S&P 500
sp500 = yf.download("^GSPC", start="2007-01-01", end="2023-03-13", interval="1d")

# Combine data into a single DataFrame
data = pd.concat([vix['Close'], cpi, gdp, unrate, umcsent, indpro, houst, tenyr, threem, yield_diff, sp500['Close']], axis=1)

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
data.columns = ['VIX', 'Inflation', 'GDP Growth', 'Unemployment Rate', 'Consumer Confidence', 'Industrial Production', 'Housing Starts', '10-Year Rate', '3-Month Rate', 'Yield Difference', 'S&P 500']

# Save data to CSV file
data.to_csv('data.csv', index=True)

# Create a new DataFrame with the independent variables
X = sm.add_constant(data[['Inflation', 'GDP Growth', 'Unemployment Rate', 'Consumer Confidence', 'Industrial Production', 'Housing Starts', '10-Year Rate', '3-Month Rate', 'Yield Difference', 'S&P 500']])

# Fit a linear regression model with VIX as the dependent variable
model = sm.OLS(data['VIX'], X).fit()

# Print the model summary
print(model.summary())

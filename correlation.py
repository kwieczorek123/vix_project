import pandas as pd
import yfinance as yf

# Load the data from the Excel files
vvix_data = pd.read_excel('VVIX_monthly_data.xlsx')
vix_data = pd.read_excel('VIX_monthly_data.xlsx')
gvz_data = pd.read_excel('GVZ_monthly_data.xlsx')
evz_data = pd.read_excel('EVZ_monthly_data.xlsx')

# Calculate the correlation coefficients between the 'Close' columns
monthly_corr_matrix = pd.DataFrame({
    'VVIX': [1,
             vvix_data['Close'].corr(vix_data['Close']),
             vvix_data['Close'].corr(gvz_data['Close']),
             vvix_data['Close'].corr(evz_data['Close'])],
    'VIX': [vix_data['Close'].corr(vvix_data['Close']),
            1,
            vix_data['Close'].corr(gvz_data['Close']),
            vix_data['Close'].corr(evz_data['Close'])],
    'GVZ': [gvz_data['Close'].corr(vvix_data['Close']),
            gvz_data['Close'].corr(vix_data['Close']),
            1,
            gvz_data['Close'].corr(evz_data['Close'])],
    'EVZ': [evz_data['Close'].corr(vvix_data['Close']),
            evz_data['Close'].corr(vix_data['Close']),
            evz_data['Close'].corr(gvz_data['Close']),
            1]
}, index=['VVIX', 'VIX', 'GVZ', 'EVZ'])

# Download the data from Yahoo Finance for each symbol
data_vix = yf.download("^VIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_vvix = yf.download("^VVIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_gvz = yf.download("^GVZ", start="2010-01-01", end="2023-02-28", interval="1d")
data_evz = yf.download("^EVZ", start="2010-01-01", end="2023-02-28", interval="1d")

# Create a new dataframe with the 'Close' columns from each symbol's data
close_df = pd.DataFrame({
    'VIX': data_vix['Close'],
    'VVIX': data_vvix['Close'],
    'GVZ': data_gvz['Close'],
    'EVZ': data_evz['Close']
})

# Calculate the correlation matrix between the 'Close' columns
daily_corr_matrix = close_df.corr()

# Create a new Excel file and write the correlation matrices to different sheets
with pd.ExcelWriter('correlation_coefficients.xlsx') as writer:
    monthly_corr_matrix.to_excel(writer, sheet_name='Monthly Correlation Matrix')
    daily_corr_matrix.to_excel(writer, sheet_name='Daily Correlation Matrix')

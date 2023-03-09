import yfinance as yf
import pandas as pd

# Download historical data for the EVZ
EVZ_data = yf.download("^EVZ", start="2010-01-01", end="2023-03-08")

# Convert the index to a datetime format
EVZ_data.index = pd.to_datetime(EVZ_data.index)

# Calculate the monthly averages of the EVZ
monthly_averages = EVZ_data.resample("M").agg({'Close': 'mean', 'High': 'max', 'Low': 'min'})

# Add a column for month
monthly_averages['Month'] = monthly_averages.index.month

# Add a column for year
monthly_averages['Year'] = monthly_averages.index.year

# Add a column for last day of the month
monthly_averages['Date'] = monthly_averages.index + pd.offsets.MonthEnd(0)

# Save to Excel file
with pd.ExcelWriter('EVZ_monthly_data.xlsx') as writer:
    monthly_averages.to_excel(writer, sheet_name='monthly_data', index=False)

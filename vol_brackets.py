import yfinance as yf
import pandas as pd
import xlsxwriter

# Download data from Yahoo Finance for VIX
vix = yf.download("^VIX", start="2007-01-01", end="2023-03-13", interval="1d")
vix['volatility_bracket'] = pd.cut(vix['Close'], bins=[-float("inf"), 20, 30, float("inf")], labels=[1, 2, 3])
vix = vix.reset_index()

# Download data from Yahoo Finance for GVZ
gvz = yf.download("^GVZ", start="2007-01-01", end="2023-03-13", interval="1d")
gvz['volatility_bracket'] = pd.cut(gvz['Close'], bins=[-float("inf"), 20, 30, float("inf")], labels=[1, 2, 3])
gvz = gvz.reset_index()

# Create a new DataFrame with the counts and percentages for each volatility_bracket value for VIX
vix_counts = vix['volatility_bracket'].value_counts().sort_index()
vix_pct = vix_counts / vix_counts.sum()
vix_table = pd.concat([vix_counts, vix_pct], axis=1, keys=['Count', 'volatility_bracket_Pct'])

# Create a new DataFrame with the counts and percentages for each volatility_bracket value for GVZ
gvz_counts = gvz['volatility_bracket'].value_counts().sort_index()
gvz_pct = gvz_counts / gvz_counts.sum()
gvz_table = pd.concat([gvz_counts, gvz_pct], axis=1, keys=['Count', 'volatility_bracket_Pct'])

# Save the data to a CSV file
vix.to_csv('vix_data.csv', index=False)
gvz.to_csv('gvz_data.csv', index=False)

# Save the data to an Excel file
with pd.ExcelWriter('volatility_indices_data.xlsx', engine='xlsxwriter') as writer:
    # Write the VIX daily data to a sheet
    vix.to_excel(writer, sheet_name='VIX Daily Data', index=False)

    # Write the VIX pivot table to a new sheet
    vix_table.sort_values('Count', ascending=False).to_excel(writer, sheet_name='VIX Pivot Table')

    # Write the GVZ daily data to a sheet
    gvz.to_excel(writer, sheet_name='GVZ Daily Data', index=False)

    # Write the GVZ pivot table to a new sheet
    gvz_table.sort_values('Count', ascending=False).to_excel(writer, sheet_name='GVZ Pivot Table')

    # Write the legend to a new sheet
    legend = pd.DataFrame({'volatility_bracket': [1, 2, 3], 'Description': ['Close < 20', '20 <= Close <= 30', 'Close > 30']})
    legend.to_excel(writer, sheet_name='Legend', index=False)

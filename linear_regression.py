import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Download data
data_vix = yf.download("^VIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_vvix = yf.download("^VVIX", start="2010-01-01", end="2023-02-28", interval="1d")

# Merge data
combined_df = pd.merge(data_vix['Close'], data_vvix['Close'], left_index=True, right_index=True)
combined_df.columns = ['VIX', 'VVIX']

# Remove any missing or infinite values
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna()

# Split into training and testing sets
train_data = combined_df.iloc[:int(0.8 * len(combined_df)), :]
test_data = combined_df.iloc[int(0.8 * len(combined_df)):, :]

# Fit linear regression model
X = sm.add_constant(train_data['VVIX'])
model = sm.OLS(train_data['VIX'], X).fit()

# Predict using the model
X_test = sm.add_constant(test_data['VVIX'])
predictions = model.predict(X_test)

# Save summary to CSV file
with open('linear_regression_summary.csv', 'w') as f:
    f.write(model.summary().as_csv())

# Save R-squared and mean squared error to CSV file
results_df = pd.DataFrame({
    'R-squared': [model.rsquared],
    'Mean squared error': [model.mse_model]
})
results_df.to_csv('linear_regression_results.csv', index=False)

# Print summary of the model
print(model.summary())

# Print R-squared and mean squared error
print("R-squared: ", model.rsquared)
print("Mean squared error: ", model.mse_model)

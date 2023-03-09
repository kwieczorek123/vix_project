import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

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

# Fit Random Forest Regression model
X_train = train_data[['VVIX']]
y_train = train_data['VIX']

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

# Predict using the model
X_test = test_data[['VVIX']]
y_test = test_data['VIX']

predictions = model.predict(X_test)

# Evaluate model performance
print("R-squared: ", r2_score(y_test, predictions))
print("Mean squared error: ", mean_squared_error(y_test, predictions))

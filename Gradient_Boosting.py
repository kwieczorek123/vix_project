import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
X_train, X_test, y_train, y_test = train_test_split(combined_df[['VVIX']], combined_df['VIX'], test_size=0.2, random_state=42)

# Fit Gradient Boosting Regression model
model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)

# Predict using the model
predictions = model.predict(X_test)

# Evaluate model performance
print("R-squared: ", r2_score(y_test, predictions))
print("Mean squared error: ", mean_squared_error(y_test, predictions))

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download data
data_vix = yf.download("^VIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_vvix = yf.download("^VVIX", start="2010-01-01", end="2023-02-28", interval="1d")

# Merge data
combined_df = pd.merge(data_vix['Close'], data_vvix['Close'], left_index=True, right_index=True)
combined_df.columns = ['VIX', 'VVIX']

# Remove any missing or infinite values
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_df)

# Split into training and testing sets
train_data = scaled_data[:int(0.8 * len(scaled_data)), :]
test_data = scaled_data[int(0.8 * len(scaled_data)):, :]

# Split into X and y variables
X_train = train_data[:, 1:]
y_train = train_data[:, 0]

X_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Define the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
mse, r_squared = model.evaluate(X_test, y_test)
print("Mean squared error: ", mse)
print("R-squared: ", r_squared)

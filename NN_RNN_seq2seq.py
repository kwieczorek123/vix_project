import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
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
X_train = []
y_train = []
for i in range(30, len(train_data)):
    X_train.append(train_data[i-30:i, :])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(30, len(test_data)):
    X_test.append(test_data[i-30:i, :])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print("Mean squared error: ", mse)

# Make predictions
new_data = combined_df[-30:].values
new_data = scaler.transform(new_data)
new_data = np.reshape(new_data, (1, 30, 2))
predicted_vix = model.predict(new_data)
predicted_vix = scaler.inverse_transform(predicted_vix.reshape((1, 1)))


print("Predicted VIX: ", predicted_vix[0, 0])

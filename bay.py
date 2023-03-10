import pandas as pd
import numpy as np
import yfinance as yf
from pomegranate import BayesianNetwork
from pomegranate import MaximumLikelihoodEstimator

# Download data
data_vix = yf.download("^VIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_vvix = yf.download("^VVIX", start="2010-01-01", end="2023-02-28", interval="1d")
data_spy = yf.download("SPY", start="2010-01-01", end="2023-02-28", interval="1d")
data_xlp = yf.download("XLP", start="2010-01-01", end="2023-02-28", interval="1d")
data_gld = yf.download("GLD", start="2010-01-01", end="2023-02-28", interval="1d")
data_uso = yf.download("USO", start="2010-01-01", end="2023-02-28", interval="1d")
data_pkw = yf.download("PKW", start="2010-01-01", end="2023-02-28", interval="1d")
data_vym = yf.download("VYM", start="2010-01-01", end="2023-02-28", interval="1d")

# Merge data
combined_df = pd.merge(data_vix['Close'], data_vvix['Close'], left_index=True, right_index=True)
combined_df = pd.merge(combined_df, data_spy['Close'], left_index=True, right_index=True)
combined_df = pd.merge(combined_df, data_xlp['Close'], left_index=True, right_index=True)
combined_df = pd.merge(combined_df, data_gld['Close'], left_index=True, right_index=True)
combined_df = pd.merge(combined_df, data_uso['Close'], left_index=True, right_index=True)
combined_df = pd.merge(combined_df, data_pkw['Close'], left_index=True, right_index=True)
combined_df = pd.merge(combined_df, data_vym['Close'], left_index=True, right_index=True)
combined_df.columns = ['VIX', 'VVIX', 'SPY', 'XLP', 'GLD', 'USO', 'PKW', 'VYM']

# Remove any missing or infinite values
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna()

# Define the symbols we want to predict and the predictors
predicted_symbol = 'VIX'
predictor_symbols = ['VVIX', 'SPY', 'XLP', 'GLD', 'USO', 'PKW', 'VYM']
relative_dict = {'SPY': 'XLP', 'GLD': 'USO', 'PKW': 'VYM'}

# Add relative symbols as columns to the data
for symbol, relative in relative_dict.items():
    combined_df[symbol + '/' + relative] = combined_df[symbol] / combined_df[relative]

# Define the Bayesian network structure
model = BayesianNetwork.from_structure(name='bn', nodes=['VVIX', 'SPY', 'VIX', 'XLP', 'USO', 'GLD', 'PKW', 'VYM'], edges=[('VVIX', 'VIX'), ('SPY', 'VIX'), ('XLP', 'SPY'), ('USO', 'GLD'), ('VYM', 'PKW')])

# Define the parameter estimation method
# Train the model using the maximum likelihood estimator

model.fit(combined_df.values)
# Predict the VIX values

predictions = model.predict(combined_df.values[:, :-1])
# Print the accuracy of the model

accuracy = (predictions[:, 2] == combined_df.values[:, -1]).mean()
print("Accuracy of the model:", accuracy)
# Define the query variables

query = {}
for symbol in predictor_symbols:
    query[symbol] = combined_df[symbol].values[-1]
# Define the inference engine

inference = model.predict_proba(query)
# Print the probabilities of VIX values

for i in range(10):
    print(f"Probability of VIX value {i}: {inference[2].parameters[0][i]:.6f}")
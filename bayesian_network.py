import pandas as pd
import numpy as np
import yfinance as yf
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

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
model = BayesianModel([('VVIX', 'VIX'), ('SPY', 'VIX'), ('XLP', 'SPY'), ('USO', 'GLD'), ('PKW', 'VYM')])

# Define the conditional probability tables for each variable

cpd_vvix = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('VVIX')
cpd_spy = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('SPY')
cpd_xlp = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('XLP')
cpd_uso = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('USO')
cpd_gld = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('GLD')
cpd_pkw = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('PKW')
cpd_vym = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('VYM')
cpd_vix = MaximumLikelihoodEstimator(model, combined_df).estimate_cpd('VIX')

# Create an inference object

inference = VariableElimination(model)


# Define a function to make predictions

def predict_vix(vvix, spy, xlp, gld, uso, pkw, vym):
    evidence = {'VVIX': vvix, 'SPY': spy, 'XLP': xlp, 'GLD': gld, 'USO': uso, 'PKW': pkw, 'VYM': vym}
    query = inference.query(variables=['VIX'], evidence=evidence)
    return query['VIX'].values


# Test the function on the last 10 days of data

test_data = combined_df.iloc[-10:]
predictions = []
for index, row in test_data.iterrows():
    vvix = row['VVIX']
    spy = row['SPY']
    xlp = row['XLP']
    gld = row['GLD']
    uso = row['USO']
    pkw = row['PKW']
    vym = row['VYM']
    prediction = predict_vix(vvix, spy, xlp, gld, uso, pkw, vym)
    predictions.append(prediction[0])
# Compare the predictions to the actual values

actual_values = test_data['VIX'].tolist()
for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}, Actual Value: {actual_values[i]}")

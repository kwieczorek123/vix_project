import pandas as pd
import yfinance as yf
from arch import arch_model

# Define symbols and frequency
symbols = ['GC=F', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
freq = ['D', 'W', 'M']

for symbol in symbols:
    # Retrieve data from Yahoo Finance
    data = yf.download(symbol, start='2000-01-01', end='2023-04-25')

    # Loop over frequencies
    for f in freq:
        # Resample data to specified frequency
        data_f = data['Adj Close'].resample(f).last().dropna()

        # Fit GARCH(1,1) model
        am = arch_model(data_f, p=1, q=1)
        res = am.fit(update_freq=5)

        # Create DataFrame for visualization
        df = pd.DataFrame({symbol: data_f, 'GARCH(1,1) Volatility': res.conditional_volatility})

        # Visualize and save results
        title = f'GARCH(1,1) Model Applied to {symbol} ({f.capitalize()})'
        fig = df.plot(title=title)
        fig.get_figure().savefig(f'{symbol}_{f}.png')
        df.to_csv(f'{symbol}_{f}.csv')


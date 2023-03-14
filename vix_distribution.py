import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Download data from Yahoo Finance for VIX
vix = yf.download("^VIX", start="2007-01-01", end="2023-03-13", interval="1d")

# Calculate daily returns of VIX
vix['Returns'] = vix['Close'].pct_change()

# Remove first row with NaN value
vix = vix.dropna()

# Calculate mean and standard deviation of returns
mu, std = stats.norm.fit(vix['Returns'])

# Print fit results
print("Fit results:")
print("Mean: {:.4f}".format(mu))
print("Standard deviation: {:.4f}".format(std))

# Plot histogram of returns with fitted normal distribution
plt.hist(vix['Returns'], bins=50, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = {:.2e},  std = {:.2e}".format(mu, std)
plt.title(title)
plt.savefig('vix_normal_distribution.png')
plt.show()


# Generate the normal probability plot
fig, ax = plt.subplots(figsize=(8, 4))
stats.probplot(vix['Returns'], plot=ax)

# Add title and axis labels
ax.set_title('Normal Probability Plot of VIX Returns')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')
plt.savefig('vix_Q-Q plot_normal_distribution.png')
plt.show()

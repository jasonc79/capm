import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import datetime
import yfinance as yf


start_date = datetime.datetime(2020, 1, 1)

end_date = datetime.datetime(2024, 1, 1)

stock_symbol = 'AAPL'  # Example: Apple Inc.
market_index_symbol = '^GSPC'  # Example: S&P 500 index
yf.pdr_override()
stock_data = web.get_data_yahoo(stock_symbol, start=start_date, end=end_date)
market_data = web.get_data_yahoo(market_index_symbol, start=start_date, end=end_date)

# Step 2: Calculate Returns
stock_returns = stock_data['Adj Close'].pct_change().dropna()
market_returns = market_data['Adj Close'].pct_change().dropna()

# Step 3: Calculate Excess Returns
risk_free_rate = 0.02  # Example risk-free rate (2%)
excess_stock_returns = stock_returns - risk_free_rate
excess_market_returns = market_returns - risk_free_rate

# Step 4: Estimate Beta
X = sm.add_constant(excess_market_returns)
model = sm.OLS(excess_stock_returns, X).fit()
beta = model.params[1]

# Step 5: Estimate Cost of Equity using CAPM
market_return = excess_market_returns.mean()
cost_of_equity = risk_free_rate + beta * (market_return)

print("Beta:", beta)
print("Cost of Equity (CAPM):", cost_of_equity)
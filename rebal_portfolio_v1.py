# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:01:42 2021

@author: ivelu
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib

tickers = ["SPY", "TLT", "GLD"]
weights = [0.7, 0.2, 0.1]
start = '2019-01-1'
end = '2021-09-30'
rebalance_frequency = '1wk'
trading_period_for_vol_calc = {"1d":252, "1wk":52, "1mo":12, "3mo":4, "1y":1}
rf = 0

## Eliminate np divide by 0 error to avoid issues when making log returns == 0 
## FIND FIX LATER
np.seterr(divide='ignore')


## Calculate returns for S&P 500 (assume S&P is benchmark)
sp500 = yf.download("^GSPC",start=start, end=end,interval=rebalance_frequency)
sp500["weighted_portfolio_return"] = np.diff(np.log(np.concatenate(([0],sp500['Adj Close']))))
sp500.replace([np.inf, -np.inf], 0, inplace=True)
sp500["cumulative_returns"] = (1+sp500["weighted_portfolio_return"]).cumprod()

# Create stock return dataframe and return
def rebalanced_portfolio(tickers, weights, start, end, rebalance_frequency):
   
    ticker_weights = dict(zip(tickers, weights))
    
    ## Download data for all tickers
    stock_data = yf.download(tickers,start=start, end=end,interval=rebalance_frequency)
    stock_data = stock_data.dropna()     
        
    ## Calculate log_returns for all assets
    stock_returns = pd.DataFrame(index = sp500.index)
    
    for ticker in tickers:
        stock_returns[ticker] = np.diff(np.log(np.concatenate((
            [0],stock_data['Adj Close'][ticker]))))
        
    stock_returns.replace([np.inf, -np.inf], 0, inplace=True)
    stock_returns = stock_returns.dropna()
    
    ## Calculate weighted returns for tickers and add portfolio returns to stock returns
    weighted_returns = pd.DataFrame(index = sp500.index)
    for ticker in tickers:
        weighted_returns[f"{ticker} {ticker_weights[ticker]}"] = stock_returns[ticker]*ticker_weights[ticker]
    
    weighted_returns["weighted_portfolio_return"] = weighted_returns.iloc[:,0:len(weighted_returns.columns)
                                                                          ].sum(axis = 1)
    
    stock_returns["weighted_portfolio_return"] = weighted_returns["weighted_portfolio_return"]
    
    ## Calculate cumulative returns for portfolio
    stock_returns["cumulative_returns"] = (1+stock_returns["weighted_portfolio_return"]).cumprod()

    return stock_returns


# Add data for analysis
def CAGR(data):
    df = data.copy()
    trading_period = trading_period_for_vol_calc[rebalance_frequency]
    n = len(df)/ trading_period
    cagr = (df['cumulative_returns'][len(df)-1])**(1/n) - 1
    return cagr

def volatility(data):
    df = data.copy()
    trading_period = trading_period_for_vol_calc[rebalance_frequency]
    vol = df['weighted_portfolio_return'].std() * np.sqrt(trading_period)
    return vol

def sharpe_ratio(data, rf):
    df = data.copy()
    sharpe = (CAGR(df) - rf)/ volatility(df)
    return sharpe 

def maximum_drawdown(data):
    df = data.copy()
    df['cumulative_returns'] =  (1 + df['weighted_portfolio_return']).cumprod()
    df['cumulative_max'] = df['cumulative_returns'].cummax()
    df['drawdown'] = df['cumulative_max'] - df['cumulative_returns']
    df['drawdown_pct'] = df['drawdown'] / df['cumulative_max']
    max_dd = df['drawdown_pct'].max()
    return max_dd


## Calcs + charts
rebalanced_portfolio = rebalanced_portfolio(tickers, weights, start, end, rebalance_frequency)

print("Rebalanced Portfolio Performance")
print("Portfolio Assets and Weights:")
print(str(dict(zip(tickers, weights))) + f" with a {rebalance_frequency} rebal period")
print("CAGR: " + str(CAGR(rebalanced_portfolio)))
print("Sharpe Ratio: " + str(sharpe_ratio(rebalanced_portfolio, 0.03)))
print("Maximum Drawdown: " + str(maximum_drawdown(rebalanced_portfolio) ))

print("\n")

print("S&P500 Index Performance")
print("CAGR: " + str(CAGR(sp500)))
print("Sharpe Ratio: " + str(sharpe_ratio(sp500, rf)))
print("Maximum Drawdown: " + str(maximum_drawdown(sp500) ))

fig, ax = plt.subplots()
ax = plt.gca()
plt.plot(rebalanced_portfolio["cumulative_returns"])
plt.plot(sp500["cumulative_returns"])
plt.title("S&P500 Index Return vs Rebalancing Strategy Return")
plt.ylabel("cumulative return")
plt.xlabel(f"Period = {rebalance_frequency}")
ax.legend([f"Strategy Return % for period: {(rebalanced_portfolio['cumulative_returns'][-1]-1)*100:.2f}",f"Index Return % for period: {(sp500['cumulative_returns'][-1]-1)*100:.2f}"])
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval = 3))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%y'))
plt.xticks(rotation = 70)

# -*- coding: utf-8 -*-

"""
Created on Thu Oct 28 11:48:08 2021

@author: ivelu
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib

tickers = ["SPY", "TLT", "GLD"]
weights = [0.4, 0.3, 0.3]
start = '2010-01-1'
end = '2021-09-30'
rebalance_frequency = '1wk'
trading_period_for_vol_calc = {"1d":252, "1wk":52, "1mo":12, "3mo":4, "1y":1}
rf = 0
ticker_weights = dict(zip(tickers, weights))

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
       
    ## Download data for all tickers
    stock_data = yf.download(tickers,start=start, end=end,interval=rebalance_frequency)
    stock_data = stock_data.dropna()     
        
    ## Calculate log_returns for all assets. DECIDE LOG VS PCT CHANGE
    stock_returns = pd.DataFrame(index = sp500.index)
    
    for ticker in tickers:
        stock_returns[ticker] = np.diff(np.log(np.concatenate((
            [0],stock_data['Adj Close'][ticker]))))
        
    stock_returns.replace([np.inf, -np.inf], 0, inplace=True)
    stock_returns = stock_returns.dropna()
    
    ## Create asset returns in order to build covariance matrix and optimization 
    asset_returns = stock_returns.copy()
    
    ## Calculate weighted portfolio returns for tickers and add portfolio returns to stock returns
    stock_returns["weighted_portfolio_return"] = stock_returns.dot(weights)
    
    ## Calculate cumulative returns for portfolio
    stock_returns["cumulative_returns"] = (1+stock_returns["weighted_portfolio_return"]).cumprod()
    
    return stock_returns, asset_returns


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

def variance_matrix(asset_returns, trading_period_for_vol_calc):
    var_matrix = asset_returns.cov()*trading_period_for_vol_calc[rebalance_frequency]
    return var_matrix


def mpt_optimization(tickers, asset_returns, number_of_sims):
    print(f"Optimizing portfolio of {tickers} and {number_of_sims} simulations")
    ind_asset_exp_return = pd.DataFrame(asset_returns.mean()*trading_period_for_vol_calc[rebalance_frequency])
    ind_asset_exp_return.transpose()
    
    port_returns = []
    port_volatility = []
    port_weights = []
    num_assets = len(tickers)
    var_matrix = variance_matrix(asset_returns, trading_period_for_vol_calc)
    
    for i in range(10000):
        nweights = np.random.random(num_assets)
        #normalize to sum to 1
        nweights = nweights/np.sum(nweights)
        port_weights.append(nweights)
        returns = nweights@ind_asset_exp_return
        port_returns.append(returns)
        
        #Calc port variance
        var = var_matrix.mul(nweights, axis = 0).mul(nweights, axis = 1).sum().sum()
        sd = np.sqrt(var)
        port_volatility.append(sd)
    
    ## convert port returns from pandas series to list
    port_returns = list(map(lambda x:float(x), port_returns))
    
    #dict of returns and vol
    data = {'Returns':port_returns, 'Volatility':port_volatility}
    
    for counter, symbol in enumerate(asset_returns.columns):
        data[symbol + ' weight'] = [w[counter] for w in port_weights]
    
    #convert dict to dataframe
    possible_portfolios = pd.DataFrame(data)
    
    #Calculate MIN VOL portfolio using idxmin()
    min_vol_portfolio = possible_portfolios.iloc[possible_portfolios['Volatility'].idxmin()]
    print(f"Portfolio with minimum vol for assets {tickers} is:")
    print(min_vol_portfolio)
    
    #Calculate portfolio with highest Sharpe Ratio
    max_sharpe_port = possible_portfolios.iloc[
        ((possible_portfolios['Returns'] - rf)/possible_portfolios['Volatility']).idxmax()]
    print(f"Portfolio with max Sharpe for assets {tickers} is:")
    print(max_sharpe_port)
    
    #Plot portfolios
    plt.subplots(figsize=(8,8))
    plt.scatter(possible_portfolios['Volatility'],possible_portfolios['Returns'],
                marker = 'o', s=10, alpha=0.3, color = "blue" )
    plt.scatter(min_vol_portfolio[1], min_vol_portfolio[0], color = 'y', marker = '*', s=500)
    plt.scatter(max_sharpe_port[1], max_sharpe_port[0], color = 'green', marker = '*', s=500)
    plt.xlabel("Volatility")
    plt.ylabel("Expected Returns")
    plt.suptitle("MPT Optimization. Yellow Star = Min. Vol, Green Star = Max Sharpe")
    plt.title(f"Assets: {tickers}")    
    
    return min_vol_portfolio, max_sharpe_port

## Calcs + charts
rebalanced_portfolio,asset_returns = rebalanced_portfolio(tickers, weights, start, end, rebalance_frequency)

def strategy_return_plots(rebalanced_portfolio, tickers, weights):
    print("Calculating portfolio performance...")
    
    print("\n")
    
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

## Optimize
## mpt_optimization(tickers, asset_returns, 10000)
## strategy_return_plots(rebalanced_portfolio, tickers, weights)

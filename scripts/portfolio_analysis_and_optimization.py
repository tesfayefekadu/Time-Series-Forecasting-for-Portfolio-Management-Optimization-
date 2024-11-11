import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from tabulate import tabulate
import yfinance as yf


def calculate_portfolio_return(weights, returns):
 
    return np.sum(returns.mean() * weights) * 252

def calculate_portfolio_volatility(weights, returns):
 
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def portfolio_statistics(weights, returns, risk_free_rate=0.01):
    
    port_return = calculate_portfolio_return(weights, returns)
    port_volatility = calculate_portfolio_volatility(weights, returns)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def generate_random_portfolios(num_portfolios, returns, risk_free_rate=0.01):
   
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    weight_matrix = np.zeros((num_portfolios, num_assets))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weight_matrix[i, :] = weights
        port_return, port_volatility, sharpe_ratio = portfolio_statistics(weights, returns, risk_free_rate)
        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio

    return results, weight_matrix

def plot_efficient_frontier(results):
   
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.show()

def optimize_portfolio(returns, risk_free_rate=0.01):
  
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def neg_sharpe_ratio(weights, returns, risk_free_rate):
      
        return -portfolio_statistics(weights, returns, risk_free_rate)[2]

    initial_guess = num_assets * [1. / num_assets]
    result = minimize(neg_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def display_optimal_portfolio(weights, returns):
   
    port_return, port_volatility, sharpe_ratio = portfolio_statistics(weights, returns)
    print(f"Optimal Portfolio Return: {port_return:.2f}")
    print(f"Optimal Portfolio Volatility: {port_volatility:.2f}")
    print(f"Optimal Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Optimal Weights: {weights}")

def create_portfolio_dataframe(tsla_forecast):
    df = pd.DataFrame({
        'TSLA': tsla_forecast['Close']
        
    })
    return df


def calculate_annual_returns(df):
   
    daily_returns = df.pct_change().dropna()  
    annual_returns = daily_returns.mean() * 252 
    return annual_returns, daily_returns  

def calculate_covariance_matrix(daily_returns):
    return daily_returns.cov() * 252  
def calculate_portfolio_performance(weights, annual_returns, cov_matrix):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe(weights, annual_returns, cov_matrix, risk_free_rate=0.02):
    port_return, port_volatility = calculate_portfolio_performance(weights, annual_returns, cov_matrix)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return -sharpe_ratio


def optimize_portfolio(annual_returns, cov_matrix, initial_weights, risk_free_rate=0.02):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  
    bounds = tuple((0, 1) for _ in range(len(initial_weights)))

    optimized = minimize(
        negative_sharpe, initial_weights,
        args=(annual_returns, cov_matrix, risk_free_rate),
        bounds=bounds, constraints=constraints
    )
    return optimized.x  


def calculate_var(daily_returns, confidence_level=0.95):
    tsla_mean_return = daily_returns['TSLA'].mean()
    tsla_std_dev = daily_returns['TSLA'].std()
    VaR_TSLA = norm.ppf(1 - confidence_level) * tsla_std_dev - tsla_mean_return
    return VaR_TSLA


def plot_cumulative_returns(daily_returns, optimal_weights):
    cumulative_returns = (1 + daily_returns).cumprod()
    portfolio_cumulative_return = cumulative_returns.dot(optimal_weights)

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_cumulative_return, label="Optimized Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Performance")
    plt.legend()
    plt.show()


def summarize_portfolio_performance(optimal_weights, annual_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(optimal_weights, annual_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

 
    summary = {
        "Expected Annual Return": portfolio_return,
        "Annualized Volatility": portfolio_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Optimal Weights": optimal_weights
    }

    summary_table = [(k, v) for k, v in summary.items()]
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    return summary
def calculate_daily_returns(data):
    """Calculate daily returns for each ticker."""
    returns = data.pct_change().dropna()
    return returns

def clean_data(data):
    """Handle missing values by forward filling and then dropping any remaining NaNs."""
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data
def download_data(tickers, start_date, end_date):
    """Download adjusted close price data for specified tickers and date range."""
    data = yf.download(tickers, start=start_date, end=end_date)
    data = data['Adj Close']
    data.columns = tickers
    return data
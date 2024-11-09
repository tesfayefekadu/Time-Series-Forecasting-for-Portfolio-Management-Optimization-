import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

def fetch_data(assets, start_date, end_date):
    """Fetches historical data from YFinance."""
    data = yf.download(assets, start=start_date, end=end_date, group_by="ticker")
    return data

def clean_data(data):
    """Cleans the dataset by handling missing values."""
    data.dropna(inplace=True)
    return data

def save_to_csv(data, filename):
    """Saves the DataFrame to a CSV file."""
    data.to_csv(filename)

def display_statistics(data):
    """Displays basic statistics and data information."""
    print(data.describe())
    print(data.info())

def normalize_data(data, assets):
    """Normalizes the closing prices using StandardScaler."""
    scaler = StandardScaler()
    data_scaled = data.copy()
    for asset in assets:
        data_scaled[(asset, 'Close')] = scaler.fit_transform(data_scaled[(asset, 'Close')].values.reshape(-1, 1))
    return data_scaled

def plot_closing_prices(data, assets):
    """Plots closing prices over time for each asset."""
    plt.figure(figsize=(14, 8))
    for asset in assets:
        plt.plot(data[(asset, 'Close')], label=f"{asset} Close Price")
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def calculate_daily_pct_change(data, assets):
    """Calculates the daily percentage change for each asset."""
    for asset in assets:
        data[(asset, 'Daily_Pct_Change')] = data[(asset, 'Close')].pct_change() * 100
    return data

def plot_daily_pct_change(data, assets):
    """Plots the daily percentage change over time for each asset."""
    plt.figure(figsize=(14, 8))
    for asset in assets:
        plt.plot(data[(asset, 'Daily_Pct_Change')], label=f"{asset} Daily % Change")
    plt.title("Daily Percentage Change Over Time")
    plt.xlabel("Date")
    plt.ylabel("Daily % Change")
    plt.legend()
    plt.show()

def calculate_rolling_stats(data, assets, window=20):
    """Calculates rolling mean and standard deviation for closing prices."""
    for asset in assets:
        data[(asset, 'Rolling_Mean')] = data[(asset, 'Close')].rolling(window=window).mean()
        data[(asset, 'Rolling_STD')] = data[(asset, 'Close')].rolling(window=window).std()
    return data

def plot_rolling_std(data, assets):
    """Plots rolling standard deviation (volatility) over time for each asset."""
    plt.figure(figsize=(14, 8))
    for asset in assets:
        plt.plot(data[(asset, 'Rolling_STD')], label=f"{asset} Rolling Std Dev")
    plt.title("Rolling Standard Deviation (Volatility) Over Time")
    plt.xlabel("Date")
    plt.ylabel("Rolling Std Dev")
    plt.legend()
    plt.show()

def detect_outliers(data, assets, threshold=3):
    """Detects outliers in daily percentage change using Z-score method."""
    outliers = {}
    for asset in assets:
        # Calculate Z-scores and align the index
        daily_pct_change = data[(asset, 'Daily_Pct_Change')].dropna()
        z_scores = pd.Series(stats.zscore(daily_pct_change), index=daily_pct_change.index)
        
        # Identify outliers based on the threshold
        outliers[asset] = daily_pct_change[(z_scores > threshold) | (z_scores < -threshold)]
    return outliers


def decompose_seasonality(data, asset):
    """Performs seasonal decomposition on the closing price of a specified asset."""
    ts = data[(asset, 'Close')].dropna()
    decomposition = seasonal_decompose(ts, model='multiplicative', period=365)
    decomposition.plot()
    plt.show()
    return decomposition

def calculate_var(data, assets, confidence_level=0.05):
    """Calculates Value at Risk (VaR) for each asset."""
    VaR = {}
    for asset in assets:
        VaR[asset] = data[(asset, 'Daily_Pct_Change')].quantile(confidence_level)
    return VaR

def calculate_sharpe_ratio(data, assets, risk_free_rate=0.02, trading_days=252):
    """Calculates the Sharpe Ratio for each asset."""
    sharpe_ratios = {}
    for asset in assets:
        mean_return = data[(asset, 'Daily_Pct_Change')].mean() * trading_days
        std_dev = data[(asset, 'Daily_Pct_Change')].std() * (trading_days ** 0.5)
        sharpe_ratios[asset] = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratios

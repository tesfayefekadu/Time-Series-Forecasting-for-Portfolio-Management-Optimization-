# Time-Series-Forecasting-for-Portfolio-Management-Optimization-

## Project Overview
This project focuses on analyzing historical data and forecasting stock prices using machine learning and deep learning techniques. The primary assets analyzed are Tesla (TSLA), S&P 500 ETF (SPY), and Vanguard Total Bond Market ETF (BND). We use various forecasting models (ARIMA, SARIMA, and LSTM) to predict future prices and apply portfolio optimization methods to maximize returns and minimize risk. The project also includes generating valuable financial insights through metrics like Sharpe Ratio and Value at Risk (VaR).

### Goals
Forecast Future Stock Prices: Predict future values for TSLA, SPY, and BND using time series models.
Optimize Portfolio: Identify the optimal allocation of assets to maximize returns while balancing risk.
Calculate Financial Metrics: Assess portfolio performance through metrics like annualized return, volatility, Sharpe Ratio, and Value at Risk.

## Project Structure
The main components of the project are:

#### Data Collection and Preparation: 
Fetching historical data from YFinance for TSLA, SPY, and BND, covering January 1, 2015, to October 31, 2024.
#### Forecasting Models: 
Applying ARIMA, SARIMA, and LSTM models for forecasting, with evaluation metrics including MAE, RMSE, and MAPE.
#### Portfolio Optimization: 
Calculating optimal weights to maximize returns and minimize risk through metrics like Sharpe Ratio and Value at Risk.


project_root/
├── data/                  # Folder for storing datasets
├── notebooks/             # Jupyter notebooks for exploration and model training
├── models/                # Saved trained models (pickle files)
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
└── scripts/               # Python scripts for data preprocessing, training, and evaluation

## Results and Conclusion
This project demonstrates that LSTM outperformed traditional ARIMA and SARIMA models in forecasting stock prices. Portfolio analysis revealed the portfolio's risk profile, allowing for a more informed approach to asset allocation based on the Sharpe Ratio and VaR metrics. Future improvements could include additional asset classes or further tuning of deep learning models to enhance predictive accuracy.
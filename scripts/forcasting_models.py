import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Function to split data into training and testing sets
def split_data(data, split_ratio=0.8):
    split_idx = int(len(data) * split_ratio)
    train, test = data[:split_idx], data[split_idx:]
    return train, test

# Function to train an ARIMA or SARIMA model
def train_sarima(train_data, seasonal=False, seasonal_period=12):
    if seasonal:
        model = auto_arima(train_data, seasonal=True, m=seasonal_period, trace=True, stepwise=True)
    else:
        model = auto_arima(train_data, seasonal=False, trace=True, stepwise=True)
    return model

# Function to train an LSTM model
def train_lstm(train_data, n_input=10, n_neurons=50, epochs=20, batch_size=1):
    train_generator = TimeseriesGenerator(train_data.values, train_data.values, length=n_input, batch_size=batch_size)
    model = Sequential([
        LSTM(n_neurons, activation='relu', input_shape=(n_input, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_generator, epochs=epochs, verbose=1)
    return model

# Function to forecast using SARIMA
def forecast_sarima(model, steps):
    forecast = model.predict(n_periods=steps)
    return forecast

# Function to forecast using LSTM
def forecast_lstm(model, data, n_input=10, steps=10):
    forecast = []
    input_seq = data[-n_input:].values.reshape((1, n_input, 1))
    for _ in range(steps):
        prediction = model.predict(input_seq, verbose=0)
        forecast.append(prediction[0][0])
        input_seq = np.append(input_seq[:, 1:, :], [[prediction]], axis=1)
    return forecast

# Function to calculate evaluation metrics
def evaluate_model(test, forecast):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    return mae, rmse, mape

# Example usage in a Jupyter notebook:
# import time_series_forecasting as tsf
# data = ... # Load your data here
# train, test = tsf.split_data(data['Close'])
# sarima_model = tsf.train_sarima(train, seasonal=True, seasonal_period=12)
# sarima_forecast = tsf.forecast_sarima(sarima_model, steps=len(test))
# mae, rmse, mape = tsf.evaluate_model(test, sarima_forecast)
# print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")

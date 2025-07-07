#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:23:12 2025

@author: charliewhite
"""

import numpy as np
import pandas as pd
import yfinance as yf
import random
import talib
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler

random.seed(50)

# Fetch stock data
tickers = ['NVDA']
df = yf.download(tickers[0], start='2015-01-01', end='2025-01-01', auto_adjust=True)

def plot_bollinger_bands(data, stock, window):
    plt.figure(figsize=(12, 6))
    
    # Plot stock closing price
    plt.plot(data['Close'], label='Close Price', color='green')

    # Calculating 20-day rolling average
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=window).std()
    data['Upper_band'] = data['MA20'] + 2 * data['Rolling_Std']
    data['Lower_band'] = data['MA20'] - 2 * data['Rolling_Std']

    # Drop NaN values after calculations
    data = data.dropna()

    # Plotting
    plt.plot(data['MA20'], label=f'{window}-Day Rolling Average', color='white')
    plt.plot(data['Upper_band'], label='Upper Bollinger Band', linestyle='--', color='orange')
    plt.plot(data['Lower_band'], label='Lower Bollinger Band', linestyle='--', color='orange')

    # Improve visibility
    plt.xlabel('Date', color='white')
    plt.ylabel('Price', color='white')
    plt.title(f'Stock Data of {stock} with Bollinger Bands', color='white')
    plt.legend()
    
    plt.show()

plot_bollinger_bands(df , tickers[0], 50)

#%% Establish variables and refine df

# Convert index to datetime
df.index = pd.to_datetime(df.index)

# Feature Engineering
df["High_Low_Diff"] = df["High"] - df["Low"]
df["Open_Close_Diff"] = df["Close"] - df["Open"]
df["Volatility"] = df["Close"].rolling(window=5).std()

# Ensure df["Close"].values is a 1D array and clean NaN/infinite values
close_prices = df["Close"].values
close_prices = close_prices[~np.isnan(close_prices)]  # Remove NaN values

# Calculate RSI
df["RSI"] = talib.RSI(close_prices, timeperiod=9)

# Calculate Williams %R
high_prices = df["High"].to_numpy(dtype=float)
low_prices = df["Low"].to_numpy(dtype=float)
close_prices = df["Close"].to_numpy(dtype=float)

# Clean NaN/infinite values for Williams %R
high_prices = high_prices[~np.isnan(high_prices)]
low_prices = low_prices[~np.isnan(low_prices)]
close_prices = close_prices[~np.isnan(close_prices)]

df["Williams_R"] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=7)

#Calculate price increase based on forward returns
df["Future_Return"] = df["Close"].pct_change().shift(-1)
df["Price_Increase"] = np.where(df["Future_Return"] > 0, 1, 0)

# Add indicators
df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
df["ATR"] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
df["ADX"] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)

#Drop Nan values from rolling calculations
df = df.dropna()

#%%

# Splitting dataset into training and testing sets
features = df[["High_Low_Diff", "Open_Close_Diff", "Volatility", "RSI", "Williams_R"]]
target = df["Price_Increase"]

split_idx = int(len(df) * 0.8)
X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

# Standardize the feature set
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Neural Network Model
model = Sequential([
    Dense(128, activation="relu", kernel_initializer="he_uniform", input_dim=X_train.shape[1]),
    Dense(units=128, activation="relu", kernel_initializer="he_uniform"),
    Dense(units=1, activation="sigmoid", kernel_initializer="glorot_uniform")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train_scaled, y_train, batch_size=10, epochs=100, verbose=1)

# Predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Store predictions in the original dataset
df.loc[df.index[split_idx:], "Predicted_Signal"] = y_pred.flatten()
df = df.dropna()  # Drop NaN values

# Compute returns
df["Daily_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df["Daily_Return"] = df["Daily_Return"].shift(-1)  # Align returns

# Compute strategy returns (buy if predicted up, short if predicted down)
df["Strategy_Return"] = np.where(df["Predicted_Signal"] == 1, df["Daily_Return"], -df["Daily_Return"])

# Compute cumulative returns
df["Cumulative_Market_Return"] = np.cumsum(df["Daily_Return"])
df["Cumulative_Strategy_Return"] = np.cumsum(df["Strategy_Return"])

#%%

# Plot Market vs Strategy Returns
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Cumulative_Market_Return"], color="red", label="Market Returns")
plt.plot(df.index, df["Cumulative_Strategy_Return"], color="green", label="Strategy Returns")

plt.title(f"Market vs Strategy Returns ({tickers[0]})", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

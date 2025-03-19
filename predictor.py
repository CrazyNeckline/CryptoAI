# predictor.py
# This file contains functions for calculating technical indicators and making trade predictions.

import pandas as pd
import numpy as np
import ta

def calculate_indicators(df):
    """
    Calculate technical indicators (SMA, RSI, ATR) for the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume).
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for SMA_20, RSI, and ATR.
    """
    # Calculate 20-period Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate Relative Strength Index (RSI) with a 14-period lookback
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # Calculate Average True Range (ATR) with a 14-period lookback
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()
    
    return df

def predict_signal(models, timeframe, latest):
    """
    Predict the trading signal (Long, Short, or None) using the trained models.
    
    Args:
        models (dict): Dictionary of trained models for each timeframe (direction, SL, TP).
        timeframe (str): Timeframe ('1m', '1h', '1d').
        latest (pd.Series): Latest row of the DataFrame with OHLCV and indicator data.
    
    Returns:
        tuple: (signal, entry_price, sl_price, tp_price)
            - signal (str): 'Long', 'Short', or None.
            - entry_price (float): Entry price for the trade.
            - sl_price (float): Stop-loss price.
            - tp_price (float): Take-profit price.
    """
    # Extract features for prediction (must match the 8 features used during training)
    features = np.array([[
        latest['Open'],
        latest['High'],
        latest['Low'],
        latest['Close'],
        latest['Volume'],
        latest['SMA_20'] if not pd.isna(latest['SMA_20']) else latest['Close'],
        latest['RSI'] if not pd.isna(latest['RSI']) else 50.0,
        latest['ATR'] if not pd.isna(latest['ATR']) else 0.0
    ]])
    
    # Predict direction (Long, Short, or None)
    direction_model = models[timeframe]['direction']
    direction_pred = direction_model.predict(features)[0]
    signal = 'Long' if direction_pred == 1 else 'Short' if direction_pred == -1 else None
    
    if signal:
        # Predict stop-loss and take-profit distances
        sl_model = models[timeframe]['sl']
        tp_model = models[timeframe]['tp']
        sl_distance = sl_model.predict(features)[0]
        tp_distance = tp_model.predict(features)[0]
        
        # Calculate entry, stop-loss, and take-profit prices
        entry_price = latest['Close']
        if signal == 'Long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # Short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return signal, entry_price, sl_price, tp_price
    else:
        return None, None, None, None
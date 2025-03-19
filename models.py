# CryptoAI/models.py
# This file contains functions to train machine learning models for the CryptoAI application.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_models(df):
    """
    Train machine learning models for each timeframe.
    
    Args:
        df (dict): Dictionary of DataFrames for each timeframe.
    
    Returns:
        dict: Dictionary of trained models for each timeframe.
    """
    models = {}
    for timeframe in ['1m', '1h', '1d']:
        print(f"Training models for {timeframe}...")
        data = df[timeframe].copy()
        
        # Calculate indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = compute_rsi(data['Close'], 14)
        data['ATR'] = compute_atr(data, 14)
        
        # Create target variables
        # Direction: 1 for Long (price increases), -1 for Short (price decreases), 0 for no trade
        data['Future_Close'] = data['Close'].shift(-5)
        data['Direction'] = 0
        # Lower threshold to 0.3% to increase likelihood of signals
        data.loc[data['Future_Close'] > data['Close'] * 1.003, 'Direction'] = 1  # Long if price increases by 0.3%
        data.loc[data['Future_Close'] < data['Close'] * 0.997, 'Direction'] = -1  # Short if price decreases by 0.3%
        
        # SL and TP (in dollars, based on ATR)
        data['SL'] = data['ATR'] * 2  # Example: SL is 2x ATR
        data['TP'] = data['ATR'] * 3  # Example: TP is 3x ATR
        
        # Drop rows with NaN values
        data = data.dropna()
        print(f"Data for {timeframe} after preprocessing: {len(data)} rows")
        print(f"Direction distribution for {timeframe}: {data['Direction'].value_counts().to_dict()}")
        
        if len(data) < 50:  # Ensure enough data for training
            print(f"Not enough data for {timeframe} to train models.")
            models[timeframe] = {'direction': None, 'sl': None, 'tp': None}
            continue
        
        # Features and targets
        features = data[['SMA_20', 'RSI', 'ATR']]
        direction_target = data['Direction']
        sl_target = data['SL']
        tp_target = data['TP']
        
        # Split data
        X_train, X_test, y_train_direction, y_test_direction = train_test_split(features, direction_target, test_size=0.2, random_state=42)
        _, _, y_train_sl, y_test_sl = train_test_split(features, sl_target, test_size=0.2, random_state=42)
        _, _, y_train_tp, y_test_tp = train_test_split(features, tp_target, test_size=0.2, random_state=42)
        
        # Train direction model (classification)
        print(f"Training direction model for {timeframe}...")
        direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        direction_model.fit(X_train, y_train_direction)
        direction_score = direction_model.score(X_test, y_test_direction)
        print(f"Direction model score for {timeframe}: {direction_score}")
        # Log some predictions to understand model behavior
        sample_predictions = direction_model.predict(X_test[:5])
        sample_proba = direction_model.predict_proba(X_test[:5])
        print(f"Sample predictions for {timeframe}: {sample_predictions}")
        print(f"Sample probabilities for {timeframe} ([-1, 0, 1]): {sample_proba}")
        
        # Train SL model (regression)
        print(f"Training SL model for {timeframe}...")
        sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
        sl_model.fit(X_train, y_train_sl)
        sl_score = sl_model.score(X_test, y_test_sl)
        print(f"SL model score for {timeframe}: {sl_score}")
        
        # Train TP model (regression)
        print(f"Training TP model for {timeframe}...")
        tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        tp_model.fit(X_train, y_train_tp)
        tp_score = tp_model.score(X_test, y_test_tp)
        print(f"TP model score for {timeframe}: {tp_score}")
        
        models[timeframe] = {
            'direction': direction_model,
            'sl': sl_model,
            'tp': tp_model
        }
    
    print("Training complete!")
    return models

def compute_rsi(data, periods=14):
    """
    Compute the Relative Strength Index (RSI) for a given data series.
    
    Args:
        data (pd.Series): Series of closing prices.
        periods (int): Number of periods for RSI calculation.
    
    Returns:
        pd.Series: RSI values.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, periods=14):
    """
    Compute the Average True Range (ATR) for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with High, Low, and Close columns.
        periods (int): Number of periods for ATR calculation.
    
    Returns:
        pd.Series: ATR values.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=periods).mean()
    return atr
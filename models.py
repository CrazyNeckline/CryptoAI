# models.py
# This file handles the training and prediction of machine learning models for trading signals.

import pandas as np
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_models(df):
    """
    Train RandomForest models for direction, stop-loss (SL), and take-profit (TP) predictions for each timeframe.
    
    Args:
        df (dict): Dictionary of DataFrames for each timeframe ('1m', '1h', '1d') with historical data.
    
    Returns:
        dict: Dictionary of trained models for each timeframe, with keys 'direction', 'sl', and 'tp'.
    """
    models = {}
    for timeframe in ['1m', '1h', '1d']:
        # Prepare features and target for direction prediction
        X = df[timeframe][['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'ATR']]
        y_direction = np.where(df[timeframe]['Close'].shift(-1) > df[timeframe]['Close'], 1, 0)[:-1]
        X = X[:-1]
        X_train, X_test, y_dir_train, y_dir_test = train_test_split(X, y_direction, test_size=0.2, shuffle=False)
        X_train_sl, X_test_sl = X_train, X_test
        
        # Prepare targets for SL and TP regression
        y_sl_train = np.abs(X_train['Close'] - X_train['Low']) * (0.5 if timeframe == '1m' else 2 if timeframe == '1h' else 5)
        y_tp_train = np.abs(X_train['High'] - X_train['Close']) * (0.5 if timeframe == '1m' else 2 if timeframe == '1h' else 5)

        # Train direction model (classification)
        print(f"Training direction model for {timeframe}...")
        models[timeframe] = {'direction': RandomForestClassifier(n_estimators=100, random_state=42)}
        models[timeframe]['direction'].fit(X_train, y_dir_train)
        
        # Train SL model (regression)
        print(f"Training SL model for {timeframe}...")
        models[timeframe]['sl'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models[timeframe]['sl'].fit(X_train_sl, y_sl_train)
        
        # Train TP model (regression)
        print(f"Training TP model for {timeframe}...")
        models[timeframe]['tp'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models[timeframe]['tp'].fit(X_train_sl, y_tp_train)
    
    print("Training complete!")
    return models

def predict_signal(models, timeframe, latest):
    """
    Predict trading signal, stop-loss, and take-profit using trained models for a given timeframe.
    
    Args:
        models (dict): Dictionary of trained models for each timeframe.
        timeframe (str): Timeframe identifier ('1m', '1h', '1d').
        latest (pd.Series): Latest candle data with features for prediction.
    
    Returns:
        tuple: (signal, entry_price, sl_price, tp_price)
            - signal (str): 'Long' or 'Short'.
            - entry_price (float): Entry price for the trade.
            - sl_price (float): Stop-loss price.
            - tp_price (float): Take-profit price.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'ATR']
    X = latest[features].values.reshape(1, -1)
    signal = 'Long' if models[timeframe]['direction'].predict_proba(X)[0][1] > 0.55 else 'Short'  # Tighter signal threshold
    sl_distance = models[timeframe]['sl'].predict(X)[0]  # Scaled by timeframe in training
    tp_distance = models[timeframe]['tp'].predict(X)[0]  # Scaled by timeframe in training
    entry_price = latest['Close']
    sl_price = round(entry_price - sl_distance if signal == 'Long' else entry_price + sl_distance, 2)
    tp_price = round(entry_price + tp_distance if signal == 'Long' else entry_price - tp_distance, 2)
    return signal, entry_price, sl_price, tp_price
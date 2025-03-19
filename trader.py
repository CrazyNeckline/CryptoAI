# trader.py
# This file handles WebSocket messages and trading logic for the CryptoAI application.

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from CryptoAI.utils import format_dollar, save_trade_history
from CryptoAI.state import trade_history, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length, indicators_at_entry, current_prediction

def process_kline(msg, timeframe, models, df, trade_history_file, local_tz, use_rule_based=False):
    """
    Process WebSocket kline messages and execute trading logic.
    
    Args:
        msg (dict): WebSocket message containing kline data.
        timeframe (str): Timeframe of the kline ('1m', '1h', '1d').
        models (dict): Trained models for direction, SL, and TP.
        df (dict): DataFrames for each timeframe.
        trade_history_file (str): Path to the trade history file.
        local_tz (str): Local timezone for datetime conversion (e.g., 'Europe/Berlin').
        use_rule_based (bool): If True, use rule-based trading instead of model prediction.
    """
    kline = msg['k']
    is_candle_closed = kline['x']
    close_price = float(kline['c'])
    timestamp = int(kline['t'])
    print(f"Candle status - Closed: {is_candle_closed}, Time: {timestamp}, Close: {close_price:.8f}")

    # Update the DataFrame with the latest candle
    new_row = pd.DataFrame({
        'Date': [pd.to_datetime(timestamp, unit='ms', utc=True)],
        'Open': [float(kline['o'])],
        'High': [float(kline['h'])],
        'Low': [float(kline['l'])],
        'Close': [close_price],
        'Volume': [float(kline['v'])]
    })
    df[timeframe] = pd.concat([df[timeframe], new_row], ignore_index=True)
    df[timeframe] = df[timeframe].tail(200)  # Keep only the last 200 rows
    print(f"Current state for {timeframe} - Position: {position[timeframe]}, Entry: {entry_price[timeframe]}, SL: {sl_price[timeframe]}, TP: {tp_price[timeframe]}")
    print(f"df[{timeframe}] size after update: {len(df[timeframe])} rows")
    print(f"Raw df[{timeframe}] tail: {df[timeframe][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(1).to_dict()}")

    # Update current BTC price
    current_btc_price = close_price
    print(f"Updated current_btc_price: ${format_dollar(current_btc_price)}")

    # Calculate indicators
    df[timeframe]['SMA_20'] = df[timeframe]['Close'].rolling(window=20).mean()
    df[timeframe]['RSI'] = compute_rsi(df[timeframe]['Close'], 14)
    df[timeframe]['ATR'] = compute_atr(df[timeframe], 14)
    print(f"df[{timeframe}] size after indicators: {len(df[timeframe])} rows")
    print(f"Indicators tail for {timeframe}: {df[timeframe][['SMA_20', 'RSI', 'ATR']].tail(1).to_dict()}")

    if not is_candle_closed:
        print(f"Candle not closed yet for {timeframe}, skipping trade logic...")
        return

    # Prepare features for model prediction
    features = df[timeframe][['SMA_20', 'RSI', 'ATR']].tail(1).copy()
    print(f"Features for prediction in {timeframe}: {features.to_dict()}")
    if features.isna().any().any():
        print(f"Features contain NaN values for {timeframe}, skipping prediction...")
        return

    signal = None
    entry = None
    sl = None
    tp = None
    if not use_rule_based:
        # Use model-based prediction
        features_array = features.values  # Convert to numpy array
        print(f"Features array for {timeframe}: {features_array}")
        direction_model = models[timeframe]['direction']
        sl_model = models[timeframe]['sl']
        tp_model = models[timeframe]['tp']

        try:
            # Predict direction
            direction = direction_model.predict(features_array)[0]
            # Add probability for more insight
            direction_proba = direction_model.predict_proba(features_array)[0]
            print(f"Direction probabilities for {timeframe} ([-1, 0, 1]): {direction_proba}")
            signal = 'Long' if direction == 1 else 'Short' if direction == -1 else None
            print(f"Direction predicted for {timeframe}: {direction}, Signal: {signal}")

            if signal:
                # Predict SL and TP
                sl = sl_model.predict(features_array)[0]
                tp = tp_model.predict(features_array)[0]
                entry = close_price
                current_prediction[timeframe] = {
                    'signal': signal,
                    'entry': entry,
                    'sl': entry - sl if signal == 'Long' else entry + sl,
                    'tp': entry + tp if signal == 'Long' else entry - tp,
                    'datetime': datetime.now(pytz.timezone(local_tz)).strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"SL predicted: {sl}, TP predicted: {tp}")
            else:
                current_prediction[timeframe] = {}
            print(f"Model prediction for {timeframe} - Signal: {signal}, Entry: {entry if signal else None}, SL: {current_prediction[timeframe].get('sl')}, TP: {current_prediction[timeframe].get('tp')}")
        except Exception as e:
            print(f"Error in model prediction for {timeframe}: {e}")
            return
    else:
        # Use rule-based trading (RSI-based)
        rsi = features['RSI'].iloc[0]
        atr = features['ATR'].iloc[0]
        print(f"RSI for {timeframe}: {rsi}, ATR: {atr}")
        if rsi < 30:  # Oversold
            signal = 'Long'
            entry = close_price
            sl = entry - 2 * atr
            tp = entry + 3 * atr
            current_prediction[timeframe] = {
                'signal': signal,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'datetime': datetime.now(pytz.timezone(local_tz)).strftime('%Y-%m-%d %H:%M:%S')
            }
            print(f"Rule-based signal for {timeframe}: {signal}, Entry: {entry}, SL: {sl}, TP: {tp}")
        elif rsi > 70:  # Overbought
            signal = 'Short'
            entry = close_price
            sl = entry + 2 * atr
            tp = entry - 3 * atr
            current_prediction[timeframe] = {
                'signal': signal,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'datetime': datetime.now(pytz.timezone(local_tz)).strftime('%Y-%m-%d %H:%M:%S')
            }
            print(f"Rule-based signal for {timeframe}: {signal}, Entry: {entry}, SL: {sl}, TP: {tp}")
        else:
            current_prediction[timeframe] = {}
            print(f"No rule-based signal for {timeframe} (RSI: {rsi})")

    # Trading logic
    print(f"Checking trade conditions - Position: {position[timeframe]}, Signal: {signal}")
    if position[timeframe] == 0 and signal:
        # Open a new position
        position[timeframe] = 1 if signal == 'Long' else -1
        entry_price[timeframe] = close_price
        sl_price[timeframe] = current_prediction[timeframe]['sl']
        tp_price[timeframe] = current_prediction[timeframe]['tp']
        entry_datetime[timeframe] = datetime.now(pytz.timezone(local_tz))
        indicators_at_entry[trade_number + 1] = {
            'SMA_20': features['SMA_20'].iloc[0],
            'RSI': features['RSI'].iloc[0],
            'ATR': features['ATR'].iloc[0]
        }
        print(f"Opened {signal} position for {timeframe} at {entry_price[timeframe]}")
    elif position[timeframe] != 0:
        # Check for SL/TP hit
        if position[timeframe] == 1:  # Long position
            if close_price <= sl_price[timeframe] or close_price >= tp_price[timeframe]:
                exit_price = sl_price[timeframe] if close_price <= sl_price[timeframe] else tp_price[timeframe]
                result = (exit_price - entry_price[timeframe]) / entry_price[timeframe] * 100
                profit = result * balance / 100
                balance += profit
                trade_number += 1
                exit_datetime = datetime.now(pytz.timezone(local_tz))
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds() / 60
                trade = {
                    'trade_id': trade_number,
                    'type': 'Long',
                    'entry': entry_price[timeframe],
                    'exit': exit_price,
                    'result': result,
                    'profit': profit,
                    'balance': balance,
                    'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': f"{duration:.1f} m",
                    'sl': sl_price[timeframe],
                    'tp': tp_price[timeframe],
                    'timeframe': timeframe,
                    'status': 'Completed'
                }
                trade_history.append(trade)
                save_trade_history(trade_history_file, trade_history)
                print(f"Closed Long position for {timeframe} - Profit: {profit}, Balance: {balance}")
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                entry_datetime[timeframe] = None
        elif position[timeframe] == -1:  # Short position
            if close_price >= sl_price[timeframe] or close_price <= tp_price[timeframe]:
                exit_price = sl_price[timeframe] if close_price >= sl_price[timeframe] else tp_price[timeframe]
                result = (entry_price[timeframe] - exit_price) / entry_price[timeframe] * 100
                profit = result * balance / 100
                balance += profit
                trade_number += 1
                exit_datetime = datetime.now(pytz.timezone(local_tz))
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds() / 60
                trade = {
                    'trade_id': trade_number,
                    'type': 'Short',
                    'entry': entry_price[timeframe],
                    'exit': exit_price,
                    'result': result,
                    'profit': profit,
                    'balance': balance,
                    'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': f"{duration:.1f} m",
                    'sl': sl_price[timeframe],
                    'tp': tp_price[timeframe],
                    'timeframe': timeframe,
                    'status': 'Completed'
                }
                trade_history.append(trade)
                save_trade_history(trade_history_file, trade_history)
                print(f"Closed Short position for {timeframe} - Profit: {profit}, Balance: {balance}")
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                entry_datetime[timeframe] = None
    else:
        print(f"No trade opened for {timeframe} - Position: {position[timeframe]}, Signal: {signal}")

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
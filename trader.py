# trader.py
# This file contains the trading logic for processing WebSocket messages and executing trades.

import pandas as pd
import json
import os
from datetime import datetime
from CryptoAI.utils import format_dollar
from CryptoAI.predictor import predict_signal, calculate_indicators

# Global variables to track trading state
trade_history = []
trade_lock = False
balance = 10000  # Will be updated from config
position = {'1m': 0, '1h': 0, '1d': 0}  # 0: no position, 1: long, -1: short
entry_datetime = {'1m': None, '1h': None, '1d': None}
trade_number = 0
entry_price = {'1m': None, '1h': None, '1d': None}
sl_price = {'1m': None, '1h': None, '1d': None}
tp_price = {'1m': None, '1h': None, '1d': None}
current_btc_price = None
last_btc_price = None
last_chart_data_length = 0
indicators_at_entry = {}
current_prediction = {'1m': {}, '1h': {}, '1d': {}}  # Store predictions for each timeframe

def process_kline(msg, timeframe, models, df, current_prediction, indicators_at_entry, trade_history_file, local_tz):
    global trade_history, trade_lock, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length
    
    print(f"WebSocket message received for {timeframe}:", msg)
    kline = msg.get('k', {})
    print(f"Candle status - Closed: {kline.get('x')}, Time: {kline.get('t')}, Close: {kline.get('c')}")
    print(f"Current state for {timeframe} - Position: {position[timeframe]}, Entry: {entry_price[timeframe]}, SL: {sl_price[timeframe]}, TP: {tp_price[timeframe]}")
    
    # Update DataFrame with the latest kline data
    new_row = pd.DataFrame({
        'Date': [pd.to_datetime(kline['t'], unit='ms').tz_localize('UTC')],
        'Open': [float(kline['o'])],
        'High': [float(kline['h'])],
        'Low': [float(kline['l'])],
        'Close': [float(kline['c'])],
        'Volume': [float(kline['v'])]
    })
    df[timeframe] = pd.concat([df[timeframe], new_row], ignore_index=True)
    df[timeframe] = df[timeframe].tail(200)  # Keep only the last 200 rows
    print(f"df[{timeframe}] size after update: {len(df[timeframe])} rows")
    print(f"Raw df[{timeframe}] tail:", df[timeframe][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(1).to_dict())
    
    # Update current Bitcoin price
    current_btc_price = float(kline['c'])
    print(f"Updated current_btc_price: ${format_dollar(current_btc_price)}")
    
    # Calculate indicators
    df[timeframe] = calculate_indicators(df[timeframe])
    print(f"df[{timeframe}] size after indicators: {len(df[timeframe])} rows")
    print(f"Indicators tail for {timeframe}:", df[timeframe][['SMA_20', 'RSI', 'ATR']].tail(1).to_dict())
    
    # Process trade logic only if the candle is closed
    if kline.get('x'):  # Only process closed candles
        latest = df[timeframe].iloc[-1]
        signal, entry_price_new, sl_price_new, tp_price_new = predict_signal(models, timeframe, latest)
        print(f"Model prediction for {timeframe} - Signal: {signal}, Entry: {entry_price_new}, SL: {sl_price_new}, TP: {tp_price_new}")
        
        # Update current prediction for the dashboard
        current_prediction[timeframe] = {
            'signal': signal,
            'entry': entry_price_new,
            'sl': sl_price_new,
            'tp': tp_price_new,
            'datetime': pd.to_datetime(kline['t'], unit='ms').tz_localize('UTC').tz_convert(local_tz).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Execute trade if there's a signal and no current position
        if signal and position[timeframe] == 0 and not trade_lock:
            trade_lock = True
            position[timeframe] = 1 if signal == 'Long' else -1
            entry_price[timeframe] = entry_price_new
            sl_price[timeframe] = sl_price_new
            tp_price[timeframe] = tp_price_new
            entry_datetime[timeframe] = pd.to_datetime(kline['t'], unit='ms').tz_localize('UTC').tz_convert(local_tz)
            trade_number += 1
            indicators_at_entry[trade_number] = {
                'SMA_20': latest['SMA_20'],
                'RSI': latest['RSI'],
                'ATR': latest['ATR']
            }
            print(f"Entering position for {timeframe} based on model prediction...")
            print(f"{signal} at {format_dollar(entry_price[timeframe])}, SL: {format_dollar(sl_price[timeframe])}, TP: {format_dollar(tp_price[timeframe])} for {timeframe}")
    
    # Check for stop-loss or take-profit if in a position
    if position[timeframe] != 0:
        current_price = float(kline['c'])
        if position[timeframe] == 1:  # Long position
            if current_price <= sl_price[timeframe] or current_price >= tp_price[timeframe]:
                exit_price = sl_price[timeframe] if current_price <= sl_price[timeframe] else tp_price[timeframe]
                profit = (exit_price - entry_price[timeframe]) * (balance / entry_price[timeframe])
                result = (exit_price - entry_price[timeframe]) / entry_price[timeframe] * 100
                exit_datetime = pd.to_datetime(kline['E'], unit='ms').tz_localize('UTC').tz_convert(local_tz)
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds() / 60  # Duration in minutes
                trade = {
                    'trade_id': trade_number,
                    'type': 'Long',
                    'entry': entry_price[timeframe],
                    'exit': exit_price,
                    'result': result,
                    'profit': profit,
                    'balance': balance + profit,
                    'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': f"{duration:.1f} m",
                    'sl': sl_price[timeframe],
                    'tp': tp_price[timeframe],
                    'timeframe': timeframe
                }
                trade_history.append(trade)
                with open(trade_history_file, 'w') as f:
                    json.dump(trade_history, f, indent=4)
                print(f"Closed Long position for {timeframe} at {format_dollar(exit_price)}, Profit: {format_dollar(profit)} ({result:.2f}%)")
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                entry_datetime[timeframe] = None
                balance += profit
        elif position[timeframe] == -1:  # Short position
            if current_price >= sl_price[timeframe] or current_price <= tp_price[timeframe]:
                exit_price = sl_price[timeframe] if current_price >= sl_price[timeframe] else tp_price[timeframe]
                profit = (entry_price[timeframe] - exit_price) * (balance / entry_price[timeframe])
                result = (entry_price[timeframe] - exit_price) / entry_price[timeframe] * 100
                exit_datetime = pd.to_datetime(kline['E'], unit='ms').tz_localize('UTC').tz_convert(local_tz)
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds() / 60  # Duration in minutes
                trade = {
                    'trade_id': trade_number,
                    'type': 'Short',
                    'entry': entry_price[timeframe],
                    'exit': exit_price,
                    'result': result,
                    'profit': profit,
                    'balance': balance + profit,
                    'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': f"{duration:.1f} m",
                    'sl': sl_price[timeframe],
                    'tp': tp_price[timeframe],
                    'timeframe': timeframe
                }
                trade_history.append(trade)
                with open(trade_history_file, 'w') as f:
                    json.dump(trade_history, f, indent=4)
                print(f"Closed Short position for {timeframe} at {format_dollar(exit_price)}, Profit: {format_dollar(profit)} ({result:.2f}%)")
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                entry_datetime[timeframe] = None
                balance += profit
    
    trade_lock = False
    print(f"Candle not closed yet for {timeframe}, skipping trade logic...")
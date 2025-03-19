# trader.py
# This file manages trading logic, state, and WebSocket processing for the trading bot.

import pandas as pd
from threading import Lock
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from CryptoAI.utils import format_number, format_duration, save_trade_history

# Global state variables for trading
trade_history = []  # List to store all trade records
trade_lock = Lock()  # Lock for thread-safe updates to trade_history
balance = 10000  # Current balance, updated with each trade
indicators_at_entry = {}  # Store indicators at trade entry for each trade ID
position = {'1m': 0, '1h': 0, '1d': 0}  # Position for each timeframe (1 = Long, -1 = Short, 0 = None)
entry_datetime = {'1m': None, '1h': None, '1d': None}  # Entry datetime for each timeframe
trade_number = 0  # Current trade ID, incremented with each trade
entry_price = {'1m': None, '1h': None, '1d': None}  # Entry price for each timeframe
sl_price = {'1m': None, '1h': None, '1d': None}  # Stop-loss price for each timeframe
tp_price = {'1m': None, '1h': None, '1d': None}  # Take-profit price for each timeframe
current_btc_price = None  # Current BTC price, updated from WebSocket
last_btc_price = None  # Previous BTC price for calculating price change
last_chart_data_length = 0  # Track chart data length for updates
current_prediction = {  # Current prediction state for each timeframe
    '1m': {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None},
    '1h': {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None},
    '1d': {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}
}

def process_kline(msg, timeframe, models, df, current_prediction, indicators_at_entry, trade_history_file, local_tz):
    """
    Process a WebSocket kline message, update data, and execute trading logic for a given timeframe.
    
    Args:
        msg (dict): WebSocket message containing kline data.
        timeframe (str): Timeframe identifier ('1m', '1h', '1d').
        models (dict): Trained models for prediction.
        df (dict): Dictionary of DataFrames for each timeframe.
        current_prediction (dict): Current prediction state for each timeframe.
        indicators_at_entry (dict): Indicators at entry for each trade.
        trade_history_file (str): Path to the trade history JSON file.
        local_tz (pytz.timezone): Timezone for datetime conversion.
    """
    global trade_history, trade_lock, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length
    
    print(f"WebSocket message received for {timeframe}:", msg)
    kline = msg.get('k', {})
    print(f"Candle status - Closed: {kline.get('x')}, Time: {kline.get('t')}, Close: {kline.get('c')}")
    print(f"Current state for {timeframe} - Position: {position[timeframe]}, Entry: {entry_price[timeframe]}, SL: {sl_price[timeframe]}, TP: {tp_price[timeframe]}")
    
    # Process EVERY candle
    new_data = {
        'Date': pd.to_datetime(kline['t'], unit='ms').tz_localize('UTC'),  # Localize to UTC
        'Open': float(kline['o']),
        'High': float(kline['h']),
        'Low': float(kline['l']),
        'Close': float(kline['c']),
        'Volume': float(kline['v'])
    }
    df[timeframe] = pd.concat([df[timeframe], pd.DataFrame([new_data])], ignore_index=True)
    if len(df[timeframe]) > 200:
        df[timeframe] = df[timeframe].tail(200)

    # Update current BTC price from 1m timeframe
    if timeframe == '1m':
        current_btc_price = new_data['Close']
        print(f"Updated current_btc_price: ${format_dollar(current_btc_price)}")

    print(f"df[{timeframe}] size after update: {len(df[timeframe])} rows")
    print(f"Raw df[{timeframe}] tail: {df[timeframe].tail(1).to_dict()}")
    df[timeframe]['SMA_20'] = SMAIndicator(df[timeframe]['Close'], window=20).sma_indicator().fillna(0)
    df[timeframe]['RSI'] = RSIIndicator(df[timeframe]['Close'], window=14).rsi().fillna(0)
    df[timeframe]['ATR'] = AverageTrueRange(df[timeframe]['High'], df[timeframe]['Low'], df[timeframe]['Close'], window=14).average_true_range().fillna(0)
    print(f"df[{timeframe}] size after indicators: {len(df[timeframe])} rows")
    print(f"Indicators tail for {timeframe}: {df[timeframe][['SMA_20', 'RSI', 'ATR']].tail(1).to_dict()}")

    if kline.get('x'):  # Only process closed candles
        latest = df[timeframe].iloc[-1]
        signal, entry_price_new, sl_price_new, tp_price_new = predict_signal(models, timeframe, latest)
        print(f"Model prediction for {timeframe} - Signal: {signal}, Entry: {entry_price_new}, SL: {sl_price_new}, TP: {tp_price_new}")

        if position[timeframe] == 0:
            print(f"Entering position for {timeframe} based on model prediction...")
            position[timeframe] = 1 if signal == 'Long' else -1
            entry_price[timeframe] = entry_price_new
            sl_price[timeframe] = sl_price_new
            tp_price[timeframe] = tp_price_new
            entry_datetime[timeframe] = latest['Date'].astimezone(local_tz)
            current_prediction[timeframe] = {'signal': signal, 'entry': format_number(entry_price[timeframe]), 
                                             'sl': format_number(sl_price[timeframe]), 'tp': format_number(tp_price[timeframe]), 
                                             'datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S')}
            indicators_at_entry[trade_number + 1] = {'SMA_20': latest['SMA_20'], 'RSI': latest['RSI'], 'ATR': latest['ATR']}
            print(f"{signal} at {format_number(entry_price[timeframe])}, SL: {format_number(sl_price[timeframe])}, TP: {format_number(tp_price[timeframe])} for {timeframe}")
        
        elif position[timeframe] == 1:  # Long
            print(f"Long Check for {timeframe} - High: {latest['High']}, Low: {latest['Low']}, SL: {sl_price[timeframe]}, TP: {tp_price[timeframe]}, Entry: {entry_price[timeframe]}")
            print(f"SL Check - Low: {latest['Low']}, SL: {sl_price[timeframe]}, Condition: {latest['Low'] <= sl_price[timeframe]}")
            print(f"TP Check - High: {latest['High']}, TP: {tp_price[timeframe]}, Condition: {latest['High'] >= tp_price[timeframe]}")
            if latest['Low'] <= sl_price[timeframe]:
                profit_pct = (sl_price[timeframe] - entry_price[timeframe]) / entry_price[timeframe]
                profit_dollar = profit_pct * entry_price[timeframe]
                balance += profit_dollar  # Update balance directly
                trade_number += 1
                exit_datetime = latest['Date'].astimezone(local_tz)
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds()
                with trade_lock:
                    trade_history.append({
                        'trade_id': trade_number, 'type': 'Long', 'entry': entry_price[timeframe], 'exit': sl_price[timeframe], 
                        'result': profit_pct * 100, 'profit': profit_dollar, 'balance': balance, 
                        'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'), 
                        'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                        'duration': format_duration(duration),
                        'sl': sl_price[timeframe], 'tp': tp_price[timeframe],
                        'timeframe': timeframe,
                        'status': 'Completed'
                    })
                print(f"#SL hit at {format_number(sl_price[timeframe])}, Balance: {format_number(balance)}")
                print(f"Trade history updated: {len(trade_history)} trades, Latest: {trade_history[-1] if trade_history else 'None'}")
                save_trade_history(trade_history_file, trade_history)
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                current_prediction[timeframe] = {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}
            elif latest['High'] >= tp_price[timeframe]:
                profit_pct = (tp_price[timeframe] - entry_price[timeframe]) / entry_price[timeframe]
                profit_dollar = profit_pct * entry_price[timeframe]
                balance += profit_dollar  # Update balance directly
                trade_number += 1
                exit_datetime = latest['Date'].astimezone(local_tz)
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds()
                with trade_lock:
                    trade_history.append({
                        'trade_id': trade_number, 'type': 'Long', 'entry': entry_price[timeframe], 'exit': tp_price[timeframe], 
                        'result': profit_pct * 100, 'profit': profit_dollar, 'balance': balance, 
                        'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'), 
                        'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                        'duration': format_duration(duration),
                        'sl': sl_price[timeframe], 'tp': tp_price[timeframe],
                        'timeframe': timeframe,
                        'status': 'Completed'
                    })
                print(f"#TP hit at {format_number(tp_price[timeframe])}, Balance: {format_number(balance)}")
                print(f"Trade history updated: {len(trade_history)} trades, Latest: {trade_history[-1] if trade_history else 'None'}")
                save_trade_history(trade_history_file, trade_history)
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                current_prediction[timeframe] = {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}
        
        elif position[timeframe] == -1:  # Short
            print(f"Short Check for {timeframe} - High: {latest['High']}, Low: {latest['Low']}, SL: {sl_price[timeframe]}, TP: {tp_price[timeframe]}, Entry: {entry_price[timeframe]}")
            print(f"SL Check - High: {latest['High']}, SL: {sl_price[timeframe]}, Condition: {latest['High'] >= sl_price[timeframe]}")
            print(f"TP Check - Low: {latest['Low']}, TP: {tp_price[timeframe]}, Condition: {latest['Low'] <= tp_price[timeframe]}")
            if latest['High'] >= sl_price[timeframe]:
                profit_pct = (entry_price[timeframe] - sl_price[timeframe]) / entry_price[timeframe]
                profit_dollar = profit_pct * entry_price[timeframe]
                balance += profit_dollar  # Update balance directly
                trade_number += 1
                exit_datetime = latest['Date'].astimezone(local_tz)
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds()
                with trade_lock:
                    trade_history.append({
                        'trade_id': trade_number, 'type': 'Short', 'entry': entry_price[timeframe], 'exit': sl_price[timeframe], 
                        'result': profit_pct * 100, 'profit': profit_dollar, 'balance': balance, 
                        'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'), 
                        'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                        'duration': format_duration(duration),
                        'sl': sl_price[timeframe], 'tp': tp_price[timeframe],
                        'timeframe': timeframe,
                        'status': 'Completed'
                    })
                print(f"#SL hit at {format_number(sl_price[timeframe])}, Balance: {format_number(balance)}")
                print(f"Trade history updated: {len(trade_history)} trades, Latest: {trade_history[-1] if trade_history else 'None'}")
                save_trade_history(trade_history_file, trade_history)
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                current_prediction[timeframe] = {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}
            elif latest['Low'] <= tp_price[timeframe]:
                profit_pct = (entry_price[timeframe] - tp_price[timeframe]) / entry_price[timeframe]
                profit_dollar = profit_pct * entry_price[timeframe]
                balance += profit_dollar  # Update balance directly
                trade_number += 1
                exit_datetime = latest['Date'].astimezone(local_tz)
                duration = (exit_datetime - entry_datetime[timeframe]).total_seconds()
                with trade_lock:
                    trade_history.append({
                        'trade_id': trade_number, 'type': 'Short', 'entry': entry_price[timeframe], 'exit': tp_price[timeframe], 
                        'result': profit_pct * 100, 'profit': profit_dollar, 'balance': balance, 
                        'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'), 
                        'exit_datetime': exit_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                        'duration': format_duration(duration),
                        'sl': sl_price[timeframe], 'tp': tp_price[timeframe],
                        'timeframe': timeframe,
                        'status': 'Completed'
                    })
                print(f"#TP hit at {format_number(tp_price[timeframe])}, Balance: {format_number(balance)}")
                print(f"Trade history updated: {len(trade_history)} trades, Latest: {trade_history[-1] if trade_history else 'None'}")
                save_trade_history(trade_history_file, trade_history)
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                current_prediction[timeframe] = {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}
        print(f"Position after for {timeframe}: {position[timeframe]}")
    else:
        print(f"Candle not closed yet for {timeframe}, skipping trade logic...")
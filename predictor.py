import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import numpy as np
from flask import Flask, render_template, jsonify
from binance.client import Client
from binance import ThreadedWebsocketManager
import json
import os
from datetime import datetime
import pytz
from threading import Lock

app = Flask(__name__)

# Global vars
current_prediction = {'1m': {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None},
                      '1h': {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None},
                      '1d': {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}}
trade_history = []  # Global trade history, persisted to file
trade_lock = Lock()  # Lock for thread-safe trade_history updates
balance = 10000
initial_balance = 10000
indicators_at_entry = {}
df = {'1m': pd.DataFrame(), '1h': pd.DataFrame(), '1d': pd.DataFrame()}  # Preloaded with historical data
position = {'1m': 0, '1h': 0, '1d': 0}
entry_datetime = {'1m': None, '1h': None, '1d': None}
trade_number = 0
entry_price = {'1m': None, '1h': None, '1d': None}
sl_price = {'1m': None, '1h': None, '1d': None}
tp_price = {'1m': None, '1h': None, '1d': None}
current_btc_price = None
last_btc_price = None  # Track previous BTC price
last_chart_data_length = 0  # Track chart data length

# Binance API keys (your live keys)
BINANCE_API_KEY = 'IhjebRD6OLbcMdOqmL1c62sSuz4MLyMM5CtYQwG4JMN8B0itY7zHbewun8FEJ4la'
BINANCE_API_SECRET = 'REDACTED_API_SECRET'
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# File for persisting trade history
TRADE_HISTORY_FILE = 'trade_history.json'

# Timezone (CET)
LOCAL_TZ = pytz.timezone('Europe/Berlin')

def format_number(number):
    return f"{int(number):,}".replace(',', "'") + f".{number:.2f}".split('.')[1] if number is not None else '-'

def format_dollar(number):
    return f"{int(number):,}".replace(',', "'") + f".{number:.2f}".split('.')[1] if number is not None else '-'

def calculate_stats():
    global trade_history
    total_trades = len(trade_history)
    if total_trades == 0:
        return {'total_win_pct': 0, 'total_trades': 0, 'short_win_pct': 0, 'short_trades': 0, 'long_win_pct': 0, 'long_trades': 0, 
                'short_profit': 0, 'long_profit': 0}
    
    wins = sum(1 for trade in trade_history if trade.get('result', 0) > 0)
    shorts = [t for t in trade_history if t['type'] == 'Short']
    longs = [t for t in trade_history if t['type'] == 'Long']
    short_wins = sum(1 for t in shorts if t.get('result', 0) > 0)
    long_wins = sum(1 for t in longs if t.get('result', 0) > 0)
    short_profit = sum(t.get('profit', 0) for t in shorts if 'status' not in t or t.get('status') == 'Completed')
    long_profit = sum(t.get('profit', 0) for t in longs if 'status' not in t or t.get('status') == 'Completed')
    
    return {
        'total_win_pct': (wins / total_trades * 100) if total_trades > 0 else 0,
        'total_trades': total_trades,
        'short_win_pct': (short_wins / len(shorts) * 100) if len(shorts) > 0 else 0,
        'short_trades': len(shorts),
        'long_win_pct': (long_wins / len(longs) * 100) if len(longs) > 0 else 0,
        'long_trades': len(longs),
        'short_profit': short_profit,
        'long_profit': long_profit
    }

def load_trade_history():
    global trade_history, trade_number
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                trade_history = json.load(f)
                if trade_history:
                    trade_number = max(trade['trade_id'] for trade in trade_history)
                print(f"Loaded {len(trade_history)} historic trades from {TRADE_HISTORY_FILE}, max trade_id: {trade_number}")
        except Exception as e:
            print(f"Error loading {TRADE_HISTORY_FILE}: {e}")
            trade_history = []
    else:
        print(f"No historic trade file found at {TRADE_HISTORY_FILE}")
        trade_history = []

def save_trade_history():
    global trade_history
    try:
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trade_history, f)
        print(f"Saved {len(trade_history)} trades to {TRADE_HISTORY_FILE}")
    except Exception as e:
        print(f"Error saving {TRADE_HISTORY_FILE}: {e}")

# Preload dfs with historical candles
def preload_df(timeframe, interval, minutes):
    print(f"Preloading df for {timeframe} with {minutes} historical candles...")
    klines = client.get_historical_klines('BTCUSDT', interval, f"{minutes} minutes ago UTC")
    df[timeframe] = pd.DataFrame(klines, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 
                                                  'Number_of_trades', 'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
    df[timeframe] = df[timeframe][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df[timeframe]['Date'] = pd.to_datetime(df[timeframe]['Date'], unit='ms').dt.tz_localize('UTC')
    print(f"Preloaded df[{timeframe}] with {len(df[timeframe])} rows")
    df[timeframe]['SMA_20'] = SMAIndicator(df[timeframe]['Close'], window=20).sma_indicator().fillna(0)
    df[timeframe]['RSI'] = RSIIndicator(df[timeframe]['Close'], window=14).rsi().fillna(0)
    df[timeframe]['ATR'] = AverageTrueRange(df[timeframe]['High'], df[timeframe]['Low'], df[timeframe]['Close'], window=14).average_true_range().fillna(0)

preload_df('1m', Client.KLINE_INTERVAL_1MINUTE, 200)
preload_df('1h', Client.KLINE_INTERVAL_1HOUR, 200 * 60)
preload_df('1d', Client.KLINE_INTERVAL_1DAY, 200 * 60 * 24)

# Train models per timeframe
models = {}
for timeframe in ['1m', '1h', '1d']:
    X = df[timeframe][['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'ATR']]
    y_direction = np.where(df[timeframe]['Close'].shift(-1) > df[timeframe]['Close'], 1, 0)[:-1]
    X = X[:-1]
    X_train, X_test, y_dir_train, y_dir_test = train_test_split(X, y_direction, test_size=0.2, shuffle=False)
    X_train_sl, X_test_sl = X_train, X_test
    y_sl_train = np.abs(X_train['Close'] - X_train['Low']) * (0.5 if timeframe == '1m' else 2 if timeframe == '1h' else 5)
    y_tp_train = np.abs(X_train['High'] - X_train['Close']) * (0.5 if timeframe == '1m' else 2 if timeframe == '1h' else 5)

    print(f"Training direction model for {timeframe}...")
    models[timeframe] = {'direction': RandomForestClassifier(n_estimators=100, random_state=42)}
    models[timeframe]['direction'].fit(X_train, y_dir_train)
    print(f"Training SL model for {timeframe}...")
    models[timeframe]['sl'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models[timeframe]['sl'].fit(X_train_sl, y_sl_train)
    print(f"Training TP model for {timeframe}...")
    models[timeframe]['tp'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models[timeframe]['tp'].fit(X_train_sl, y_tp_train)

print("Training complete!")

# Load historic trades from file
load_trade_history()

@app.route('/test')
def test():
    print("Hit /test route")
    return "Fuck yeah, Flask is alive!"

@app.route('/')
def dashboard():
    global current_prediction, trade_history, balance, initial_balance, position, current_btc_price, last_btc_price, trade_number
    # Recalculate balance and total profit from all completed trades
    balance = initial_balance
    total_profit = 0
    for trade in trade_history:
        if trade.get('status', 'Completed') == 'Completed':  # Default to 'Completed' if missing
            profit = trade.get('profit', 0)
            balance += profit
            total_profit += profit
    print(f"Balance calculation: Initial = {initial_balance}, Total Profit = {total_profit}, Final Balance = {balance}")
    print(f"Sample profits from trade_history: {[trade.get('profit', 0) for trade in trade_history[:5]]}")
    profit_pct = (total_profit / initial_balance) * 100 if initial_balance != 0 else 0
    balance_str = format_dollar(balance)
    profit_str = format_dollar(total_profit)
    profit_pct_str = f"{profit_pct:.2f}%"
    btc_price_str = format_dollar(current_btc_price) if current_btc_price else 'N/A'
    btc_change_pct = ((current_btc_price - last_btc_price) / last_btc_price * 100) if last_btc_price and current_btc_price else 0
    btc_change_str = f"{btc_change_pct:+.1f} %"
    stats = calculate_stats()
    # Use full trade_history for chart_data, not limited to 20 trades
    chart_data = {
        'labels': [trade.get('exit_datetime', '-') for trade in sorted(trade_history, key=lambda x: x['trade_id'])],  # Sort ascending for chart
        'balances': [trade.get('balance', balance) for trade in sorted(trade_history, key=lambda x: x['trade_id'])]
    }
    # Add current open trades if active (for table only)
    trades_with_status = sorted(trade_history, key=lambda x: x['trade_id'], reverse=True)[:20]  # Sort by trade_id, take 20 newest for table
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and current_btc_price:
            profit_pct_live = ((current_btc_price - entry_price[timeframe]) / entry_price[timeframe]) * 100 if position[timeframe] == 1 else ((entry_price[timeframe] - current_btc_price) / entry_price[timeframe]) * 100
            profit_dollar_live = profit_pct_live * entry_price[timeframe] / 100
            current_trade = {
                'trade_id': trade_number + 1,
                'type': 'Long' if position[timeframe] == 1 else 'Short',
                'entry': entry_price[timeframe],
                'exit': None,
                'result': profit_pct_live,  # Live Profit %
                'profit': profit_dollar_live,  # Live Profit $
                'balance': balance,  # Will be updated below
                'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_datetime': None,
                'duration': None,
                'status': 'Open',
                'sl': sl_price[timeframe],
                'tp': tp_price[timeframe],
                'timeframe': timeframe
            }
            trades_with_status.insert(0, current_trade)  # Add open trade at top
            trade_number += 1  # Increment trade_number for each open trade
    for trade in trades_with_status:
        if 'status' not in trade:
            trade['status'] = 'Completed'
        if 'timeframe' not in trade:
            trade['timeframe'] = '1m'  # Default for older trades
    # Calculate running balance for each trade in the table (newest to oldest)
    running_balance = balance
    for trade in trades_with_status:
        trade['balance'] = running_balance
        if trade['status'] == 'Completed':
            running_balance -= trade.get('profit', 0)  # Subtract profit since we're going newest to oldest
    print("Dashboard HIT - Prediction:", current_prediction)
    print("Trades in dashboard:", len(trades_with_status))
    print("Trade History in dashboard:", trades_with_status)
    print("Balance:", balance_str, "Profit:", profit_str, "Profit %:", profit_pct_str, "BTC Price:", btc_price_str, "BTC Change:", btc_change_str)
    last_btc_price = current_btc_price  # Update last price
    return render_template('dashboard.html', prediction=current_prediction, trades=trades_with_status, 
                          balance=balance_str, profit=profit_str, profit_pct=profit_pct_str, 
                          stats=stats, chart_data=json.dumps(chart_data), btc_price=btc_price_str, btc_change=btc_change_str)

@app.route('/trades')
def get_trades():
    global current_prediction, trade_history, balance, initial_balance, position, current_btc_price, last_btc_price, last_chart_data_length, trade_number
    # Recalculate balance and total profit from all completed trades
    balance = initial_balance
    total_profit = 0
    for trade in trade_history:
        if trade.get('status', 'Completed') == 'Completed':  # Default to 'Completed' if missing
            profit = trade.get('profit', 0)
            balance += profit
            total_profit += profit
    print(f"Trades route - Balance calculation: Initial = {initial_balance}, Total Profit = {total_profit}, Final Balance = {balance}")
    profit_pct = (total_profit / initial_balance) * 100 if initial_balance != 0 else 0
    balance_str = format_dollar(balance)
    profit_str = format_dollar(total_profit)
    profit_pct_str = f"{profit_pct:.2f}%"
    btc_price_str = format_dollar(current_btc_price) if current_btc_price else 'N/A'
    btc_change_pct = ((current_btc_price - last_btc_price) / last_btc_price * 100) if last_btc_price and current_btc_price else 0
    btc_change_str = f"{btc_change_pct:+.1f} %"
    stats = calculate_stats()
    # Use full trade_history for chart_data
    chart_data = {
        'labels': [trade.get('exit_datetime', '-') for trade in sorted(trade_history, key=lambda x: x['trade_id'])],  # Sort ascending for chart
        'balances': [trade.get('balance', balance) for trade in sorted(trade_history, key=lambda x: x['trade_id'])]
    }
    update_chart = len(chart_data['labels']) != last_chart_data_length
    last_chart_data_length = len(chart_data['labels'])
    # Add current open trades if active (for table only)
    trades_with_status = sorted(trade_history, key=lambda x: x['trade_id'], reverse=True)[:20]  # Sort by trade_id, take 20 newest for table
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and current_btc_price:
            profit_pct_live = ((current_btc_price - entry_price[timeframe]) / entry_price[timeframe]) * 100 if position[timeframe] == 1 else ((entry_price[timeframe] - current_btc_price) / entry_price[timeframe]) * 100
            profit_dollar_live = profit_pct_live * entry_price[timeframe] / 100
            current_trade = {
                'trade_id': trade_number + 1,
                'type': 'Long' if position[timeframe] == 1 else 'Short',
                'entry': entry_price[timeframe],
                'exit': None,
                'result': profit_pct_live,  # Live Profit %
                'profit': profit_dollar_live,  # Live Profit $
                'balance': balance,  # Will be updated below
                'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_datetime': None,
                'duration': None,
                'status': 'Open',
                'sl': sl_price[timeframe],
                'tp': tp_price[timeframe],
                'timeframe': timeframe
            }
            trades_with_status.insert(0, current_trade)
            trade_number += 1  # Increment trade_number for each open trade
    for trade in trades_with_status:
        if 'status' not in trade:
            trade['status'] = 'Completed'
        if 'timeframe' not in trade:
            trade['timeframe'] = '1m'  # Default for older trades
    # Calculate running balance for each trade in the table (newest to oldest)
    running_balance = balance
    for trade in trades_with_status:
        trade['balance'] = running_balance
        if trade['status'] == 'Completed':
            running_balance -= trade.get('profit', 0)  # Subtract profit since we're going newest to oldest
    last_btc_price = current_btc_price  # Update last price
    return jsonify({
        'prediction': current_prediction,
        'trades': trades_with_status,
        'balance': balance_str,
        'profit': profit_str,
        'profit_pct': profit_pct_str,
        'stats': stats,
        'chart_data': chart_data,
        'update_chart': update_chart,
        'btc_price': btc_price_str,
        'btc_change': btc_change_str
    })

@app.route('/history')
def history():
    global trade_history, balance, initial_balance, position, current_btc_price, last_btc_price, trade_number
    # Recalculate balance and total profit from all completed trades
    balance = initial_balance
    total_profit = 0
    for trade in trade_history:
        if trade.get('status', 'Completed') == 'Completed':  # Default to 'Completed' if missing
            profit = trade.get('profit', 0)
            balance += profit
            total_profit += profit
    print(f"History route - Balance calculation: Initial = {initial_balance}, Total Profit = {total_profit}, Final Balance = {balance}")
    profit_pct = (total_profit / initial_balance) * 100 if initial_balance != 0 else 0
    balance_str = format_dollar(balance)
    profit_str = format_dollar(total_profit)
    profit_pct_str = f"{profit_pct:.2f}%"
    btc_price_str = format_dollar(current_btc_price) if current_btc_price else 'N/A'
    btc_change_pct = ((current_btc_price - last_btc_price) / last_btc_price * 100) if last_btc_price and current_btc_price else 0
    btc_change_str = f"{btc_change_pct:+.1f} %"
    stats = calculate_stats()
    chart_data = {
        'labels': [trade.get('exit_datetime', '-') for trade in sorted(trade_history, key=lambda x: x['trade_id'])],  # Sort ascending for chart
        'balances': [trade.get('balance', balance) for trade in sorted(trade_history, key=lambda x: x['trade_id'])]
    }
    # Add current open trades if active
    trades_with_status = sorted(trade_history, key=lambda x: x['trade_id'], reverse=True)  # Sort all trades by trade_id descending
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and current_btc_price:
            profit_pct_live = ((current_btc_price - entry_price[timeframe]) / entry_price[timeframe]) * 100 if position[timeframe] == 1 else ((entry_price[timeframe] - current_btc_price) / entry_price[timeframe]) * 100
            profit_dollar_live = profit_pct_live * entry_price[timeframe] / 100
            current_trade = {
                'trade_id': trade_number + 1,
                'type': 'Long' if position[timeframe] == 1 else 'Short',
                'entry': entry_price[timeframe],
                'exit': None,
                'result': profit_pct_live,  # Live Profit %
                'profit': profit_dollar_live,  # Live Profit $
                'balance': balance,  # Will be updated below
                'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_datetime': None,
                'duration': None,
                'status': 'Open',
                'sl': sl_price[timeframe],
                'tp': tp_price[timeframe],
                'timeframe': timeframe
            }
            trades_with_status.insert(0, current_trade)
            trade_number += 1  # Increment trade_number for each open trade
    for trade in trades_with_status:
        if 'status' not in trade:
            trade['status'] = 'Completed'
        if 'timeframe' not in trade:
            trade['timeframe'] = '1m'  # Default for older trades
    # Calculate running balance for each trade in the table (newest to oldest)
    running_balance = balance
    for trade in trades_with_status:
        trade['balance'] = running_balance
        if trade['status'] == 'Completed':
            running_balance -= trade.get('profit', 0)  # Subtract profit since we're going newest to oldest
    print("History HIT - Trades:", len(trades_with_status))
    last_btc_price = current_btc_price  # Update last price
    return render_template('history.html', trades=trades_with_status, 
                          balance=balance_str, profit=profit_str, profit_pct=profit_pct_str, stats=stats, 
                          chart_data=json.dumps(chart_data), btc_price=btc_price_str, btc_change=btc_change_str)

@app.route('/trade_details/<int:trade_id>')
def trade_details(trade_id):
    global trade_history, sl_price, tp_price, position, trade_number, indicators_at_entry, entry_datetime
    trade = next((t for t in trade_history if t['trade_id'] == trade_id), None)
    if trade:
        indicators = indicators_at_entry.get(trade_id, {})
        sl = trade.get('sl', sl_price[trade.get('timeframe', '1m')] if trade_id == trade_number + 1 and position.get(trade.get('timeframe', '1m')) != 0 else None)
        tp = trade.get('tp', tp_price[trade.get('timeframe', '1m')] if trade_id == trade_number + 1 and position.get(trade.get('timeframe', '1m')) != 0 else None)
        return jsonify({
            'trade': trade,
            'indicators': indicators,
            'stop_loss': sl,
            'take_profit': tp
        })
    # Handle open trade not yet in trade_history
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and trade_id == trade_number + 1:
            indicators = indicators_at_entry.get(trade_id, indicators_at_entry.get(trade_number, {}))  # Fallback to latest if not set
            return jsonify({
                'trade': {
                    'trade_id': trade_id,
                    'type': 'Long' if position[timeframe] == 1 else 'Short',
                    'entry': entry_price[timeframe],
                    'exit': None,
                    'result': 0,
                    'profit': 0,
                    'balance': balance,
                    'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_datetime': None,
                    'duration': None,
                    'status': 'Open',
                    'sl': sl_price[timeframe],
                    'tp': tp_price[timeframe],
                    'timeframe': timeframe
                },
                'indicators': indicators,
                'stop_loss': sl_price[timeframe],
                'take_profit': tp_price[timeframe]
            })
    return jsonify({'error': 'Trade not found'}), 404

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    else:
        return f"{seconds / 86400:.1f} d"

def process_kline(msg, timeframe):
    global df, current_prediction, trade_history, balance, position, entry_datetime, trade_number, indicators_at_entry, entry_price, sl_price, tp_price, current_btc_price
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

    print(f"df[{timeframe}] size after update: {len(df[timeframe])} rows")
    print(f"Raw df[{timeframe}] tail: {df[timeframe].tail(1).to_dict()}")
    df[timeframe]['SMA_20'] = SMAIndicator(df[timeframe]['Close'], window=20).sma_indicator().fillna(0)
    df[timeframe]['RSI'] = RSIIndicator(df[timeframe]['Close'], window=14).rsi().fillna(0)
    df[timeframe]['ATR'] = AverageTrueRange(df[timeframe]['High'], df[timeframe]['Low'], df[timeframe]['Close'], window=14).average_true_range().fillna(0)
    print(f"df[{timeframe}] size after indicators: {len(df[timeframe])} rows")
    print(f"Indicators tail for {timeframe}: {df[timeframe][['SMA_20', 'RSI', 'ATR']].tail(1).to_dict()}")

    if kline.get('x'):  # Only process closed candles
        latest = df[timeframe].iloc[-1]
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'ATR']
        X = latest[features].values.reshape(1, -1)
        signal = 'Long' if models[timeframe]['direction'].predict_proba(X)[0][1] > 0.55 else 'Short'  # Tighter signal threshold
        sl_distance = models[timeframe]['sl'].predict(X)[0]  # Scaled by timeframe in training
        tp_distance = models[timeframe]['tp'].predict(X)[0]  # Scaled by timeframe in training
        entry_price_new = latest['Close']
        sl_price_new = round(entry_price_new - sl_distance if signal == 'Long' else entry_price_new + sl_distance, 2)
        tp_price_new = round(entry_price_new + tp_distance if signal == 'Long' else entry_price_new - tp_distance, 2)

        print(f"Model prediction for {timeframe} - Signal: {signal}, Entry: {entry_price_new}, SL: {sl_price_new}, TP: {tp_price_new}")

        if position[timeframe] == 0:
            print(f"Entering position for {timeframe} based on model prediction...")
            position[timeframe] = 1 if signal == 'Long' else -1
            entry_price[timeframe] = entry_price_new
            sl_price[timeframe] = sl_price_new
            tp_price[timeframe] = tp_price_new
            entry_datetime[timeframe] = latest['Date'].astimezone(LOCAL_TZ)
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
                exit_datetime = latest['Date'].astimezone(LOCAL_TZ)
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
                save_trade_history()
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
                exit_datetime = latest['Date'].astimezone(LOCAL_TZ)
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
                save_trade_history()
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
                exit_datetime = latest['Date'].astimezone(LOCAL_TZ)
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
                save_trade_history()
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
                exit_datetime = latest['Date'].astimezone(LOCAL_TZ)
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
                save_trade_history()
                position[timeframe] = 0
                entry_price[timeframe] = None
                sl_price[timeframe] = None
                tp_price[timeframe] = None
                current_prediction[timeframe] = {'signal': None, 'entry': None, 'sl': None, 'tp': None, 'datetime': None}
        print(f"Position after for {timeframe}: {position[timeframe]}")
    else:
        print(f"Candle not closed yet for {timeframe}, skipping trade logic...")

# Start WebSocket for multiple timeframes
twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
twm.start()
twm.start_kline_socket(callback=lambda msg: process_kline(msg, '1m'), symbol='BTCUSDT', interval='1m')
twm.start_kline_socket(callback=lambda msg: process_kline(msg, '1h'), symbol='BTCUSDT', interval='1h')
twm.start_kline_socket(callback=lambda msg: process_kline(msg, '1d'), symbol='BTCUSDT', interval='1d')

if __name__ == '__main__':
    print(f"\nInitial Balance: ${format_number(balance)}")
    print("Starting Flask server...")
    app.run(debug=False, host='127.0.0.1', port=5000)
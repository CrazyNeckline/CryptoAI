# app.py
# This file sets up the Flask application and defines the routes for the trading dashboard.

import sys
import os
import json

# Add the parent directory to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, jsonify
from binance.client import Client
from binance import ThreadedWebsocketManager
from CryptoAI.config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADE_HISTORY_FILE, LOCAL_TZ, INITIAL_BALANCE
from CryptoAI.utils import format_dollar, format_number, load_trade_history
from CryptoAI.data import preload_df
from CryptoAI.models import train_models
from CryptoAI.trader import process_kline, trade_history, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length, indicators_at_entry, current_prediction

app = Flask(__name__)

# Initialize Binance client for API access
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Fetch the initial Bitcoin price on startup
try:
    ticker = client.get_symbol_ticker(symbol="BTCUSDT")
    current_btc_price = float(ticker['price'])
    print(f"Initial Bitcoin price fetched: ${format_dollar(current_btc_price)}")
except Exception as e:
    print(f"Error fetching initial Bitcoin price: {e}")
    current_btc_price = None

# Load historical data for each timeframe
df = {
    '1m': preload_df(client, '1m', Client.KLINE_INTERVAL_1MINUTE, 200),
    '1h': preload_df(client, '1h', Client.KLINE_INTERVAL_1HOUR, 200 * 60),
    '1d': preload_df(client, '1d', Client.KLINE_INTERVAL_1DAY, 200 * 60 * 24)
}

# Train models for each timeframe
models = train_models(df)

# Load trade history from file
trade_history, trade_number = load_trade_history(TRADE_HISTORY_FILE, trade_history, trade_number)

@app.route('/test')
def test():
    """Test route to verify Flask is running."""
    print("Hit /test route")
    return "Fuck yeah, Flask is alive!"

@app.route('/')
def dashboard():
    """
    Render the main dashboard page with trade predictions, stats, and recent trades.
    
    Returns:
        str: Rendered HTML template for the dashboard.
    """
    global trade_history, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length, indicators_at_entry, current_prediction
    
    # Recalculate balance and total profit from all completed trades
    balance = INITIAL_BALANCE
    total_profit = 0
    for trade in trade_history:
        if trade.get('status', 'Completed') == 'Completed':
            profit = trade.get('profit', 0)
            balance += profit
            total_profit += profit
    print(f"Balance calculation: Initial = {INITIAL_BALANCE}, Total Profit = {total_profit}, Final Balance = {balance}")
    print(f"Sample profits from trade_history: {[trade.get('profit', 0) for trade in trade_history[:5]]}")
    profit_pct = (total_profit / INITIAL_BALANCE) * 100 if INITIAL_BALANCE != 0 else 0
    balance_str = format_dollar(balance)
    profit_str = format_dollar(total_profit)
    profit_pct_str = f"{profit_pct:.2f}%"

    # Fallback: Fetch the Bitcoin price if current_btc_price is None
    btc_price = current_btc_price
    if btc_price is None:
        try:
            ticker = client.get_symbol_ticker(symbol="BTCUSDT")
            btc_price = float(ticker['price'])
            print(f"Fallback Bitcoin price fetched: ${format_dollar(btc_price)}")
        except Exception as e:
            print(f"Error fetching fallback Bitcoin price: {e}")
            btc_price = None

    btc_price_str = format_dollar(btc_price) if btc_price is not None else 'N/A'
    btc_change_pct = ((btc_price - last_btc_price) / last_btc_price * 100) if last_btc_price and btc_price else 0
    btc_change_str = f"{btc_change_pct:+.1f} %"
    stats = calculate_stats()
    # Use full trade_history for chart_data, not limited to 20 trades
    chart_data = {
        'labels': [trade.get('exit_datetime', '-') for trade in sorted(trade_history, key=lambda x: x['trade_id'])],
        'balances': [trade.get('balance', balance) for trade in sorted(trade_history, key=lambda x: x['trade_id'])]
    }
    # Add current open trades if active (for table only)
    trades_with_status = sorted(trade_history, key=lambda x: x['trade_id'], reverse=True)[:20]
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and btc_price:
            profit_pct_live = ((btc_price - entry_price[timeframe]) / entry_price[timeframe]) * 100 if position[timeframe] == 1 else ((entry_price[timeframe] - btc_price) / entry_price[timeframe]) * 100
            profit_dollar_live = profit_pct_live * entry_price[timeframe] / 100
            current_trade = {
                'trade_id': trade_number + 1,
                'type': 'Long' if position[timeframe] == 1 else 'Short',
                'entry': entry_price[timeframe],
                'exit': None,
                'result': profit_pct_live,
                'profit': profit_dollar_live,
                'balance': balance,
                'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_datetime': None,
                'duration': None,
                'status': 'Open',
                'sl': sl_price[timeframe],
                'tp': tp_price[timeframe],
                'timeframe': timeframe
            }
            trades_with_status.insert(0, current_trade)
            trade_number += 1
    for trade in trades_with_status:
        if 'status' not in trade:
            trade['status'] = 'Completed'
        if 'timeframe' not in trade:
            trade['timeframe'] = '1m'
    # Calculate running balance for each trade in the table (newest to oldest)
    running_balance = balance
    for trade in trades_with_status:
        trade['balance'] = running_balance
        if trade['status'] == 'Completed':
            running_balance -= trade.get('profit', 0)
    print("Dashboard HIT - Prediction:", current_prediction)
    print("Trades in dashboard:", len(trades_with_status))
    print("Trade History in dashboard:", trades_with_status)
    print("Balance:", balance_str, "Profit:", profit_str, "Profit %:", profit_pct_str, "BTC Price:", btc_price_str, "BTC Change:", btc_change_str)
    last_btc_price = btc_price
    return render_template('dashboard.html', prediction=current_prediction, trades=trades_with_status, 
                          balance=balance_str, profit=profit_str, profit_pct=profit_pct_str, 
                          stats=stats, chart_data=json.dumps(chart_data), btc_price=btc_price_str, btc_change=btc_change_str)

@app.route('/trades')
def get_trades():
    """
    API endpoint to fetch trade data for the dashboard (used for periodic updates).
    
    Returns:
        dict: JSON response with trade data, predictions, stats, and chart data.
    """
    global trade_history, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length, indicators_at_entry, current_prediction
    
    # Recalculate balance and total profit from all completed trades
    balance = INITIAL_BALANCE
    total_profit = 0
    for trade in trade_history:
        if trade.get('status', 'Completed') == 'Completed':
            profit = trade.get('profit', 0)
            balance += profit
            total_profit += profit
    print(f"Trades route - Balance calculation: Initial = {INITIAL_BALANCE}, Total Profit = {total_profit}, Final Balance = {balance}")
    profit_pct = (total_profit / INITIAL_BALANCE) * 100 if INITIAL_BALANCE != 0 else 0
    balance_str = format_dollar(balance)
    profit_str = format_dollar(total_profit)
    profit_pct_str = f"{profit_pct:.2f}%"

    # Fallback: Fetch the Bitcoin price if current_btc_price is None
    btc_price = current_btc_price
    if btc_price is None:
        try:
            ticker = client.get_symbol_ticker(symbol="BTCUSDT")
            btc_price = float(ticker['price'])
            print(f"Fallback Bitcoin price fetched: ${format_dollar(btc_price)}")
        except Exception as e:
            print(f"Error fetching fallback Bitcoin price: {e}")
            btc_price = None

    btc_price_str = format_dollar(btc_price) if btc_price is not None else 'N/A'
    btc_change_pct = ((btc_price - last_btc_price) / last_btc_price * 100) if last_btc_price and btc_price else 0
    btc_change_str = f"{btc_change_pct:+.1f} %"
    stats = calculate_stats()
    # Use full trade_history for chart_data
    chart_data = {
        'labels': [trade.get('exit_datetime', '-') for trade in sorted(trade_history, key=lambda x: x['trade_id'])],
        'balances': [trade.get('balance', balance) for trade in sorted(trade_history, key=lambda x: x['trade_id'])]
    }
    update_chart = len(chart_data['labels']) != last_chart_data_length
    last_chart_data_length = len(chart_data['labels'])
    # Add current open trades if active (for table only)
    trades_with_status = sorted(trade_history, key=lambda x: x['trade_id'], reverse=True)[:20]
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and btc_price:
            profit_pct_live = ((btc_price - entry_price[timeframe]) / entry_price[timeframe]) * 100 if position[timeframe] == 1 else ((entry_price[timeframe] - btc_price) / entry_price[timeframe]) * 100
            profit_dollar_live = profit_pct_live * entry_price[timeframe] / 100
            current_trade = {
                'trade_id': trade_number + 1,
                'type': 'Long' if position[timeframe] == 1 else 'Short',
                'entry': entry_price[timeframe],
                'exit': None,
                'result': profit_pct_live,
                'profit': profit_dollar_live,
                'balance': balance,
                'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_datetime': None,
                'duration': None,
                'status': 'Open',
                'sl': sl_price[timeframe],
                'tp': tp_price[timeframe],
                'timeframe': timeframe
            }
            trades_with_status.insert(0, current_trade)
            trade_number += 1
    for trade in trades_with_status:
        if 'status' not in trade:
            trade['status'] = 'Completed'
        if 'timeframe' not in trade:
            trade['timeframe'] = '1m'
    # Calculate running balance for each trade in the table (newest to oldest)
    running_balance = balance
    for trade in trades_with_status:
        trade['balance'] = running_balance
        if trade['status'] == 'Completed':
            running_balance -= trade.get('profit', 0)
    last_btc_price = btc_price
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
    """
    Render the trade history page with all trades and stats.
    
    Returns:
        str: Rendered HTML template for the history page.
    """
    global trade_history, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length, indicators_at_entry, current_prediction
    
    # Recalculate balance and total profit from all completed trades
    balance = INITIAL_BALANCE
    total_profit = 0
    for trade in trade_history:
        if trade.get('status', 'Completed') == 'Completed':
            profit = trade.get('profit', 0)
            balance += profit
            total_profit += profit
    print(f"History route - Balance calculation: Initial = {INITIAL_BALANCE}, Total Profit = {total_profit}, Final Balance = {balance}")
    profit_pct = (total_profit / INITIAL_BALANCE) * 100 if INITIAL_BALANCE != 0 else 0
    balance_str = format_dollar(balance)
    profit_str = format_dollar(total_profit)
    profit_pct_str = f"{profit_pct:.2f}%"

    # Fallback: Fetch the Bitcoin price if current_btc_price is None
    btc_price = current_btc_price
    if btc_price is None:
        try:
            ticker = client.get_symbol_ticker(symbol="BTCUSDT")
            btc_price = float(ticker['price'])
            print(f"Fallback Bitcoin price fetched: ${format_dollar(btc_price)}")
        except Exception as e:
            print(f"Error fetching fallback Bitcoin price: {e}")
            btc_price = None

    btc_price_str = format_dollar(btc_price) if btc_price is not None else 'N/A'
    btc_change_pct = ((btc_price - last_btc_price) / last_btc_price * 100) if last_btc_price and btc_price else 0
    btc_change_str = f"{btc_change_pct:+.1f} %"
    stats = calculate_stats()
    chart_data = {
        'labels': [trade.get('exit_datetime', '-') for trade in sorted(trade_history, key=lambda x: x['trade_id'])],
        'balances': [trade.get('balance', balance) for trade in sorted(trade_history, key=lambda x: x['trade_id'])]
    }
    # Add current open trades if active
    trades_with_status = sorted(trade_history, key=lambda x: x['trade_id'], reverse=True)
    for timeframe in ['1m', '1h', '1d']:
        if position[timeframe] != 0 and btc_price:
            profit_pct_live = ((btc_price - entry_price[timeframe]) / entry_price[timeframe]) * 100 if position[timeframe] == 1 else ((entry_price[timeframe] - btc_price) / entry_price[timeframe]) * 100
            profit_dollar_live = profit_pct_live * entry_price[timeframe] / 100
            current_trade = {
                'trade_id': trade_number + 1,
                'type': 'Long' if position[timeframe] == 1 else 'Short',
                'entry': entry_price[timeframe],
                'exit': None,
                'result': profit_pct_live,
                'profit': profit_dollar_live,
                'balance': balance,
                'entry_datetime': entry_datetime[timeframe].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_datetime': None,
                'duration': None,
                'status': 'Open',
                'sl': sl_price[timeframe],
                'tp': tp_price[timeframe],
                'timeframe': timeframe
            }
            trades_with_status.insert(0, current_trade)
            trade_number += 1
    for trade in trades_with_status:
        if 'status' not in trade:
            trade['status'] = 'Completed'
        if 'timeframe' not in trade:
            trade['timeframe'] = '1m'
    # Calculate running balance for each trade in the table (newest to oldest)
    running_balance = balance
    for trade in trades_with_status:
        trade['balance'] = running_balance
        if trade['status'] == 'Completed':
            running_balance -= trade.get('profit', 0)
    print("History HIT - Trades:", len(trades_with_status))
    last_btc_price = btc_price
    return render_template('history.html', trades=trades_with_status, 
                          balance=balance_str, profit=profit_str, profit_pct=profit_pct_str, stats=stats, 
                          chart_data=json.dumps(chart_data), btc_price=btc_price_str, btc_change=btc_change_str)

@app.route('/trade_details/<int:trade_id>')
def trade_details(trade_id):
    """
    API endpoint to fetch details for a specific trade by ID.
    
    Args:
        trade_id (int): ID of the trade to fetch details for.
    
    Returns:
        dict: JSON response with trade details, indicators, stop-loss, and take-profit.
    """
    global trade_history, balance, position, entry_datetime, trade_number, entry_price, sl_price, tp_price, current_btc_price, last_btc_price, last_chart_data_length, indicators_at_entry, current_prediction
    
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
            indicators = indicators_at_entry.get(trade_id, indicators_at_entry.get(trade_number, {}))
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

def calculate_stats():
    """
    Calculate trading statistics (win percentages, total trades, profits) from trade history.
    
    Returns:
        dict: Statistics including win percentages and profits for short and long trades.
    """
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

if __name__ == '__main__':
    # Start WebSocket for multiple timeframes
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    twm.start()

    # Function to start a WebSocket stream with retry and error handling
    def start_kline_socket_with_retry(twm, timeframe, interval, max_attempts=5, delay=5):
        for attempt in range(max_attempts):
            try:
                twm.start_kline_socket(
                    callback=lambda msg: process_kline(msg, timeframe, models, df, current_prediction, indicators_at_entry, TRADE_HISTORY_FILE, LOCAL_TZ),
                    symbol='BTCUSDT',
                    interval=interval
                )
                print(f"Started {timeframe} WebSocket stream")
                return True
            except Exception as e:
                print(f"Failed to start {timeframe} WebSocket stream (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    print(f"Retrying in {delay} seconds...")
                    import time
                    time.sleep(delay)
                else:
                    print(f"Failed to start {timeframe} WebSocket stream after {max_attempts} attempts")
                    return False

    # Start WebSocket streams for each timeframe
    start_kline_socket_with_retry(twm, '1m', '1m')
    start_kline_socket_with_retry(twm, '1h', '1h')
    start_kline_socket_with_retry(twm, '1d', '1d')

    print(f"\nInitial Balance: ${format_number(balance)}")
    print("Starting Flask server...")
    app.run(debug=False, host='127.0.0.1', port=5000)
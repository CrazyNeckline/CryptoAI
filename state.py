# CryptoAI/state.py
# This file holds the global state variables for the CryptoAI application.

# Global variables to track trading state
trade_history = []
balance = 10000  # Initial balance
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
current_prediction = {'1m': {}, '1h': {}, '1d': {}}
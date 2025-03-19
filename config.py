# config.py
# This file contains constants and configuration settings for the trading bot.

import pytz
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Binance API credentials for accessing market data and WebSocket streams
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# File path for persisting trade history
TRADE_HISTORY_FILE = 'trade_history.json'

# Timezone for displaying dates (CET)
LOCAL_TZ = pytz.timezone('Europe/Berlin')

# Initial balance for the trading account
INITIAL_BALANCE = 10000
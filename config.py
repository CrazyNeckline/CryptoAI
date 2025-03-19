# config.py
# This file contains constants and configuration settings for the trading bot.

import pytz  # Import pytz for timezone handling

# Binance API credentials for accessing market data and WebSocket streams
BINANCE_API_KEY = 'IhjebRD6OLbcMdOqmL1c62sSuz4MLyMM5CtYQwG4JMN8B0itY7zHbewun8FEJ4la'
BINANCE_API_SECRET = 'REDACTED_API_SECRET'

# File path for persisting trade history
TRADE_HISTORY_FILE = 'trade_history.json'

# Timezone for displaying dates (CET)
LOCAL_TZ = pytz.timezone('Europe/Berlin')

# Initial balance for the trading account
INITIAL_BALANCE = 10000
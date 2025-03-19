# data.py
# This file handles data loading and preprocessing for the trading bot, including historical candle data.

import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from binance.client import Client

def preload_df(client, timeframe, interval, minutes):
    """
    Preload historical candle data for a given timeframe into a DataFrame and calculate technical indicators.
    
    Args:
        client (Client): Binance API client instance.
        timeframe (str): Timeframe identifier ('1m', '1h', '1d').
        interval (str): Binance interval for fetching candles (e.g., Client.KLINE_INTERVAL_1MINUTE).
        minutes (int): Number of minutes of historical data to fetch.
    
    Returns:
        pd.DataFrame: DataFrame with historical candles and technical indicators (SMA_20, RSI, ATR).
    """
    print(f"Preloading df for {timeframe} with {minutes} historical candles...")
    klines = client.get_historical_klines('BTCUSDT', interval, f"{minutes} minutes ago UTC")
    df = pd.DataFrame(klines, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 
                                       'Number_of_trades', 'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms').dt.tz_localize('UTC')
    print(f"Preloaded df[{timeframe}] with {len(df)} rows")
    
    # Calculate technical indicators
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator().fillna(0)
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi().fillna(0)
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().fillna(0)
    return df
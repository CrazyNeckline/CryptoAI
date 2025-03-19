# utils.py
# This file contains helper functions for formatting numbers, durations, and file I/O operations.

import json
import os
from datetime import datetime
import pytz

def format_number(number):
    """
    Format a number with thousands separators and two decimal places.
    
    Args:
        number (float): The number to format.
    
    Returns:
        str: Formatted string (e.g., "10'000.00") or '-' if number is None.
    """
    if number is None:
        return '-'
    return f"{int(number):,}".replace(',', "'") + f".{number:.2f}".split('.')[1]

def format_dollar(number):
    """
    Format a dollar amount with thousands separators and exactly two decimal places.
    
    Args:
        number (float): The dollar amount to format.
    
    Returns:
        str: Formatted string (e.g., "84'236.42") or '-' if number is None.
    """
    if number is None:
        return '-'
    # Format the number with two decimal places
    formatted = f"{number:.2f}"
    # Split into integer and decimal parts
    integer_part, decimal_part = formatted.split('.')
    # Add thousands separators to the integer part
    integer_part = f"{int(integer_part):,}".replace(',', "'")
    # Combine with the decimal part
    return f"{integer_part}.{decimal_part}"

def format_duration(seconds):
    """
    Format a duration in seconds into a human-readable string (seconds, minutes, hours, or days).
    
    Args:
        seconds (float): Duration in seconds.
    
    Returns:
        str: Formatted duration (e.g., "5.0 s", "2.5 m", "1.2 h", "3.0 d").
    """
    if seconds < 60:
        return f"{seconds:.0f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    else:
        return f"{seconds / 86400:.1f} d"

def load_trade_history(trade_history_file, trade_history, trade_number):
    """
    Load trade history from a JSON file into the global trade_history list and set trade_number.
    
    Args:
        trade_history_file (str): Path to the trade history JSON file.
        trade_history (list): Global list to store trade history.
        trade_number (int): Global variable to store the maximum trade ID.
    
    Returns:
        tuple: Updated (trade_history, trade_number).
    """
    if os.path.exists(trade_history_file):
        try:
            with open(trade_history_file, 'r') as f:
                trade_history.extend(json.load(f))
                if trade_history:
                    trade_number = max(trade['trade_id'] for trade in trade_history)
                print(f"Loaded {len(trade_history)} historic trades from {trade_history_file}, max trade_id: {trade_number}")
        except Exception as e:
            print(f"Error loading {trade_history_file}: {e}")
            trade_history = []
    else:
        print(f"No historic trade file found at {trade_history_file}")
        trade_history = []
    return trade_history, trade_number

def save_trade_history(trade_history_file, trade_history):
    """
    Save the trade history to a JSON file.
    
    Args:
        trade_history_file (str): Path to the trade history JSON file.
        trade_history (list): List of trade records to save.
    """
    try:
        with open(trade_history_file, 'w') as f:
            json.dump(trade_history, f)
        print(f"Saved {len(trade_history)} trades to {trade_history_file}")
    except Exception as e:
        print(f"Error saving {trade_history_file}: {e}")
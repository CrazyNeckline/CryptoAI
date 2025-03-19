# CryptoAI/utils.py
# This file contains utility functions for the CryptoAI application.

import json
import os

def format_dollar(amount):
    """
    Format a number as a dollar amount with commas and two decimal places.
    
    Args:
        amount (float): The amount to format.
    
    Returns:
        str: Formatted dollar amount.
    """
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"

def format_number(number):
    """
    Format a number with commas.
    
    Args:
        number (float): The number to format.
    
    Returns:
        str: Formatted number.
    """
    return f"{number:,}"

def save_trade_history(file_path, trade_history):
    """
    Save the trade history to a JSON file.
    
    Args:
        file_path (str): Path to the trade history file.
        trade_history (list): List of trade dictionaries to save.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(trade_history, f, indent=4)
        print(f"Saved trade history to {file_path}")
    except Exception as e:
        print(f"Error saving trade history to {file_path}: {e}")

def load_trade_history(file_path, trade_history, trade_number):
    """
    Load the trade history from a JSON file.
    
    Args:
        file_path (str): Path to the trade history file.
        trade_history (list): Current trade history list.
        trade_number (int): Current trade number.
    
    Returns:
        tuple: Updated trade history and trade number.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                loaded_history = json.load(f)
            trade_history.extend(loaded_history)
            if trade_history:
                trade_number = max(trade['trade_id'] for trade in trade_history)
            print(f"Loaded trade history from {file_path}")
        except Exception as e:
            print(f"Error loading trade history from {file_path}: {e}")
    return trade_history, trade_number
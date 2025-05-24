import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # ATR (Average True Range)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df

def find_support_resistance(df, window=20, threshold=0.02):
    """
    Find support and resistance levels using local minima and maxima.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        window (int): Window size for finding local extrema
        threshold (float): Minimum price change threshold for levels
    
    Returns:
        tuple: (support_levels, resistance_levels)
    """
    # Find local minima (potential support levels)
    df['is_min'] = df['Low'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] == min(x), raw=True
    )
    
    # Find local maxima (potential resistance levels)
    df['is_max'] = df['High'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] == max(x), raw=True
    )
    
    # Get support and resistance levels using boolean mask
    support_levels = df[df['is_min'] == 1.0]['Low'].tolist()
    resistance_levels = df[df['is_max'] == 1.0]['High'].tolist()
    
    # Filter levels based on threshold
    filtered_support = []
    filtered_resistance = []
    
    for level in support_levels:
        if not filtered_support or abs(level - filtered_support[-1]) / filtered_support[-1] > threshold:
            filtered_support.append(level)
    
    for level in resistance_levels:
        if not filtered_resistance or abs(level - filtered_resistance[-1]) / filtered_resistance[-1] > threshold:
            filtered_resistance.append(level)
    
    # Clean up temp columns
    df.drop(['is_min', 'is_max'], axis=1, inplace=True)
    
    return filtered_support, filtered_resistance

def calculate_candlestick_levels(df, price_tolerance=0.01, lookback_periods=104, decay_factor=0.98):
    """
    Calculate support and resistance levels based on candlestick analysis with weighted importance.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        price_tolerance (float): Percentage tolerance for grouping price levels
        lookback_periods (int): Number of periods to look back for analysis
        decay_factor (float): Factor to decay importance of older data (0-1)
    
    Returns:
        tuple: (support_levels, resistance_levels) where each is a list of (price, strength) tuples
    """
    # Make a copy of the recent data
    recent_data = df.tail(lookback_periods).copy()
    
    # Initialize price clusters
    price_clusters = defaultdict(lambda: {'opens': 0, 'closes': 0, 'wicks': 0, 'strength': 0})
    
    # Calculate weights for each period (more recent = higher weight)
    period_weights = np.array([decay_factor ** i for i in range(len(recent_data))])
    period_weights = period_weights / period_weights.sum()  # Normalize weights
    
    # Analyze each candlestick
    for idx, (_, row) in enumerate(recent_data.iterrows()):
        weight = period_weights[idx]
        
        # Get price levels for this candle
        levels = {
            'open': row['Open'],
            'close': row['Close'],
            'high': row['High'],
            'low': row['Low']
        }
        
        # Group price levels within tolerance
        for price in levels.values():
            # Find the base level (rounded to nearest tolerance)
            base_level = round(price / price_tolerance) * price_tolerance
            
            # Add weighted contribution to the cluster
            if price in [row['Open'], row['Close']]:
                price_clusters[base_level]['opens'] += weight
                price_clusters[base_level]['closes'] += weight
            else:  # High/Low (wicks)
                price_clusters[base_level]['wicks'] += weight
    
    # Calculate strength for each level
    support_levels = []
    resistance_levels = []
    
    for level, data in price_clusters.items():
        # Calculate total strength (weighted sum of opens, closes, and wicks)
        total_strength = (
            data['opens'] * 1.0 +  # Full weight for opens
            data['closes'] * 1.0 +  # Full weight for closes
            data['wicks'] * 0.5     # Half weight for wicks
        )
        
        # Only consider levels with significant strength
        if total_strength > 0.05:  # Lowered minimum strength threshold
            # Determine if it's support or resistance based on price action
            price_above_level = recent_data['Close'] > level
            price_below_level = recent_data['Close'] < level
            
            # If price has bounced off this level multiple times, it's a significant level
            if sum(price_above_level) > len(recent_data) * 0.2 and sum(price_below_level) > len(recent_data) * 0.2:  # Lowered threshold
                if recent_data['Close'].iloc[-1] > level:
                    resistance_levels.append((level, total_strength))
                else:
                    support_levels.append((level, total_strength))
    
    # Sort levels by strength
    support_levels.sort(key=lambda x: x[1], reverse=True)
    resistance_levels.sort(key=lambda x: x[1], reverse=True)
    
    return support_levels, resistance_levels

def analyze_current_levels(df, current_price):
    """
    Analyze which support and resistance levels are currently active.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        current_price (float): Current price to analyze
    
    Returns:
        dict: Dictionary with active support and resistance levels
    """
    # Get traditional support/resistance levels
    traditional_support, traditional_resistance = find_support_resistance(df)
    
    # Get candlestick-based levels
    candlestick_support, candlestick_resistance = calculate_candlestick_levels(df)
    
    # Combine and sort levels
    all_support = [(price, 'traditional') for price in traditional_support] + \
                 [(price, f'candlestick (strength: {strength:.2f})') for price, strength in candlestick_support]
    all_resistance = [(price, 'traditional') for price in traditional_resistance] + \
                    [(price, f'candlestick (strength: {strength:.2f})') for price, strength in candlestick_resistance]
    
    # Add moving averages as support/resistance
    ma_labels = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']
    for ma in ma_labels:
        ma_value = df[ma].iloc[-1]
        if not np.isnan(ma_value):
            label = f"{ma} (moving average)"
            if ma_value < current_price:
                all_support.append((ma_value, label))
            else:
                all_resistance.append((ma_value, label))
    
    # Filter to only levels within ±20% of current price
    lower_bound = current_price * 0.8
    upper_bound = current_price * 1.2
    all_support = [(price, desc) for price, desc in all_support if lower_bound <= price < current_price]
    all_resistance = [(price, desc) for price, desc in all_resistance if current_price < price <= upper_bound]
    
    # Find nearest levels
    nearest_support = max([(price, desc) for price, desc in all_support], key=lambda x: x[0], default=(None, None))
    nearest_resistance = min([(price, desc) for price, desc in all_resistance], key=lambda x: x[0], default=(None, None))
    
    return {
        'nearest_support': nearest_support[0] if nearest_support[0] is not None else None,
        'nearest_support_type': nearest_support[1] if nearest_support[0] is not None else None,
        'nearest_resistance': nearest_resistance[0] if nearest_resistance[0] is not None else None,
        'nearest_resistance_type': nearest_resistance[1] if nearest_resistance[0] is not None else None,
        'all_support_levels': all_support,
        'all_resistance_levels': all_resistance
    }

if __name__ == "__main__":
    # Load the weekly Bitcoin data
    df = pd.read_csv('bitcoin_data_weekly.csv', index_col='Timestamp', parse_dates=True)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Save the enhanced dataset
    df.to_csv('bitcoin_weekly_technical.csv')
    print("Technical analysis completed and saved to bitcoin_weekly_technical.csv")
    
    # Get current price and analyze levels
    current_price = df['Close'].iloc[-1]
    levels = analyze_current_levels(df, current_price)
    
    # Print current analysis
    print("\nCurrent Price Analysis:")
    print(f"Current Price: ${current_price:,.2f}")
    
    # Print support levels around $100,000
    print("\nSupport Levels around $100,000:")
    print("-" * 50)
    print(f"{'Price':>15} | {'Type':<30}")
    print("-" * 50)
    for price, desc in sorted(levels['all_support_levels'], key=lambda x: x[0], reverse=True):
        if 80000 <= price <= 120000:  # Show levels within ±20% of $100,000
            print(f"${price:>13,.2f} | {desc:<30}")
    
    # Print resistance levels around $100,000
    print("\nResistance Levels around $100,000:")
    print("-" * 50)
    print(f"{'Price':>15} | {'Type':<30}")
    print("-" * 50)
    for price, desc in sorted(levels['all_resistance_levels']):
        if 80000 <= price <= 120000:  # Show levels within ±20% of $100,000
            print(f"${price:>13,.2f} | {desc:<30}")
    
    # Print some key technical indicators
    print("\nLatest Technical Indicators:")
    latest = df.iloc[-1]
    print(f"RSI (14): {latest['RSI']:.2f}")
    print(f"MACD: {latest['MACD']:.2f}")
    print(f"MACD Signal: {latest['MACD_Signal']:.2f}")
    print(f"MACD Histogram: {latest['MACD_Histogram']:.2f}")
    print(f"20-week SMA: ${latest['SMA_20']:,.2f}")
    print(f"50-week SMA: ${latest['SMA_50']:,.2f}")
    print(f"200-week SMA: ${latest['SMA_200']:,.2f}") 
import pandas as pd
import numpy as np

# --- Candlestick Pattern Recognition ---
def is_doji(open_price, high_price, low_price, close_price, body_threshold=0.05):
    """Identifies a Doji pattern."""
    body_size = abs(open_price - close_price)
    total_range = high_price - low_price
    if total_range == 0: # Avoid division by zero if H=L
        return body_size == 0 # True if O=C and H=L
    return (body_size / total_range) < body_threshold

def is_hammer(open_price, high_price, low_price, close_price, body_max_ratio=0.33, lower_shadow_min_ratio=2.0, upper_shadow_max_ratio=0.1):
    """Identifies a Hammer pattern (assumes context of downtrend is checked elsewhere)."""
    body = abs(open_price - close_price)
    total_range = high_price - low_price
    if total_range == 0: return False # Cannot be a hammer if no range

    lower_shadow = min(open_price, close_price) - low_price
    upper_shadow = high_price - max(open_price, close_price)

    is_small_body = (body / total_range) <= body_max_ratio if total_range > 0 else True
    is_long_lower_shadow = lower_shadow >= (body * lower_shadow_min_ratio) if body > 0 else lower_shadow > 0 # Ensure shadow exists if body is zero
    is_short_upper_shadow = (upper_shadow / total_range) < upper_shadow_max_ratio if total_range > 0 else True
    
    return is_small_body and is_long_lower_shadow and is_short_upper_shadow

def is_hanging_man(open_price, high_price, low_price, close_price, body_max_ratio=0.33, lower_shadow_min_ratio=2.0, upper_shadow_max_ratio=0.1):
    """Identifies a Hanging Man pattern (assumes context of uptrend is checked elsewhere)."""
    # Same visual characteristics as Hammer
    return is_hammer(open_price, high_price, low_price, close_price, body_max_ratio, lower_shadow_min_ratio, upper_shadow_max_ratio)

def is_bullish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    """Identifies a Bullish Engulfing pattern (o1,c1 are previous; o2,c2 are current)."""
    is_prev_bearish = c1 < o1
    is_curr_bullish = c2 > o2
    engulfs_body = (c2 > o1) and (o2 < c1)
    # Optional: current candle engulfs entire previous candle range
    # engulfs_range = (c2 > h1) and (o2 < l1) 
    return is_prev_bearish and is_curr_bullish and engulfs_body

def is_bearish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    """Identifies a Bearish Engulfing pattern (o1,c1 are previous; o2,c2 are current)."""
    is_prev_bullish = c1 > o1
    is_curr_bearish = c2 < o2
    engulfs_body = (o2 > c1) and (c2 < o1)
    # Optional: current candle engulfs entire previous candle range
    # engulfs_range = (o2 > h1) and (c2 < l1)
    return is_prev_bullish and is_curr_bearish and engulfs_body


# --- Support and Resistance Level Identification ---

def get_pivot_points(prev_high, prev_low, prev_close):
    """Calculates Pivot Points, Support and Resistance levels."""
    pp = (prev_high + prev_low + prev_close) / 3
    s1 = (pp * 2) - prev_high
    r1 = (pp * 2) - prev_low
    s2 = pp - (prev_high - prev_low)
    r2 = pp + (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    r3 = prev_high + 2 * (pp - prev_low)
    return {"pp": pp, "s1": s1, "s2": s2, "s3": s3, "r1": r1, "r2": r2, "r3": r3}

def find_swing_highs_lows(prices: pd.Series, window: int = 5):
    """Identifies swing highs and lows based on a rolling window.
       A swing high is a price peak higher than prices within the window on either side.
       A swing low is a price trough lower than prices within the window on either side.
       Returns two series: swing_highs and swing_lows with prices at swing points, else NaN.
    """
    # Ensure window is odd for a central point
    if window % 2 == 0:
        window += 1
    
    half_window = window // 2
    
    highs = prices.rolling(window=window, center=True).apply(lambda x: x[half_window] == np.max(x), raw=True).fillna(0).astype(bool)
    lows = prices.rolling(window=window, center=True).apply(lambda x: x[half_window] == np.min(x), raw=True).fillna(0).astype(bool)
    
    swing_highs = pd.Series(np.where(highs, prices, np.nan), index=prices.index)
    swing_lows = pd.Series(np.where(lows, prices, np.nan), index=prices.index)
    
    return swing_highs.dropna(), swing_lows.dropna()


if __name__ == '__main__':
    # Sample data for candlestick patterns
    # Data: O, H, L, C
    doji_data = [100, 105, 95, 100.1] # Classic Doji
    hammer_data = [100, 102, 90, 101] # Hammer (assuming downtrend)
    hanging_man_data = [100, 102, 90, 101] # Hanging Man (assuming uptrend)
    bullish_eng_prev = [100, 101, 98, 99] # Previous candle for Bullish Engulfing (Bearish)
    bullish_eng_curr = [98.5, 102, 98, 101.5] # Current candle for Bullish Engulfing (Bullish, engulfs prev body)
    bearish_eng_prev = [100, 102, 99, 101] # Previous candle for Bearish Engulfing (Bullish)
    bearish_eng_curr = [101.5, 102, 98, 98.5] # Current candle for Bearish Engulfing (Bearish, engulfs prev body)

    print("--- Candlestick Tests ---")
    print(f"Is Doji: {is_doji(*doji_data)}")
    print(f"Is Hammer: {is_hammer(*hammer_data)}")
    print(f"Is Hanging Man: {is_hanging_man(*hanging_man_data)}")
    print(f"Is Bullish Engulfing: {is_bullish_engulfing(*bullish_eng_prev, *bullish_eng_curr)}")
    print(f"Is Bearish Engulfing: {is_bearish_engulfing(*bearish_eng_prev, *bearish_eng_curr)}")

    # Sample data for Support/Resistance
    prev_day_data = {"high": 110, "low": 90, "close": 105} # Previous day OHLC for Pivot Points
    price_series_data = [100, 102, 98, 105, 103, 107, 100, 95, 99, 103, 101, 106, 104]
    price_series = pd.Series(price_series_data)

    print("\n--- Support/Resistance Tests ---")
    pivots = get_pivot_points(prev_day_data['high'], prev_day_data['low'], prev_day_data['close'])
    print(f"Pivot Points: {pivots}")

    swing_highs, swing_lows = find_swing_highs_lows(price_series, window=5)
    print(f"Swing Highs:\n{swing_highs}")
    print(f"Swing Lows:\n{swing_lows}")

    # Example with a more complex series for swing points
    data_ohlc = {
        'open':  [10, 11, 10, 12, 13, 12, 11, 10, 9, 8, 9, 10, 11, 10, 9],
        'high':  [12, 12, 11, 13, 14, 13, 12, 11, 10, 9, 10, 11, 12, 11, 10],
        'low':   [9,  10, 9,  11, 12, 11, 10, 9,  8, 7, 8, 9,  10, 9,  8],
        'close': [11, 10, 10, 13, 12, 12, 11, 10, 8, 9, 9, 10, 11, 10, 9]
    }
    df_ohlc = pd.DataFrame(data_ohlc)

    print("\n--- Candlestick detection on DataFrame ---")
    df_ohlc['is_doji'] = df_ohlc.apply(lambda row: is_doji(row['open'], row['high'], row['low'], row['close']), axis=1)
    print(df_ohlc[['open', 'high', 'low', 'close', 'is_doji']].head())

    # For engulfing, you'd typically iterate and compare row i with row i-1
    # Example: is_bullish_engulfing(df_ohlc.iloc[i-1]['open'], ..., df_ohlc.iloc[i]['open'], ...)


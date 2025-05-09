import pandas as pd

# Assume other modules are available for import if this were part of a larger package
# from technical_indicators import calculate_macd, calculate_rsi, calculate_stochastic_oscillator, calculate_adx
# from market_analysis import is_bullish_engulfing, is_bearish_engulfing, get_pivot_points, find_swing_highs_lows
# from additional_metrics import get_funding_rate_history, get_order_book, get_mvrv_z_score_mock

# --- Trading Strategy Logic ---

def generate_signals(ohlc_data, macd_line, macd_signal, rsi, slow_k, slow_d, adx, plus_di, minus_di,
                     candlestick_patterns, support_levels, resistance_levels,
                     funding_rate_data, order_book_data, mvrv_z_score_data):
    """
    Generates trading signals based on a combination of technical indicators and market metrics.
    This is a simplified example strategy.

    Args:
        ohlc_data (pd.DataFrame): DataFrame with columns [open, high, low, close, volume]
        macd_line (pd.Series): MACD line values.
        macd_signal (pd.Series): MACD signal line values.
        rsi (pd.Series): RSI values.
        slow_k (pd.Series): Stochastic %K values.
        slow_d (pd.Series): Stochastic %D values.
        adx (pd.Series): ADX values.
        plus_di (pd.Series): +DI values.
        minus_di (pd.Series): -DI values.
        candlestick_patterns (dict): Dictionary of identified candlestick patterns for the latest period.
                                     e.g., {"is_bullish_engulfing": True, "is_doji": False}
        support_levels (list): List of identified support price levels.
        resistance_levels (list): List of identified resistance price levels.
        funding_rate_data (pd.DataFrame or dict): Data on funding rates. Structure depends on implementation.
                                                May be empty or mock due to API limitations.
        order_book_data (dict): Processed order book data (bids, asks). May be None or mock.
        mvrv_z_score_data (dict): MVRV Z-Score data. May be mock.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    latest_close = ohlc_data["close"].iloc[-1]
    signal = "HOLD" # Default signal

    # --- Entry Conditions (BUY) ---
    buy_score = 0

    # 1. MACD Bullish Crossover
    if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-2] <= macd_signal.iloc[-2]:
        buy_score += 1
        print("Signal: MACD Bullish Crossover")

    # 2. RSI Oversold (e.g., < 30) and turning up
    if rsi.iloc[-1] < 30 and rsi.iloc[-1] > rsi.iloc[-2]:
        buy_score += 1
        print("Signal: RSI Oversold and rising")
    elif rsi.iloc[-1] < 40: # General bullish bias if RSI is low but not extremely oversold
        buy_score += 0.5

    # 3. Stochastic Oversold (%K < 20) and Bullish Crossover (%K > %D)
    if slow_k.iloc[-1] < 20 and slow_d.iloc[-1] < 20 and slow_k.iloc[-1] > slow_d.iloc[-1] and slow_k.iloc[-2] <= slow_d.iloc[-2]:
        buy_score += 1
        print("Signal: Stochastic Oversold Bullish Crossover")

    # 4. ADX indicates trend strength (e.g., ADX > 20-25) and +DI > -DI
    if adx.iloc[-1] > 20 and plus_di.iloc[-1] > minus_di.iloc[-1]:
        buy_score += 1
        print("Signal: ADX strong trend, +DI dominant")

    # 5. Bullish Candlestick Pattern
    if candlestick_patterns.get("is_bullish_engulfing") or candlestick_patterns.get("is_hammer") or candlestick_patterns.get("is_morning_star") :
        buy_score += 1.5 # Stronger signal
        print(f"Signal: Bullish Candlestick Pattern detected: { {k:v for k,v in candlestick_patterns.items() if v} }")

    # 6. Price near Support Level
    for s_level in support_levels:
        if abs(latest_close - s_level) / latest_close < 0.01: # Within 1% of support
            buy_score += 1
            print(f"Signal: Price near Support Level {s_level}")
            break
            
    # 7. MVRV-Z Score (if available and indicates undervaluation)
    if mvrv_z_score_data and mvrv_z_score_data.get("status") != "mock_data":
        if mvrv_z_score_data.get("mvrv_z_score", 0) < 0: # Example: Green zone
            buy_score += 1
            print("Signal: MVRV-Z Score indicates undervaluation")
    elif mvrv_z_score_data and mvrv_z_score_data.get("status") == "mock_data" and mvrv_z_score_data.get("mvrv_z_score", 0) < 0.5:
        buy_score += 0.5 # Mild positive bias for mock data in green-ish zone
        print("Signal: MVRV-Z Score (mock) in potentially undervalued zone")

    # 8. Funding Rates (if available and not excessively positive or are negative)
    # This is highly dependent on how funding_rate_data is structured and if it is live.
    # Example: if funding_rate_data is not empty and latest rate < 0.0005 (0.05%)
    if funding_rate_data is not None and not funding_rate_data.empty:
        latest_funding_rate = funding_rate_data["fundingRate"].iloc[-1]
        if latest_funding_rate <= 0: # Negative or zero funding favorable for longs
            buy_score += 0.5
            print(f"Signal: Favorable funding rate for longs: {latest_funding_rate}")

    # --- Exit Conditions (SELL) ---
    sell_score = 0

    # 1. MACD Bearish Crossover
    if macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_line.iloc[-2] >= macd_signal.iloc[-2]:
        sell_score += 1
        print("Signal: MACD Bearish Crossover")

    # 2. RSI Overbought (e.g., > 70) and turning down
    if rsi.iloc[-1] > 70 and rsi.iloc[-1] < rsi.iloc[-2]:
        sell_score += 1
        print("Signal: RSI Overbought and falling")
    elif rsi.iloc[-1] > 60:
        sell_score += 0.5
        
    # 3. Stochastic Overbought (%K > 80) and Bearish Crossover (%K < %D)
    if slow_k.iloc[-1] > 80 and slow_d.iloc[-1] > 80 and slow_k.iloc[-1] < slow_d.iloc[-1] and slow_k.iloc[-2] >= slow_d.iloc[-2]:
        sell_score += 1
        print("Signal: Stochastic Overbought Bearish Crossover")

    # 4. ADX indicates trend strength (e.g., ADX > 20-25) and -DI > +DI
    if adx.iloc[-1] > 20 and minus_di.iloc[-1] > plus_di.iloc[-1]:
        sell_score += 1
        print("Signal: ADX strong trend, -DI dominant")

    # 5. Bearish Candlestick Pattern
    if candlestick_patterns.get("is_bearish_engulfing") or candlestick_patterns.get("is_hanging_man") or candlestick_patterns.get("is_evening_star"):
        sell_score += 1.5 # Stronger signal
        print(f"Signal: Bearish Candlestick Pattern detected: { {k:v for k,v in candlestick_patterns.items() if v} }")

    # 6. Price near Resistance Level
    for r_level in resistance_levels:
        if abs(latest_close - r_level) / latest_close < 0.01: # Within 1% of resistance
            sell_score += 1
            print(f"Signal: Price near Resistance Level {r_level}")
            break
            
    # 7. MVRV-Z Score (if available and indicates overvaluation)
    if mvrv_z_score_data and mvrv_z_score_data.get("status") != "mock_data":
        if mvrv_z_score_data.get("mvrv_z_score", 0) > 2.5: # Example: Red zone
            sell_score += 1
            print("Signal: MVRV-Z Score indicates overvaluation")
    elif mvrv_z_score_data and mvrv_z_score_data.get("status") == "mock_data" and mvrv_z_score_data.get("mvrv_z_score", 0) > 2.0:
        sell_score += 0.5 # Mild negative bias for mock data in red-ish zone
        print("Signal: MVRV-Z Score (mock) in potentially overvalued zone")

    # --- Decision Logic ---
    # This is a very basic threshold system. Real strategies are more nuanced.
    print(f"Final Buy Score: {buy_score}, Final Sell Score: {sell_score}")
    if buy_score >= 3.0 and buy_score > sell_score: # Threshold for BUY signal
        signal = "BUY"
    elif sell_score >= 3.0 and sell_score > buy_score:
        signal = "SELL"
    
    return signal

if __name__ == "__main__":
    print("--- Testing Trading Strategy Logic ---")

    # Create Sample Data (mimicking outputs from other modules)
    sample_ohlc = pd.DataFrame({
        "open": [100, 101, 102, 101, 103, 105, 104, 106, 107, 105],
        "high": [102, 103, 103, 104, 105, 106, 107, 108, 109, 108],
        "low":  [99, 100, 101, 100, 102, 104, 103, 105, 106, 104],
        "close":[101, 102, 101, 103, 105, 104, 106, 107, 105, 107],
        "volume":[1000, 1100, 1200, 1050, 1300, 1250, 1400, 1350, 1450, 1500]
    })
    sample_macd_line = pd.Series([-0.5, -0.2, 0.1, 0.3, 0.5, 0.4, 0.6, 0.7, 0.5, 0.8]) # Bullish crossover at index 2
    sample_macd_signal = pd.Series([-0.4, -0.3, -0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6])
    sample_rsi = pd.Series([40, 45, 25, 60, 65, 55, 75, 78, 60, 65]) # Oversold at index 2
    sample_slow_k = pd.Series([30, 35, 15, 60, 70, 65, 85, 88, 70, 75]) # Oversold at index 2
    sample_slow_d = pd.Series([32, 33, 18, 55, 65, 66, 80, 85, 75, 72]) # Bullish crossover at index 2 for stoch
    sample_adx = pd.Series([15, 18, 22, 25, 28, 26, 24, 22, 20, 23]) # Trend strength > 20 from index 2
    sample_plus_di = pd.Series([18, 20, 25, 28, 30, 28, 25, 22, 20, 26]) # +DI > -DI from index 2
    sample_minus_di = pd.Series([22, 20, 18, 15, 12, 14, 16, 18, 22, 19])
    
    sample_candlesticks = {"is_bullish_engulfing": False, "is_hammer": True} # Example: Hammer detected
    sample_supports = [100, 95]
    sample_resistances = [110, 115]
    
    # Mocked data for other metrics (as they might fail in sandbox)
    sample_funding_rates = pd.DataFrame({"fundingTime": [pd.Timestamp.now()], "symbol": ["BTCUSDT"], "fundingRate": [-0.0001]})
    sample_order_book = {"bids": pd.DataFrame({"price": [106.5], "quantity": [10]}), "asks": pd.DataFrame({"price": [107.5], "quantity": [12]})}
    sample_mvrv = {"symbol": "BTC", "mvrv_z_score": -0.5, "status": "mock_data"} # Undervalued mock

    print("\n--- Scenario 1: Potential BUY ---")
    # Simulate conditions for a BUY signal
    # Make MACD cross up recently
    test_macd_line = sample_macd_line.copy()
    test_macd_line.iloc[-1] = 0.8
    test_macd_line.iloc[-2] = 0.5
    test_macd_signal = sample_macd_signal.copy()
    test_macd_signal.iloc[-1] = 0.7
    test_macd_signal.iloc[-2] = 0.6
    
    # RSI low
    test_rsi = sample_rsi.copy()
    test_rsi.iloc[-1] = 35
    test_rsi.iloc[-2] = 28 # Rising from oversold
    
    # Stoch low and crossing
    test_slow_k = sample_slow_k.copy()
    test_slow_k.iloc[-1] = 25
    test_slow_k.iloc[-2] = 18
    test_slow_d = sample_slow_d.copy()
    test_slow_d.iloc[-1] = 22
    test_slow_d.iloc[-2] = 20
    
    # ADX trending and +DI dominant
    test_adx = sample_adx.copy()
    test_adx.iloc[-1] = 25
    test_plus_di = sample_plus_di.copy()
    test_plus_di.iloc[-1] = 30
    test_minus_di = sample_minus_di.copy()
    test_minus_di.iloc[-1] = 15
    
    test_candlesticks = {"is_bullish_engulfing": True}
    test_supports = [sample_ohlc["close"].iloc[-1] - 0.5] # Price near support

    signal = generate_signals(sample_ohlc, test_macd_line, test_macd_signal, test_rsi, test_slow_k, test_slow_d, 
                              test_adx, test_plus_di, test_minus_di,
                              test_candlesticks, test_supports, sample_resistances,
                              sample_funding_rates, sample_order_book, sample_mvrv)
    print(f"Generated Signal for Scenario 1: {signal}")

    print("\n--- Scenario 2: Potential SELL ---")
    # Simulate conditions for a SELL signal
    test_macd_line.iloc[-1] = -0.5
    test_macd_line.iloc[-2] = -0.2
    test_macd_signal.iloc[-1] = -0.4
    test_macd_signal.iloc[-2] = -0.3 # Bearish crossover

    test_rsi.iloc[-1] = 75 
    test_rsi.iloc[-2] = 78 # Falling from overbought
    
    test_slow_k.iloc[-1] = 75
    test_slow_k.iloc[-2] = 82
    test_slow_d.iloc[-1] = 78
    test_slow_d.iloc[-2] = 80 # Bearish crossover
    
    test_adx.iloc[-1] = 25
    test_plus_di.iloc[-1] = 15
    test_minus_di.iloc[-1] = 30 # -DI dominant
    
    test_candlesticks = {"is_bearish_engulfing": True}
    test_resistances = [sample_ohlc["close"].iloc[-1] + 0.5] # Price near resistance
    sample_mvrv_sell = {"symbol": "BTC", "mvrv_z_score": 2.8, "status": "mock_data"} # Overvalued mock

    signal_sell = generate_signals(sample_ohlc, test_macd_line, test_macd_signal, test_rsi, test_slow_k, test_slow_d, 
                                 test_adx, test_plus_di, test_minus_di,
                                 test_candlesticks, sample_supports, test_resistances,
                                 sample_funding_rates, sample_order_book, sample_mvrv_sell)
    print(f"Generated Signal for Scenario 2: {signal_sell}")

    print("\nNote: The scoring and thresholds in this strategy are examples and should be tuned.")
    print("Note: Live data for funding rates and order book may be affected by sandbox limitations (451 errors).")


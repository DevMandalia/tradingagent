import json
import pandas as pd
import numpy as np
import sys

# Add the directory containing custom modules to the Python path
sys.path.append("/home/ubuntu/")

from technical_indicators import calculate_macd, calculate_rsi, calculate_stochastic_oscillator, calculate_adx
from market_analysis import is_doji, is_hammer, is_hanging_man, is_bullish_engulfing, is_bearish_engulfing, get_pivot_points, find_swing_highs_lows
from additional_metrics import get_funding_rate_history, get_order_book, get_mvrv_z_score_mock # funding and order book will be mocked effectively due to sandbox limitations
from trading_strategy import generate_signals

def load_historical_data(filepath="/home/ubuntu/btc_historical_data.json"):
    """Loads and preprocesses historical data from the JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not (data and data.get("chart") and data["chart"].get("result") and data["chart"]["result"][0]):
            print("Historical data JSON is not in the expected format or is empty.")
            return None

        chart_data = data["chart"]["result"][0]
        timestamps = chart_data["timestamp"]
        ohlcv = chart_data["indicators"]["quote"][0]
        adjclose = chart_data["indicators"]["adjclose"][0]["adjclose"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="s"),
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv["volume"],
            "adjclose": adjclose
        })
        df = df.dropna() # Remove any rows with NaN values, common at the start of indicator calculations
        df = df.set_index("timestamp")
        print(f"Loaded {len(df)} rows of historical data.")
        return df
    except FileNotFoundError:
        print(f"Error: Historical data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading or processing historical data: {e}")
        return None

def run_backtest(historical_df):
    """Runs the backtesting simulation."""
    if historical_df is None or historical_df.empty:
        print("No historical data to backtest.")
        return

    # Portfolio simulation variables
    initial_capital = 10000.0
    capital = initial_capital
    position = 0 # Number of units of asset held
    trades = []
    min_indicator_period = 30 # Minimum periods needed for some indicators like ADX, longer EMAs in MACD

    if len(historical_df) < min_indicator_period:
        print(f"Not enough historical data for indicator calculation. Need at least {min_indicator_period} periods, got {len(historical_df)}.")
        return

    print(f"Starting backtest with initial capital: ${initial_capital:.2f}")

    for i in range(min_indicator_period, len(historical_df)):
        current_data_window = historical_df.iloc[:i+1] # Data up to current point
        latest_ohlc_row = current_data_window.iloc[-1]
        prev_ohlc_row = current_data_window.iloc[-2] # For candlestick patterns and pivot points

        # 1. Calculate Technical Indicators
        close_prices = current_data_window["close"]
        high_prices = current_data_window["high"]
        low_prices = current_data_window["low"]

        macd_line, macd_signal, _ = calculate_macd(close_prices)
        rsi = calculate_rsi(close_prices)
        slow_k, slow_d = calculate_stochastic_oscillator(high_prices, low_prices, close_prices)
        adx, plus_di, minus_di = calculate_adx(high_prices, low_prices, close_prices)

        # Ensure indicators have values for the current period
        if any(s.empty or len(s) <= i for s in [macd_line, macd_signal, rsi, slow_k, slow_d, adx, plus_di, minus_di]):
            # print(f"Skipping period {current_data_window.index[i]} due to insufficient indicator data.")
            continue

        # 2. Candlestick Patterns (for the latest candle)
        candlestick_patterns = {
            "is_doji": is_doji(latest_ohlc_row["open"], latest_ohlc_row["high"], latest_ohlc_row["low"], latest_ohlc_row["close"]),
            "is_hammer": is_hammer(latest_ohlc_row["open"], latest_ohlc_row["high"], latest_ohlc_row["low"], latest_ohlc_row["close"]),
            "is_hanging_man": is_hanging_man(latest_ohlc_row["open"], latest_ohlc_row["high"], latest_ohlc_row["low"], latest_ohlc_row["close"]),
            "is_bullish_engulfing": is_bullish_engulfing(prev_ohlc_row["open"], prev_ohlc_row["high"], prev_ohlc_row["low"], prev_ohlc_row["close"],
                                                     latest_ohlc_row["open"], latest_ohlc_row["high"], latest_ohlc_row["low"], latest_ohlc_row["close"]),
            "is_bearish_engulfing": is_bearish_engulfing(prev_ohlc_row["open"], prev_ohlc_row["high"], prev_ohlc_row["low"], prev_ohlc_row["close"], 
                                                     latest_ohlc_row["open"], latest_ohlc_row["high"], latest_ohlc_row["low"], latest_ohlc_row["close"])
        }

        # 3. Support/Resistance
        # Using pivot points from previous day's actual data
        pivots = get_pivot_points(prev_ohlc_row["high"], prev_ohlc_row["low"], prev_ohlc_row["close"])
        support_levels = [pivots["s1"], pivots["s2"], pivots["s3"]]
        resistance_levels = [pivots["r1"], pivots["r2"], pivots["r3"]]
        # Swing highs/lows could also be used but require more context or a longer lookback on the series passed to them

        # 4. Additional Metrics (Mocked/Placeholder)
        # In a live scenario, these would be fetched for the current time or latest available
        funding_rate_data = pd.DataFrame() # Mock empty, as live API calls fail in sandbox
        order_book_data = None # Mock None
        mvrv_z_score_data = get_mvrv_z_score_mock() # Uses mock data

        # 5. Generate Signal
        # Pass only the series for indicators, not the full history if not needed by strategy
        signal = generate_signals(
            current_data_window, macd_line, macd_signal, rsi, slow_k, slow_d, adx, plus_di, minus_di,
            candlestick_patterns, support_levels, resistance_levels,
            funding_rate_data, order_book_data, mvrv_z_score_data
        )
        
        current_price = latest_ohlc_row["close"]
        current_time = current_data_window.index[i]

        # 6. Simulate Trading (Simplified)
        if signal == "BUY" and position == 0:
            position = capital / current_price
            capital = 0
            trades.append({"time": current_time, "type": "BUY", "price": current_price, "units": position, "capital_after": capital + position * current_price})
            print(f"{current_time}: BUY {position:.4f} units at ${current_price:.2f}")
        elif signal == "SELL" and position > 0:
            capital = position * current_price
            trades.append({"time": current_time, "type": "SELL", "price": current_price, "units": position, "capital_after": capital})
            print(f"{current_time}: SELL {position:.4f} units at ${current_price:.2f}, Capital: ${capital:.2f}")
            position = 0
        
    # End of backtest loop
    final_portfolio_value = capital + (position * historical_df["close"].iloc[-1] if position > 0 else 0)
    profit_loss = final_portfolio_value - initial_capital
    profit_loss_percent = (profit_loss / initial_capital) * 100

    print("\n--- Backtest Results ---")
    print(f"Period: {historical_df.index[min_indicator_period]} to {historical_df.index[-1]}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Trades: {len(trades)}")
    print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_percent:.2f}%)")
    
    # Save trades to a file for review
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("/home/ubuntu/backtest_trades.csv", index=False)
        print("Trades saved to /home/ubuntu/backtest_trades.csv")
    else:
        print("No trades were executed during the backtest.")

if __name__ == "__main__":
    historical_data_df = load_historical_data()
    if historical_data_df is not None:
        run_backtest(historical_data_df)
    else:
        print("Could not run backtest due to data loading issues.")


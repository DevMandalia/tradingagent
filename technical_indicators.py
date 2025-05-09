import pandas as pd
import numpy as np

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices: pd.Series, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    Returns MACD line, Signal line, and MACD Histogram.
    """
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    macd_line = ema_short - ema_long
    signal_line = calculate_ema(macd_line, signal_period)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean() # First value is simple MA
    avg_loss = loss.rolling(window=period, min_periods=1).mean() # First value is simple MA

    # Wilder's smoothing for subsequent values
    # For a more accurate Wilder's smoothing, the first value is a simple average,
    # then: AvgGain_t = ((AvgGain_t-1 * (n-1)) + CurrentGain_t) / n
    # Pandas' ewm with alpha = 1/period is equivalent to Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0) # Fill initial NaNs if any, or handle appropriately
    return rsi

def calculate_stochastic_oscillator(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, k_period: int = 14, d_period: int = 3, smooth_k_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Slow Stochastic Oscillator (%K and %D).
    """
    lowest_low_k = low_prices.rolling(window=k_period).min()
    highest_high_k = high_prices.rolling(window=k_period).max()

    fast_k = ((close_prices - lowest_low_k) / (highest_high_k - lowest_low_k)) * 100
    slow_k = fast_k.rolling(window=smooth_k_period).mean() # This is the %K for Slow Stochastic
    slow_d = slow_k.rolling(window=d_period).mean()      # This is the %D for Slow Stochastic

    return slow_k, slow_d

def calculate_adx(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI.
    Uses Wilder's smoothing (EMA with alpha = 1/period).
    """
    # Calculate +DM, -DM, and TR
    move_up = high_prices.diff()
    move_down = -low_prices.diff()

    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high_prices.index)
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=high_prices.index)

    tr1 = pd.DataFrame(high_prices - low_prices)
    tr2 = pd.DataFrame(abs(high_prices - close_prices.shift(1)))
    tr3 = pd.DataFrame(abs(low_prices - close_prices.shift(1)))
    tr_df = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = tr_df.max(axis=1)

    # Smooth +DM, -DM, and TR using Wilder's smoothing (EMA with alpha = 1/N)
    # The first value of a Wilder's MA is a simple N-period average.
    # However, using ewm(alpha=1/N, adjust=False) is a common approximation.
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    smooth_tr = true_range.ewm(alpha=1/period, adjust=False).mean()

    # Calculate +DI and -DI
    plus_di = (smooth_plus_dm / smooth_tr) * 100
    minus_di = (smooth_minus_dm / smooth_tr) * 100

    # Calculate DX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    dx = dx.fillna(0) # Handle potential division by zero if plus_di + minus_di is 0

    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di

if __name__ == '__main__':
    # Create sample data for testing
    data = {
        'high': [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21],
        'low':  [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19],
        'close':[9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    }
    df = pd.DataFrame(data)

    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']

    # Test MACD
    macd_line, signal_line, macd_hist = calculate_macd(close_prices)
    print("MACD Line:\n", macd_line.tail())
    print("Signal Line:\n", signal_line.tail())
    print("MACD Histogram:\n", macd_hist.tail())

    # Test RSI
    rsi = calculate_rsi(close_prices)
    print("\nRSI:\n", rsi.tail())

    # Test Stochastic Oscillator
    slow_k, slow_d = calculate_stochastic_oscillator(high_prices, low_prices, close_prices)
    print("\nSlow %K:\n", slow_k.tail())
    print("Slow %D:\n", slow_d.tail())

    # Test ADX
    adx, plus_di, minus_di = calculate_adx(high_prices, low_prices, close_prices)
    print("\nADX:\n", adx.tail())
    print("+DI:\n", plus_di.tail())
    print("-DI:\n", minus_di.tail())


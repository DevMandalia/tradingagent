# Technical Indicator Formulas

This document outlines the calculation formulas for the technical indicators to be used in the Bitcoin trading bot.

## 1. Moving Average Convergence Divergence (MACD)

*   **MACD Line**: 12-period Exponential Moving Average (EMA) - 26-period EMA of the price.
*   **Signal Line**: 9-period EMA of the MACD Line.
*   **MACD Histogram**: MACD Line - Signal Line.

    *Reference: Investopedia, StockCharts*

## 2. Relative Strength Index (RSI)

1.  **Calculate Price Changes**: Upward changes (U) and downward changes (D) for each period.
    *   If Close(current) > Close(previous), then U = Close(current) - Close(previous), D = 0.
    *   If Close(current) < Close(previous), then U = 0, D = Close(previous) - Close(current).
    *   If Close(current) == Close(previous), then U = 0, D = 0.
2.  **Calculate Average Gains and Average Losses**: Typically over a 14-period timeframe. For the first calculation, it's a simple average. Subsequent calculations use a smoothed average:
    *   Average Gain = [(Previous Average Gain) * (N-1) + Current Gain] / N
    *   Average Loss = [(Previous Average Loss) * (N-1) + Current Loss] / N
    *   Where N is the RSI period (e.g., 14).
3.  **Calculate Relative Strength (RS)**: RS = Average Gain / Average Loss.
4.  **Calculate RSI**: RSI = 100 - [100 / (1 + RS)].

    *Reference: Investopedia, StockCharts*

## 3. Stochastic Oscillator

*   **%K**: [(Current Close - Lowest Low_N) / (Highest High_N - Lowest Low_N)] * 100
    *   Lowest Low_N: The lowest price over the last N periods (typically 14 periods).
    *   Highest High_N: The highest price over the last N periods (typically 14 periods).
*   **%D (Slow Stochastic)**: 3-period Simple Moving Average (SMA) of %K.
*   A **Fast Stochastic Oscillator** uses the raw %K and a %D that is a 3-period SMA of that raw %K. The **Slow Stochastic Oscillator** (more commonly used) first smooths the initial %K by taking a 3-period SMA of it (this smoothed %K is often then referred to as the new %K or sometimes %D_fast), and then the %D line is a 3-period SMA of this smoothed %K.

    For clarity in implementation, we will aim for the Slow Stochastic Oscillator:
    1.  Calculate initial %K (Fast %K) using the formula above (e.g., 14 periods).
    2.  Calculate Slow %K by taking a 3-period SMA of the initial %K.
    3.  Calculate Slow %D by taking a 3-period SMA of the Slow %K.

    *Reference: Investopedia, StockCharts*

## 4. Average Directional Index (ADX)

The ADX calculation involves several steps and components: Directional Movement (+DM and -DM), True Range (TR), Directional Indicators (+DI and -DI), and finally the ADX itself.

1.  **Calculate Directional Movement (+DM, -DM) and True Range (TR)** for each period:
    *   UpMove = Current High - Previous High
    *   DownMove = Previous Low - Current Low
    *   If UpMove > DownMove and UpMove > 0, then +DM = UpMove, else +DM = 0.
    *   If DownMove > UpMove and DownMove > 0, then -DM = DownMove, else -DM = 0.
    *   (Note: if UpMove == DownMove or both are < 0, then +DM = 0 and -DM = 0. Some versions set both to 0 if UpMove and DownMove are equal, or if UpMove < 0 and DownMove < 0. Wilder's original method: if (Current High - Previous High) > (Previous Low - Current Low) then +DM = max(Current High - Previous High, 0). If (Previous Low - Current Low) > (Current High - Previous High) then -DM = max(Previous Low - Current Low, 0). If they are equal or both negative, +DM=0 and -DM=0.)
    *   TR = Max of:
        *   Current High - Current Low
        *   Absolute Value (Current High - Previous Close)
        *   Absolute Value (Current Low - Previous Close)

2.  **Smooth +DM, -DM, and TR**: Typically using a 14-period Wilder's smoothing method (which is an EMA variant: Smoothed Value = Previous Smoothed Value - (Previous Smoothed Value / N) + Current Value).
    *   For the first value: Sum the first N periods and divide by N.
    *   Smoothed +DM_N = [(Previous Smoothed +DM_N) * (N-1)/N] + Current +DM
    *   Smoothed -DM_N = [(Previous Smoothed -DM_N) * (N-1)/N] + Current -DM
    *   Smoothed TR_N = [(Previous Smoothed TR_N) * (N-1)/N] + Current TR

3.  **Calculate Directional Indicators (+DI_N, -DI_N)**:
    *   +DI_N = (Smoothed +DM_N / Smoothed TR_N) * 100
    *   -DI_N = (Smoothed -DM_N / Smoothed TR_N) * 100

4.  **Calculate Directional Movement Index (DX)**:
    *   DX = [Absolute Value (+DI_N - -DI_N) / (+DI_N + -DI_N)] * 100

5.  **Calculate Average Directional Index (ADX)**:
    *   ADX is a smoothed average of DX, typically over N periods (e.g., 14 periods).
    *   First ADX_N = Simple average of the first N DX values.
    *   Subsequent ADX_N = [(Previous ADX_N * (N-1)) + Current DX] / N

    *Reference: Investopedia, StockCharts, Wikipedia*


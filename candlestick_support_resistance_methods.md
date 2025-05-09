# Candlestick Patterns and Support/Resistance Analysis

This document outlines the methods for identifying common candlestick patterns and support/resistance levels for the Bitcoin trading bot.

## 1. Candlestick Pattern Recognition

Based on research from sources like IG.com and Investopedia, the following key candlestick patterns will be implemented. The input for these functions will typically be a pandas DataFrame containing Open, High, Low, and Close (OHLC) data for a series of periods.

### 1.1. Bullish Patterns

*   **Hammer**:
    *   Forms at the bottom of a downtrend.
    *   Short body, long lower shadow (at least 2x body length), very short or no upper shadow.
    *   Indicates potential bullish reversal.
*   **Bullish Engulfing**:
    *   Two-candlestick pattern.
    *   First candle: short red (bearish) body.
    *   Second candle: larger green (bullish) body that completely engulfs the first candle's body.
    *   Appears in a downtrend, signals potential reversal upwards.
*   **Piercing Line**:
    *   Two-candlestick pattern in a downtrend.
    *   First candle: long red (bearish) body.
    *   Second candle: long green (bullish) body, opens below the low of the first candle and closes above the midpoint of the first candle's body.
    *   Signals potential bullish reversal.
*   **Morning Star**:
    *   Three-candlestick pattern in a downtrend.
    *   First candle: long red (bearish) body.
    *   Second candle: short body (star), gaps down from the first candle. Color can be red or green.
    *   Third candle: long green (bullish) body, closes well into the body of the first candle.
    *   Signals strong bullish reversal.
*   **Three White Soldiers**:
    *   Three consecutive long green (bullish) candles.
    *   Each candle opens within the previous candle's body and closes higher than the previous candle's high, with small upper shadows.
    *   Appears after a downtrend, strong bullish reversal signal.

### 1.2. Bearish Patterns

*   **Hanging Man**:
    *   Forms at the top of an uptrend.
    *   Shape is identical to a Hammer (short body, long lower shadow at least 2x body, short/no upper shadow).
    *   Indicates potential bearish reversal.
*   **Bearish Engulfing**:
    *   Two-candlestick pattern.
    *   First candle: short green (bullish) body.
    *   Second candle: larger red (bearish) body that completely engulfs the first candle's body.
    *   Appears in an uptrend, signals potential reversal downwards.
*   **Evening Star**:
    *   Three-candlestick pattern in an uptrend.
    *   First candle: long green (bullish) body.
    *   Second candle: short body (star), gaps up from the first candle. Color can be red or green.
    *   Third candle: long red (bearish) body, closes well into the body of the first candle.
    *   Signals strong bearish reversal.
*   **Three Black Crows**:
    *   Three consecutive long red (bearish) candles.
    *   Each candle opens within the previous candle's body and closes lower than the previous candle's low, with small lower shadows.
    *   Appears after an uptrend, strong bearish reversal signal.
*   **Dark Cloud Cover**:
    *   Two-candlestick pattern in an uptrend.
    *   First candle: long green (bullish) body.
    *   Second candle: red (bearish) body, opens above the high of the first candle and closes below the midpoint of the first candle's body.
    *   Signals potential bearish reversal.

### 1.3. Continuation/Indecision Patterns

*   **Doji**:
    *   Open and Close prices are very close or identical, resulting in a very small or non-existent body.
    *   Shadows can vary in length.
    *   Indicates indecision in the market. Its significance depends on the preceding trend and subsequent candles.
    *   Variations: Long-legged Doji, Dragonfly Doji, Gravestone Doji.

## 2. Support and Resistance Level Identification

Support and resistance levels are key price areas where the price tends to stop and reverse. Common methods to identify them include:

*   **Historical Price Levels (Swing Highs and Lows)**:
    *   **Support**: A price level where a downtrend can be expected to pause due to a concentration of demand. Identified by previous troughs (swing lows).
    *   **Resistance**: A price level where an uptrend can be expected to pause temporarily, due to a concentration of supply. Identified by previous peaks (swing highs).
    *   Implementation: Look for local minima and maxima in the price data over a defined lookback period.

*   **Moving Averages**:
    *   Moving averages (e.g., 50-day, 200-day SMA or EMA) can act as dynamic support or resistance levels.
    *   When price approaches a significant moving average from above, it may find support. When it approaches from below, it may find resistance.

*   **Fibonacci Retracement Levels**:
    *   Identify a significant swing high and swing low.
    *   Calculate Fibonacci retracement levels (e.g., 23.6%, 38.2%, 50%, 61.8%, 78.6%) between these two points.
    *   These levels can act as potential support or resistance.

*   **Pivot Points**:
    *   Calculated based on the previous period's high, low, and close prices.
    *   Standard pivot point (PP) = (Previous High + Previous Low + Previous Close) / 3
    *   Support levels (S1, S2, S3) and resistance levels (R1, R2, R3) are then calculated based on the PP.
        *   S1 = (PP * 2) - Previous High
        *   R1 = (PP * 2) - Previous Low
        *   S2 = PP - (Previous High - Previous Low)
        *   R2 = PP + (Previous High - Previous Low)
        *   S3 = Previous Low - 2 * (Previous High - PP)
        *   R3 = Previous High + 2 * (PP - Previous Low)

*   **Psychological Levels (Round Numbers)**:
    *   Prices often find support or resistance at round numbers (e.g., $50,000, $60,000 for Bitcoin) as traders often place orders at these levels.

For the initial implementation, we will focus on identifying support and resistance using **Swing Highs/Lows** and potentially **Pivot Points** due to their objective calculation from price data. Moving averages identified in the technical indicators module can also be re-purposed.

## 3. Implementation Approach

*   Functions will be created in a new Python file, e.g., `market_analysis.py`.
*   Each candlestick pattern will have its own detection function that takes OHLC data as input and returns a boolean or a signal strength.
*   Support and resistance functions will take price data and return a list of identified levels or zones.
*   The code will be modular and use pandas for data manipulation.


# Bitcoin Trading Bot

## 1. Introduction

This document provides a summary of the Bitcoin trading bot. The bot is designed to analyze market data using various technical indicators and metrics to make trading decisions on the Binance platform. This report outlines its features, architecture, implementation details, backtesting results, and important limitations.

## 2. Bot Features

The trading bot incorporates the following analytical features:

*   **Technical Indicators**: 
    *   Moving Average Convergence Divergence (MACD)
    *   Relative Strength Index (RSI)
    *   Stochastic Oscillator (Slow %K and %D)
    *   Average Directional Index (ADX) with +DI and -DI
*   **Candlestick Pattern Analysis**: Detection of common bullish and bearish patterns, including:
    *   Doji
    *   Hammer & Hanging Man
    *   Bullish & Bearish Engulfing
    *   (Further patterns like Piercing Line, Morning/Evening Star, Three White Soldiers/Black Crows were documented and can be added to the `market_analysis.py` module if desired).
*   **Support and Resistance Levels**: Identification using:
    *   Pivot Points (calculated from previous day's OHLC)
    *   Swing Highs and Lows
*   **Additional Market Metrics**:
    *   **Funding Rates**: Fetched from Binance Futures API (Note: API calls may be restricted in some environments, see Limitations).
    *   **Order Book Analysis**: Fetched from Binance Spot/Futures API (Note: API calls may be restricted, see Limitations).
    *   **MVRV-Z Score**: Currently implemented as a mock function due to the lack of free, direct public API access. Requires integration with a dedicated on-chain data provider.
*   **Trading Logic**: A configurable, score-based system that aggregates signals from the above analyses to generate BUY, SELL, or HOLD decisions.

## 3. Bot Architecture

The bot is designed with a modular architecture, with key components separated into different Python files:

*   `technical_indicators.py`: Functions for calculating MACD, RSI, Stochastic Oscillator, and ADX.
*   `market_analysis.py`: Functions for candlestick pattern recognition and support/resistance level identification (Pivot Points, Swing Highs/Lows).
*   `additional_metrics.py`: Functions to fetch funding rates, order book data (from Binance API), and a mock function for MVRV-Z score.
*   `trading_strategy.py`: Contains the core logic for generating trading signals by combining inputs from all analysis modules.
*   `fetch_btc_data.py`: A utility script to download historical Bitcoin data using the YahooFinance API (used for backtesting).
*   `backtester.py`: A script to simulate the trading strategy on historical data, calculating performance metrics.

Detailed architecture and formula documentation can be found in:
*   `trading_bot_architecture.md`
*   `technical_indicator_formulas.md`
*   `candlestick_support_resistance_methods.md`
*   `additional_metrics_sources.md`

## 4. Setup and Usage (Brief Overview)

1.  **Environment**: Ensure Python 3.x is installed with necessary libraries (pandas, numpy, requests). These are standard and were used during development.
2.  **API Keys**: For live trading (not implemented in this phase beyond fetching data), you would need to generate API keys from your Binance account and securely configure them. The current scripts for fetching funding rates and order book data do not require API keys for public endpoints but might be subject to IP-based rate limits or regional restrictions.
3.  **Configuration**: The trading strategy in `trading_strategy.py` has example parameters (e.g., RSI thresholds, ADX levels, scoring weights). These should be reviewed and tuned based on further testing and your risk appetite.
4.  **Data for MVRV-Z Score**: The `get_mvrv_z_score_mock()` function in `additional_metrics.py` needs to be replaced with a connection to a live data source for the MVRV-Z score.
5.  **Running the Backtester**: 
    *   First, run `python3.11 fetch_btc_data.py` to get historical data (saves to `btc_historical_data.json`).
    *   Then, run `python3.11 backtester.py` to execute the backtest. Results and trades will be printed, and trades saved to `backtest_trades.csv`.

## 5. Backtesting Results (Summary from 1-Year Daily BTC-USD Data)

The provided `backtester.py` script was run on approximately one year of daily BTC-USD data (from 2024-06-07 to 2025-05-08).

*   **Initial Capital**: $10,000.00
*   **Final Portfolio Value**: $14,925.02
*   **Total Trades**: 6
*   **Profit/Loss**: +$4,925.02
*   **Profit/Loss Percentage**: +49.25%

**Note**: These backtesting results are based on a specific historical period and a sample strategy. Past performance is not indicative of future results. The strategy parameters, risk management, and handling of transaction costs/slippage would need significant refinement for live trading.

## 6. Important Limitations and Considerations

*   **API Access in Sandbox**: During development, direct API calls to Binance (for funding rates and order book) from the sandbox environment resulted in HTTP 451 errors (Unavailable For Legal Reasons). This indicates a regional restriction. When you run the bot, ensure your environment does not have these restrictions, or you may need to use a VPN or a server in a different region. The code is structured to make these calls, but they were not live during the final backtest simulation for these specific metrics (they were mocked as empty/None).
*   **MVRV-Z Score Data Source**: The MVRV-Z score is a critical metric requested but is not available via a simple, free, public API. The current implementation uses a mock function. For live trading, you will need to subscribe to an on-chain data provider (e.g., Glassnode, CryptoQuant, CoinGlass) that offers an API for this metric or find a reliable free source and adapt the `additional_metrics.py` script.
*   **Strategy Simplicity**: The trading strategy implemented in `trading_strategy.py` is a basic example using a scoring system. It should be considered a starting point. Real-world trading requires more sophisticated strategies, risk management (e.g., stop-loss, take-profit, position sizing), and parameter optimization.
*   **Backtesting Scope**: The backtester is simplified. It does not account for: 
    *   Trading fees (Binance commissions).
    *   Slippage (difference between expected trade price and actual execution price).
    *   Latency in order execution.
    *   Detailed order book dynamics for order fills.
    For serious deployment, a more advanced backtesting framework is recommended.
*   **No Live Trading Module**: This project focused on the analytical components and a backtesting framework. A live trading execution module (connecting to Binance to place actual orders) has not been implemented.
*   **Error Handling and Robustness**: While basic error handling is present, a production-grade bot would require more extensive error handling, logging, and monitoring.

## 7. Conclusion and Next Steps

The developed modules provide a foundational framework for a Bitcoin trading bot with analytical capabilities. To move towards live trading, the following steps are recommended:

1.  Secure a reliable environment for Binance API calls.
2.  Integrate a live data feed for the MVRV-Z score.
3.  Thoroughly test and optimize the trading strategy parameters.
4.  Implement robust risk management rules.
5.  Develop and test a live order execution module.
6.  Consider a more advanced backtesting platform.

This is a good starting point for trading bot development.


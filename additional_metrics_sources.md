# Additional Market Metrics: Funding Rates, Order Book, and MVRV-Z Score

This document outlines the sources and methods for obtaining funding rates, order book data, and the MVRV-Z score for the Bitcoin trading bot.

## 1. Funding Rates

*   **Source**: Binance API (Futures).
*   **Endpoint**: `/fapi/v1/fundingRate` (for historical funding rates) or `/fapi/v1/premiumIndex` (for real-time mark price and last funding rate).
*   **Description**: Funding rates are periodic payments either to traders that are long or short based on the difference between perpetual contract markets and spot prices. Positive funding rates imply longs pay shorts, suggesting bullish sentiment. Negative rates imply shorts pay longs, suggesting bearish sentiment.
*   **Data to Fetch**: Symbol (e.g., BTCUSDT), funding rate, funding time.
*   **Relevance**: High funding rates can make holding positions expensive, while very low or negative rates might indicate crowded trades or extreme sentiment.

## 2. Order Book Analysis

*   **Source**: Binance API (Spot or Futures, depending on trading context).
*   **Endpoint**: `/api/v3/depth` (for Spot) or `/fapi/v1/depth` (for Futures).
*   **Description**: The order book shows a list of buy (bids) and sell (asks) orders for a specific asset at various price levels. Analyzing the depth and spread can provide insights into market liquidity and potential short-term price movements.
*   **Data to Fetch**: Bids (price, quantity), Asks (price, quantity), last update ID.
*   **Analysis**: 
    *   **Depth**: Total quantity of bids vs. asks at different levels.
    *   **Spread**: Difference between the best ask and best bid.
    *   **Order Imbalance**: Significant differences in volume on the bid or ask side.
    *   **Large Orders (Walls)**: Unusually large orders that might act as temporary support or resistance.
*   **Relevance**: Helps gauge immediate supply and demand, potential slippage, and areas of liquidity.

## 3. MVRV-Z Score (Market Value to Realized Value Z-Score)

*   **Source**: External on-chain data providers (e.g., Glassnode, CoinGlass, LookIntoBitcoin, CryptoQuant). Binance API does not directly provide this metric as it requires extensive on-chain data processing.
*   **Description**: The MVRV-Z Score is used to assess when Bitcoin is over/undervalued relative to its "fair value". It compares Bitcoin's market capitalization (Market Value) to its realized capitalization (Realized Value, where coins are valued at the price they last moved on-chain) and uses standard deviation to show how many Z-scores away the current Market Cap is from the Realized Cap.
    *   **High Values (e.g., Z-score > 3, often in red zone on charts)**: Historically indicate market tops or overvaluation.
    *   **Low Values (e.g., Z-score < 0, often in green zone on charts)**: Historically indicate market bottoms or undervaluation.
*   **Data to Fetch**: The MVRV-Z score value itself.
*   **Challenges**: 
    *   **API Access**: Many specialized on-chain data providers require paid subscriptions for API access.
    *   **Web Scraping**: Scraping data from websites like Glassnode or CoinGlass is possible but can be unreliable due to website structure changes and potential blocking. It also might violate terms of service.
    *   **Calculation**: Calculating MVRV-Z from scratch requires access to full Bitcoin blockchain data and significant computational resources, which is beyond the scope of a typical trading bot API integration.
*   **Proposed Approach**: 
    1.  Investigate if any of the reputable sources (Glassnode, CoinGlass, etc.) offer a free or trial API tier that provides the MVRV-Z score for Bitcoin.
    2.  If direct API access is not feasible without cost, inform the user about the limitations and potential need for manual input or a subscription to a data provider for this metric.
    3.  As a fallback, the bot could be designed to accept this value as a manual input or from a local file updated by the user.
*   **Relevance**: Provides a long-term macro perspective on Bitcoin's valuation, helping to identify potentially overbought or oversold conditions.

## 4. Implementation Plan

*   Create a new Python file, e.g., `additional_metrics.py`.
*   Implement functions to fetch funding rates and order book data from the Binance API.
*   Research and implement the best feasible method for obtaining the MVRV-Z score.
*   Ensure all functions are modular and return data in a usable format (e.g., pandas DataFrames or dictionaries).


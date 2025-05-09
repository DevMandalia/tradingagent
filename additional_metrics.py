import requests
import pandas as pd
import json

# Configuration (Ideally, API keys would be in a config file or environment variables)
BINANCE_SPOT_API_URL = "https://api.binance.com/api/v3"
BINANCE_FUTURES_API_URL = "https://fapi.binance.com/fapi/v1"

# --- Funding Rate --- 
def get_funding_rate_history(symbol="BTCUSDT", limit=100):
    """Fetches historical funding rates for a given symbol from Binance Futures.
       Note: The user's request is for Bitcoin trading, which could be Spot or Futures.
       Funding rates are specific to perpetual futures contracts.
    """
    endpoint = f"{BINANCE_FUTURES_API_URL}/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = pd.to_numeric(df["fundingRate"])
            return df[["fundingTime", "symbol", "fundingRate"]]
        else:
            print(f"No funding rate data returned for {symbol}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching funding rates for {symbol}: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for funding rates {symbol}: {e} - Response: {response.text}")
        return pd.DataFrame()

# --- Order Book --- 
def get_order_book(symbol="BTCUSDT", limit=100, market_type="spot"):
    """Fetches the order book (depth) for a given symbol from Binance.
       market_type can be 'spot' or 'futures'.
    """
    if market_type == "spot":
        endpoint = f"{BINANCE_SPOT_API_URL}/depth"
    elif market_type == "futures":
        endpoint = f"{BINANCE_FUTURES_API_URL}/depth"
    else:
        print("Invalid market type specified for order book. Use 'spot' or 'futures'.")
        return None

    params = {"symbol": symbol, "limit": limit}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        # Process bids and asks into DataFrames for easier analysis
        bids_df = pd.DataFrame(data.get("bids", []), columns=["price", "quantity"], dtype=float)
        asks_df = pd.DataFrame(data.get("asks", []), columns=["price", "quantity"], dtype=float)
        return {"bids": bids_df, "asks": asks_df, "lastUpdateId": data.get("lastUpdateId")}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching order book for {symbol} ({market_type}): {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for order book {symbol} ({market_type}): {e} - Response: {response.text}")
        return None

# --- MVRV-Z Score --- 
# MVRV Z-Score is not directly available from Binance API.
# It typically requires on-chain data providers like Glassnode, CoinGlass, etc.
# Many of these require API keys and subscriptions.
# For this implementation, we will simulate fetching it or provide a placeholder.
# In a real bot, this would need a robust solution (paid API, reliable free source if found, or manual input).

def get_mvrv_z_score_mock(symbol="BTC"):
    """Mock function for MVRV Z-Score.
       In a real scenario, this would fetch data from an external API or data source.
       This function will try to fetch from a known public endpoint if available and simple,
       otherwise, it will return a placeholder and a note about the data source.
    """
    # Attempt to fetch from a public source if one is simple and reliable (e.g. lookintobitcoin.com, but they often have Cloudflare)
    # For now, let's try a known public API if one exists and is simple, otherwise, placeholder.
    # Example: Glassnode has an API, but it's typically authenticated and tiered.
    # Coinglass might have some public charts, but direct API for raw MVRV Z-score might be tricky without a key.
    
    # Placeholder for demonstration if no simple public API is readily available without keys/complex parsing:
    print("MVRV Z-Score: Data typically sourced from specialized on-chain providers (e.g., Glassnode, CoinGlass).")
    print("This function is a placeholder. For a live bot, integrate a reliable MVRV Z-Score data feed.")
    # As an example, let's try to fetch from a public API that might provide this, if one exists that's simple.
    # After quick check, direct, free, unauthenticated API endpoints for raw MVRV Z-score are not common or stable.
    # Many chart sites (like lookintobitcoin, glassnode) protect their data or require logins for API access.
    
    # For demonstration, we'll return a mock value and a status.
    # In a real bot, you'd replace this with actual API call to Glassnode, CryptoQuant, etc. (likely requiring a paid key)
    # or a web scraping solution (less reliable).
    return {"symbol": symbol, "mvrv_z_score": 1.5, "status": "mock_data", "notes": "Using mock data. Integrate a real MVRV-Z provider."}

if __name__ == '__main__':
    print("--- Testing Additional Metrics ---")

    # Test Funding Rate (BTCUSDT is a perpetual future on Binance)
    print("\nFetching Funding Rate History for BTCUSDT...")
    funding_rates_df = get_funding_rate_history(symbol="BTCUSDT", limit=5)
    if not funding_rates_df.empty:
        print(funding_rates_df)
    else:
        print("Could not fetch funding rates or no data available.")

    # Test Order Book (Spot)
    print("\nFetching Spot Order Book for BTCUSDT...")
    spot_order_book = get_order_book(symbol="BTCUSDT", limit=5, market_type="spot")
    if spot_order_book:
        print("Spot Bids (Top 5):")
        print(spot_order_book["bids"].head())
        print("\nSpot Asks (Top 5):")
        print(spot_order_book["asks"].head())
        print(f"Last Update ID: {spot_order_book['lastUpdateId']}")

    # Test Order Book (Futures)
    print("\nFetching Futures Order Book for BTCUSDT...")
    futures_order_book = get_order_book(symbol="BTCUSDT", limit=5, market_type="futures")
    if futures_order_book:
        print("Futures Bids (Top 5):")
        print(futures_order_book["bids"].head())
        print("\nFutures Asks (Top 5):")
        print(futures_order_book["asks"].head())
        print(f"Last Update ID: {futures_order_book['lastUpdateId']}")

    # Test MVRV-Z Score (Mock)
    print("\nFetching MVRV Z-Score (Mock for BTC)...")
    mvrv_data = get_mvrv_z_score_mock(symbol="BTC")
    print(mvrv_data)


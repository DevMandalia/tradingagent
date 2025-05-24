import requests
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def to_snake_case(text: str) -> str:
    """Convert text to snake case."""
    replacements = {' ': '_', '-': '_', '(': '', ')': '', ">": '', ',': ''}
    return text.lower().translate(str.maketrans(replacements)).replace("___", '_').replace("__", '_')

def create_session():
    """Create a session with realistic browser headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    })
    return session

def lib_post_request(session, endpoint: str, payload: dict, max_retries: int = 5, base_delay: int = 2):
    """Make a POST request to LookIntoBitcoin API with improved error handling."""
    base_url = f"https://www.lookintobitcoin.com/django_plotly_dash/app/{endpoint}/_dash-update-component"
    
    # Update headers for the specific request
    session.headers.update({
        "Content-Type": "application/json",
        "Origin": "https://www.lookintobitcoin.com",
        "Referer": f"https://www.lookintobitcoin.com/charts/{endpoint}/"
    })
    
    for attempt in range(max_retries):
        try:
            # First, visit the page to get any necessary cookies
            page_url = f"https://www.lookintobitcoin.com/charts/{endpoint}/"
            session.get(page_url, timeout=30)
            
            # Add a small delay before making the POST request
            time.sleep(random.uniform(1, 2))
            
            response = session.post(base_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "response" in data and "chart" in data["response"]:
                        return data["response"]["chart"]["figure"]["data"]
                    else:
                        print(f"Unexpected response format for {endpoint}")
                        print(f"Response: {data}")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {endpoint}: {str(e)}")
                    print(f"Response text: {response.text[:200]}...")
            else:
                print(f"HTTP {response.status_code} for {endpoint}")
                print(f"Response: {response.text[:200]}...")
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            
        except requests.exceptions.RequestException as e:
            print(f"Request error for {endpoint}: {str(e)}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for {endpoint}")
                return None
    
    return None

def scrape_lib_fear_greed(session) -> pd.DataFrame:
    """Scrape Bitcoin Fear & Greed Index data."""
    payload = {
        "output": "..chart.figure...indicator.figure...last_update.children...now.children...now.style..",
        "outputs": [
            {"id": "chart", "property": "figure"},
            {"id": "indicator", "property": "figure"},
            {"id": "last_update", "property": "children"},
            {"id": "now", "property": "children"},
            {"id": "now", "property": "style"}
        ],
        "inputs": [{"id": "url", "property": "pathname", "value": "/charts/bitcoin-fear-and-greed-index/"}]
    }
    
    request_data = lib_post_request(session, "fear_and_greed", payload)
    if not request_data:
        return pd.DataFrame()
    
    try:
        data = request_data[1]["customdata"]
        df = pd.DataFrame([[c[i] for i in (0, 2, 3)] for c in data],
                         columns=["datetime", "fear_greed_value", "fear_greed_category"])
        return df
    except Exception as e:
        print(f"Error processing fear & greed data: {str(e)}")
        return pd.DataFrame()

def scrape_lib_mvrv(session) -> pd.DataFrame:
    """Scrape MVRV Z-Score data."""
    payload = {
        "inputs": [{"id": "url", "property": "pathname", "value": "/charts/mvrv-zscore/"}]
    }
    
    request_data = lib_post_request(session, "mvrv_zscore", payload)
    if not request_data:
        return pd.DataFrame()
    
    try:
        mcap_data = next((data for data in request_data if data["name"] == "Market Cap"), None)
        rcap_data = next((data for data in request_data if data["name"] == "Realized Cap"), None)
        
        if mcap_data and rcap_data:
            column_values = np.array([mcap_data['y'], rcap_data['y']]).T
            index_values = mcap_data["x"][:len(mcap_data['y'])]
            df = pd.DataFrame(column_values, 
                            columns=["market_cap_usd", "realised_cap_usd"], 
                            index=index_values).reset_index(names="datetime")
            return df
    except Exception as e:
        print(f"Error processing MVRV data: {str(e)}")
    
    return pd.DataFrame()

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Starting LookIntoBitcoin data collection...")
    
    # Create a session for all requests
    session = create_session()
    
    # Collect Fear & Greed Index data
    print("\nCollecting Fear & Greed Index data...")
    fear_greed_df = scrape_lib_fear_greed(session)
    if not fear_greed_df.empty:
        fear_greed_df.to_csv('data/fear_greed_index.csv', index=False)
        print(f"Saved {len(fear_greed_df)} records to data/fear_greed_index.csv")
    else:
        print("Failed to collect Fear & Greed Index data")
    
    # Add a delay between different data collections
    time.sleep(random.uniform(2, 4))
    
    # Collect MVRV data
    print("\nCollecting MVRV data...")
    mvrv_df = scrape_lib_mvrv(session)
    if not mvrv_df.empty:
        mvrv_df.to_csv('data/mvrv_data.csv', index=False)
        print(f"Saved {len(mvrv_df)} records to data/mvrv_data.csv")
    else:
        print("Failed to collect MVRV data")
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main() 
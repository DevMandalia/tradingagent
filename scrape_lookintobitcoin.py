import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os
from datetime import datetime
import time
import random

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

def extract_json_data(html_content, chart_name):
    """Extract JSON data from the page's JavaScript."""
    try:
        # Look for the chart data in the page source
        start_marker = f'var {chart_name} = '
        end_marker = ';'
        
        start_idx = html_content.find(start_marker)
        if start_idx == -1:
            return None
            
        start_idx += len(start_marker)
        end_idx = html_content.find(end_marker, start_idx)
        
        if end_idx == -1:
            return None
            
        json_str = html_content[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error extracting {chart_name} data: {str(e)}")
        return None

def scrape_fear_greed():
    """Scrape Fear & Greed Index data."""
    session = create_session()
    url = "https://www.lookintobitcoin.com/charts/bitcoin-fear-and-greed-index/"
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract the chart data
        chart_data = extract_json_data(response.text, 'fearGreedData')
        if not chart_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(chart_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={
            'value': 'fear_greed_value',
            'classification': 'fear_greed_category'
        })
        
        return df
    except Exception as e:
        print(f"Error scraping Fear & Greed data: {str(e)}")
        return pd.DataFrame()

def scrape_mvrv():
    """Scrape MVRV Z-Score data."""
    session = create_session()
    url = "https://www.lookintobitcoin.com/charts/mvrv-zscore/"
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract the chart data
        chart_data = extract_json_data(response.text, 'mvrvData')
        if not chart_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(chart_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={
            'marketCap': 'market_cap_usd',
            'realizedCap': 'realised_cap_usd'
        })
        
        return df
    except Exception as e:
        print(f"Error scraping MVRV data: {str(e)}")
        return pd.DataFrame()

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Starting LookIntoBitcoin data collection...")
    
    # Scrape Fear & Greed Index data
    print("\nCollecting Fear & Greed Index data...")
    fear_greed_df = scrape_fear_greed()
    if not fear_greed_df.empty:
        fear_greed_df.to_csv('data/fear_greed_index.csv', index=False)
        print(f"Saved {len(fear_greed_df)} records to data/fear_greed_index.csv")
    else:
        print("Failed to collect Fear & Greed Index data")
    
    # Add a delay between different data collections
    time.sleep(random.uniform(2, 4))
    
    # Scrape MVRV data
    print("\nCollecting MVRV data...")
    mvrv_df = scrape_mvrv()
    if not mvrv_df.empty:
        mvrv_df.to_csv('data/mvrv_data.csv', index=False)
        print(f"Saved {len(mvrv_df)} records to data/mvrv_data.csv")
    else:
        print("Failed to collect MVRV data")
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main() 
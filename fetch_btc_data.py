import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import json

def fetch_and_save_btc_data():
    client = ApiClient()
    try:
        print("Fetching BTC-USD historical data from YahooFinance API...")
        # Query for 1 year of daily data for BTC-USD
        btc_data = client.call_api('YahooFinance/get_stock_chart', query={'symbol': 'BTC-USD', 'interval': '1d', 'range': '1y', 'includeAdjustedClose': True})
        
        # Check if data is valid
        if btc_data and btc_data.get("chart") and btc_data["chart"].get("result") and btc_data["chart"]["result"][0]:
            filepath = "/home/ubuntu/btc_historical_data.json"
            with open(filepath, 'w') as f:
                json.dump(btc_data, f)
            print(f"BTC historical data saved to {filepath}")
            return True
        else:
            print("Failed to fetch valid data from YahooFinance API or data is empty.")
            if btc_data and btc_data.get("chart") and btc_data["chart"].get("error"):
                print(f"API Error: {btc_data['chart']['error']}")
            elif btc_data:
                print(f"Received data structure: {json.dumps(btc_data, indent=2)}")
            return False
            
    except Exception as e:
        print(f"An error occurred during data fetching: {e}")
        return False

if __name__ == "__main__":
    fetch_and_save_btc_data()


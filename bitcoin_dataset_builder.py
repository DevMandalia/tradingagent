import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import json

def get_blockchain_data(start_date, end_date):
    """
    Fetch Bitcoin blockchain data from Blockchain.info API
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize empty list to store data
    data = []
    
    # API endpoint
    base_url = "https://api.blockchain.info/charts/"
    
    # Metrics to fetch
    metrics = {
        'market-price': 'price',
        'total-bitcoins': 'total_btc',
        'market-cap': 'market_cap',
        'trade-volume': 'volume',
        'transactions-per-block': 'tx_per_block',
        'median-confirmation-time': 'median_conf_time',
        'hash-rate': 'hash_rate',
        'difficulty': 'difficulty',
        'miners-revenue': 'miners_revenue',
        'transaction-fees': 'tx_fees',
        'cost-per-transaction': 'cost_per_tx',
        'cost-per-transaction-percent': 'cost_per_tx_percent',
        'n-transactions': 'n_transactions',
        'n-unique-addresses': 'n_unique_addresses',
        'n-transactions-excluding-popular': 'n_tx_ex_popular',
        'n-transactions-excluding-chains-longer-than-100': 'n_tx_ex_long_chains',
        'output-volume': 'output_volume',
        'estimated-transaction-volume': 'est_tx_volume',
        'estimated-transaction-volume-usd': 'est_tx_volume_usd'
    }
    
    # Fetch data for each metric
    for metric, column_name in metrics.items():
        print(f"Fetching {metric}...")
        
        try:
            # Make API request
            url = f"{base_url}{metric}?timespan=all&format=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                metric_data = response.json()
                
                # Check if the response has the expected format
                if 'values' in metric_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(metric_data['values'])
                    
                    # Check if the DataFrame has the expected columns
                    if 'x' in df.columns and 'y' in df.columns:
                        df['date'] = pd.to_datetime(df['x'], unit='s')
                        df = df[['date', 'y']]
                        df.columns = ['date', column_name]
                        
                        # Filter by date range
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                        
                        # Add to data list
                        data.append(df)
                    else:
                        print(f"Unexpected columns in response for {metric}")
                else:
                    print(f"Unexpected response format for {metric}")
            else:
                print(f"Error fetching {metric}: {response.status_code}")
                
            # Sleep to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {metric}: {str(e)}")
            continue
    
    if not data:
        raise Exception("No data was successfully fetched")
    
    # Merge all DataFrames
    final_df = data[0]
    for df in data[1:]:
        final_df = pd.merge(final_df, df, on='date', how='outer')
    
    # Sort by date
    final_df = final_df.sort_values('date')
    
    return final_df

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Set date range (from 2012 to present)
    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching Bitcoin data from {start_date} to {end_date}...")
    
    try:
        # Get blockchain data
        df = get_blockchain_data(start_date, end_date)
        
        # Save to CSV (new file)
        output_file = 'data/on_chain_data.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        
        # Display NaN statistics
        print("\nNaN values per column:")
        nan_stats = df.isna().sum()
        for column, count in nan_stats.items():
            print(f"{column}: {count} NaN values")
        
        # Display head and tail of the dataset
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        print("\nLast 5 rows of the dataset:")
        print(df.tail())
        
        # Display basic statistics
        print("\nDataset Statistics:")
        print(f"Number of records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
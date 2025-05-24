import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import zipfile
import io
import os
import kaggle

def create_resampled_datasets(df):
    """
    Create resampled datasets at different time intervals from the 1-minute data.
    
    Args:
        df (pd.DataFrame): Original 1-minute data DataFrame
    
    Returns:
        dict: Dictionary containing resampled DataFrames
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Define resampling intervals
    intervals = {
        '15min': '15T',
        'hourly': 'H',
        'daily': 'D',
        'weekly': 'W'
    }
    
    resampled_dfs = {}
    
    for interval_name, interval_code in intervals.items():
        print(f"\nCreating {interval_name} dataset...")
        
        # Resample the data
        resampled = df.resample(interval_code).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Remove rows with all NaN values
        resampled = resampled.dropna(how='all')
        
        # Save to CSV
        filename = f'bitcoin_data_{interval_name}.csv'
        resampled.to_csv(filename)
        
        print(f"Saved {interval_name} data to {filename}")
        print(f"Date range: {resampled.index.min()} to {resampled.index.max()}")
        print(f"Number of records: {len(resampled)}")
        
        resampled_dfs[interval_name] = resampled
    
    return resampled_dfs

def download_kaggle_bitcoin_data():
    """
    Download Bitcoin historical data from Kaggle dataset using the Kaggle API.
    This dataset contains comprehensive Bitcoin data from 2012 to 2021.
    
    Returns:
        pd.DataFrame: DataFrame containing Bitcoin historical data
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download the dataset using Kaggle API
    print("Downloading Kaggle dataset...")
    try:
        kaggle.api.dataset_download_files(
            'mczielinski/bitcoin-historical-data',
            path='data',
            unzip=True
        )
        
        # Find the CSV file in the data directory
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the data directory")
        
        # Read the CSV file
        df = pd.read_csv(os.path.join('data', csv_files[0]))
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        
        # Set timestamp as index
        df.set_index('Timestamp', inplace=True)
        
        # Save to CSV
        df.to_csv('bitcoin_data_1min.csv')
        print(f"Data downloaded and saved to bitcoin_data_1min.csv")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Number of records: {len(df)}")
        
        return df
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

def download_bitcoin_data(start_date='2015-01-01', end_date=None):
    """
    Download Bitcoin historical data from Yahoo Finance.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format. If None, uses current date
    
    Returns:
        pd.DataFrame: DataFrame containing Bitcoin historical data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Bitcoin-USD ticker symbol
    btc = yf.Ticker("BTC-USD")
    
    # Download historical data
    df = btc.history(start=start_date, end=end_date)
    
    # Save to CSV
    df.to_csv('bitcoin_data.csv')
    print(f"Data downloaded and saved to bitcoin_data.csv")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of records: {len(df)}")
    
    return df

if __name__ == "__main__":
    # Try to download Kaggle dataset first
    df = download_kaggle_bitcoin_data()
    
    # If Kaggle download fails, fall back to Yahoo Finance
    if df is None:
        print("\nFalling back to Yahoo Finance data...")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        df = download_bitcoin_data(start_date=start_date)
    
    # Create resampled datasets
    if df is not None:
        resampled_dfs = create_resampled_datasets(df)
        
        # Display sample of each timeframe
        for interval_name, resampled_df in resampled_dfs.items():
            print(f"\nSample of {interval_name} data:")
            print(resampled_df.head())
            print("\nLast few rows:")
            print(resampled_df.tail()) 
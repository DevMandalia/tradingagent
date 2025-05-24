import pandas as pd
import os

def merge_bitcoin_files():
    print("Loading datasets...")
    
    # Load both datasets
    btc_daily = pd.read_csv('bitcoin_data_daily.csv')
    lib_daily = pd.read_csv('look_into_bitcoin_daily_data.csv')
    
    # Convert date columns to datetime
    btc_daily['Timestamp'] = pd.to_datetime(btc_daily['Timestamp'])
    lib_daily['datetime'] = pd.to_datetime(lib_daily['datetime'])
    
    # Rename columns for consistency
    btc_daily = btc_daily.rename(columns={'Timestamp': 'date'})
    lib_daily = lib_daily.rename(columns={'datetime': 'date'})
    
    print("\nDataset shapes:")
    print(f"Bitcoin daily data: {btc_daily.shape}")
    print(f"LookIntoBitcoin data: {lib_daily.shape}")
    
    # Merge datasets on date
    print("\nMerging datasets...")
    merged_df = btc_daily.merge(lib_daily, on='date', how='outer')
    
    # Sort by date
    merged_df = merged_df.sort_values('date')
    
    # Display NaN statistics
    print("\nNaN values per column:")
    nan_stats = merged_df.isna().sum()
    print(nan_stats)
    
    # Display dataset info
    print("\nDataset statistics:")
    print(f"Total records: {len(merged_df)}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    # Save merged dataset
    output_file = 'bitcoin_merged_daily_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved merged dataset to {output_file}")
    
    # Display first and last few rows
    print("\nFirst 5 rows:")
    print(merged_df.head())
    print("\nLast 5 rows:")
    print(merged_df.tail())

if __name__ == "__main__":
    merge_bitcoin_files() 
import pandas as pd
import os

def merge_bitcoin_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Loading datasets...")
    
    # Load Bitcoin daily data
    btc_daily = pd.read_csv('data/bitcoin_data_daily.csv')
    btc_daily['date'] = pd.to_datetime(btc_daily['date'])
    
    # Load LookIntoBitcoin data
    fear_greed = pd.read_csv('data/fear_greed_index.csv')
    fear_greed['datetime'] = pd.to_datetime(fear_greed['datetime'])
    fear_greed = fear_greed.rename(columns={'datetime': 'date'})
    
    mvrv = pd.read_csv('data/mvrv_data.csv')
    mvrv['datetime'] = pd.to_datetime(mvrv['datetime'])
    mvrv = mvrv.rename(columns={'datetime': 'date'})
    
    print("\nDataset shapes:")
    print(f"Bitcoin daily data: {btc_daily.shape}")
    print(f"Fear & Greed data: {fear_greed.shape}")
    print(f"MVRV data: {mvrv.shape}")
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_df = btc_daily.merge(fear_greed, on='date', how='left')
    merged_df = merged_df.merge(mvrv, on='date', how='left')
    
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
    output_file = 'data/bitcoin_merged_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved merged dataset to {output_file}")
    
    # Display first and last few rows
    print("\nFirst 5 rows:")
    print(merged_df.head())
    print("\nLast 5 rows:")
    print(merged_df.tail())

if __name__ == "__main__":
    merge_bitcoin_data() 
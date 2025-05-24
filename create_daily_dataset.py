import pandas as pd
import numpy as np

def create_daily_dataset():
    # Read the hourly dataset
    print("Loading hourly Bitcoin data...")
    df = pd.read_csv('data/bitcoin_data_hourly.csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index for resampling
    df.set_index('date', inplace=True)
    
    # Resample to daily frequency, taking max for each column
    print("\nResampling to daily frequency...")
    daily_df = df.resample('D').max()
    
    # Remove any duplicate dates (if they exist)
    daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
    
    # Forward fill NaN values within a 7-day window
    print("\nFilling missing values...")
    daily_df = daily_df.fillna(method='ffill', limit=7)
    
    # Sort by date
    daily_df.sort_index(inplace=True)
    
    # Save the daily dataset
    output_file = 'data/bitcoin_data_daily.csv'
    daily_df.to_csv(output_file)
    print(f"\nDaily dataset saved to: {output_file}")
    
    # Display NaN statistics
    print("\nNaN values per column:")
    nan_stats = daily_df.isna().sum()
    for column, count in nan_stats.items():
        print(f"{column}: {count} NaN values")
    
    # Display head and tail of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(daily_df.head())
    
    print("\nLast 5 rows of the dataset:")
    print(daily_df.tail())
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Number of records: {len(daily_df)}")
    print(f"Date range: {daily_df.index.min()} to {daily_df.index.max()}")
    print("\nColumns:")
    for col in daily_df.columns:
        print(f"- {col}")
    
    # Display data quality metrics
    print("\nData Quality Metrics:")
    print(f"Total number of records: {len(daily_df)}")
    print(f"Total number of NaN values: {daily_df.isna().sum().sum()}")
    print(f"Percentage of complete records: {((len(daily_df) - daily_df.isna().sum().sum() / len(daily_df.columns)) / len(daily_df) * 100):.2f}%")

if __name__ == "__main__":
    create_daily_dataset() 
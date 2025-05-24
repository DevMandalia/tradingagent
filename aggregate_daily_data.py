import pandas as pd
import numpy as np

def aggregate_daily_data():
    # Read the forward-filled dataset
    print("Loading forward-filled data...")
    df = pd.read_csv('data/on_chain_data_ff.csv')
    
    # Convert date column to datetime and remove time component
    print("\nConverting dates and removing time component...")
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Group by date and take max value for each column
    print("\nAggregating by day (taking max values)...")
    daily_df = df.groupby('date').max()
    
    # Reset index to make date a column again
    daily_df = daily_df.reset_index()
    
    # Remove the first row
    print("\nRemoving first row...")
    daily_df = daily_df.iloc[1:]
    
    # Save to CSV file
    output_file = 'data/on_chain_data_daily_agg.csv'
    daily_df.to_csv(output_file, index=False)
    print(f"\nAggregated data saved to: {output_file}")
    
    # Display NaN statistics
    print("\nNaN values per column:")
    nan_stats = daily_df.isna().sum()
    for column, count in nan_stats.items():
        print(f"{column}: {count} NaN values")
    
    # Display total NaN values
    total_nans = daily_df.isna().sum().sum()
    print(f"\nTotal NaN values across entire dataset: {total_nans}")
    
    # Display head and tail of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(daily_df.head())
    
    print("\nLast 5 rows of the dataset:")
    print(daily_df.tail())
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Number of records: {len(daily_df)}")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print("\nColumns:")
    for col in daily_df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    aggregate_daily_data() 
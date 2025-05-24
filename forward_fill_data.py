import pandas as pd
import numpy as np

def forward_fill_data():
    # Read the original dataset
    print("Loading on-chain data...")
    df = pd.read_csv('data/on_chain_data.csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Forward fill NaN values
    print("\nForward filling missing values...")
    df = df.fillna(method='ffill')
    
    # Group by date and take the first 5 rows for each date
    print("\nLimiting to 5 rows per date...")
    df = df.groupby('date').head(5)
    
    # Save to new CSV file
    output_file = 'data/on_chain_data_ff.csv'
    df.to_csv(output_file, index=False)
    print(f"\nForward-filled data saved to: {output_file}")
    
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

if __name__ == "__main__":
    forward_fill_data() 
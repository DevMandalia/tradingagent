import pandas as pd
from collections import Counter

def nan_row_summary():
    df = pd.read_csv('bitcoin_merged_daily_data.csv')
    nan_counts = df.isna().sum(axis=1)
    count_by_nan = Counter(nan_counts)
    print("Number of rows by NaN count:")
    for n_nan, count in sorted(count_by_nan.items()):
        print(f"Rows with {int(n_nan)} NaN values: {count}")
    print(f"\nTotal rows: {len(df)}")

if __name__ == "__main__":
    nan_row_summary() 
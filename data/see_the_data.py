import pandas as pd

# Load the dataset
df = pd.read_csv('data/bitcoin_data_hourly.csv')

# Show the top 100 rows
print(df.head(100))
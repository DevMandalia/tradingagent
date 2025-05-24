import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the backend to interactive
plt.switch_backend('TkAgg')  # This should work on macOS

def plot_mvrv_ratio():
    # Read the merged dataset
    df = pd.read_csv('bitcoin_merged_daily_data.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate MVRV ratio
    df['mvrv_ratio'] = df['market_cap_usd'] / df['realised_cap_usd']
    
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot MVRV ratio on primary y-axis
    ax1.plot(df['date'], df['mvrv_ratio'], 'b-', label='MVRV Ratio')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MVRV Ratio', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for Bitcoin price
    ax2 = ax1.twinx()
    ax2.plot(df['date'], np.log(df['Close']), 'r-', label='Log Bitcoin Price')
    ax2.set_ylabel('Log Bitcoin Price (USD)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add title and legend
    plt.title('MVRV Ratio vs Log Bitcoin Price Over Time')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('mvrv_ratio_vs_price.png')
    print("Plot saved as 'mvrv_ratio_vs_price.png'")
    
    # Display some statistics
    print("\nMVRV Ratio Statistics:")
    print(df['mvrv_ratio'].describe())
    
    # Display correlation between MVRV ratio and log price
    correlation = df['mvrv_ratio'].corr(np.log(df['Close']))
    print(f"\nCorrelation between MVRV ratio and log price: {correlation:.4f}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_mvrv_ratio() 
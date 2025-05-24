import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def create_plots(weekly_ohlc, positions_df, trades_df, initial_capital, strategy_sharpe, buy_hold_sharpe, total_return, buy_and_hold_return, max_drawdown, num_trades, win_rate, profit_factor):
    """Create and save all visualization plots"""
    
    # Create directories if they don't exist
    os.makedirs('data/plots', exist_ok=True)
    os.makedirs('data/backtest_results', exist_ok=True)
    os.makedirs('data/trade_analysis', exist_ok=True)
    
    # Plot 1: MACD with log scale
    plt.figure(figsize=(15, 7))
    plt.title('MACD Analysis (Log Scale)')
    
    # Check if we need to offset MACD values for log scale
    min_macd = min(weekly_ohlc['macd'].min(), weekly_ohlc['macd_signal'].min())
    if min_macd <= 0:
        offset = abs(min_macd) + 1
        macd_for_log = weekly_ohlc['macd'] + offset
        macd_signal_for_log = weekly_ohlc['macd_signal'] + offset
        
        plt.plot(weekly_ohlc['time'], macd_for_log, label=f'MACD (offset +{offset:.2f})', color='blue')
        plt.plot(weekly_ohlc['time'], macd_signal_for_log, label=f'Signal Line (offset +{offset:.2f})', color='red')
    else:
        plt.plot(weekly_ohlc['time'], weekly_ohlc['macd'], label='MACD', color='blue')
        plt.plot(weekly_ohlc['time'], weekly_ohlc['macd_signal'], label='Signal Line', color='red')
    
    # Plot MACD histogram as bars
    plt.bar(weekly_ohlc['time'], weekly_ohlc['macd_histogram'], label='MACD Histogram', color='gray', alpha=0.3)
    
    # Highlight buy and sell signals
    buy_signals = weekly_ohlc[weekly_ohlc['buy_signal'] == 1]
    sell_signals = weekly_ohlc[weekly_ohlc['sell_signal'] == 1]
    
    if not buy_signals.empty:
        plt.scatter(buy_signals['time'], buy_signals['macd'] + (offset if min_macd <= 0 else 0), 
                   color='green', label='Buy Signal', marker='^', s=100)
    
    if not sell_signals.empty:
        plt.scatter(sell_signals['time'], sell_signals['macd'] + (offset if min_macd <= 0 else 0), 
                   color='red', label='Sell Signal', marker='v', s=100)
    
    plt.xlabel('Date')
    plt.ylabel('MACD Value (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/plots/weekly_mvrv_macd_signals.png')
    plt.close()
    
    # Plot 2: Equity curve (log scale)
    plt.figure(figsize=(15, 7))
    plt.title('Strategy Equity Curve vs Buy-and-Hold (Long-Only, 2012-Present)')
    plt.plot(positions_df['date'] if not positions_df.empty else [], 
             positions_df['portfolio_value'] if not positions_df.empty else [], 
             label='MVRV-MACD Long-Only Strategy', color='blue')
    
    # Calculate buy-and-hold equity curve
    buy_hold_btc = initial_capital / weekly_ohlc.iloc[1]['Open']
    buy_hold_equity = [buy_hold_btc * price for price in weekly_ohlc['Close'].iloc[1:]]
    plt.plot(weekly_ohlc['time'].iloc[1:], buy_hold_equity, label='Buy and Hold', color='gray', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/plots/weekly_mvrv_macd_equity_curve.png')
    plt.close()
    
    # Plot 3: Position type over time with BTC price (log scale)
    plt.figure(figsize=(15, 7))
    plt.title('Position Type Over Time (Long-Only, 2012-Present)')
    
    # Create a numeric representation of position type for plotting
    position_type_numeric = positions_df['position_type'].map({'NONE': 0, 'LONG': 1})
    
    # Plot position type
    plt.plot(positions_df['date'], position_type_numeric, label='Position Type', color='blue', drawstyle='steps-post')
    
    # Add Bitcoin price on secondary y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(weekly_ohlc['time'].iloc[1:], weekly_ohlc['Close'].iloc[1:], label='BTC Price', color='gray', alpha=0.5)
    ax2.set_yscale('log')
    
    # Add horizontal lines for position types
    plt.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Long')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3, label='None')
    
    # Set y-axis ticks and labels
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['NONE', 'LONG'])
    
    plt.xlabel('Date')
    ax1.set_ylabel('Position Type')
    ax2.set_ylabel('BTC Price (USD)')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True)
    plt.savefig('data/plots/weekly_mvrv_macd_position_type.png')
    plt.close()
    
    # Plot 4: Sharpe ratio comparison
    plt.figure(figsize=(10, 6))
    plt.title('Sharpe Ratio Comparison: Long-Only vs Buy-and-Hold (2012-Present)')
    strategies = ['MVRV-MACD Long-Only', 'Buy-and-Hold']
    sharpe_values = [strategy_sharpe, buy_hold_sharpe]
    
    plt.bar(strategies, sharpe_values, color=['blue', 'gray'])
    plt.ylabel('Sharpe Ratio')
    plt.grid(axis='y')
    
    # Add values on top of bars
    for i, v in enumerate(sharpe_values):
        plt.text(i, v + 0.1, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('data/plots/weekly_mvrv_macd_sharpe.png')
    plt.close()
    
    # Plot 5: MVRV ratio
    plt.figure(figsize=(15, 7))
    plt.title('MVRV Ratio Over Time')
    plt.plot(weekly_ohlc['time'], weekly_ohlc['mvrv_ratio'], label='MVRV Ratio', color='blue')
    plt.axhline(y=3.6, color='r', linestyle='--', label='Sell Threshold (3.6)')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Buy Threshold (1.0)')
    plt.xlabel('Date')
    plt.ylabel('MVRV Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/plots/weekly_mvrv_macd_mvrv_ratio.png')
    plt.close()
    
    # Save positions data
    positions_df.to_csv('data/backtest_results/weekly_mvrv_macd_positions.csv', index=False)
    
    # Create trade spreadsheet
    print("Creating trade spreadsheet...")
    trade_data = []
    
    # Process each buy signal
    for i, row in weekly_ohlc[weekly_ohlc['buy_signal'] == 1].iterrows():
        # Find matching trade in trades_df
        matching_trade = trades_df[trades_df['date'] == row['time']]
        
        if not matching_trade.empty:
            trade = matching_trade.iloc[0]
            trade_data.append({
                'Date': row['time'],
                'Action': 'BUY',
                'Bitcoin Price': row['Open'],
                'MACD': row['macd'],
                'MACD Signal Line': row['macd_signal'],
                'MACD Histogram': row['macd_histogram'],
                'MACD Diff': row['macd_diff'],
                'MACD Histogram Diff': row['macd_histogram_diff'],
                'Fast EMA': row['fast_ema'],
                'Slow EMA': row['slow_ema'],
                'Below 40-week SMA': 'Yes' if row['below_sma_40'] == 1 else 'No',
                'Bullish Side': 'Yes' if row['bullish_side'] == 1 else 'No',
                'MVRV Ratio': row['mvrv_ratio'],
                'Signal Logic': 'MVRV < 1, Price below SMA40, MACD histogram positive for second time in 6 months',
                'Portfolio Value': trade['portfolio_value'],
                'BTC Amount': trade['btc_amount'],
                'USD Value': trade['usd_value']
            })
    
    # Process each sell signal
    for i, row in weekly_ohlc[weekly_ohlc['sell_signal'] == 1].iterrows():
        matching_trade = trades_df[trades_df['date'] == row['time']]
        
        if not matching_trade.empty:
            trade = matching_trade.iloc[0]
            
            # Calculate trade return
            if len(trade_data) > 0 and trade_data[-1]['Action'] == 'BUY':
                buy_price = trade_data[-1]['Bitcoin Price']
                sell_price = row['Open']
                trade_return = (sell_price / buy_price - 1) * 100
                trade_return_str = f"{trade_return:.2f}%"
            else:
                trade_return_str = None
            
            trade_data.append({
                'Date': row['time'],
                'Action': 'SELL',
                'Bitcoin Price': row['Open'],
                'MACD': row['macd'],
                'MACD Signal Line': row['macd_signal'],
                'MACD Histogram': row['macd_histogram'],
                'MACD Diff': row['macd_diff'],
                'MACD Histogram Diff': row['macd_histogram_diff'],
                'Fast EMA': row['fast_ema'],
                'Slow EMA': row['slow_ema'],
                'Below 40-week SMA': 'Yes' if row['below_sma_40'] == 1 else 'No',
                'Bullish Side': 'Yes' if row['bullish_side'] == 1 else 'No',
                'MVRV Ratio': row['mvrv_ratio'],
                'Signal Logic': 'MVRV > 3.6, Bullish side, MACD difference negative in 6 months',
                'Portfolio Value': trade['portfolio_value'],
                'BTC Amount': trade['btc_amount'],
                'USD Value': trade['usd_value'],
                'Trade Return': trade_return_str
            })
    
    # Convert to dataframe and sort by date
    trade_df = pd.DataFrame(trade_data)
    if not trade_df.empty:
        trade_df = trade_df.sort_values('Date')
        trade_df.to_csv('data/trade_analysis/weekly_mvrv_macd_trades.csv', index=False)
    
    print("All plots and analysis files have been saved to the data directory.")

def generate_signals(df):
    # Initialize signal columns
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    
    # Process each week
    for i in range(1, len(df)):
        current_week = df.iloc[i]
        previous_week = df.iloc[i-1]
        
        # Check if we're before 2014
        is_before_2014 = current_week['time'] < pd.Timestamp('2014-01-01')
        
        # Add forced sell signal at the start of 2014
        if current_week['time'] == pd.Timestamp('2014-01-05'):  # First week of 2014
            df.loc[df.index[i], 'sell_signal'] = 1
            continue
        
        # Generate buy signals
        if is_before_2014:
            # For 2012-2014, only use MACD conditions
            if current_week['macd_histogram'] > 0 and current_week['macd_diff'] > 0:
                df.loc[df.index[i], 'buy_signal'] = 1
        else:
            # After 2014, use MVRV-MACD logic
            if current_week['mvrv_ratio'] < 1 and current_week['below_sma_40'] == 1:
                # Check if MACD histogram is positive for second time in 6 months
                six_months_ago = current_week['time'] - pd.DateOffset(months=6)
                recent_data = df[(df['time'] >= six_months_ago) & (df['time'] <= current_week['time'])]
                positive_macd_count = len(recent_data[recent_data['macd_histogram'] > 0])
                
                if positive_macd_count >= 2:
                    df.loc[df.index[i], 'buy_signal'] = 1
        
        # Generate sell signals
        if is_before_2014:
            # For 2012-2014, only use MACD conditions
            if previous_week['bullish_side'] == 1 and current_week['macd_diff'] < 0:
                df.loc[df.index[i], 'sell_signal'] = 1
        else:
            # After 2014, use MVRV-MACD logic
            if current_week['mvrv_ratio'] > 3.6 and current_week['bullish_side'] == 1:
                # Check if MACD difference is negative in 6 months
                six_months_ago = current_week['time'] - pd.DateOffset(months=6)
                recent_data = df[(df['time'] >= six_months_ago) & (df['time'] <= current_week['time'])]
                negative_macd_diff = len(recent_data[recent_data['macd_diff'] < 0])
                
                if negative_macd_diff > 0:
                    df.loc[df.index[i], 'sell_signal'] = 1
    
    return df

def main():
    # Load the hourly Bitcoin data
    df = pd.read_csv('bitcoin_merged_daily_data.csv')
    df['time'] = pd.to_datetime(df['date'])
    df.set_index('time', inplace=True)
    
    # Resample to weekly data
    weekly_ohlc = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'market_cap_usd': 'last',
        'realised_cap_usd': 'last'
    })
    weekly_ohlc.reset_index(inplace=True)
    
    # Filter data from 2012 onwards
    weekly_ohlc = weekly_ohlc[weekly_ohlc['time'] >= '2012-01-01']
    
    # Calculate weekly moving averages and indicators
    weekly_ohlc['fast_ema'] = weekly_ohlc['Close'].ewm(span=12, adjust=False).mean()
    weekly_ohlc['slow_ema'] = weekly_ohlc['Close'].ewm(span=26, adjust=False).mean()
    weekly_ohlc['sma_40'] = weekly_ohlc['Close'].rolling(window=40).mean()
    weekly_ohlc['macd'] = weekly_ohlc['fast_ema'] - weekly_ohlc['slow_ema']
    weekly_ohlc['macd_signal'] = weekly_ohlc['macd'].ewm(span=9, adjust=False).mean()
    weekly_ohlc['macd_histogram'] = weekly_ohlc['macd'] - weekly_ohlc['macd_signal']
    weekly_ohlc['macd_diff'] = weekly_ohlc['macd'].diff()
    weekly_ohlc['macd_histogram_diff'] = weekly_ohlc['macd_histogram'].diff()
    weekly_ohlc['mvrv_ratio'] = weekly_ohlc['market_cap_usd'] / weekly_ohlc['realised_cap_usd']
    
    # Identify bullish and bearish sides
    weekly_ohlc['bullish_side'] = (weekly_ohlc['fast_ema'] > weekly_ohlc['slow_ema']).astype(int)
    weekly_ohlc['below_sma_40'] = (weekly_ohlc['Close'] < weekly_ohlc['sma_40']).astype(int)
    
    # Generate signals
    weekly_ohlc = generate_signals(weekly_ohlc)
    
    # Save the weekly dataset with signals
    os.makedirs('data/weekly', exist_ok=True)
    weekly_data_path = 'data/weekly/BTC_USD_weekly_with_mvrv_macd_signals_from_2012.csv'
    weekly_ohlc.to_csv(weekly_data_path, index=False)
    
    # Run long-only backtest with position check
    initial_capital = 10000  # $10,000 USD
    capital = initial_capital
    btc_holdings = 0
    positions = []
    equity_curve = []
    trades = []
    position_type = "NONE"  # Track current position type: "LONG" or "NONE"
    
    # Run through each week
    for i in range(1, len(weekly_ohlc)):
        week = weekly_ohlc.iloc[i]
        prev_week = weekly_ohlc.iloc[i-1]
        
        # Calculate portfolio value at the start of the week
        if position_type == "LONG":
            portfolio_value = capital + btc_holdings * week['Open']
        else:  # NONE
            portfolio_value = capital
        
        # Check if we're before or after 2014
        is_before_2014 = week['time'] < pd.Timestamp('2014-01-01')
        
        # Force sell at the start of 2014 if in a position
        if week['time'] == pd.Timestamp('2014-01-05') and position_type == "LONG":
            # Close the long position and go to cash
            if btc_holdings > 0:
                # Calculate profit/loss from long position
                long_profit = btc_holdings * (week['Open'] - prev_week['Close'])
                capital += btc_holdings * week['Open']
                
                trades.append({
                    'date': week['time'],
                    'type': 'SELL',
                    'price': week['Open'],
                    'btc_amount': btc_holdings,
                    'usd_value': btc_holdings * week['Open'],
                    'portfolio_value': portfolio_value,
                    'profit_loss': long_profit
                })
                
                btc_holdings = 0
                position_type = "NONE"
        
        if is_before_2014:
            # Before 2014: Use MACD-only signals
            # Check for buy signal - only register if we're not already in a long position
            if week['buy_signal'] == 1 and position_type != "LONG":
                # Open a long position with all available capital
                if capital > 0:
                    btc_bought = capital / week['Open']
                    btc_holdings = btc_bought
                    
                    trades.append({
                        'date': week['time'],
                        'type': 'BUY',
                        'price': week['Open'],
                        'btc_amount': btc_bought,
                        'usd_value': capital,
                        'portfolio_value': portfolio_value
                    })
                    
                    capital = 0
                    position_type = "LONG"
            
            # Check for sell signal - only register if we're in a long position
            elif week['sell_signal'] == 1 and position_type == "LONG":
                # Close the long position and go to cash
                if btc_holdings > 0:
                    # Calculate profit/loss from long position
                    long_profit = btc_holdings * (week['Open'] - prev_week['Close'])
                    capital += btc_holdings * week['Open']
                    
                    trades.append({
                        'date': week['time'],
                        'type': 'SELL',
                        'price': week['Open'],
                        'btc_amount': btc_holdings,
                        'usd_value': btc_holdings * week['Open'],
                        'portfolio_value': portfolio_value,
                        'profit_loss': long_profit
                    })
                    
                    btc_holdings = 0
                    position_type = "NONE"
        else:
            # From 2014 onwards: Use MVRV-MACD strategy
            # Check for buy signal - only register if we're not already in a long position
            if week['buy_signal'] == 1 and position_type != "LONG":
                # Open a long position with all available capital
                if capital > 0:
                    btc_bought = capital / week['Open']
                    btc_holdings = btc_bought
                    
                    trades.append({
                        'date': week['time'],
                        'type': 'BUY',
                        'price': week['Open'],
                        'btc_amount': btc_bought,
                        'usd_value': capital,
                        'portfolio_value': portfolio_value
                    })
                    
                    capital = 0
                    position_type = "LONG"
            
            # Check for sell signal - only register if we're in a long position
            elif week['sell_signal'] == 1 and position_type == "LONG":
                # Close the long position and go to cash
                if btc_holdings > 0:
                    # Calculate profit/loss from long position
                    long_profit = btc_holdings * (week['Open'] - prev_week['Close'])
                    capital += btc_holdings * week['Open']
                    
                    trades.append({
                        'date': week['time'],
                        'type': 'SELL',
                        'price': week['Open'],
                        'btc_amount': btc_holdings,
                        'usd_value': btc_holdings * week['Open'],
                        'portfolio_value': portfolio_value,
                        'profit_loss': long_profit
                    })
                    
                    btc_holdings = 0
                    position_type = "NONE"
        
        # Calculate end-of-week portfolio value
        if position_type == "LONG":
            end_portfolio_value = capital + btc_holdings * week['Close']
        else:  # NONE
            end_portfolio_value = capital
        
        # Record position and portfolio value
        positions.append({
            'date': week['time'],
            'capital': capital,
            'btc_holdings': btc_holdings,
            'position_type': position_type,
            'btc_price': week['Close'],
            'portfolio_value': end_portfolio_value
        })
        
        equity_curve.append(end_portfolio_value)
    
    # Convert positions and trades to DataFrames
    positions_df = pd.DataFrame(positions)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['date', 'type', 'price', 'btc_amount', 'usd_value', 'portfolio_value'])
    
    # Calculate buy-and-hold performance for comparison
    buy_and_hold_btc = initial_capital / weekly_ohlc.iloc[1]['Open']
    buy_and_hold_value = buy_and_hold_btc * weekly_ohlc.iloc[-1]['Close']
    buy_and_hold_return = (buy_and_hold_value / initial_capital - 1) * 100
    
    # Calculate strategy performance metrics
    final_portfolio_value = positions_df.iloc[-1]['portfolio_value'] if not positions_df.empty else 0
    total_return = (final_portfolio_value / initial_capital - 1) * 100
    max_drawdown = 0
    peak = positions_df['portfolio_value'].iloc[0] if not positions_df.empty else 0
    
    for value in positions_df['portfolio_value'] if not positions_df.empty else []:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate additional metrics
    num_trades = len(trades_df)
    num_winning_trades = 0
    total_profit = 0
    total_loss = 0
    
    # Count trade types
    buy_trades = trades_df[trades_df['type'].str.contains('BUY')].shape[0] if not trades_df.empty else 0
    sell_trades = trades_df[trades_df['type'] == 'SELL'].shape[0] if not trades_df.empty else 0
    
    # Calculate trade profitability
    if 'profit_loss' in trades_df.columns and not trades_df.empty:
        profitable_trades = trades_df[trades_df['profit_loss'] > 0]
        num_winning_trades = len(profitable_trades)
        total_profit = profitable_trades['profit_loss'].sum() if not profitable_trades.empty else 0
        
        losing_trades = trades_df[trades_df['profit_loss'] < 0]
        total_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
    
    win_rate = (num_winning_trades / sell_trades * 100) if sell_trades > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
    
    # Calculate Sharpe ratio
    positions_df['weekly_return'] = positions_df['portfolio_value'].pct_change()
    strategy_returns = positions_df['weekly_return'].dropna()
    strategy_mean_return = strategy_returns.mean()
    strategy_std_return = strategy_returns.std()
    strategy_sharpe = (strategy_mean_return / strategy_std_return) * np.sqrt(52)  # Annualized for weekly data
    
    # Calculate buy-and-hold Sharpe ratio
    weekly_ohlc['buy_hold_value'] = buy_and_hold_btc * weekly_ohlc['Close']
    weekly_ohlc['buy_hold_return'] = weekly_ohlc['buy_hold_value'].pct_change()
    buy_hold_returns = weekly_ohlc['buy_hold_return'].dropna()
    buy_hold_mean_return = buy_hold_returns.mean()
    buy_hold_std_return = buy_hold_returns.std()
    buy_hold_sharpe = (buy_hold_mean_return / buy_hold_std_return) * np.sqrt(52)  # Annualized for weekly data
    
    # Create performance report
    report = f"""# Weekly MVRV-MACD Long-Only Strategy Backtest Results (2012-Present)

## Strategy Overview
- **Strategy**: Weekly MACD signal-based long-only trading with MVRV ratio conditions
- **Long Entry Conditions**: 
  1. When price is below 40-week SMA and MACD difference turned positive for the second time AND MVRV ratio < 1
  2. When on bullish side (fast EMA > slow EMA) and MACD histogram difference turns positive AND MVRV ratio < 1
- **Exit Condition**: When on bullish side (fast EMA > slow EMA) and MACD difference turns negative AND MVRV ratio > 3.6
- **Position Management**: Full allocation, only register signals when not already in that position type
- **Initial Capital**: ${initial_capital:,.2f}
- **Test Period**: {weekly_ohlc['time'].min()} to {weekly_ohlc['time'].max()}

## Performance Metrics

### Returns
- **Final Portfolio Value**: ${final_portfolio_value:,.2f}
- **Total Return**: {total_return:.2f}%
- **Buy-and-Hold Return**: {buy_and_hold_return:.2f}%
- **Outperformance vs Buy-and-Hold**: {total_return - buy_and_hold_return:.2f}%

### Risk Metrics
- **Maximum Drawdown**: {max_drawdown:.2f}%
- **Sharpe Ratio**: {strategy_sharpe:.4f}
- **Buy-and-Hold Sharpe Ratio**: {buy_hold_sharpe:.4f}
- **Sharpe Ratio Difference**: {strategy_sharpe - buy_hold_sharpe:.4f}

### Trading Statistics
- **Number of Trades**: {num_trades}
- **Buy Trades**: {buy_trades}
- **Sell Trades**: {sell_trades}
- **Win Rate**: {win_rate:.2f}%
- **Profit Factor**: {profit_factor:.2f}
"""
    
    # Save performance report
    os.makedirs('data/backtest_results', exist_ok=True)
    with open('data/backtest_results/weekly_mvrv_macd_performance_report.md', 'w') as f:
        f.write(report)
    
    # Create plots
    create_plots(weekly_ohlc, positions_df, trades_df, initial_capital, strategy_sharpe, buy_hold_sharpe, 
                total_return, buy_and_hold_return, max_drawdown, num_trades, win_rate, profit_factor)
    
    # Print and save results for Weekly MVRV MACD strategy and Buy & Hold separately
    print("\n==== Weekly MVRV MACD Strategy Results ====")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {strategy_sharpe:.4f}")

    print("\n==== Buy & Hold Strategy Results ====")
    print(f"Final Portfolio Value: ${buy_and_hold_value:,.2f}")
    print(f"Total Return: {buy_and_hold_return:.2f}%")
    print(f"Maximum Drawdown: {weekly_ohlc['buy_hold_value'].cummax().sub(weekly_ohlc['buy_hold_value']).div(weekly_ohlc['buy_hold_value'].cummax()).max() * 100:.2f}%")
    print(f"Sharpe Ratio: {buy_hold_sharpe:.4f}")

    # Save results to markdown
    with open('data/backtest_results/weekly_mvrv_macd_vs_buyhold_results.md', 'w') as f:
        f.write("# Weekly MVRV MACD vs Buy & Hold Results\n\n")
        f.write("## Weekly MVRV MACD Strategy\n")
        f.write(f"Final Portfolio Value: ${final_portfolio_value:,.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Maximum Drawdown: {max_drawdown:.2f}%\n")
        f.write(f"Number of Trades: {num_trades}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Sharpe Ratio: {strategy_sharpe:.4f}\n\n")
        f.write("## Buy & Hold Strategy\n")
        f.write(f"Final Portfolio Value: ${buy_and_hold_value:,.2f}\n")
        f.write(f"Total Return: {buy_and_hold_return:.2f}%\n")
        f.write(f"Maximum Drawdown: {weekly_ohlc['buy_hold_value'].cummax().sub(weekly_ohlc['buy_hold_value']).div(weekly_ohlc['buy_hold_value'].cummax()).max() * 100:.2f}%\n")
        f.write(f"Sharpe Ratio: {buy_hold_sharpe:.4f}\n")

if __name__ == "__main__":
    main() 
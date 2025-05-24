import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class WeeklyMACDTradingBot:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, initial_capital=10000):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_holdings = 0
        self.position_type = "NONE"  # "NONE" or "LONG"
        self.trades = []
        self.equity_curve = [initial_capital]
        
    def calculate_macd(self, df):
        """Calculate MACD indicators"""
        # Calculate MACD line
        exp1 = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        # Calculate Signal line
        df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate MACD histogram difference
        df['macd_histogram_diff'] = df['macd_histogram'].diff()
        
        # Calculate 40-week SMA
        df['sma_40'] = df['Close'].rolling(window=40).mean()
        
        # Identify bull and bear sides
        df['bull_side'] = (df['macd'] > df['macd_signal']).astype(int)
        
        return df
    
    def generate_signals(self, df):
        """Generate buy and sell signals based on MACD"""
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        
        # Track consecutive positive/negative histogram differences
        df['consecutive_positive_diff'] = 0
        df['consecutive_negative_diff'] = 0
        
        for i in range(1, len(df)):
            current_week = df.iloc[i]
            prev_week = df.iloc[i-1]
            
            # Update consecutive differences
            if current_week['macd_histogram_diff'] > 0:
                df.loc[df.index[i], 'consecutive_positive_diff'] = prev_week['consecutive_positive_diff'] + 1
                df.loc[df.index[i], 'consecutive_negative_diff'] = 0
            elif current_week['macd_histogram_diff'] < 0:
                df.loc[df.index[i], 'consecutive_negative_diff'] = prev_week['consecutive_negative_diff'] + 1
                df.loc[df.index[i], 'consecutive_positive_diff'] = 0
            else:
                df.loc[df.index[i], 'consecutive_positive_diff'] = 0
                df.loc[df.index[i], 'consecutive_negative_diff'] = 0
            
            # On bull side (MACD above signal line)
            if current_week['bull_side'] == 1:
                # Generate sell signal when histogram difference turns negative
                if current_week['macd_histogram_diff'] < 0 and prev_week['macd_histogram_diff'] >= 0:
                    df.loc[df.index[i], 'sell_signal'] = 1
                
                # Generate buy signal when histogram difference turns positive again
                if current_week['macd_histogram_diff'] > 0 and prev_week['macd_histogram_diff'] <= 0:
                    df.loc[df.index[i], 'buy_signal'] = 1
            
            # On bear side (MACD below signal line)
            else:
                # Generate buy signal when price is below 40-week SMA and histogram difference turns positive for second time
                if (current_week['Close'] < current_week['sma_40'] and 
                    current_week['consecutive_positive_diff'] == 2):
                    df.loc[df.index[i], 'buy_signal'] = 1
                
                # Generate sell signal when histogram difference turns negative again
                if current_week['macd_histogram_diff'] < 0 and prev_week['macd_histogram_diff'] >= 0:
                    df.loc[df.index[i], 'sell_signal'] = 1
        
        return df
    
    def backtest(self, df):
        """Run backtest and calculate performance metrics"""
        self.current_capital = self.initial_capital
        self.btc_holdings = 0
        self.position_type = "NONE"
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        # Find the first valid index where we have all indicators
        first_valid_idx = max(
            df['macd'].first_valid_index(),
            df['macd_signal'].first_valid_index()
        )
        
        # Filter data up to current date
        current_date = pd.Timestamp.now()
        df = df[df.index <= current_date]
        
        # Run through each week
        for i in range(df.index.get_loc(first_valid_idx) + 1, len(df)):
            week = df.iloc[i]
            prev_week = df.iloc[i-1]
            
            # Calculate portfolio value at the start of the week
            if self.position_type == "LONG":
                portfolio_value = self.current_capital + self.btc_holdings * week['Open']
            else:  # NONE
                portfolio_value = self.current_capital
            
            # Check for buy signal
            if week['buy_signal'] == 1 and self.position_type != "LONG":
                # Open a long position with all available capital
                if self.current_capital > 0:
                    btc_bought = self.current_capital / week['Open']
                    self.btc_holdings = btc_bought
                    
                    self.trades.append({
                        'date': week.name,
                        'type': 'BUY',
                        'price': week['Open'],
                        'btc_amount': btc_bought,
                        'usd_value': self.current_capital,
                        'portfolio_value': portfolio_value
                    })
                    
                    self.current_capital = 0
                    self.position_type = "LONG"
            
            # Check for sell signal
            elif week['sell_signal'] == 1 and self.position_type == "LONG":
                # Close the long position and go to cash
                if self.btc_holdings > 0:
                    # Calculate profit/loss from long position
                    entry_price = self.trades[-1]['price']  # Get the entry price from the last buy trade
                    profit_loss = self.btc_holdings * (week['Open'] - entry_price)
                    self.current_capital += self.btc_holdings * week['Open']
                    
                    self.trades.append({
                        'date': week.name,
                        'type': 'SELL',
                        'price': week['Open'],
                        'btc_amount': self.btc_holdings,
                        'usd_value': self.btc_holdings * week['Open'],
                        'portfolio_value': portfolio_value,
                        'profit_loss': profit_loss
                    })
                    
                    self.btc_holdings = 0
                    self.position_type = "NONE"
            
            # Calculate end-of-week portfolio value
            if self.position_type == "LONG":
                end_portfolio_value = self.current_capital + self.btc_holdings * week['Close']
            else:  # NONE
                end_portfolio_value = self.current_capital
            
            self.equity_curve.append(end_portfolio_value)
        
        return self.calculate_metrics(df)
    
    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate returns
        final_portfolio_value = self.equity_curve[-1]
        total_return = (final_portfolio_value / self.initial_capital - 1) * 100
        
        # Calculate buy and hold returns
        buy_hold_btc = self.initial_capital / df.iloc[0]['Open']
        buy_hold_value = buy_hold_btc * df.iloc[-1]['Close']
        buy_hold_return = (buy_hold_value / self.initial_capital - 1) * 100
        
        # Calculate max drawdown
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdowns = (rolling_max - equity_curve) / rolling_max * 100
        max_drawdown = drawdowns.max()
        
        # Calculate win rate and profit factor
        if len(trades_df) >= 2:
            buy_trades = trades_df[trades_df['type'] == 'BUY'].shape[0]
            sell_trades = trades_df[trades_df['type'] == 'SELL'].shape[0]
            
            if 'profit_loss' in trades_df.columns:
                profitable_trades = trades_df[trades_df['profit_loss'] > 0]
                num_winning_trades = len(profitable_trades)
                total_profit = profitable_trades['profit_loss'].sum() if not profitable_trades.empty else 0
                
                losing_trades = trades_df[trades_df['profit_loss'] < 0]
                total_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
                
                win_rate = (num_winning_trades / sell_trades * 100) if sell_trades > 0 else 0
                profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
        
        # Calculate Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(52) * returns.mean() / returns.std()  # Annualized
        
        return {
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_results(self, df):
        """Create and save visualization plots"""
        # Create directories if they don't exist
        os.makedirs('data/plots', exist_ok=True)
        os.makedirs('data/backtest_results', exist_ok=True)
        os.makedirs('data/trade_analysis', exist_ok=True)
        
        # Plot 1: MACD with log scale
        plt.figure(figsize=(15, 7))
        plt.title('MACD Analysis (Log Scale)')
        
        # Check if we need to offset MACD values for log scale
        min_macd = min(df['macd'].min(), df['macd_signal'].min())
        if min_macd <= 0:
            offset = abs(min_macd) + 1
            macd_for_log = df['macd'] + offset
            macd_signal_for_log = df['macd_signal'] + offset
            
            plt.plot(df.index, macd_for_log, label=f'MACD (offset +{offset:.2f})', color='blue')
            plt.plot(df.index, macd_signal_for_log, label=f'Signal Line (offset +{offset:.2f})', color='red')
        else:
            plt.plot(df.index, df['macd'], label='MACD', color='blue')
            plt.plot(df.index, df['macd_signal'], label='Signal Line', color='red')
        
        # Plot MACD histogram as bars
        plt.bar(df.index, df['macd_histogram'], label='MACD Histogram', color='gray', alpha=0.3)
        
        # Highlight buy and sell signals
        buy_signals = df[df['buy_signal'] == 1]
        sell_signals = df[df['sell_signal'] == 1]
        
        if not buy_signals.empty:
            plt.scatter(buy_signals.index, buy_signals['macd'] + (offset if min_macd <= 0 else 0), 
                       color='green', label='Buy Signal', marker='^', s=100)
        
        if not sell_signals.empty:
            plt.scatter(sell_signals.index, sell_signals['macd'] + (offset if min_macd <= 0 else 0), 
                       color='red', label='Sell Signal', marker='v', s=100)
        
        plt.xlabel('Date')
        plt.ylabel('MACD Value (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('data/plots/weekly_macd_signals.png')
        plt.close()
        
        # Plot 2: Equity curve (log scale)
        plt.figure(figsize=(15, 7))
        plt.title('Strategy Equity Curve vs Buy-and-Hold')
        
        # Create date range for equity curve
        dates = df.index[1:len(self.equity_curve)]
        plt.plot(dates, self.equity_curve[1:], label='MACD Strategy', color='blue')
        
        # Calculate buy-and-hold equity curve
        buy_hold_btc = self.initial_capital / df.iloc[0]['Open']
        buy_hold_equity = [buy_hold_btc * price for price in df['Close'].iloc[1:len(self.equity_curve)]]
        plt.plot(dates, buy_hold_equity, label='Buy and Hold', color='gray', linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/plots/weekly_macd_equity_curve.png')
        plt.close()
        
        # Save trades to CSV
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv('data/trade_analysis/weekly_macd_trades.csv', index=False)

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
        'Volume': 'sum'
    })
    
    # Filter data from 2012 onwards up to current date
    current_date = pd.Timestamp.now()
    weekly_ohlc = weekly_ohlc[(weekly_ohlc.index >= '2012-01-01') & (weekly_ohlc.index <= current_date)]
    
    # Initialize and run the trading bot
    bot = WeeklyMACDTradingBot(initial_capital=10000)
    weekly_ohlc = bot.calculate_macd(weekly_ohlc)
    weekly_ohlc = bot.generate_signals(weekly_ohlc)
    metrics = bot.backtest(weekly_ohlc)
    bot.plot_results(weekly_ohlc)
    
    # Print results
    print("\n==== Weekly MACD Strategy Results ====")
    print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    print("\n==== Buy & Hold Strategy Results ====")
    print(f"Final Portfolio Value: ${metrics['buy_hold_return'] + 10000:,.2f}")
    print(f"Total Return: {metrics['buy_hold_return']:.2f}%")
    
    # Save results to markdown
    os.makedirs('data/backtest_results', exist_ok=True)
    with open('data/backtest_results/weekly_macd_results.md', 'w') as f:
        f.write("# Weekly MACD Strategy Results\n\n")
        f.write("## Strategy Performance\n")
        f.write(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}\n")
        f.write(f"Total Return: {metrics['total_return']:.2f}%\n")
        f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%\n")
        f.write(f"Number of Trades: {metrics['num_trades']}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n\n")
        f.write("## Buy & Hold Strategy\n")
        f.write(f"Final Portfolio Value: ${metrics['buy_hold_return'] + 10000:,.2f}\n")
        f.write(f"Total Return: {metrics['buy_hold_return']:.2f}%\n")

if __name__ == "__main__":
    main() 
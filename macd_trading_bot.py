import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from datetime import datetime
import seaborn as sns

class MACDTradingBot:
    def __init__(self, initial_capital=10000, slope_threshold=0.01):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # 0: no position, 1: long position
        self.trades = []
        self.equity_curve = []
        self.slope_threshold = slope_threshold
        
    def calculate_macd(self, df):
        """Calculate MACD and its components"""
        macd = MACD(
            close=df['Close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Calculate MACD slope
        df['MACD_Slope'] = df['MACD_Histogram'].diff()
        
        # Calculate 40-week SMA
        df['SMA_40'] = df['Close'].rolling(window=40).mean()
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals based on MACD and price action"""
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # MACD Crossover signals
        df['MACD_Crossover'] = np.where(
            (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)),
                -1,  # Bearish crossover
                0
            )
        )
        
        # Slope change signals (with threshold)
        df['Slope_Change'] = np.where(
            (df['MACD_Slope'] > self.slope_threshold) & (df['MACD_Slope'].shift(1) <= self.slope_threshold),
            1,  # Positive slope change
            np.where(
                (df['MACD_Slope'] < -self.slope_threshold) & (df['MACD_Slope'].shift(1) >= -self.slope_threshold),
                -1,  # Negative slope change
                0
            )
        )
        
        # Track consecutive negative slope changes
        negative_slope_count = 0
        
        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            sma_40 = df['SMA_40'].iloc[i]
            slope = df['MACD_Slope'].iloc[i]
            prev_slope = df['MACD_Slope'].iloc[i-1]
            
            # Buy when price is below 40-week SMA and slope turns positive (crosses above threshold)
            if price < sma_40 and prev_slope <= self.slope_threshold and slope > self.slope_threshold:
                if self.position == 0:
                    df.loc[df.index[i], 'Signal'] = 1
                    self.position = 1
                    negative_slope_count = 0  # Reset counter on buy
            # Otherwise, use original logic (with threshold)
            elif (df['MACD_Crossover'].iloc[i] == 1 or  # Bullish crossover
                  (df['Slope_Change'].iloc[i] == 1 and price >= sma_40)):
                if self.position == 0:
                    df.loc[df.index[i], 'Signal'] = 1
                    self.position = 1
                    negative_slope_count = 0  # Reset counter on buy
            
            # Sell: after bullish crossover, when slope turns negative for the second time
            elif df['Slope_Change'].iloc[i] == -1 and self.position == 1:
                negative_slope_count += 1
                if negative_slope_count >= 2:  # Sell on second negative slope change
                    df.loc[df.index[i], 'Signal'] = -1
                    self.position = 0
                    negative_slope_count = 0  # Reset counter after sell
            else:
                # Reset counter if slope is not negative
                if slope >= -self.slope_threshold:
                    negative_slope_count = 0
        
        return df
    
    def backtest(self, df):
        """Run backtest and calculate performance metrics"""
        self.position = 0
        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        # Find the first valid index where we have all indicators
        first_valid_idx = max(
            df['MACD'].first_valid_index(),
            df['SMA_40'].first_valid_index()
        )
        
        # Start with a buy position at the beginning
        if first_valid_idx is not None:
            self.position = 1
            self.trades.append({
                'type': 'buy',
                'date': first_valid_idx,
                'price': df.loc[first_valid_idx, 'Close'],
                'capital': self.current_capital
            })
            self.equity_curve = [self.initial_capital] * (df.index.get_loc(first_valid_idx) + 1)
        
        # Continue with the rest of the backtest
        for i in range(df.index.get_loc(first_valid_idx) + 1, len(df)):
            if df['Signal'].iloc[i] == 1:  # Buy signal
                if self.position == 0:  # Only buy if no position
                    self.trades.append({
                        'type': 'buy',
                        'date': df.index[i],
                        'price': df['Close'].iloc[i],
                        'capital': self.current_capital
                    })
                    self.position = 1
            
            elif df['Signal'].iloc[i] == -1:  # Sell signal
                if self.position == 1:  # Only sell if we have a position
                    last_trade = self.trades[-1]
                    if last_trade['type'] == 'buy':
                        # Calculate returns
                        returns = (df['Close'].iloc[i] - last_trade['price']) / last_trade['price']
                        self.current_capital *= (1 + returns)
                        
                        self.trades.append({
                            'type': 'sell',
                            'date': df.index[i],
                            'price': df['Close'].iloc[i],
                            'capital': self.current_capital
                        })
                        self.position = 0
            
            self.equity_curve.append(self.current_capital)
        
        # Close any open position at the end
        if self.position == 1 and self.trades:
            last_trade = self.trades[-1]
            if last_trade['type'] == 'buy':
                returns = (df['Close'].iloc[-1] - last_trade['price']) / last_trade['price']
                self.current_capital *= (1 + returns)
                
                self.trades.append({
                    'type': 'sell',
                    'date': df.index[-1],
                    'price': df['Close'].iloc[-1],
                    'capital': self.current_capital
                })
        
        return self.calculate_metrics(df)
    
    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Calculate buy and hold returns
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        # Calculate win rate
        if len(trades_df) >= 2:
            trades_df['returns'] = trades_df['capital'].pct_change()
            win_rate = len(trades_df[trades_df['returns'] > 0]) / len(trades_df[trades_df['returns'].notna()])
        else:
            win_rate = 0
        
        # Calculate max drawdown
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate Sharpe ratio (assuming weekly returns)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(52) * returns.mean() / returns.std()  # Annualized
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades_df[trades_df['type'] == 'buy'])
        }
    
    def plot_results(self, df):
        """Plot trading results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price and equity curve
        ax1.plot(df.index, df['Close'], label='Bitcoin Price', alpha=0.5)
        ax1.plot(df.index, df['SMA_40'], label='40-week SMA', alpha=0.5)
        
        # Plot buy and sell points
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            buy_points = trades_df[trades_df['type'] == 'buy']
            sell_points = trades_df[trades_df['type'] == 'sell']
            
            ax1.scatter(buy_points['date'], buy_points['price'], 
                       marker='^', color='g', label='Buy', alpha=1)
            ax1.scatter(sell_points['date'], sell_points['price'], 
                       marker='v', color='r', label='Sell', alpha=1)
        
        ax1.set_title('Bitcoin Price and Trading Signals')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')
        
        # Plot equity curve
        ax2.plot(df.index, self.equity_curve, label='MACD Strategy Equity Curve', color='purple')
        # Plot buy and hold equity curve
        buy_hold_equity = self.initial_capital * (df['Close'] / df['Close'].iloc[0])
        ax2.plot(df.index, buy_hold_equity, label='Buy & Hold Equity Curve', color='orange', linestyle='--')
        ax2.set_title('Equity Curve')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        # Plot MACD
        ax3.bar(df.index, df['MACD_Histogram'], label='MACD Histogram', color='blue', alpha=0.5)
        ax3.plot(df.index, df['MACD'], label='MACD', color='red')
        ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='green')
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Load data
    df = pd.read_csv('bitcoin_weekly_technical.csv', index_col='Timestamp', parse_dates=True)
    
    # Initialize and run trading bot
    bot = MACDTradingBot(initial_capital=10000)
    df = bot.calculate_macd(df)
    df = bot.generate_signals(df)
    metrics = bot.backtest(df)
    
    # Print results
    print("\nTrading Bot Performance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {metrics['num_trades']}")
    
    # Plot results
    bot.plot_results(df)

if __name__ == "__main__":
    main() 
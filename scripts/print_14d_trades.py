import pandas as pd
from pathlib import Path

def print_trade_list(trades_df):
    print("\n" + "="*50)
    print("DETAILED TRADE LIST (14-Day Backtest)")
    print("="*50)
    
    # Sort by timestamp
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp')
    
    for _, t in trades_df.iterrows():
        # Format time
        time_str = t['timestamp'].strftime("%Y-%m-%d %H:%M")
        
        # Clean pair name
        pair_clean = t['pair'].replace('USDT_USDT', 'USDT')
        
        # Add emoji based on result
        emoji = "🚀" if t['pnl_pct'] > 20 else "✅" if t['net_profit'] > 0 else "❌"
        if t['net_profit'] > 0 and t['pnl_pct'] < 1: emoji = "🛡️" # Breakeven/Small profit
        
        print(f"{pair_clean} ({t['direction']}) {time_str} — Profit: ${t['net_profit']:+,.2f} ({t['pnl_pct']:+.1f}%) {emoji}")
        print(f"   Entry: {t['entry_price']:.5f} | Exit: {t['exit_price']:.5f} | Reason: {t['outcome']}")
        print("-" * 30)

if __name__ == "__main__":
    file_path = Path("models/v7_sniper/backtest_trades_14d.csv")
    if file_path.exists():
        df = pd.read_csv(file_path)
        print_trade_list(df)
    else:
        print(f"File not found: {file_path}")

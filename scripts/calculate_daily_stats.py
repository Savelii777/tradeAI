import pandas as pd
import sys

def analyze_day(file_path, date_label):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    total_trades = len(df)
    if total_trades == 0:
        print(f"No trades for {date_label}")
        return

    wins = df[df['net_profit'] > 0]
    losses = df[df['net_profit'] <= 0]
    
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades) * 100
    
    total_pnl_usd = df['net_profit'].sum()
    total_pnl_pct = df['pnl_pct'].sum()
    
    # Average PnL per trade
    avg_pnl_usd = total_pnl_usd / total_trades
    
    print(f"=== Статистика за {date_label} ===")
    print(f"Всего сделок: {total_trades}")
    print(f"Прибыльных:   {num_wins} ({win_rate:.1f}%)")
    print(f"Убыточных:    {num_losses}")
    print(f"Общий доход:  ${total_pnl_usd:,.2f}")
    print(f"Суммарный %:  {total_pnl_pct:.2f}%")
    print(f"Средний PnL:  ${avg_pnl_usd:,.2f}")
    print("-" * 30)
    print("Топ профитных сделок:")
    for _, row in wins.sort_values('net_profit', ascending=False).head(3).iterrows():
        print(f"  {row['pair']} ({row['direction']}): +${row['net_profit']:,.2f} (+{row['pnl_pct']:.2f}%)")
    print("\n")

def main():
    analyze_day("models/v7_sniper/backtest_trades_dec25.csv", "25 Декабря")
    analyze_day("models/v7_sniper/backtest_trades_dec26.csv", "26 Декабря")

if __name__ == "__main__":
    main()

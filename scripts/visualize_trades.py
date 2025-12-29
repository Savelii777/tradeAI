import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os

# Config
TRADES_FILES = [
    "models/v7_sniper/backtest_trades_dec25.csv",
    "models/v7_sniper/backtest_trades_dec26.csv"
]
DATA_DIR = "data/candles"
OUTPUT_DIR = "results/charts"
TIMEFRAME = "5m"

def load_candles(pair, timeframe):
    safe_pair = pair.replace('/', '_').replace(':', '_')
    path = Path(DATA_DIR) / f"{safe_pair}_{timeframe}.csv"
    if not path.exists():
        print(f"Warning: Data file not found for {pair}: {path}")
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def visualize_trade(trade, trade_idx, date_label):
    pair = trade['pair']
    entry_time = pd.to_datetime(trade['timestamp'])
    exit_time = pd.to_datetime(trade['exit_time'])
    
    # Load data
    df = load_candles(pair, TIMEFRAME)
    if df is None:
        return

    # Define window dynamic based on trade duration
    duration = exit_time - entry_time
    # Minimum view context: 2 hours, or 3x trade duration
    context_duration = max(pd.Timedelta(hours=2), duration * 3)
    
    start_window = entry_time - (context_duration / 2)
    end_window = exit_time + (context_duration / 2)
    
    mask = (df.index >= start_window) & (df.index <= end_window)
    subset = df.loc[mask]
    
    if subset.empty:
        print(f"No data found for trade {trade_idx} in window {start_window} - {end_window}")
        return

    # Calculate levels
    sl_dist = trade['sl_dist']
    entry_price = trade['entry_price']
    direction = trade['direction']
    exit_price = trade['exit_price']
    pnl_pct = trade['pnl_pct']
    
    # Determine colors and symbols
    is_win = pnl_pct > 0
    trade_color = '#00ff00' if is_win else '#ff0000'  # Bright Green or Red
    
    # Estimate RR based on standard 1:3 (from backtest config)
    rr_ratio = 3.0 
    
    if direction == 'LONG':
        sl_price = entry_price - sl_dist
        tp_price = entry_price + (sl_dist * rr_ratio)
        entry_arrow_color = '#00ccff' # Cyan for Long Entry
        entry_ay = 40 # Text below, pointing up
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - (sl_dist * rr_ratio)
        entry_arrow_color = '#ff9900' # Orange for Short Entry
        entry_ay = -40 # Text above, pointing down

    fig = go.Figure(data=[go.Candlestick(x=subset.index,
                open=subset['open'],
                high=subset['high'],
                low=subset['low'],
                close=subset['close'],
                name=pair,
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'
    )])

    # 1. Trade Path Line (Connect Entry to Exit)
    fig.add_shape(
        type="line",
        x0=entry_time, y0=entry_price,
        x1=exit_time, y1=exit_price,
        line=dict(color=trade_color, width=3, dash="solid"),
        opacity=0.8
    )

    # 2. Entry Annotation (Arrow)
    fig.add_annotation(
        x=entry_time,
        y=entry_price,
        xref="x", yref="y",
        text=f"<b>ENTRY {direction}</b><br>{entry_price:.4f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=entry_arrow_color,
        ax=0,
        ay=entry_ay,
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=entry_arrow_color,
        font=dict(color="white", size=10)
    )

    # 3. Exit Annotation (Arrow)
    exit_ay = -entry_ay # Opposite side of entry usually, or based on price?
    # Let's base it on price relative to candle? 
    # Simpler: If Win, Exit is usually 'above' entry for Long.
    # Let's just put it opposite to the candle direction or fixed.
    # Let's stick to: Long Exit (Sell) -> Text Above. Short Exit (Buy) -> Text Below.
    
    if direction == 'LONG':
        exit_ay_val = -40 # Text above
    else:
        exit_ay_val = 40 # Text below

    fig.add_annotation(
        x=exit_time,
        y=exit_price,
        xref="x", yref="y",
        text=f"<b>EXIT ({trade['outcome']})</b><br>{exit_price:.4f}<br>PnL: {pnl_pct:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=trade_color,
        ax=0,
        ay=exit_ay_val,
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=trade_color,
        font=dict(color="white", size=10)
    )

    # 4. SL and TP lines (Extended slightly)
    fig.add_shape(type="line",
        x0=subset.index[0], y0=sl_price, x1=subset.index[-1], y1=sl_price,
        line=dict(color='red', width=1, dash="dash"),
    )
    # TP Line - Visual Reference Only (Bot uses Trailing Stop)
    fig.add_shape(type="line",
        x0=subset.index[0], y0=tp_price, x1=subset.index[-1], y1=tp_price,
        line=dict(color='green', width=1, dash="dot"), # Dotted to indicate "soft" target
    )
    fig.add_annotation(
        x=subset.index[-1], y=tp_price,
        text="Target 1:3 (Ref)",
        showarrow=False,
        font=dict(color="green", size=8),
        xanchor="left"
    )
    
    # Labels for SL/TP on the right axis
    fig.add_annotation(x=subset.index[-1], y=sl_price, text="SL", showarrow=False, xanchor="left", font=dict(color='red'))
    fig.add_annotation(x=subset.index[-1], y=tp_price, text="TP", showarrow=False, xanchor="left", font=dict(color='green'))

    # Info Box (Improved)
    info_text = (
        f"<b>{pair} {direction}</b><br>"
        f"Lev: {trade['leverage']:.1f}x | Risk: {trade.get('atr', 0)*100:.2f}%<br>"
        f"Entry: {entry_time.strftime('%H:%M')}<br>"
        f"Exit:  {exit_time.strftime('%H:%M')}<br>"
        f"Duration: {str(duration).split('.')[0]}<br>"
        f"<b>PnL: {trade['pnl_pct']:.2f}%</b>"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=info_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(20, 20, 20, 0.8)",
        bordercolor=trade_color,
        borderwidth=2,
        font=dict(size=12, color="white")
    )

    fig.update_layout(
        title=f"Trade #{trade_idx+1} ({date_label}) | {pair} | {trade['outcome']}",
        xaxis_title='Time (UTC)',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Save as HTML
    filename = f"{OUTPUT_DIR}/trade_{date_label}_{trade_idx+1}_{pair.replace('/','_').replace(':','_')}.html"
    fig.write_html(filename)
    print(f"Saved chart: {filename}")

def main():
    for trades_file in TRADES_FILES:
        if not os.path.exists(trades_file):
            print(f"File not found: {trades_file}")
            continue
            
        date_label = "dec25" if "dec25" in trades_file else "dec26"
        print(f"Processing {trades_file}...")
        trades = pd.read_csv(trades_file)
        print(f"Found {len(trades)} trades.")
        
        for i, trade in trades.iterrows():
            visualize_trade(trade, i, date_label)

if __name__ == "__main__":
    main()

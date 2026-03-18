"""
Fix OI fetch — Binance limits openInterestHist to 30-day chunks.
Fetch in 30-day windows, then merge.
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

API_BASE = "https://fapi.binance.com"


def pair_to_binance(pair_config):
    return pair_config['base'] + 'USDT'


def fetch_open_interest_hist(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance in 30-day chunks.
    Period: 5m. Max 500 records per request (500 * 5min = ~1.7 days).
    """
    all_data = []
    now = datetime.now(timezone.utc)
    total_start = now - timedelta(days=days)
    
    # Process in 1.5-day chunks (500 * 5min = 2500min ≈ 1.7 days)
    chunk_size = timedelta(days=1, hours=12)
    current_start = total_start
    
    while current_start < now:
        chunk_end = min(current_start + chunk_size, now)
        
        try:
            resp = requests.get(f"{API_BASE}/futures/data/openInterestHist", params={
                'symbol': symbol,
                'period': '5m',
                'startTime': int(current_start.timestamp() * 1000),
                'endTime': int(chunk_end.timestamp() * 1000),
                'limit': 500
            }, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    all_data.extend(data)
            elif resp.status_code == 429:
                print(f"    Rate limited, waiting 10s...")
                time.sleep(10)
                continue  # Retry this chunk
            else:
                # Try without startTime constraints (some symbols have limited history)
                pass
                
        except Exception as e:
            print(f"    Error: {e}")
        
        current_start = chunk_end
        time.sleep(0.3)  # Rate limit
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'time'
    df['timestamp'] = pd.to_datetime(df[ts_col].astype(int), unit='ms', utc=True)
    df['open_interest'] = df['sumOpenInterest'].astype(float)
    df['oi_value'] = df['sumOpenInterestValue'].astype(float)
    df = df[['timestamp', 'open_interest', 'oi_value']].sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def main():
    data_dir = Path(__file__).parent.parent / 'data' / 'candles'
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_20.json'
    
    with open(pairs_file) as f:
        pairs_config = json.load(f)
    
    pairs = pairs_config['pairs']
    
    print("=" * 60)
    print("FETCHING OPEN INTEREST DATA (5m)")
    print("=" * 60)
    
    for pair in pairs:
        symbol = pair_to_binance(pair)
        base = pair['base']
        
        oi_path = data_dir / f"{base}_USDT_open_interest.csv"
        
        print(f"\n  {base} ({symbol})...", end=" ", flush=True)
        oi_df = fetch_open_interest_hist(symbol, days=100)
        
        if len(oi_df) > 0:
            oi_df.to_csv(oi_path, index=False)
            print(f"✅ {len(oi_df)} records ({oi_df['timestamp'].min().strftime('%Y-%m-%d')} to {oi_df['timestamp'].max().strftime('%Y-%m-%d')})")
        else:
            print(f"❌ No data")
    
    print("\n✅ DONE")


if __name__ == "__main__":
    main()

"""
Fetch historical funding rate and open interest data from Binance Futures API.
No API key required — these are public endpoints.

Output: CSV files per pair in data/candles/ directory:
  - {PAIR}_funding_rate.csv  (timestamp, funding_rate)
  - {PAIR}_open_interest.csv (timestamp, open_interest, oi_value)
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

API_BASE = "https://fapi.binance.com"

# Map our pair format to Binance symbol
def pair_to_binance(pair_config):
    """Convert 'BTC' to 'BTCUSDT'"""
    return pair_config['base'] + 'USDT'


def fetch_funding_rate(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Fetch historical funding rate from Binance.
    Binance funding rate is settled every 8 hours.
    Max 1000 records per request.
    """
    all_data = []
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    end_time = int(datetime.utcnow().timestamp() * 1000)
    
    current_start = start_time
    while current_start < end_time:
        try:
            resp = requests.get(f"{API_BASE}/fapi/v1/fundingRate", params={
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000
            }, timeout=10)
            
            if resp.status_code != 200:
                print(f"  ⚠️ Funding rate error for {symbol}: {resp.status_code}")
                break
            
            data = resp.json()
            if not data:
                break
                
            all_data.extend(data)
            
            # Move start past the last record
            current_start = data[-1]['fundingTime'] + 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"  ⚠️ Error fetching funding rate for {symbol}: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['funding_rate'] = df['fundingRate'].astype(float)
    df = df[['timestamp', 'funding_rate']].sort_values('timestamp').reset_index(drop=True)
    return df


def fetch_open_interest_hist(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance.
    Period: 5m (matches M5 candles).
    Max 500 records per request.
    """
    all_data = []
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    end_time = int(datetime.utcnow().timestamp() * 1000)
    
    current_start = start_time
    while current_start < end_time:
        try:
            resp = requests.get(f"{API_BASE}/futures/data/openInterestHist", params={
                'symbol': symbol,
                'period': '5m',
                'startTime': current_start,
                'endTime': end_time,
                'limit': 500
            }, timeout=10)
            
            if resp.status_code != 200:
                print(f"  ⚠️ OI error for {symbol}: {resp.status_code} - {resp.text[:200]}")
                break
            
            data = resp.json()
            if not data:
                break
                
            all_data.extend(data)
            
            # Move start past the last record
            last_ts = data[-1].get('timestamp', data[-1].get('time', 0))
            current_start = int(last_ts) + 1
            
            if len(data) < 500:
                break
                
            time.sleep(0.2)  # Rate limit (stricter for OI endpoint)
            
        except Exception as e:
            print(f"  ⚠️ Error fetching OI for {symbol}: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'time'
    df['timestamp'] = pd.to_datetime(df[ts_col].astype(int), unit='ms', utc=True)
    df['open_interest'] = df['sumOpenInterest'].astype(float)
    df['oi_value'] = df['sumOpenInterestValue'].astype(float)
    df = df[['timestamp', 'open_interest', 'oi_value']].sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    data_dir = Path(__file__).parent.parent / 'data' / 'candles'
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_20.json'
    
    with open(pairs_file) as f:
        pairs_config = json.load(f)
    
    pairs = pairs_config['pairs']
    
    print("=" * 60)
    print("FETCHING FUNDING RATE + OPEN INTEREST DATA")
    print("=" * 60)
    print(f"Pairs: {len(pairs)}")
    print(f"Data directory: {data_dir}")
    print()
    
    for pair in pairs:
        symbol = pair_to_binance(pair)
        base = pair['base']
        print(f"\n{'='*40}")
        print(f"Processing {base} ({symbol})...")
        
        # 1. Funding Rate
        print(f"  Fetching funding rate...")
        fr_df = fetch_funding_rate(symbol, days=100)
        if len(fr_df) > 0:
            fr_path = data_dir / f"{base}_USDT_funding_rate.csv"
            fr_df.to_csv(fr_path, index=False)
            print(f"  ✅ Funding rate: {len(fr_df)} records -> {fr_path.name}")
        else:
            print(f"  ❌ No funding rate data")
        
        # 2. Open Interest
        print(f"  Fetching open interest (5m)...")
        oi_df = fetch_open_interest_hist(symbol, days=100)
        if len(oi_df) > 0:
            oi_path = data_dir / f"{base}_USDT_open_interest.csv"
            oi_df.to_csv(oi_path, index=False)
            print(f"  ✅ Open interest: {len(oi_df)} records -> {oi_path.name}")
        else:
            print(f"  ❌ No open interest data")
        
        time.sleep(0.3)  # Rate limit between pairs
    
    print("\n" + "=" * 60)
    print("DONE! All data saved to data/candles/")
    print("=" * 60)


if __name__ == "__main__":
    main()

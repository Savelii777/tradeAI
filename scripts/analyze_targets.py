#!/usr/bin/env python3
"""Analyze TARGET distribution (not predictions) across periods."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

data_dir = Path(__file__).parent.parent / 'data' / 'candles'

# Load BTC data
m1 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_1m.parquet')
m5 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_5m.parquet')
m15 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_15m.parquet')

print(f"Data range: {m5.index.min().date()} to {m5.index.max().date()}")

# Copy target creation logic
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

LOOKAHEAD = 12

def create_targets(df):
    df = df.copy()
    df['atr'] = calculate_atr(df)
    
    vol_short = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    vol_medium = df['close'].pct_change().rolling(window=50, min_periods=25).std()
    vol_long = df['close'].pct_change().rolling(window=100, min_periods=50).std()
    
    combined_vol = np.maximum(vol_short, (vol_medium + vol_long) / 2)
    combined_vol = combined_vol.shift(1)
    
    threshold = np.maximum(combined_vol * 0.8, 0.003)
    
    future_return_6 = df['close'].pct_change(6).shift(-6)
    future_return_12 = df['close'].pct_change(12).shift(-12)
    future_return = 0.6 * future_return_6.fillna(0) + 0.4 * future_return_12.fillna(0)
    
    df['target_dir'] = np.where(
        future_return > threshold, 2,  # UP
        np.where(future_return < -threshold, 0, 1)  # DOWN or SIDEWAYS
    )
    
    return df

# Analyze periods
periods = [
    ('Aug 14 - Sep 14', datetime(2025, 8, 14, tzinfo=timezone.utc), datetime(2025, 9, 14, tzinfo=timezone.utc)),
    ('Sep 14 - Oct 14', datetime(2025, 9, 14, tzinfo=timezone.utc), datetime(2025, 10, 14, tzinfo=timezone.utc)),
    ('Oct 14 - Nov 14', datetime(2025, 10, 14, tzinfo=timezone.utc), datetime(2025, 11, 14, tzinfo=timezone.utc)),
    ('Nov 14 - Dec 14', datetime(2025, 11, 14, tzinfo=timezone.utc), datetime(2025, 12, 14, tzinfo=timezone.utc)),
    ('Dec 14 - Jan 11', datetime(2025, 12, 14, tzinfo=timezone.utc), datetime(2026, 1, 11, tzinfo=timezone.utc)),
]

print("\n" + "="*70)
print("TARGET DISTRIBUTION BY PERIOD (What the model should learn)")
print("="*70)

for name, start, end in periods:
    m5_p = m5[(m5.index >= start) & (m5.index < end)]
    
    if len(m5_p) < 100:
        print(f"\n{name}: Not enough data")
        continue
    
    df = create_targets(m5_p)
    df = df.dropna()
    
    total = len(df)
    up = (df['target_dir'] == 2).sum()
    down = (df['target_dir'] == 0).sum()
    side = (df['target_dir'] == 1).sum()
    
    print(f"\n{name} ({total} samples):")
    print(f"  UP (2):      {up:5d} ({up/total*100:5.1f}%)")
    print(f"  SIDEWAYS (1):{side:5d} ({side/total*100:5.1f}%)")
    print(f"  DOWN (0):    {down:5d} ({down/total*100:5.1f}%)")
    
    # Check threshold values
    vol = df['close'].pct_change().rolling(20).std().mean()
    print(f"  Avg volatility: {vol*100:.2f}%")

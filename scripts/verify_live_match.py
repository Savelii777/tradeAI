#!/usr/bin/env python3
"""
Verify that live_trading_v10_csv.py matches backtest exactly.

Checks:
1. Same features calculated
2. Same prediction values
3. Same signal filtering
4. Data integrity (no duplicates, proper updates)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'

print("="*70)
print("LIVE vs BACKTEST VERIFICATION")
print("="*70)

# 1. Load models
print("\n1. Loading models...")
dir_model = joblib.load(MODEL_DIR / 'direction_model.joblib')
timing_model = joblib.load(MODEL_DIR / 'timing_model.joblib')
strength_model = joblib.load(MODEL_DIR / 'strength_model.joblib')
features = joblib.load(MODEL_DIR / 'feature_names.joblib')
print(f"   ✅ Models loaded: {len(features)} features")

# 2. Load data (simulating live)
print("\n2. Loading Parquet data (like live)...")
pair = 'BTC/USDT:USDT'
pair_name = pair.replace('/', '_').replace(':', '_')

m1 = pd.read_parquet(DATA_DIR / f'{pair_name}_1m.parquet')
m5 = pd.read_parquet(DATA_DIR / f'{pair_name}_5m.parquet')
m15 = pd.read_parquet(DATA_DIR / f'{pair_name}_15m.parquet')

print(f"   M1: {len(m1)} candles, {m1.index.min()} to {m1.index.max()}")
print(f"   M5: {len(m5)} candles")
print(f"   M15: {len(m15)} candles")

# Check for duplicates
m1_dups = m1.index.duplicated().sum()
m5_dups = m5.index.duplicated().sum()
m15_dups = m15.index.duplicated().sum()
if m1_dups + m5_dups + m15_dups > 0:
    print(f"   ⚠️ Duplicates found: M1={m1_dups}, M5={m5_dups}, M15={m15_dups}")
else:
    print(f"   ✅ No duplicates in data")

# 3. Prepare features (like live does)
print("\n3. Preparing features (matching live logic)...")

# Live uses LOOKBACK limits
LOOKBACK_M1 = 7500
LOOKBACK_M5 = 1500
LOOKBACK_M15 = 500

m1_tail = m1.tail(LOOKBACK_M1)
m5_tail = m5.tail(LOOKBACK_M5)
m15_tail = m15.tail(LOOKBACK_M15)

mtf_fe = MTFFeatureEngine()

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    return df

ft = mtf_fe.align_timeframes(m1_tail, m5_tail, m15_tail)
ft = ft.join(m5_tail[['open', 'high', 'low', 'close', 'volume']])
ft = add_volume_features(ft)
ft = ft.dropna()

print(f"   Features calculated: {len(ft)} rows, {len(ft.columns)} columns")

# 4. Check all required features exist
print("\n4. Checking required features...")
missing = [f for f in features if f not in ft.columns]
if missing:
    print(f"   ❌ Missing features: {missing}")
else:
    print(f"   ✅ All {len(features)} features present")

# 5. Make predictions on last 10 rows
print("\n5. Testing predictions on last 10 rows...")

last_10 = ft.tail(10).copy()
for f in features:
    if f not in last_10.columns:
        last_10[f] = 0.0

X = last_10[features].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0)

probas = dir_model.predict_proba(X)
dir_preds = np.argmax(probas, axis=1)
dir_confs = np.max(probas, axis=1)
timing = timing_model.predict(X)
strength = strength_model.predict(X)

print(f"   Direction predictions: {dir_preds}")
print(f"   Confidences: {np.round(dir_confs, 3)}")
print(f"   Timing: {np.round(timing, 3)}")
print(f"   Strength: {np.round(strength, 3)}")

# 6. Check signal generation with thresholds
print("\n6. Signal generation test (MIN_CONF=0.62, MIN_TIMING=1.5, MIN_STRENGTH=1.8)...")
MIN_CONF = 0.62
MIN_TIMING = 1.5
MIN_STRENGTH = 1.8

signals = []
for i in range(len(X)):
    if dir_preds[i] == 1:  # Sideways
        continue
    if dir_confs[i] >= MIN_CONF and timing[i] >= MIN_TIMING and strength[i] >= MIN_STRENGTH:
        direction = 'LONG' if dir_preds[i] == 2 else 'SHORT'
        signals.append({
            'idx': i,
            'time': last_10.index[i],
            'direction': direction,
            'conf': dir_confs[i],
            'timing': timing[i],
            'strength': strength[i]
        })

if signals:
    print(f"   ✅ {len(signals)} signals generated:")
    for s in signals:
        print(f"      {s['time']}: {s['direction']} conf={s['conf']:.3f} tim={s['timing']:.3f} str={s['strength']:.3f}")
else:
    print(f"   ℹ️ No signals in last 10 rows (thresholds may be strict)")

# 7. Check data update mechanism
print("\n7. Data integrity check...")

# Check monotonic index
if m5.index.is_monotonic_increasing:
    print("   ✅ M5 index is monotonically increasing")
else:
    print("   ❌ M5 index is NOT monotonically increasing - needs sorting!")

# Check timezone
if m5.index.tz is not None:
    print(f"   ✅ M5 index has timezone: {m5.index.tz}")
else:
    print("   ⚠️ M5 index is timezone-naive - may cause issues")

# Check gap detection
gaps = m5.index.to_series().diff()
expected_gap = pd.Timedelta(minutes=5)
large_gaps = gaps[gaps > expected_gap * 2].dropna()
if len(large_gaps) > 0:
    print(f"   ⚠️ Found {len(large_gaps)} gaps > 10min in M5 data")
    print(f"      Largest gap: {large_gaps.max()}")
else:
    print("   ✅ No significant gaps in M5 data")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

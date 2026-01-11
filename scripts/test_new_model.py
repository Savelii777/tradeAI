#!/usr/bin/env python3
"""
FAST but CORRECT model test - computes features ONCE per pair, then predicts ALL candles
Uses MTFFeatureEngine for correct features, but vectorized for speed
"""
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
import time
sys.path.insert(0, 'scripts')
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path('models/v8_improved')
DATA = 'data/candles'

# Load model
print("Loading model...")
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
    'scaler': joblib.load(MODEL_DIR / 'scaler.joblib') if (MODEL_DIR / 'scaler.joblib').exists() else None
}

mtf_fe = MTFFeatureEngine()

print("="*60)
print("FAST CORRECT TEST (MTFFeatureEngine, vectorized)")
print(f"Scaler: {'YES' if models['scaler'] else 'NO'}")
print(f"Features: {len(models['features'])}")
print("="*60)

def test_pair_fast(pair, start, end):
    """Test pair by computing features ONCE for whole period, then predict ALL at once"""
    t0 = time.time()
    
    # Convert to UTC-aware timestamps
    start_ts = pd.Timestamp(start, tz='UTC')
    end_ts = pd.Timestamp(end, tz='UTC')
    buffer_start = start_ts - pd.Timedelta(days=30)
    
    # Load data with buffer for indicator warmup
    m1 = pd.read_parquet(f'{DATA}/{pair}_USDT_USDT_1m.parquet')
    m5 = pd.read_parquet(f'{DATA}/{pair}_USDT_USDT_5m.parquet')
    m15 = pd.read_parquet(f'{DATA}/{pair}_USDT_USDT_15m.parquet')
    
    m1 = m1[(m1.index >= buffer_start) & (m1.index <= end_ts)]
    m5 = m5[(m5.index >= buffer_start) & (m5.index <= end_ts)]
    m15 = m15[(m15.index >= buffer_start) & (m15.index <= end_ts)]
    
    if len(m5) < 100:
        print(f"\n{pair}: Not enough data")
        return None
    
    # Compute features ONCE for whole period
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open','high','low','close','volume']])
    ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
    ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
    ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
    ft = ft.ffill().dropna()
    
    # Filter to test period only (after warmup)
    ft = ft[(ft.index >= start_ts) & (ft.index <= end_ts)]
    
    if len(ft) < 10:
        print(f"\n{pair}: Not enough features")
        return None
    
    # Fill missing features
    for f in models['features']:
        if f not in ft.columns:
            ft[f] = 0.0
    
    # Predict ALL at once (vectorized - FAST!)
    X = ft[models['features']].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    
    if models['scaler']:
        X = models['scaler'].transform(X)
    
    proba = models['direction'].predict_proba(X)
    preds = np.argmax(proba, axis=1)
    confs = np.max(proba, axis=1)
    
    # Stats
    dir_counts = {0: int((preds == 0).sum()), 1: int((preds == 1).sum()), 2: int((preds == 2).sum())}
    total = len(preds)
    pct_side = 100 * dir_counts[1] / total if total > 0 else 0
    
    signals_50 = int(((preds != 1) & (confs >= 0.50)).sum())
    signals_62 = int(((preds != 1) & (confs >= 0.62)).sum())
    
    elapsed = time.time() - t0
    
    print(f"\n{pair} [{elapsed:.1f}s]:")
    print(f"  Candles: {total} | SHORT={dir_counts[0]} SIDEWAYS={dir_counts[1]} ({pct_side:.0f}%) LONG={dir_counts[2]}")
    print(f"  Conf: min={confs.min():.3f} max={confs.max():.3f} mean={confs.mean():.3f}")
    print(f"  Signals conf>=0.50: {signals_50} | conf>=0.62: {signals_62}")
    
    return signals_50, signals_62, total

# Test ALL pairs
pairs = ['BTC', 'ETH', 'SOL', 'DOGE', 'PIPPIN', 'TAO', 'HYPE', '1000PEPE', 'SUI', 'APT',
         'AAVE', 'ADA', 'AVAX', 'LINK', 'LTC', 'UNI', 'XRP', 'NEAR', 'FIL', 'ZEC']

start_date = '2025-12-01'
end_date = '2025-12-14'
days = 14

print(f"\nTesting {len(pairs)} pairs from {start_date} to {end_date}")
print("-"*60)

total_50, total_62, total_candles = 0, 0, 0

for pair in pairs:
    try:
        result = test_pair_fast(pair, start_date, end_date)
        if result:
            total_50 += result[0]
            total_62 += result[1]
            total_candles += result[2]
    except Exception as e:
        print(f"\n{pair}: Error - {e}")

print("\n" + "="*60)
print(f"TOTAL across {len(pairs)} pairs ({days} days):")
print(f"  Total candles analyzed: {total_candles}")
print(f"  Signals conf>=0.50: {total_50} ({total_50/days:.1f}/day)")
print(f"  Signals conf>=0.62: {total_62} ({total_62/days:.1f}/day)")
print("="*60)

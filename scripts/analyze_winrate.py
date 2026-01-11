#!/usr/bin/env python3
"""Analyze why walk-forward win rate is lower than backtest."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.utils.constants import CORE_20_FEATURES

data_dir = Path(__file__).parent.parent / 'data' / 'candles'
model_dir = Path(__file__).parent.parent / 'models' / 'v8_improved'

# Load current production models
dir_model = joblib.load(model_dir / 'direction_model.joblib')
timing_model = joblib.load(model_dir / 'timing_model.joblib')
strength_model = joblib.load(model_dir / 'strength_model.joblib')
features = joblib.load(model_dir / 'feature_names.joblib')

print(f"Model uses {len(features)} features")
print(f"Features: {features[:10]}...")

# Load BTC data
m1 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_1m.parquet')
m5 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_5m.parquet')
m15 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_15m.parquet')

print(f"\nData range: {m5.index.min().date()} to {m5.index.max().date()}")

mtf_fe = MTFFeatureEngine()

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    return df

# Analyze different time periods
periods = [
    ('Train (Dec 7-21)', datetime(2025, 12, 7, tzinfo=timezone.utc), datetime(2025, 12, 21, tzinfo=timezone.utc)),
    ('Test (Dec 22-28)', datetime(2025, 12, 22, tzinfo=timezone.utc), datetime(2025, 12, 28, tzinfo=timezone.utc)),
    ('Recent (Jan 5-11)', datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 11, tzinfo=timezone.utc)),
]

print("\n" + "="*70)
print("PREDICTION DISTRIBUTION BY PERIOD")
print("="*70)

for name, start, end in periods:
    # Filter data
    m1_p = m1[(m1.index >= start) & (m1.index < end)]
    m5_p = m5[(m5.index >= start) & (m5.index < end)]
    m15_p = m15[(m15.index >= start) & (m15.index < end)]
    
    if len(m5_p) < 100:
        print(f"\n{name}: Not enough data ({len(m5_p)} candles)")
        continue
    
    # Calculate features
    ft = mtf_fe.align_timeframes(m1_p, m5_p, m15_p)
    ft = ft.join(m5_p[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft = ft.dropna()
    
    # Ensure all features exist
    for f in features:
        if f not in ft.columns:
            ft[f] = 0.0
    
    X = ft[features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    
    # Predictions
    probas = dir_model.predict_proba(X)
    dir_preds = np.argmax(probas, axis=1)
    dir_confs = np.max(probas, axis=1)
    timing = timing_model.predict(X)
    strength = strength_model.predict(X)
    
    print(f"\n{name} ({len(ft)} samples):")
    print(f"  Direction dist: Long={np.sum(dir_preds==0)/len(dir_preds)*100:.1f}%, "
          f"Sideways={np.sum(dir_preds==1)/len(dir_preds)*100:.1f}%, "
          f"Short={np.sum(dir_preds==2)/len(dir_preds)*100:.1f}%")
    print(f"  Confidence: mean={dir_confs.mean():.3f}, min={dir_confs.min():.3f}, max={dir_confs.max():.3f}")
    print(f"  Timing:     mean={timing.mean():.3f}, min={timing.min():.3f}, max={timing.max():.3f}")
    print(f"  Strength:   mean={strength.mean():.3f}, min={strength.min():.3f}, max={strength.max():.3f}")
    
    # Count signals at different thresholds
    for conf_th, tim_th, str_th in [(0.62, 1.5, 1.8), (0.55, 1.3, 1.5), (0.50, 1.0, 1.0)]:
        signals = 0
        for i in range(len(X)):
            if dir_preds[i] == 1:  # Skip sideways
                continue
            if dir_confs[i] >= conf_th and timing[i] >= tim_th and strength[i] >= str_th:
                signals += 1
        print(f"  Signals (conf>{conf_th}, tim>{tim_th}, str>{str_th}): {signals} ({signals/len(ft)*100:.1f}%)")

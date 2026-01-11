#!/usr/bin/env python3
"""Debug Period_1 no signals issue."""

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

# Period 1 dates
now = datetime.now(timezone.utc)
test_end = now
test_start = test_end - timedelta(days=7)
train_end = test_start - timedelta(days=1)
train_start = train_end - timedelta(days=15)

print(f'Period 1:')
print(f'  TRAIN: {train_start.strftime("%Y-%m-%d")} → {train_end.strftime("%Y-%m-%d")}')
print(f'  TEST:  {test_start.strftime("%Y-%m-%d")} → {test_end.strftime("%Y-%m-%d")}')

# Check BTC data
m5 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_5m.parquet')
if m5.index.tz is None:
    m5.index = m5.index.tz_localize('UTC')

print(f'\nBTC M5 data range: {m5.index.min()} → {m5.index.max()}')

# Count candles in train/test periods
m5_train = m5[(m5.index >= train_start) & (m5.index < train_end)]
m5_test = m5[(m5.index >= test_start) & (m5.index < test_end)]

print(f'Train candles: {len(m5_train)}')
print(f'Test candles: {len(m5_test)}')

# Load current model and test on latest data
print('\n=== Testing current model on latest data ===')
dir_model = joblib.load(model_dir / 'direction_model.joblib')
timing_model = joblib.load(model_dir / 'timing_model.joblib')
strength_model = joblib.load(model_dir / 'strength_model.joblib')
features = joblib.load(model_dir / 'feature_names.joblib')

print(f'Model uses {len(features)} features')

# Quick test on last 100 rows
m1 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_1m.parquet')
m15 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_15m.parquet')

mtf_fe = MTFFeatureEngine()

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    return df

# Use last 1500 like live
m1_tail = m1.tail(7500)
m5_tail = m5.tail(1500)
m15_tail = m15.tail(500)

ft = mtf_fe.align_timeframes(m1_tail, m5_tail, m15_tail)
ft = ft.join(m5_tail[['open', 'high', 'low', 'close', 'volume']])
ft = add_volume_features(ft)
ft = ft.dropna()

print(f'Features calculated: {len(ft)} rows')

# Check last 100 for signals
last_100 = ft.tail(100).copy()
for f in features:
    if f not in last_100.columns:
        last_100[f] = 0.0

X = last_100[features].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0)

probas = dir_model.predict_proba(X)
dir_preds = np.argmax(probas, axis=1)
dir_confs = np.max(probas, axis=1)
timing = timing_model.predict(X)
strength = strength_model.predict(X)

print(f'\nLast 100 rows prediction stats:')
print(f'  Conf range: {dir_confs.min():.3f} - {dir_confs.max():.3f}')
print(f'  Timing range: {timing.min():.3f} - {timing.max():.3f}')
print(f'  Strength range: {strength.min():.3f} - {strength.max():.3f}')

# Thresholds
MIN_CONF = 0.62
MIN_TIMING = 1.5
MIN_STRENGTH = 1.8

signals = 0
for i in range(len(X)):
    if dir_preds[i] == 1:  # Sideways
        continue
    if dir_confs[i] >= MIN_CONF and timing[i] >= MIN_TIMING and strength[i] >= MIN_STRENGTH:
        signals += 1

print(f'\nSignals passing thresholds (conf>={MIN_CONF}, tim>={MIN_TIMING}, str>={MIN_STRENGTH}): {signals}')

# Try lower thresholds
for conf_th in [0.60, 0.55, 0.50]:
    count = 0
    for i in range(len(X)):
        if dir_preds[i] == 1:
            continue
        if dir_confs[i] >= conf_th and timing[i] >= 1.3 and strength[i] >= 1.5:
            count += 1
    print(f'  With conf>={conf_th}, tim>=1.3, str>=1.5: {count} signals')

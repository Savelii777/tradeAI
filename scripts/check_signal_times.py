#!/usr/bin/env python3
"""Check when signals happen in the backtest period."""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from backtest_mexc_live import prepare_features, load_pair_data, MIN_CONF, MIN_TIMING, MIN_STRENGTH

DATA_DIR = Path(__file__).parent.parent / 'data/candles'
MODEL_DIR = Path(__file__).parent.parent / 'models/v8_improved'

# Load models
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
    'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
}

# Check BTC signals distribution
print("Loading BTC data...")
m1 = load_pair_data('BTC/USDT:USDT', DATA_DIR, '1m')
m5 = load_pair_data('BTC/USDT:USDT', DATA_DIR, '5m')
m15 = load_pair_data('BTC/USDT:USDT', DATA_DIR, '15m')

print(f"M5 range: {m5.index.min()} to {m5.index.max()}")

mtf = MTFFeatureEngine()
features = prepare_features(m1, m5, m15, mtf)

for f in models['features']:
    if f not in features.columns:
        features[f] = 0

X = features[models['features']].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

dir_proba = models['direction'].predict_proba(X)
dir_pred = models['direction'].predict(X)
timing = models['timing'].predict(X)
strength = models['strength'].predict(X)

# Count signals
signals = []
for i, (ts, row) in enumerate(features.iterrows()):
    d = int(dir_pred[i])
    if d == 1: continue
    c = float(np.max(dir_proba[i]))
    t = float(timing[i])
    s = float(strength[i])
    if c >= MIN_CONF and t >= MIN_TIMING and s >= MIN_STRENGTH:
        signals.append(ts)

print(f'\nBTC total signals: {len(signals)}')
if signals:
    print(f'First signal: {signals[0]}')
    print(f'Last signal: {signals[-1]}')
    
    # Group by date
    sig_df = pd.DataFrame({'ts': signals})
    sig_df['date'] = sig_df['ts'].dt.date
    print(f'\nSignals per day (last 30 days):')
    
    end_date = datetime.now(timezone.utc)
    start_30 = end_date - timedelta(days=30)
    sig_df_30 = sig_df[sig_df['ts'] >= start_30]
    
    daily = sig_df_30.groupby('date').size()
    print(daily.tail(30))
    print(f"\nTotal signals in 30 days: {len(sig_df_30)}")

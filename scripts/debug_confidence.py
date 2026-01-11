#!/usr/bin/env python3
"""Debug why confidence is always low."""
import pandas as pd
import joblib
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from train_mtf import MTFFeatureEngine

# Load everything
model = joblib.load('models/v8_improved/direction_model.joblib')
features = joblib.load('models/v8_improved/feature_names.joblib')

m1 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_1m.parquet')
m5 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_5m.parquet')
m15 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_15m.parquet')

mtf_fe = MTFFeatureEngine()

# Prepare features like in live
m1_w = m1.tail(7500)
m5_w = m5.tail(1500)
m15_w = m15.tail(500)

ft = mtf_fe.align_timeframes(m1_w, m5_w, m15_w)
ft = ft.join(m5_w[['open','high','low','close','volume']])
ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
ft = ft.ffill().dropna()

print(f'Features shape: {ft.shape}')
print(f'Required features: {len(features)}')

# Check which features exist
missing = [f for f in features if f not in ft.columns]
print(f'Missing features: {len(missing)}')
if missing:
    print(f'  {missing}')

# Prepare X
row = ft.iloc[[-2]].copy()
for f in features:
    if f not in row.columns:
        row[f] = 0.0

X = row[features].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0)

print()
print('Feature values:')
for i, f in enumerate(features):
    print(f'  {f}: {X[0][i]:.4f}')

print()
print('Feature stats:')
print(f'  Min: {X.min():.4f}, Max: {X.max():.4f}, Mean: {X.mean():.4f}')
print(f'  Zeros: {(X == 0).sum()} / {len(features)}')

# Predict
proba = model.predict_proba(X)
print()
print(f'Prediction: {np.argmax(proba)} with conf {np.max(proba):.3f}')
print(f'Probabilities: SHORT={proba[0][0]:.3f} SIDEWAYS={proba[0][1]:.3f} LONG={proba[0][2]:.3f}')

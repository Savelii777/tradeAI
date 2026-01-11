#!/usr/bin/env python3
"""Check feature distribution difference between train and live"""
import pandas as pd
import joblib
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from train_mtf import MTFFeatureEngine

features = joblib.load('models/v8_improved/feature_names.joblib')
model = joblib.load('models/v8_improved/direction_model.joblib')
mtf_fe = MTFFeatureEngine()

print("CORE_20_FEATURES used by model:")
print(features)
print()

# Load PIPPIN
m1 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_1m.parquet')
m5 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_5m.parquet')
m15 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_15m.parquet')

# Dec period
m5_dec = m5[(m5.index >= '2025-12-01') & (m5.index <= '2025-12-07')]
end_idx = m5_dec.index[len(m5_dec)//2]

m5_w = m5[m5.index <= end_idx].tail(1500)
m1_w = m1[m1.index <= end_idx].tail(7500)
m15_w = m15[m15.index <= end_idx].tail(500)

ft = mtf_fe.align_timeframes(m1_w, m5_w, m15_w)
ft = ft.join(m5_w[['open','high','low','close','volume']])
ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
ft = ft.ffill().dropna()

row = ft.iloc[-2]
print('Feature values for PIPPIN (Dec):')
print('='*50)
for f in features:
    val = row.get(f, 0.0)
    print(f'{f:25}: {val:10.4f}')

print()

# Create feature vector
X = np.array([[row.get(f, 0.0) for f in features]])
X = np.nan_to_num(X, nan=0.0)

print('Feature vector stats:')
print(f'  Shape: {X.shape}')
print(f'  Min: {X.min():.4f}')
print(f'  Max: {X.max():.4f}')
print(f'  Mean: {X.mean():.4f}')

# Predict
proba = model.predict_proba(X)
print()
print(f'Prediction: {proba}')
print(f'Max conf: {proba.max():.4f}')

# Now check what happens with scaled features
print()
print('='*50)
print('Testing with STANDARDIZED features (mean=0, std=1):')
X_std = (X - X.mean()) / (X.std() + 1e-8)
proba_std = model.predict_proba(X_std)
print(f'Prediction: {proba_std}')
print(f'Max conf: {proba_std.max():.4f}')

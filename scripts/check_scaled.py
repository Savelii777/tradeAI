#!/usr/bin/env python3
"""Check scaled feature values"""
import pandas as pd
import joblib
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from train_mtf import MTFFeatureEngine

scaler = joblib.load('models/v8_improved/scaler.joblib')
model = joblib.load('models/v8_improved/direction_model.joblib')
features = joblib.load('models/v8_improved/feature_names.joblib')
mtf_fe = MTFFeatureEngine()

# Load PIPPIN Dec data
m1 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_1m.parquet')
m5 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_5m.parquet')
m15 = pd.read_parquet('data/candles/PIPPIN_USDT_USDT_15m.parquet')

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
X_raw = np.array([[row.get(f, 0.0) for f in features]])

print('RAW feature values:')
for i, f in enumerate(features):
    print(f'  {f:25}: {X_raw[0, i]:10.4f}')

X_scaled = scaler.transform(X_raw)
print()
print('SCALED feature values:')
for i, f in enumerate(features):
    print(f'  {f:25}: {X_scaled[0, i]:10.4f}')

print()
print(f'Scaled stats: min={X_scaled.min():.2f} max={X_scaled.max():.2f} mean={X_scaled.mean():.2f}')

proba = model.predict_proba(X_scaled)
print()
print('Prediction:', proba)
print('Max conf:', proba.max())

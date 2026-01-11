#!/usr/bin/env python3
"""Quick check of MTF predictions"""
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path('models/v8_improved')
DATA_DIR = Path('data/candles')

models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
    'scaler': joblib.load(MODEL_DIR / 'scaler.joblib')
}

mtf_fe = MTFFeatureEngine()

pair = 'PIPPIN'
m1 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_1m.parquet')
m5 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_5m.parquet')
m15 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_15m.parquet')

m5_dec = m5[(m5.index >= '2025-12-01') & (m5.index <= '2025-12-07')]
print(f'Testing {len(m5_dec)} 5m candles (Dec 1-7)')

confs = []
signals_50 = 0
signals_62 = 0

# Check every 6th candle (30 min)
for i in range(0, min(100, len(m5_dec)), 6):
    end_idx = m5_dec.index[i]
    m5_w = m5[m5.index <= end_idx].tail(1500)
    m1_w = m1[m1.index <= end_idx].tail(7500)
    m15_w = m15[m15.index <= end_idx].tail(500)
    
    ft = mtf_fe.align_timeframes(m1_w, m5_w, m15_w)
    ft = ft.join(m5_w[['open','high','low','close','volume']])
    ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
    ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
    ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
    ft = ft.ffill().dropna()
    
    if len(ft) < 2: continue
    
    row = ft.iloc[[-2]].copy()
    for f in models['features']:
        if f not in row.columns: row[f] = 0.0
    
    X = row[models['features']].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    X = models['scaler'].transform(X)
    
    proba = models['direction'].predict_proba(X)
    conf = float(np.max(proba))
    pred = int(np.argmax(proba))
    
    confs.append(conf)
    if pred != 1 and conf >= 0.50: signals_50 += 1
    if pred != 1 and conf >= 0.62: signals_62 += 1

print(f'Checked {len(confs)} points')
print(f'Conf: min={min(confs):.3f} max={max(confs):.3f} mean={np.mean(confs):.3f}')
print(f'Signals conf>=0.50: {signals_50}')
print(f'Signals conf>=0.62: {signals_62}')

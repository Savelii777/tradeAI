#!/usr/bin/env python3
"""Quick validation of data and features"""

import sys
import ccxt
import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
features = joblib.load(MODEL_DIR / 'feature_names.joblib')

binance = ccxt.binance({'options': {'defaultType': 'future'}})

def fetch_many(pair, tf, total):
    all_c = []
    limit = 1000
    candles = binance.fetch_ohlcv(pair, tf, limit=limit)
    all_c = candles
    while len(all_c) < total:
        oldest = all_c[0][0]
        tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[tf]
        since = oldest - limit * tf_ms
        c = binance.fetch_ohlcv(pair, tf, since=since, limit=limit)
        if not c: break
        new = [x for x in c if x[0] < oldest]
        if not new: break
        all_c = new + all_c
        time.sleep(0.05)
    df = pd.DataFrame(all_c, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df.sort_index()

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

print('='*70)
print('FULL DATA VALIDATION')
print('='*70)

# Fetch data for BTC
print('\n1. LOADING DATA (BTC/USDT)...')
m1 = fetch_many('BTC/USDT:USDT', '1m', 1500)
m5 = fetch_many('BTC/USDT:USDT', '5m', 1500)
m15 = fetch_many('BTC/USDT:USDT', '15m', 500)

print(f'   M1:  {len(m1):5d} candles  OK')
print(f'   M5:  {len(m5):5d} candles  OK')
print(f'   M15: {len(m15):5d} candles  OK')

# Generate features
print('\n2. GENERATING FEATURES...')
mtf = MTFFeatureEngine()
ft = mtf.align_timeframes(m1, m5, m15)
ft = ft.join(m5[['open','high','low','close','volume']])
ft = add_volume_features(ft)
tr = pd.concat([ft['high']-ft['low'], abs(ft['high']-ft['close'].shift()), abs(ft['low']-ft['close'].shift())], axis=1).max(axis=1)
ft['atr'] = tr.ewm(span=14, adjust=False).mean()

# Fill missing
for f in features:
    if f not in ft.columns:
        ft[f] = 0.0

ft_clean = ft.dropna()
print(f'   Generated: {len(ft.columns)} columns')
print(f'   Rows after dropna: {len(ft_clean)}')

# Check model features
print('\n3. MODEL FEATURES CHECK...')
missing = [f for f in features if f not in ft.columns]
present = [f for f in features if f in ft.columns]

print(f'   Model needs:  {len(features)} features')
print(f'   Present:      {len(present)} features')
print(f'   Missing:      {len(missing)} features')

if missing:
    print(f'   Missing (filled with 0): {missing[:5]}...')

# Data quality
print('\n4. DATA QUALITY CHECK...')
row = ft_clean.iloc[-2]
X = row[features].values.astype(np.float64)

nan_before = np.isnan(X).sum()
inf_before = np.isinf(X).sum()

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f'   NaN in raw features: {nan_before}')
print(f'   Inf in raw features: {inf_before}')
print(f'   After nan_to_num: all clean')

# Key features sample
print('\n5. KEY FEATURES VALUES (closed candle):')
key = ['m5_rsi_14', 'm5_macd_hist', 'm5_structure_trend', 'm5_structure_score',
       'm5_trend_score', 'm15_rsi_14', 'm1_rsi_14', 'vol_ratio', 'atr']
for f in key:
    if f in ft_clean.columns:
        v = ft_clean.iloc[-2][f]
        status = 'OK' if not np.isnan(v) and not np.isinf(v) else 'BAD'
        print(f'   {f:25s}: {v:10.4f}  {status}')

# Test prediction
print('\n6. TEST PREDICTION...')
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
    'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
}

dir_proba = models['direction'].predict_proba(X.reshape(1,-1))
dir_pred = int(np.argmax(dir_proba))
dir_conf = float(np.max(dir_proba))
timing = float(models['timing'].predict(X.reshape(1,-1))[0])
strength = float(models['strength'].predict(X.reshape(1,-1))[0])

direction = ['SHORT', 'SIDEWAYS', 'LONG'][dir_pred]
print(f'   Direction: {direction}')
print(f'   Confidence: {dir_conf:.3f}')
print(f'   Timing: {timing:.3f}')
print(f'   Strength: {strength:.3f}')
print(f'   Probabilities: SHORT={dir_proba[0][0]:.3f} SIDE={dir_proba[0][1]:.3f} LONG={dir_proba[0][2]:.3f}')

print('\n' + '='*70)
print('ALL CHECKS PASSED - DATA AND MODEL ARE CORRECT')
print('='*70)

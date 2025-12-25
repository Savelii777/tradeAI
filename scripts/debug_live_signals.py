#!/usr/bin/env python3
"""Debug live signal generation."""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
import joblib
import ccxt
from train_mtf import MTFFeatureEngine
import warnings
warnings.filterwarnings('ignore')

# Load models
print('Loading models...')
direction_model = joblib.load('models/v1_fresh/direction_model.joblib')
timing_model = joblib.load('models/v1_fresh/timing_model.joblib')
feature_names = list(direction_model.feature_name_)
print(f'Model expects {len(feature_names)} features')

# Initialize exchange
exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

# Fetch live data
pair = 'BTC/USDT'
mtf_engine = MTFFeatureEngine()

# Fetch OHLCV
print('\nFetching OHLCV data...')
df_1m = exchange.fetch_ohlcv(pair, '1m', limit=100)
df_5m = exchange.fetch_ohlcv(pair, '5m', limit=100)
df_15m = exchange.fetch_ohlcv(pair, '15m', limit=100)

df_1m = pd.DataFrame(df_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_5m = pd.DataFrame(df_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_15m = pd.DataFrame(df_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], unit='ms')

df_1m.set_index('timestamp', inplace=True)
df_5m.set_index('timestamp', inplace=True)
df_15m.set_index('timestamp', inplace=True)

print('Data shapes:')
print(f'  1m: {df_1m.shape}')
print(f'  5m: {df_5m.shape}')
print(f'  15m: {df_15m.shape}')

# align_timeframes (backtest method)
print('\nRunning align_timeframes...')
features_bt = mtf_engine.align_timeframes(df_1m, df_5m, df_15m)

print(f'\nFeatures after align_timeframes:')
print(f'  Shape: {features_bt.shape}')
print(f'  Columns: {len(features_bt.columns)}')
print(f'  Rows: {len(features_bt)}')

if len(features_bt) == 0:
    print('\n!!! PROBLEM: align_timeframes returned EMPTY DataFrame !!!')
    print('This means dropna() removed all rows due to NaN')
else:
    print(f'\nLast row index: {features_bt.index[-1]}')
    
    # Add missing features
    missing_count = 0
    for feat in feature_names:
        if feat not in features_bt.columns:
            features_bt[feat] = 0
            missing_count += 1
    
    print(f'Missing features filled with 0: {missing_count}')
    
    # Get predictions
    X = features_bt[feature_names].tail(1)
    
    dir_proba = direction_model.predict_proba(X)[0]
    timing_proba = timing_model.predict_proba(X)[0]
    
    direction_idx = np.argmax(dir_proba)
    direction_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
    direction = direction_map[direction_idx]
    
    print(f'\nPredictions:')
    print(f'  Direction: {direction}')
    print(f'  Direction probs: DOWN={dir_proba[0]:.3f}, HOLD={dir_proba[1]:.3f}, UP={dir_proba[2]:.3f}')
    print(f'  Timing proba: {timing_proba[1]:.4f}')
    
    # Check conditions
    is_good_timing = timing_proba[1] > 0.01
    max_prob = max(dir_proba)
    
    signal = 0
    if is_good_timing and max_prob >= 0.50:
        if direction_idx == 2:
            signal = 1
        elif direction_idx == 0:
            signal = -1
    
    print(f'\nConditions:')
    print(f'  timing > 0.01? {is_good_timing}')
    print(f'  max_prob >= 0.50? {max_prob >= 0.50} ({max_prob:.3f})')
    print(f'  direction != HOLD? {direction_idx != 1}')
    print(f'  Signal: {signal}')
    print(f'  Would open trade: {signal != 0}')

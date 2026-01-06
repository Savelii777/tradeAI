#!/usr/bin/env python3
"""Check full filter logic on PIPPIN"""

import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def main():
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }

    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()

    cumsum_patterns = ['bars_since_swing', 'consecutive_up', 'consecutive_down', 'obv', 'volume_delta_cumsum', 'swing_high_price', 'swing_low_price']
    features = [f for f in models['features'] if not any(p in f.lower() for p in cumsum_patterns)]

    pair = 'PIPPIN/USDT:USDT'
    print(f'Checking {pair} with FULL filter logic...')

    data = {}
    for tf in ['1m', '5m', '15m']:
        candles = binance.fetch_ohlcv(pair, tf, limit=1000)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        data[tf] = df

    ft = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
    ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna()

    for f in features:
        if f not in ft.columns:
            ft[f] = 0.0

    X = ft[features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    timing = models['timing'].predict(X)
    strength = models['strength'].predict(X)

    MIN_CONF = 0.50
    MIN_TIMING = 0.8
    MIN_STRENGTH = 1.4

    print(f'\nTotal candles: {len(ft)}')
    print(f'\nSignals passing ALL filters (Conf>{MIN_CONF}, Timing>{MIN_TIMING}, Strength>{MIN_STRENGTH}):')
    count = 0
    for i in range(len(ft)):
        if dir_preds[i] == 1:
            continue
        if dir_confs[i] < MIN_CONF:
            continue
        if timing[i] < MIN_TIMING:
            continue
        if strength[i] < MIN_STRENGTH:
            continue
        
        count += 1
        ts = ft.index[i]
        dir_str = 'LONG' if dir_preds[i] == 2 else 'SHORT'
        print(f'   {ts}: {dir_str} | Conf={dir_confs[i]:.3f} | Timing={timing[i]:.2f} | Strength={strength[i]:.2f}')

    print(f'\nTotal valid signals: {count}')

    # Breakdown
    print(f'\nBreakdown of rejections (non-sideways only):')
    non_sideways = dir_preds != 1
    rejected_conf = 0
    rejected_timing = 0
    rejected_strength = 0
    passed = 0
    
    for i in range(len(ft)):
        if dir_preds[i] == 1:
            continue
        if dir_confs[i] < MIN_CONF:
            rejected_conf += 1
        elif timing[i] < MIN_TIMING:
            rejected_timing += 1
        elif strength[i] < MIN_STRENGTH:
            rejected_strength += 1
        else:
            passed += 1

    print(f'   Total non-sideways: {non_sideways.sum()}')
    print(f'   Rejected by Conf < {MIN_CONF}: {rejected_conf}')
    print(f'   Rejected by Timing < {MIN_TIMING}: {rejected_timing}')
    print(f'   Rejected by Strength < {MIN_STRENGTH}: {rejected_strength}')
    print(f'   PASSED all filters: {passed}')

if __name__ == '__main__':
    main()

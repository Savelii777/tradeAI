#!/usr/bin/env python3
"""
LIVE V2 - С исключением проблемных фичей
Фичи которые зависят от длины истории исключены
"""

import sys
import time
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")

# Thresholds
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# ПРОБЛЕМНЫЕ ФИЧИ - исключаем их из предсказания
# Эти фичи зависят от длины истории (ffill, cumsum, structure detection)
PROBLEM_FEATURES = [
    'm5_structure_trend', 'm5_structure_score',
    'm5_trend_score', 'm5_trend_direction',
    'm5_atr_14_ma',  # Зависит от rolling на большом окне
    # Swing-related (уже исключены ранее)
    'swing_high_price', 'swing_low_price',
    'bars_since_swing_high', 'bars_since_swing_low',
    # Cumsum-related
    'obv', 'volume_delta_cumsum',
    'consecutive_up', 'consecutive_down',
]


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def main():
    print("="*70)
    print("LIVE V2 - Исключены проблемные фичи")
    print("="*70)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    feature_names = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    # Исключаем проблемные фичи
    features = [f for f in feature_names if not any(p in f for p in PROBLEM_FEATURES)]
    excluded = len(feature_names) - len(features)
    print(f"Features: {len(features)} (excluded {excluded} problem features)")
    
    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:20]]
    print(f"Pairs: {len(pairs)}")
    
    # Init
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    print(f"\nScanning at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*70)
    
    all_signals = []
    
    for pair in pairs:
        try:
            # Fetch data
            data = {}
            for tf in ['1m', '5m', '15m']:
                candles = binance.fetch_ohlcv(pair, tf, limit=500)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                data[tf] = df
            
            # Build features
            ft = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
            ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
            
            if len(ft) < 10:
                continue
            
            # Fill missing features with 0
            for f in features:
                if f not in ft.columns:
                    ft[f] = 0.0
            
            # Get closed candle
            row = ft.iloc[[-2]]
            X = row[features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict
            dir_proba = models['direction'].predict_proba(X)
            dir_pred = int(np.argmax(dir_proba))
            dir_conf = float(np.max(dir_proba))
            timing = float(models['timing'].predict(X)[0])
            strength = float(models['strength'].predict(X)[0])
            
            if dir_pred == 1:  # Sideways
                continue
            
            direction = 'LONG' if dir_pred == 2 else 'SHORT'
            ts = row.index[0]
            
            # Check filters
            passes = dir_conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH
            
            if passes:
                print(f"✅ {pair:<20} {ts} | {direction} | Conf={dir_conf:.3f} | T={timing:.2f} | S={strength:.2f}")
                all_signals.append({
                    'pair': pair,
                    'direction': direction,
                    'timestamp': ts,
                    'conf': dir_conf,
                    'timing': timing,
                    'strength': strength,
                })
            elif dir_conf > 0.40:  # Show near-misses
                reject = []
                if dir_conf < MIN_CONF: reject.append(f"conf<{MIN_CONF}")
                if timing < MIN_TIMING: reject.append(f"T<{MIN_TIMING}")
                if strength < MIN_STRENGTH: reject.append(f"S<{MIN_STRENGTH}")
                print(f"❌ {pair:<20} {ts} | {direction} | Conf={dir_conf:.3f} | {', '.join(reject)}")
                        
        except Exception as e:
            print(f"⚠️  {pair}: {e}")
    
    print("\n" + "="*70)
    print(f"FOUND {len(all_signals)} VALID SIGNALS")
    print("="*70)


if __name__ == '__main__':
    main()

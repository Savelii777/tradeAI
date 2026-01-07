#!/usr/bin/env python3
"""
SIMPLE LIVE TRADING v3 - РАБОТАЕТ КАК БЭКТЕСТ
Загружает достаточно данных (2000 свечей) для корректных фичей
"""

import sys
import time
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_list.json"

# Thresholds - ТОЧНО КАК В БЭКТЕСТЕ
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# Сколько свечей грузить (2000 достаточно для всех индикаторов)
CANDLES_TO_LOAD = 2000


# ============================================================
# FEATURES - КОПИЯ ИЗ train_v3_dynamic.py
# ============================================================
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


# ============================================================
# DATA LOADING - Загружает 2000+ свечей
# ============================================================
def fetch_candles(exchange, pair: str, timeframe: str, total_needed: int = 2000):
    """Загружает много свечей несколькими запросами"""
    all_candles = []
    limit = 1000  # Binance лимит
    
    # Первый запрос
    candles = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
    all_candles = candles
    
    # Дополнительные запросы если нужно больше
    while len(all_candles) < total_needed:
        oldest = all_candles[0][0]  # timestamp самой старой свечи
        tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[timeframe]
        since = oldest - limit * tf_ms
        
        candles = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        if not candles:
            break
        
        # Добавляем только новые (более старые)
        new = [c for c in candles if c[0] < oldest]
        if not new:
            break
        
        all_candles = new + all_candles
        time.sleep(0.1)  # Rate limit
    
    # Конвертируем в DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("SIMPLE LIVE v3 - Загружает 2000 свечей как бэктест")
    print("="*70)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features = joblib.load(MODEL_DIR / 'feature_names.joblib')
    print(f"Model loaded: {len(features)} features")
    
    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:20]]
    print(f"Pairs: {len(pairs)}")
    
    # Init
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    print(f"\nScanning at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Loading {CANDLES_TO_LOAD} candles per timeframe...")
    print("="*70)
    
    all_signals = []
    
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair}...", end=" ", flush=True)
        
        try:
            # Fetch data - 2000 свечей!
            data = {}
            for tf in ['1m', '5m', '15m']:
                data[tf] = fetch_candles(binance, pair, tf, CANDLES_TO_LOAD)
            
            # Build features - ТОЧНО КАК В БЭКТЕСТЕ
            ft = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
            ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
            
            if len(ft) < 10:
                print("insufficient data")
                continue
            
            # Fill missing features
            for f in features:
                if f not in ft.columns:
                    ft[f] = 0.0
            
            # Get CLOSED candle (index -2)
            row = ft.iloc[[-2]]
            X = row[features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict
            dir_proba = models['direction'].predict_proba(X)
            dir_pred = int(np.argmax(dir_proba))
            dir_conf = float(np.max(dir_proba))
            timing = float(models['timing'].predict(X)[0])
            strength = float(models['strength'].predict(X)[0])
            
            direction = ['SHORT', 'SIDEWAYS', 'LONG'][dir_pred]
            ts = row.index[0]
            close_price = row['close'].iloc[0]
            
            # Check filters
            if dir_pred == 1:  # Sideways
                print(f"SIDEWAYS conf={dir_conf:.2f}")
                continue
            
            passes = dir_conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH
            
            if passes:
                print(f"✅ {direction} @ {close_price:.6f} | Conf={dir_conf:.3f} T={timing:.2f} S={strength:.2f}")
                all_signals.append({
                    'pair': pair,
                    'direction': direction,
                    'timestamp': ts,
                    'price': close_price,
                    'conf': dir_conf,
                    'timing': timing,
                    'strength': strength,
                    'atr': row['atr'].iloc[0]
                })
            else:
                reject = []
                if dir_conf < MIN_CONF: reject.append(f"conf={dir_conf:.2f}<{MIN_CONF}")
                if timing < MIN_TIMING: reject.append(f"T={timing:.2f}<{MIN_TIMING}")
                if strength < MIN_STRENGTH: reject.append(f"S={strength:.2f}<{MIN_STRENGTH}")
                print(f"❌ {direction} | {', '.join(reject)}")
                        
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(f"FOUND {len(all_signals)} VALID SIGNALS")
    print("="*70)
    
    for sig in all_signals:
        print(f"  {sig['pair']} {sig['direction']} @ {sig['price']:.6f}")
        print(f"    Conf={sig['conf']:.3f} Timing={sig['timing']:.2f} Strength={sig['strength']:.2f}")
        print(f"    Time: {sig['timestamp']}")
        print()
    
    return all_signals


if __name__ == '__main__':
    main()

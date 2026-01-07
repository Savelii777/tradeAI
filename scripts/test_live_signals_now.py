#!/usr/bin/env python3
"""Test model on LIVE data from Binance RIGHT NOW."""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import joblib
import ccxt
from train_mtf import MTFFeatureEngine

def add_volume_features(df):
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    df["vol_zscore"] = (df["volume"] - df["vol_sma_20"]) / df["volume"].rolling(20).std()
    df["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["price_vs_vwap"] = df["close"] / df["vwap"] - 1
    df["vol_momentum"] = df["volume"].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# Загружаем модели
dir_model = joblib.load("../models/v8_improved/direction_model.joblib")
timing_model = joblib.load("../models/v8_improved/timing_model.joblib")
strength_model = joblib.load("../models/v8_improved/strength_model.joblib")
feature_cols = joblib.load("../models/v8_improved/feature_names.joblib")

# Binance
binance = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

print("="*70)
print("LIVE SIGNAL TEST - RIGHT NOW from Binance")
print("="*70)

pairs = ["HYPE/USDT:USDT", "ASTER/USDT:USDT", "ZEC/USDT:USDT", "AVAX/USDT:USDT", "PIPPIN/USDT:USDT"]
engine = MTFFeatureEngine()

for pair in pairs:
    try:
        print(f"\n{pair}:")
        
        # Fetch data from Binance
        data = {}
        for tf in ['1m', '5m', '15m']:
            candles = binance.fetch_ohlcv(pair, tf, limit=1000)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        
        # Prepare features
        ft = engine.align_timeframes(data['1m'], data['5m'], data['15m'])
        ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        ft = ft.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Last 20 bars (last ~2 hours)
        ft = ft.tail(20)
        X = ft[feature_cols]
        
        # Predictions
        proba = dir_model.predict_proba(X)
        preds = dir_model.predict(X)
        confs = np.max(proba, axis=1)
        timing = timing_model.predict(X)
        strength = strength_model.predict(X)
        
        # Show last 5 predictions
        print(f"  Last 5 bars:")
        for i in range(-5, 0):
            idx = ft.index[i]
            p = preds[i]
            c = confs[i]
            t = timing[i]
            s = strength[i]
            p_down, p_side, p_up = proba[i]
            dir_str = 'LONG' if p == 2 else ('SHORT' if p == 0 else 'SIDE')
            
            # Check if passes all filters
            MIN_CONF = 0.5
            MIN_TIMING = 0.8
            MIN_STRENGTH = 1.4
            
            passes = p != 1 and c >= MIN_CONF and t >= MIN_TIMING and s >= MIN_STRENGTH
            emoji = "✅" if passes else "❌"
            
            print(f"    {idx.strftime('%H:%M')} | {dir_str:5} | Conf:{c:.2f} | Tim:{t:.1f} | Str:{s:.1f} | {emoji}")
            print(f"           DOWN={p_down:.2f} SIDE={p_side:.2f} UP={p_up:.2f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)

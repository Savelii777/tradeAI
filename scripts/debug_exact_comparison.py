#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ ДИАГНОСТИКА: Сравнение ТОЧНО ТЕХ ЖЕ данных между методами
"""
import json
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")

models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
    'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

mtf_fe = MTFFeatureEngine()

def fetch_klines(symbol, interval, limit=500):
    clean_symbol = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {'symbol': clean_symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)
    return df

def add_volume_features(df):
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(m1, m5, m15):
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)
    return ft

# Тест
pair = 'PIPPIN/USDT:USDT'

print("="*90)
print(f"🔍 ТЕСТ: {pair}")
print("="*90)

# Fetch data
data = {}
for tf in ['1m', '5m', '15m']:
    data[tf] = fetch_klines(pair, tf, 500)

ft = prepare_features(data['1m'], data['5m'], data['15m'])

print(f"\nDataFrame shape: {ft.shape}")
print(f"Last 5 timestamps: {[t.strftime('%H:%M') for t in ft.index[-5:]]}")

# Сравним iloc[-2] и loc по тому же timestamp
row_iloc = ft.iloc[[-2]]
timestamp = row_iloc.index[0]
row_loc = ft.loc[[timestamp]]

print(f"\nTarget timestamp: {timestamp}")

# Get features
X_iloc = row_iloc[models['features']].values.astype(float)
X_loc = row_loc[models['features']].values.astype(float)

# Check if identical
print(f"\nFeatures comparison:")
print(f"  X_iloc shape: {X_iloc.shape}")
print(f"  X_loc shape: {X_loc.shape}")
print(f"  Are arrays equal: {np.array_equal(X_iloc, X_loc)}")

# Predictions
pred_iloc = models['direction'].predict_proba(X_iloc)
pred_loc = models['direction'].predict_proba(X_loc)

dir_iloc = ['SHORT', 'SIDE', 'LONG'][np.argmax(pred_iloc)]
dir_loc = ['SHORT', 'SIDE', 'LONG'][np.argmax(pred_loc)]

print(f"\nPredictions:")
print(f"  iloc[-2]: {dir_iloc} Conf: {np.max(pred_iloc):.4f}")
print(f"  loc[{timestamp.strftime('%H:%M')}]: {dir_loc} Conf: {np.max(pred_loc):.4f}")
print(f"  Match: {'✅' if dir_iloc == dir_loc else '❌'}")

# Check predictions array equality
print(f"\nPredictions array equal: {np.allclose(pred_iloc, pred_loc)}")

# Now test multiple historical candles
print("\n" + "="*90)
print("📊 ПОСЛЕДНИЕ 20 СВЕЧЕЙ:")
print("="*90)

print(f"\n{'Время':<12} | {'Направление':<10} | {'Conf':<8} | {'Timing':<8} | {'Strength':<8} | Фильтры")
print("-"*80)

for i in range(-20, 0):
    try:
        row = ft.iloc[[i]]
        t = row.index[0]
        X = row[models['features']].values.astype(float)
        
        dir_proba = models['direction'].predict_proba(X)
        dir_pred = int(np.argmax(dir_proba))
        dir_conf = float(np.max(dir_proba))
        timing = float(models['timing'].predict(X)[0])
        strength = float(models['strength'].predict(X)[0])
        dir_str = ['SHORT', 'SIDE', 'LONG'][dir_pred]
        
        # Check filters
        passes = []
        if dir_pred != 1:  # Not SIDEWAYS
            if dir_conf >= 0.50:
                passes.append("Conf✅")
            else:
                passes.append(f"Conf❌({dir_conf:.2f})")
            if timing >= 0.80:
                passes.append("Tim✅")
            else:
                passes.append(f"Tim❌({timing:.2f})")
            if strength >= 1.40:
                passes.append("Str✅")
            else:
                passes.append(f"Str❌({strength:.1f})")
        
        all_pass = dir_pred != 1 and dir_conf >= 0.50 and timing >= 0.80 and strength >= 1.40
        marker = "🎯 SIGNAL!" if all_pass else ""
        filter_str = " ".join(passes) if passes else "-"
        
        print(f"{t.strftime('%H:%M'):<12} | {dir_str:<10} | {dir_conf:<8.3f} | {timing:<8.2f} | {strength:<8.2f} | {filter_str} {marker}")
    except Exception as e:
        print(f"Error at index {i}: {e}")

print("\n" + "="*90)
print("💡 ИТОГ:")
print("   Если здесь есть LONG/SHORT сигналы с Conf>0.5, Timing>0.8, Strength>1.4")
print("   но live_trading.log их не показывает - значит проблема в LIVE скрипте")
print("="*90)


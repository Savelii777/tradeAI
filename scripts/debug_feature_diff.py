#!/usr/bin/env python3
"""
КРИТИЧЕСКАЯ ДИАГНОСТИКА: Сравнение фичей между CCXT и Direct API
"""
import json
import ccxt
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

# Load models
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

mtf_fe = MTFFeatureEngine()

def fetch_via_direct_api(symbol, interval, limit=500):
    """Метод из backtest скрипта - Direct Binance API"""
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

def fetch_via_ccxt(exchange, symbol, interval, limit=500):
    """Метод из live trading скрипта - CCXT"""
    candles = exchange.fetch_ohlcv(symbol, interval, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
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

# Initialize CCXT
binance_ccxt = ccxt.binance({
    'timeout': 10000,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# Test pair
TEST_PAIR = 'ADA/USDT:USDT'

print("="*90)
print("🔍 СРАВНЕНИЕ ДАННЫХ: Direct API vs CCXT")
print("="*90)

# Fetch via both methods
print(f"\n📊 Загрузка данных для {TEST_PAIR}...")

data_direct = {}
data_ccxt = {}

for tf in ['1m', '5m', '15m']:
    print(f"   {tf}...", end=" ", flush=True)
    data_direct[tf] = fetch_via_direct_api(TEST_PAIR, tf, 500)
    data_ccxt[tf] = fetch_via_ccxt(binance_ccxt, TEST_PAIR, tf, 500)
    print(f"Direct: {len(data_direct[tf])}, CCXT: {len(data_ccxt[tf])}")

# Compare raw M5 data
print("\n" + "="*90)
print("📊 СРАВНЕНИЕ СЫРЫХ M5 ДАННЫХ (последние 5 свечей):")
print("="*90)

m5_direct = data_direct['5m']
m5_ccxt = data_ccxt['5m']

print(f"\n{'Время':<20} | {'Direct Close':<15} | {'CCXT Close':<15} | {'Diff %':<10}")
print("-"*70)

for i in range(-5, 0):
    t_direct = m5_direct.index[i]
    t_ccxt = m5_ccxt.index[i]
    c_direct = m5_direct['close'].iloc[i]
    c_ccxt = m5_ccxt['close'].iloc[i]
    diff_pct = abs(c_direct - c_ccxt) / c_direct * 100
    match = "✅" if diff_pct < 0.001 else "❌"
    print(f"{t_direct.strftime('%H:%M'):<20} | {c_direct:<15.6f} | {c_ccxt:<15.6f} | {diff_pct:.6f}% {match}")

# Prepare features
print("\n" + "="*90)
print("📊 СРАВНЕНИЕ ПРЕДСКАЗАНИЙ:")
print("="*90)

ft_direct = prepare_features(data_direct['1m'], data_direct['5m'], data_direct['15m'])
ft_ccxt = prepare_features(data_ccxt['1m'], data_ccxt['5m'], data_ccxt['15m'])

# Get last closed candle
row_direct = ft_direct.iloc[[-2]]
row_ccxt = ft_ccxt.iloc[[-2]]

print(f"\nDirect API row time: {row_direct.index[0]}")
print(f"CCXT row time:       {row_ccxt.index[0]}")

# Make predictions
X_direct = row_direct[models['features']].values
X_ccxt = row_ccxt[models['features']].values

dir_proba_direct = models['direction'].predict_proba(X_direct)
dir_proba_ccxt = models['direction'].predict_proba(X_ccxt)

dir_pred_direct = int(np.argmax(dir_proba_direct))
dir_pred_ccxt = int(np.argmax(dir_proba_ccxt))

dir_conf_direct = float(np.max(dir_proba_direct))
dir_conf_ccxt = float(np.max(dir_proba_ccxt))

dir_str_direct = ['SHORT', 'SIDE', 'LONG'][dir_pred_direct]
dir_str_ccxt = ['SHORT', 'SIDE', 'LONG'][dir_pred_ccxt]

print(f"\n🎯 ПРЕДСКАЗАНИЯ:")
print(f"   Direct API: {dir_str_direct} Conf: {dir_conf_direct:.4f}")
print(f"   CCXT:       {dir_str_ccxt} Conf: {dir_conf_ccxt:.4f}")

if dir_str_direct == dir_str_ccxt:
    print(f"\n   ✅ Направления СОВПАДАЮТ!")
else:
    print(f"\n   ❌ НАПРАВЛЕНИЯ РАЗНЫЕ! Это КРИТИЧЕСКАЯ проблема!")

# Compare feature values
print("\n" + "="*90)
print("📊 СРАВНЕНИЕ КЛЮЧЕВЫХ ФИЧ:")
print("="*90)

# Find features with biggest difference
feature_diffs = []
for feat in models['features']:
    v_direct = row_direct[feat].values[0]
    v_ccxt = row_ccxt[feat].values[0]
    if v_direct != 0:
        diff_pct = abs(v_direct - v_ccxt) / abs(v_direct) * 100
    else:
        diff_pct = abs(v_direct - v_ccxt) * 100
    feature_diffs.append({
        'feature': feat,
        'direct': v_direct,
        'ccxt': v_ccxt,
        'diff_pct': diff_pct
    })

# Sort by difference
feature_diffs.sort(key=lambda x: x['diff_pct'], reverse=True)

print(f"\n{'Feature':<40} | {'Direct':<15} | {'CCXT':<15} | {'Diff %':<10}")
print("-"*90)

# Show top 20 most different features
for fd in feature_diffs[:20]:
    match = "❌" if fd['diff_pct'] > 1 else "✅"
    print(f"{fd['feature']:<40} | {fd['direct']:<15.6f} | {fd['ccxt']:<15.6f} | {fd['diff_pct']:.2f}% {match}")

# Summary
diff_features = [f for f in feature_diffs if f['diff_pct'] > 1]
print(f"\n{'='*90}")
print(f"📊 ИТОГ:")
print(f"   Всего фичей: {len(models['features'])}")
print(f"   Фичей с разницей >1%: {len(diff_features)}")
if len(diff_features) > 0:
    print(f"\n   ⚠️ ВНИМАНИЕ: Есть значительные расхождения в фичах!")
    print(f"   Это объясняет почему предсказания могут отличаться!")
else:
    print(f"\n   ✅ Фичи практически идентичны")
print("="*90)


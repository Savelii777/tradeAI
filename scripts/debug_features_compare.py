#!/usr/bin/env python3
"""
Compare features between live_trading style and check_same_candles style
to find why predictions differ for the same candle.
"""
import pandas as pd
import numpy as np
import joblib
import requests
import ccxt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

mtf_fe = MTFFeatureEngine()

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

def fetch_binance_api(symbol, interval, limit=500):
    """Fetch via Binance API (like check_same_candles)"""
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

def fetch_ccxt(exchange, symbol, interval, limit=500):
    """Fetch via CCXT (like live_trading)"""
    candles = exchange.fetch_ohlcv(symbol, interval, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df

# Initialize CCXT
binance = ccxt.binance({
    'timeout': 10000,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

pair = '1000PEPE/USDT:USDT'
target_time = pd.Timestamp('2026-01-05 09:00:00', tz='UTC')

print("="*70)
print(f"COMPARING FEATURES FOR {pair} @ {target_time}")
print("="*70)

# ========== METHOD 1: BINANCE API (check_same_candles style) ==========
print("\n[1] BINANCE API (check_same_candles style)...")
data_api = {}
for tf in ['1m', '5m', '15m']:
    data_api[tf] = fetch_binance_api(pair, tf, 500)

m1_api, m5_api, m15_api = data_api['1m'], data_api['5m'], data_api['15m']
ft_api = mtf_fe.align_timeframes(m1_api, m5_api, m15_api)
ft_api = ft_api.join(m5_api[['open', 'high', 'low', 'close', 'volume']])
ft_api = add_volume_features(ft_api)
ft_api['atr'] = calculate_atr(ft_api)
ft_api = ft_api.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)

# ========== METHOD 2: CCXT (live_trading style) ==========
print("[2] CCXT (live_trading style)...")
data_ccxt = {}
for tf in ['1m', '5m', '15m']:
    data_ccxt[tf] = fetch_ccxt(binance, pair, tf, 500)

m1_ccxt, m5_ccxt, m15_ccxt = data_ccxt['1m'], data_ccxt['5m'], data_ccxt['15m']
# Live-style: sort_index inplace
for df in [m1_ccxt, m5_ccxt, m15_ccxt]:
    df.sort_index(inplace=True)

ft_ccxt = mtf_fe.align_timeframes(m1_ccxt, m5_ccxt, m15_ccxt)
ft_ccxt = ft_ccxt.join(m5_ccxt[['open', 'high', 'low', 'close', 'volume']])
ft_ccxt = add_volume_features(ft_ccxt)
ft_ccxt['atr'] = calculate_atr(ft_ccxt)
ft_ccxt = ft_ccxt.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)

# ========== COMPARE ==========
print("\n" + "="*70)
print("COMPARISON FOR CANDLE @ 09:00 UTC")
print("="*70)

if target_time in ft_api.index and target_time in ft_ccxt.index:
    row_api = ft_api.loc[target_time]
    row_ccxt = ft_ccxt.loc[target_time]
    
    # Compare feature values
    print("\nFeature differences (> 0.001):")
    print("-"*70)
    diff_count = 0
    for feat in models['features']:
        val_api = float(row_api[feat])
        val_ccxt = float(row_ccxt[feat])
        if abs(val_api - val_ccxt) > 0.001:
            diff_count += 1
            print(f"  {feat}: API={val_api:.6f} vs CCXT={val_ccxt:.6f} (diff={val_api-val_ccxt:.6f})")
    
    if diff_count == 0:
        print("  NO DIFFERENCES > 0.001")
    else:
        print(f"\nTotal features with differences: {diff_count}")
    
    # Predictions
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    X_api = row_api[models['features']].values.reshape(1, -1)
    X_ccxt = row_ccxt[models['features']].values.reshape(1, -1)
    
    proba_api = models['direction'].predict_proba(X_api)[0]
    proba_ccxt = models['direction'].predict_proba(X_ccxt)[0]
    
    pred_api = np.argmax(proba_api)
    pred_ccxt = np.argmax(proba_ccxt)
    
    dir_api = ['SHORT', 'SIDE', 'LONG'][pred_api]
    dir_ccxt = ['SHORT', 'SIDE', 'LONG'][pred_ccxt]
    
    print(f"\nBINANCE API: {dir_api} | Conf: {max(proba_api):.3f}")
    print(f"   Proba: SHORT={proba_api[0]:.3f}, SIDE={proba_api[1]:.3f}, LONG={proba_api[2]:.3f}")
    
    print(f"\nCCXT:        {dir_ccxt} | Conf: {max(proba_ccxt):.3f}")
    print(f"   Proba: SHORT={proba_ccxt[0]:.3f}, SIDE={proba_ccxt[1]:.3f}, LONG={proba_ccxt[2]:.3f}")
    
    if dir_api != dir_ccxt:
        print(f"\n🔴 MISMATCH! API says {dir_api}, CCXT says {dir_ccxt}")
    else:
        print(f"\n✅ MATCH: Both say {dir_api}")

else:
    print(f"Target time {target_time} not found!")
    print(f"API range: {ft_api.index[0]} to {ft_api.index[-1]}")
    print(f"CCXT range: {ft_ccxt.index[0]} to {ft_ccxt.index[-1]}")

print("\n" + "="*70)
print("LIVE LOG REFERENCE (from actual logs at 12:05 local / 09:05 UTC)")
print("="*70)
print("Candle @ 2026-01-05 09:00:00+00:00 | Close: 0.006651")
print("→ SHORT | Conf: 0.41")
print("Proba: (not logged)")


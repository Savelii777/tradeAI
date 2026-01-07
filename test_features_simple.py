#!/usr/bin/env python3
"""Simple test for feature matching."""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.train_mtf import MTFFeatureEngine

def add_volume_features(df):
    """Add volume features required by the model (6 features)."""
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe):
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    # Generate MTF features (166 features aligned to M5)
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    
    # === ADD OHLCV for volume features & ATR calculation ===
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    
    # Add volume features (6 features required by model)
    ft = add_volume_features(ft)
    
    # Add ATR for position sizing (not used by model)
    ft['atr'] = calculate_atr(ft)
    
    return ft.dropna()

# Load model
MODEL_DIR = Path("models/v8_improved")
models = joblib.load(MODEL_DIR / 'feature_names.joblib')

print("\n" + "="*70)
print("MODEL EXPECTS")
print("="*70)
print(f"Total features: {len(models)}")

# Load sample data
data_dir = Path("data/candles")
test_pair = "BTC_USDT_USDT"

m1 = pd.read_csv(data_dir / f"{test_pair}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
m5 = pd.read_csv(data_dir / f"{test_pair}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
m15 = pd.read_csv(data_dir / f"{test_pair}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')

# Take last 500 candles
m1 = m1.iloc[-500:]
m5 = m5.iloc[-500:]
m15 = m15.iloc[-500:]

print(f"Loaded data: m1={len(m1)}, m5={len(m5)}, m15={len(m15)}")

# Generate features
mtf_fe = MTFFeatureEngine()
data = {'1m': m1, '5m': m5, '15m': m15}
ft = prepare_features(data, mtf_fe)

print(f"\nGenerated: {len(ft.columns)} columns, {len(ft)} rows")

# Check match
missing = [f for f in models if f not in ft.columns]
extra = [f for f in ft.columns if f not in models and f not in ['open', 'high', 'low', 'close', 'volume', 'atr']]

print("\n" + "="*70)
print("VALIDATION")
print("="*70)

if missing:
    print(f"\n❌ MISSING {len(missing)} features:")
    for f in missing[:10]:
        print(f"  - {f}")
else:
    print("\n✅ All 172 model features are present!")

if extra:
    print(f"\n⚠️  EXTRA {len(extra)} features:")
    for f in extra[:10]:
        print(f"  - {f}")

# Test prediction
if not missing:
    print("\n" + "="*70)
    print("TEST PREDICTION")
    print("="*70)
    
    # Load models
    dir_model = joblib.load(MODEL_DIR / 'direction_model.joblib')
    
    row = ft.iloc[[-1]]
    X = row[models].values
    
    print(f"Input shape: {X.shape}")
    has_nan = pd.isna(X).any()
    print(f"Contains NaN: {has_nan}")
    
    if not has_nan:
        dir_proba = dir_model.predict_proba(X)
        dir_pred = dir_proba.argmax()
        dir_conf = dir_proba.max()
        
        classes = ['SHORT', 'SIDEWAYS', 'LONG']
        print(f"\n✅ Prediction: {classes[dir_pred]} (confidence: {dir_conf:.2%})")
        print(f"   Probabilities:")
        print(f"   - SHORT:    {dir_proba[0][0]:.2%}")
        print(f"   - SIDEWAYS: {dir_proba[0][1]:.2%}")
        print(f"   - LONG:     {dir_proba[0][2]:.2%}")
    else:
        print("\n❌ Cannot predict - features contain NaN")

print("\n" + "="*70)


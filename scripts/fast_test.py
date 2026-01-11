#!/usr/bin/env python3
"""FAST test of new model - loads features once, iterates quickly"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path('models/v8_improved')
DATA_DIR = Path('data/candles')

# Load model
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
    'scaler': joblib.load(MODEL_DIR / 'scaler.joblib') if (MODEL_DIR / 'scaler.joblib').exists() else None
}

print("="*60)
print("FAST MODEL TEST")
print(f"Scaler: {'YES' if models['scaler'] else 'NO'}")
print(f"Features: {len(models['features'])}")
print("="*60)

def fast_test(pair, start, end):
    """Fast test using pre-computed rolling features"""
    m5 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_5m.parquet')
    
    # Filter period
    df = m5[(m5.index >= start) & (m5.index <= end)].copy()
    if len(df) == 0:
        return None
    
    # Compute CORE features directly (fast)
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_50'] = tr.rolling(50).mean()
    
    # CORE_20_FEATURES calculation
    df['m5_atr_14_pct'] = df['atr_14'] / df['close'] * 100
    df['m5_atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(20).mean()
    df['m15_atr_pct'] = df['atr_14'] / df['close'] * 100 * 1.5  # Approx
    df['m5_return_1'] = df['close'].pct_change(1)
    df['m5_return_5'] = df['close'].pct_change(5)
    df['m5_return_10'] = df['close'].pct_change(10)
    df['m5_return_20'] = df['close'].pct_change(20)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    df['m5_rsi_7'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    gain14 = delta.where(delta > 0, 0).rolling(14).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['m5_rsi_14'] = 100 - (100 / (1 + gain14 / (loss14 + 1e-10)))
    df['m15_rsi'] = df['m5_rsi_14'].rolling(3).mean()  # Approx
    
    # Position
    df['m5_close_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    
    # Bollinger
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['m5_bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-10)
    df['m5_bb_width'] = (4 * std20) / sma20 * 100
    
    # Volume
    df['m5_volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
    df['m5_volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['m5_volume_ratio_20']
    
    # Structure (simplified)
    df['m5_higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['m5_lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    
    # Trend
    df['m15_trend'] = np.sign(df['close'].rolling(15).mean() - df['close'].rolling(45).mean())
    df['m15_momentum'] = df['close'].pct_change(15)
    
    # Additional
    df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-10)
    df['m5_ema_9_dist'] = (df['close'] - df['close'].ewm(span=9).mean()) / df['close'] * 100
    df['m5_atr_vs_avg'] = df['atr_14'] / df['atr_50']
    
    df = df.dropna()
    if len(df) < 10:
        return None
    
    # Prepare features
    X = df[models['features']].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    
    if models['scaler']:
        X = models['scaler'].transform(X)
    
    # Predict all at once
    proba = models['direction'].predict_proba(X)
    preds = np.argmax(proba, axis=1)
    confs = np.max(proba, axis=1)
    
    # Stats
    dir_counts = {0: (preds == 0).sum(), 1: (preds == 1).sum(), 2: (preds == 2).sum()}
    total = len(preds)
    pct_side = 100 * dir_counts[1] / total
    
    directional = (preds != 1)
    signals_50 = ((preds != 1) & (confs >= 0.50)).sum()
    signals_62 = ((preds != 1) & (confs >= 0.62)).sum()
    
    print(f"\n{pair} ({start[:10]} to {end[:10]}):")
    print(f"  SHORT={dir_counts[0]} SIDEWAYS={dir_counts[1]} ({pct_side:.0f}%) LONG={dir_counts[2]}")
    print(f"  Conf: min={confs.min():.3f} max={confs.max():.3f} mean={confs.mean():.3f}")
    print(f"  Signals conf>=0.50: {signals_50} | conf>=0.62: {signals_62}")
    
    return signals_50, signals_62

# Test pairs
pairs = ['BTC', 'ETH', 'SOL', 'DOGE', 'PIPPIN', 'TAO', 'HYPE', '1000PEPE', 'SUI', 'APT']
total_50, total_62 = 0, 0

for pair in pairs:
    try:
        result = fast_test(pair, '2025-12-01', '2025-12-14')
        if result:
            total_50 += result[0]
            total_62 += result[1]
    except Exception as e:
        print(f"\n{pair}: Error - {e}")

print("\n" + "="*60)
print(f"TOTAL across {len(pairs)} pairs (14 days):")
print(f"  Signals conf>=0.50: {total_50} ({total_50/14:.1f}/day)")
print(f"  Signals conf>=0.62: {total_62} ({total_62/14:.1f}/day)")
print("="*60)

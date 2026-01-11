#!/usr/bin/env python3
"""Compare features from fast_test vs MTFFeatureEngine"""
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path('models/v8_improved')
DATA_DIR = Path('data/candles')

models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
    'scaler': joblib.load(MODEL_DIR / 'scaler.joblib') if (MODEL_DIR / 'scaler.joblib').exists() else None
}

mtf_fe = MTFFeatureEngine()

# Load PIPPIN
pair = 'PIPPIN'
m1 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_1m.parquet')
m5 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_5m.parquet')
m15 = pd.read_parquet(DATA_DIR / f'{pair}_USDT_USDT_15m.parquet')

# Pick a point in Dec
end_idx = m5[(m5.index >= '2025-12-03') & (m5.index <= '2025-12-03 12:00')].index[-1]

# Method 1: MTFFeatureEngine (correct, used in backtest)
m5_w = m5[m5.index <= end_idx].tail(1500)
m1_w = m1[m1.index <= end_idx].tail(7500)
m15_w = m15[m15.index <= end_idx].tail(500)

ft_mtf = mtf_fe.align_timeframes(m1_w, m5_w, m15_w)
ft_mtf = ft_mtf.join(m5_w[['open','high','low','close','volume']])
ft_mtf['vol_sma_20'] = ft_mtf['volume'].rolling(20).mean()
ft_mtf['vol_ratio'] = ft_mtf['volume'] / ft_mtf['vol_sma_20']
ft_mtf['vol_zscore'] = (ft_mtf['volume'] - ft_mtf['vol_sma_20']) / ft_mtf['volume'].rolling(20).std()
ft_mtf = ft_mtf.ffill().dropna()
row_mtf = ft_mtf.iloc[-2]

# Method 2: Fast calculation (what fast_test uses)
df = m5[m5.index <= end_idx].tail(200).copy()
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1)
df['atr_14'] = tr.rolling(14).mean()
df['atr_50'] = tr.rolling(50).mean()
df['m5_atr_14_pct'] = df['atr_14'] / df['close'] * 100
df['m5_atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(20).mean()
df['m15_atr_pct'] = df['atr_14'] / df['close'] * 100 * 1.5
df['m5_return_1'] = df['close'].pct_change(1)
df['m5_return_5'] = df['close'].pct_change(5)
df['m5_return_10'] = df['close'].pct_change(10)
df['m5_return_20'] = df['close'].pct_change(20)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(7).mean()
loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
df['m5_rsi_7'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
gain14 = delta.where(delta > 0, 0).rolling(14).mean()
loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['m5_rsi_14'] = 100 - (100 / (1 + gain14 / (loss14 + 1e-10)))
df['m15_rsi'] = df['m5_rsi_14'].rolling(3).mean()
df['m5_close_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
sma20 = df['close'].rolling(20).mean()
std20 = df['close'].rolling(20).std()
df['m5_bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-10)
df['m5_bb_width'] = (4 * std20) / sma20 * 100
df['m5_volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
df['m5_volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
df['vol_ratio'] = df['m5_volume_ratio_20']
df['m5_higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
df['m5_lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
df['m15_trend'] = np.sign(df['close'].rolling(15).mean() - df['close'].rolling(45).mean())
df['m15_momentum'] = df['close'].pct_change(15)
df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-10)
df['m5_ema_9_dist'] = (df['close'] - df['close'].ewm(span=9).mean()) / df['close'] * 100
df['m5_atr_vs_avg'] = df['atr_14'] / df['atr_50']
df = df.dropna()
row_fast = df.iloc[-1]

print("FEATURE COMPARISON: MTF vs FAST")
print("="*70)
print(f"{'Feature':<25} {'MTF':>12} {'FAST':>12} {'DIFF':>12}")
print("-"*70)

for f in models['features']:
    mtf_val = row_mtf.get(f, 0.0)
    fast_val = row_fast.get(f, 0.0)
    diff = abs(mtf_val - fast_val)
    marker = "***" if diff > 0.5 else ""
    print(f"{f:<25} {mtf_val:>12.4f} {fast_val:>12.4f} {diff:>12.4f} {marker}")

print()
print("Now checking predictions:")

# MTF prediction
X_mtf = np.array([[row_mtf.get(f, 0.0) for f in models['features']]])
X_mtf = np.nan_to_num(X_mtf, nan=0.0)
if models['scaler']:
    X_mtf_scaled = models['scaler'].transform(X_mtf)
proba_mtf = models['direction'].predict_proba(X_mtf_scaled)
print(f"MTF:  pred={np.argmax(proba_mtf)} conf={np.max(proba_mtf):.3f} proba={proba_mtf[0]}")

# Fast prediction  
X_fast = np.array([[row_fast.get(f, 0.0) for f in models['features']]])
X_fast = np.nan_to_num(X_fast, nan=0.0)
if models['scaler']:
    X_fast_scaled = models['scaler'].transform(X_fast)
proba_fast = models['direction'].predict_proba(X_fast_scaled)
print(f"FAST: pred={np.argmax(proba_fast)} conf={np.max(proba_fast):.3f} proba={proba_fast[0]}")

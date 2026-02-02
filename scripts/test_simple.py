#!/usr/bin/env python3
"""
Простая проверка - загружаем данные как в train скрипте и делаем prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES

# Import model classes for unpickling
from train_v3_dynamic import EnsembleDirectionModel

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df

# Загрузка данных
print("Loading data...")
m1 = pd.read_parquet(DATA_DIR / "LTC_USDT_USDT_1m.parquet")
m5 = pd.read_parquet(DATA_DIR / "LTC_USDT_USDT_5m.parquet")
m15 = pd.read_parquet(DATA_DIR / "LTC_USDT_USDT_15m.parquet")

print(f"M5 data range: {m5.index[0]} to {m5.index[-1]}")
print(f"Total M5 rows: {len(m5)}")

# Фильтруем до 15 октября (как в train скрипте - берём данные ДО test периода)
m1_oct14 = m1[(m1.index >= '2025-10-14') & (m1.index < '2025-10-15')]
m5_oct14 = m5[(m5.index >= '2025-10-14') & (m5.index < '2025-10-15')]
m15_oct14 = m15[(m15.index >= '2025-10-14') & (m15.index < '2025-10-15')]

print(f"\nOct 14 data:")
print(f"  M1: {len(m1_oct14)} rows")
print(f"  M5: {len(m5_oct14)} rows")
print(f"  M15: {len(m15_oct14)} rows")

# Генерация фич КАК В TRAIN СКРИПТЕ
print("\nGenerating features (like train script)...")
mtf_fe = MTFFeatureEngine()
ft = mtf_fe.align_timeframes(m1_oct14, m5_oct14, m15_oct14)
ft = ft.join(m5_oct14[['open', 'high', 'low', 'close', 'volume']])
ft = add_volume_features(ft)
ft['atr'] = calculate_atr(ft)

print(f"Features shape: {ft.shape}")
print(f"Feature range: {ft.index[0]} to {ft.index[-1]}")

# Проверяем есть ли 20:20
target_time = pd.Timestamp('2025-10-14 20:20:00', tz='UTC')
if target_time in ft.index:
    print(f"\n✅ Found 20:20 in features!")
else:
    print(f"\n❌ 20:20 NOT in features!")
    print(f"Closest times:")
    idx = ft.index.get_indexer([target_time], method='nearest')[0]
    print(f"  {ft.index[max(0, idx-2):idx+3].tolist()}")

# Загрузка модели
print("\nLoading model...")
direction_model = joblib.load(MODEL_DIR / "direction_model.joblib")
timing_model = joblib.load(MODEL_DIR / "timing_model.joblib")
strength_model = joblib.load(MODEL_DIR / "strength_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

print(f"Model expects {len(feature_names)} features:")
print(f"  {feature_names}")

# Проверяем какие фичи есть
missing = [f for f in feature_names if f not in ft.columns]
if missing:
    print(f"\n❌ Missing features: {missing}")
else:
    print(f"\n✅ All features present!")

# Prediction на 20:20
if target_time in ft.index:
    row = ft.loc[[target_time]]
    X = row[feature_names].values
    X_scaled = scaler.transform(X)
    
    dir_proba = direction_model.predict_proba(X_scaled)[0]
    timing_pred = timing_model.predict(X_scaled)[0]
    strength_pred = strength_model.predict(X_scaled)[0]
    
    dir_class = np.argmax(dir_proba)
    direction = ['LONG', 'NEUTRAL', 'SHORT'][dir_class]
    confidence = dir_proba[dir_class]
    
    print(f"\n{'='*60}")
    print(f"PREDICTION for LTC 2025-10-14 20:20:")
    print(f"{'='*60}")
    print(f"Direction: {direction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Timing: {timing_pred:.2f}")
    print(f"Strength: {strength_pred:.2f}")
    print(f"Dir proba: LONG={dir_proba[0]:.4f}, NEUTRAL={dir_proba[1]:.4f}, SHORT={dir_proba[2]:.4f}")
    print(f"{'='*60}")
    
    if confidence >= 0.55:
        print("✅ PASSES threshold!")
    else:
        print(f"❌ Below threshold (need 0.55)")

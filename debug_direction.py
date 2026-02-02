#!/usr/bin/env python3
"""
Проверка конкретных сигналов: POL 07:10 и UNI 13:55 на 24 января.
"""
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
from train_mtf import MTFFeatureEngine

# Import from check_live_signals in scripts
from scripts.check_live_signals import add_volume_features, calculate_atr, EnsembleDirectionModel

MODEL_DIR = Path("./models/v8_improved")
DATA_DIR = Path("./data/candles")

# Load model
print("Loading model...")
direction_model = joblib.load(MODEL_DIR / "direction_model.joblib")
timing_model = joblib.load(MODEL_DIR / "timing_model.joblib")
strength_model = joblib.load(MODEL_DIR / "strength_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

print(f"Model loaded: {len(feature_names)} features\n")

def check_signal(pair, timestamp_str):
    print("=" * 80)
    print(f"Checking {pair} at {timestamp_str}")
    print("=" * 80)
    
    # Load data
    safe_symbol = pair.replace('/', '_').replace(':', '_')
    m1 = pd.read_parquet(DATA_DIR / f"{safe_symbol}_1m.parquet")
    m5 = pd.read_parquet(DATA_DIR / f"{safe_symbol}_5m.parquet")
    m15 = pd.read_parquet(DATA_DIR / f"{safe_symbol}_15m.parquet")
    
    m1.index = pd.to_datetime(m1.index, utc=True)
    m5.index = pd.to_datetime(m5.index, utc=True)
    m15.index = pd.to_datetime(m15.index, utc=True)
    
    # Prepare features
    mtf_fe = MTFFeatureEngine()
    features = mtf_fe.align_timeframes(m1, m5, m15)
    features = features.join(m5[['open', 'high', 'low', 'close', 'volume']])
    features = add_volume_features(features)
    features['atr'] = calculate_atr(features)
    features = features.ffill().dropna()
    
    # Get specific timestamp
    target_time = pd.to_datetime(timestamp_str, utc=True)
    if target_time not in features.index:
        print(f"❌ Timestamp {target_time} not found in features")
        return
    
    row = features.loc[[target_time]]
    
    # Extract features for model
    X = row[feature_names].values
    X_scaled = scaler.transform(X)
    
    # Predict
    dir_proba = direction_model.predict_proba(X_scaled)[0]
    timing_pred = timing_model.predict(X_scaled)[0]
    strength_pred = strength_model.predict(X_scaled)[0]
    
    # Direction
    # 0=SHORT, 1=NEUTRAL, 2=LONG (same as train_v3_dynamic.py)
    dir_class = np.argmax(dir_proba)
    direction = ['SHORT', 'NEUTRAL', 'LONG'][dir_class]
    confidence = dir_proba[dir_class]
    
    print(f"\nPredictions:")
    print(f"  Direction: {direction} (class={dir_class})")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Dir proba: LONG={dir_proba[2]:.4f}, NEUTRAL={dir_proba[1]:.4f}, SHORT={dir_proba[0]:.4f}")
    print(f"  Timing: {timing_pred:.2f}")
    print(f"  Strength: {strength_pred:.2f}")
    print(f"  Entry price: {row['close'].iloc[0]:.6f}")
    print(f"  ATR: {row['atr'].iloc[0]:.6f}")
    print()

# Check the two problematic signals
check_signal('POL/USDT:USDT', '2026-01-24 07:10:00')
check_signal('UNI/USDT:USDT', '2026-01-24 13:55:00')

print("=" * 80)
print("Expected from backtest CSV:")
print("  POL 07:10: LONG conf=0.600")
print("  UNI 13:55: LONG conf=0.625")
print("=" * 80)

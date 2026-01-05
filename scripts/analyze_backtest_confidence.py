#!/usr/bin/env python3
"""
Analyze confidence values in backtest to understand why live trading has low confidence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime, timezone, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_mtf import MTFFeatureEngine
from train_v3_dynamic import add_volume_features, calculate_atr

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

def load_models():
    """Load trained models."""
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }

def analyze_backtest_period(pair_name: str, days_back: int = 7):
    """Analyze confidence values for a specific pair in recent period."""
    print(f"\n{'='*70}")
    print(f"Analyzing {pair_name} for last {days_back} days")
    print(f"{'='*70}")
    
    # Load models
    models = load_models()
    mtf_fe = MTFFeatureEngine()
    
    # Load data
    pair_file = pair_name.replace('/', '_').replace(':', '_')
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_file}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_file}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_file}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"❌ Data not found for {pair_name}")
        return
    
    # Localize to UTC if timezone-naive
    if m1.index.tz is None:
        m1.index = m1.index.tz_localize('UTC')
    if m5.index.tz is None:
        m5.index = m5.index.tz_localize('UTC')
    if m15.index.tz is None:
        m15.index = m15.index.tz_localize('UTC')
    
    # Filter to last N days (use last available data if CSV is old)
    if len(m5) > 0:
        end_time = m5.index[-1]  # Use last available timestamp
        start_time = end_time - timedelta(days=days_back)
    else:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)
    
    m1 = m1[(m1.index >= start_time) & (m1.index <= end_time)]
    m5 = m5[(m5.index >= start_time) & (m5.index <= end_time)]
    m15 = m15[(m15.index >= start_time) & (m15.index <= end_time)]
    
    if len(m5) < 100:
        print(f"❌ Not enough data: {len(m5)} candles")
        return
    
    # Prepare features
    df = mtf_fe.align_timeframes(m1, m5, m15)
    df = df.join(m5[['open', 'high', 'low', 'close', 'volume']])
    df = add_volume_features(df)
    df['atr'] = calculate_atr(df)
    
    # Get features
    feature_cols = models['features']
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing[:5]}...")
        return
    
    df = df[feature_cols + ['close', 'atr']].dropna()
    
    if len(df) < 50:
        print(f"❌ Not enough data after feature prep: {len(df)} rows")
        return
    
    # Predict
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    
    # Get class probabilities
    prob_short = dir_proba[:, 0]  # Class 0 = SHORT
    prob_sideways = dir_proba[:, 1]  # Class 1 = SIDEWAYS
    prob_long = dir_proba[:, 2]  # Class 2 = LONG
    
    # Timing and strength
    timing_preds = models['timing'].predict(X)
    strength_preds = models['strength'].predict(X)
    
    # Analyze
    long_mask = dir_preds == 2
    short_mask = dir_preds == 0
    sideways_mask = dir_preds == 1
    
    print(f"\n📊 Prediction Distribution:")
    print(f"   LONG: {np.sum(long_mask)} ({np.sum(long_mask)/len(df)*100:.1f}%)")
    print(f"   SHORT: {np.sum(short_mask)} ({np.sum(short_mask)/len(df)*100:.1f}%)")
    print(f"   SIDEWAYS: {np.sum(sideways_mask)} ({np.sum(sideways_mask)/len(df)*100:.1f}%)")
    
    print(f"\n📊 Confidence Statistics (max probability):")
    print(f"   All: min={dir_confs.min():.3f}, max={dir_confs.max():.3f}, mean={dir_confs.mean():.3f}, median={np.median(dir_confs):.3f}")
    
    if np.sum(long_mask) > 0:
        long_confs = dir_confs[long_mask]
        print(f"   LONG only: min={long_confs.min():.3f}, max={long_confs.max():.3f}, mean={long_confs.mean():.3f}, median={np.median(long_confs):.3f}")
        print(f"   LONG with conf >= 0.5: {np.sum(long_confs >= 0.5)} ({np.sum(long_confs >= 0.5)/len(long_confs)*100:.1f}%)")
        print(f"   LONG with conf >= 0.4: {np.sum(long_confs >= 0.4)} ({np.sum(long_confs >= 0.4)/len(long_confs)*100:.1f}%)")
        print(f"   LONG with conf >= 0.35: {np.sum(long_confs >= 0.35)} ({np.sum(long_confs >= 0.35)/len(long_confs)*100:.1f}%)")
    
    if np.sum(short_mask) > 0:
        short_confs = dir_confs[short_mask]
        print(f"   SHORT only: min={short_confs.min():.3f}, max={short_confs.max():.3f}, mean={short_confs.mean():.3f}, median={np.median(short_confs):.3f}")
        print(f"   SHORT with conf >= 0.5: {np.sum(short_confs >= 0.5)} ({np.sum(short_confs >= 0.5)/len(short_confs)*100:.1f}%)")
        print(f"   SHORT with conf >= 0.4: {np.sum(short_confs >= 0.4)} ({np.sum(short_confs >= 0.4)/len(short_confs)*100:.1f}%)")
        print(f"   SHORT with conf >= 0.35: {np.sum(short_confs >= 0.35)} ({np.sum(short_confs >= 0.35)/len(short_confs)*100:.1f}%)")
    
    # Check if using class-specific probability would help
    print(f"\n📊 Class-Specific Probabilities:")
    if np.sum(long_mask) > 0:
        long_class_probs = prob_long[long_mask]
        print(f"   LONG class prob (when predicted LONG): min={long_class_probs.min():.3f}, max={long_class_probs.max():.3f}, mean={long_class_probs.mean():.3f}")
        print(f"   LONG class prob >= 0.5: {np.sum(long_class_probs >= 0.5)} ({np.sum(long_class_probs >= 0.5)/len(long_class_probs)*100:.1f}%)")
    
    if np.sum(short_mask) > 0:
        short_class_probs = prob_short[short_mask]
        print(f"   SHORT class prob (when predicted SHORT): min={short_class_probs.min():.3f}, max={short_class_probs.max():.3f}, mean={short_class_probs.mean():.3f}")
        print(f"   SHORT class prob >= 0.5: {np.sum(short_class_probs >= 0.5)} ({np.sum(short_class_probs >= 0.5)/len(short_class_probs)*100:.1f}%)")
    
    # Show examples of signals that would pass filters
    print(f"\n📊 Signals that would pass filters (Conf >= 0.5, Timing >= 0.8, Strength >= 1.4):")
    valid_mask = (dir_preds != 1) & (dir_confs >= 0.5) & (timing_preds >= 0.8) & (strength_preds >= 1.4)
    valid_count = np.sum(valid_mask)
    print(f"   Valid signals: {valid_count} ({valid_count/len(df)*100:.1f}%)")
    
    if valid_count > 0:
        valid_df = df[valid_mask].copy()
        valid_df['direction'] = ['LONG' if dir_preds[i] == 2 else 'SHORT' for i in range(len(df)) if valid_mask[i]]
        valid_df['confidence'] = dir_confs[valid_mask]
        valid_df['timing'] = timing_preds[valid_mask]
        valid_df['strength'] = strength_preds[valid_mask]
        
        print(f"\n   First 5 valid signals:")
        for idx, row in valid_df.head(5).iterrows():
            print(f"      {idx} | {row['direction']} | Conf: {row['confidence']:.3f} | Timing: {row['timing']:.2f} | Strength: {row['strength']:.1f}")
    
    # Show recent predictions
    print(f"\n📊 Last 10 predictions:")
    recent_df = df.tail(10).copy()
    for i, idx in enumerate(recent_df.index):
        j = len(df) - 10 + i
        direction = 'LONG' if dir_preds[j] == 2 else ('SHORT' if dir_preds[j] == 0 else 'SIDEWAYS')
        print(f"   {idx} | {direction} | Conf: {dir_confs[j]:.3f} | Timing: {timing_preds[j]:.2f} | Strength: {strength_preds[j]:.1f}")

if __name__ == "__main__":
    # Analyze a few pairs
    pairs = ['UNI/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT']
    
    for pair in pairs:
        try:
            analyze_backtest_period(pair, days_back=7)
        except Exception as e:
            print(f"❌ Error analyzing {pair}: {e}")
            import traceback
            traceback.print_exc()


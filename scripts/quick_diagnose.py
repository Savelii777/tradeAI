#!/usr/bin/env python3
"""
Quick Diagnostic: Compare predictions on recent data (backtest vs live style)

This script helps identify WHY the model gives low confidence on live.
It compares predictions using:
1. BACKTEST style: Full historical data, same as training
2. LIVE style: Limited window, same as live_trading_mexc_v8.py

Usage:
    python scripts/quick_diagnose.py --pair BTC_USDT_USDT --bars 100
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES, DEFAULT_EXCLUDE_FEATURES
)

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'


def add_volume_features(df):
    """Add volume features (same as train_v3_dynamic.py)"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_models():
    """Load trained models"""
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }


def prepare_features(m1, m5, m15, mtf_fe, exclude_cumsum=True, exclude_absolute=True):
    """Prepare features (combined backtest + live logic)"""
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    if len(ft) == 0:
        return None
    
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    # Drop critical NaN
    ft = ft.dropna(subset=['close', 'atr'])
    
    # Exclude cumsum features
    if exclude_cumsum:
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        if cols_to_drop:
            ft = ft.drop(columns=cols_to_drop)
    
    # Exclude absolute features
    if exclude_absolute:
        cols_to_drop = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        if cols_to_drop:
            ft = ft.drop(columns=cols_to_drop)
    
    # Forward fill non-critical
    for col in ft.columns:
        if col not in ['close', 'atr']:
            ft[col] = ft[col].ffill()
    
    ft = ft.dropna()
    return ft


def main():
    parser = argparse.ArgumentParser(description='Quick diagnostic for live vs backtest')
    parser.add_argument('--pair', type=str, default='BTC_USDT_USDT', help='Pair to analyze')
    parser.add_argument('--bars', type=int, default=100, help='Number of bars to analyze')
    args = parser.parse_args()
    
    pair_name = args.pair
    
    print("="*70)
    print(f"QUICK DIAGNOSTIC: {pair_name}")
    print("="*70)
    
    # Load data
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure data files exist in {DATA_DIR}")
        return
    
    print(f"Data loaded: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
    print(f"Date range: {m5.index[0]} to {m5.index[-1]}")
    
    # Load models
    models = load_models()
    feature_cols = models['features']
    print(f"Model expects {len(feature_cols)} features")
    
    mtf_fe = MTFFeatureEngine()
    
    # Prepare features using FULL data (backtest style)
    print("\n" + "="*70)
    print("PREPARING FEATURES (Full Data - Backtest Style)")
    print("="*70)
    
    ft = prepare_features(m1, m5, m15, mtf_fe)
    if ft is None or len(ft) == 0:
        print("Error: Could not prepare features")
        return
    
    print(f"Features prepared: {ft.shape}")
    
    # Get last N bars
    ft_last = ft.tail(args.bars + 1).iloc[:-1]  # Exclude last (forming) candle
    print(f"Analyzing last {len(ft_last)} closed candles")
    
    # Check for missing features
    missing = [f for f in feature_cols if f not in ft_last.columns]
    if missing:
        print(f"\n⚠️ MISSING FEATURES ({len(missing)}):")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
        
        # Fill missing with 0
        for mf in missing:
            ft_last[mf] = 0.0
    
    # Make predictions
    print("\n" + "="*70)
    print("PREDICTIONS ON LAST BARS")
    print("="*70)
    
    X = ft_last[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba = models['direction'].predict_proba(X)
    dir_confs = np.max(dir_proba, axis=1)
    dir_preds = np.argmax(dir_proba, axis=1)
    
    timing_preds = models['timing'].predict(X)
    strength_preds = models['strength'].predict(X)
    
    # Stats
    print(f"\nConfidence Stats:")
    print(f"  Mean: {dir_confs.mean():.4f}")
    print(f"  Min:  {dir_confs.min():.4f}")
    print(f"  Max:  {dir_confs.max():.4f}")
    print(f"  Std:  {dir_confs.std():.4f}")
    
    print(f"\nConfidence Distribution:")
    print(f"  < 0.35: {(dir_confs < 0.35).sum()} bars ({(dir_confs < 0.35).mean()*100:.1f}%)")
    print(f"  0.35-0.40: {((dir_confs >= 0.35) & (dir_confs < 0.40)).sum()} bars")
    print(f"  0.40-0.45: {((dir_confs >= 0.40) & (dir_confs < 0.45)).sum()} bars")
    print(f"  0.45-0.50: {((dir_confs >= 0.45) & (dir_confs < 0.50)).sum()} bars")
    print(f"  >= 0.50: {(dir_confs >= 0.50).sum()} bars ({(dir_confs >= 0.50).mean()*100:.1f}%)")
    
    # Signals that would pass filters
    signals_conf = dir_confs >= 0.50
    signals_timing = timing_preds >= 0.8
    signals_strength = strength_preds >= 1.4
    signals_direction = dir_preds != 1  # Not SIDEWAYS
    
    all_pass = signals_conf & signals_timing & signals_strength & signals_direction
    
    print(f"\nSignal Filter Stats:")
    print(f"  Direction != SIDEWAYS: {signals_direction.sum()} ({signals_direction.mean()*100:.1f}%)")
    print(f"  Confidence >= 0.50: {signals_conf.sum()} ({signals_conf.mean()*100:.1f}%)")
    print(f"  Timing >= 0.8: {signals_timing.sum()} ({signals_timing.mean()*100:.1f}%)")
    print(f"  Strength >= 1.4: {signals_strength.sum()} ({signals_strength.mean()*100:.1f}%)")
    print(f"  ALL PASS: {all_pass.sum()} ({all_pass.mean()*100:.1f}%)")
    
    # Show top confidence bars
    print("\n" + "="*70)
    print("TOP 10 HIGHEST CONFIDENCE BARS")
    print("="*70)
    
    top_idx = np.argsort(dir_confs)[-10:][::-1]
    for i in top_idx:
        ts = ft_last.index[i]
        conf = dir_confs[i]
        direction = 'LONG' if dir_preds[i] == 2 else ('SHORT' if dir_preds[i] == 0 else 'SIDE')
        timing = timing_preds[i]
        strength = strength_preds[i]
        passes = "✅" if all_pass[i] else "❌"
        print(f"  {ts} | {direction:5} | Conf:{conf:.4f} | Timing:{timing:.2f} | Str:{strength:.2f} | {passes}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    high_conf_rate = (dir_confs >= 0.50).mean() * 100
    
    if high_conf_rate < 5:
        print(f"""
⚠️ VERY LOW HIGH-CONFIDENCE RATE: {high_conf_rate:.1f}%

This means the model rarely gives confidence >= 50%.
This is EXPECTED with anti-overfitting parameters.

IMPLICATIONS FOR LIVE:
- If you scan 20 pairs every 5 minutes (12 times/hour)
- 240 predictions per hour
- At {high_conf_rate:.1f}% rate → expect ~{int(240*high_conf_rate/100)} high-conf per hour
- After timing/strength filters → even fewer signals

This is NOT a bug - it's the model being conservative.
High-confidence signals are RARE but should be higher quality.
""")
    else:
        print(f"""
✅ Reasonable high-confidence rate: {high_conf_rate:.1f}%

The model gives >= 50% confidence on {high_conf_rate:.1f}% of bars.
This suggests the issue may be:
1. Current market conditions are different
2. Live data differs from historical CSV
3. Timing of data fetch

Try running this script on freshly downloaded data to compare.
""")


if __name__ == '__main__':
    main()

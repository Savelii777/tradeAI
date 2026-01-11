#!/usr/bin/env python3
"""
Check for Look-Ahead Bias Script

This script specifically checks if the model might be using future data
by comparing predictions at the same timestamp using:
1. Full data (as in backtest)
2. Truncated data up to that point only (as in live)

If predictions differ significantly, it indicates look-ahead bias.

Usage:
    python scripts/check_lookahead_bias.py --pair PIPPIN_USDT_USDT
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'


# ============================================================
# HELPERS
# ============================================================
def add_volume_features(df):
    """Add volume features - MUST match train_v3_dynamic.py EXACTLY"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df, period=14):
    """Calculate ATR - MUST match train_v3_dynamic.py EXACTLY"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features(m1, m5, m15, mtf_engine):
    """Prepare features matching train_v3_dynamic.py"""
    ft = mtf_engine.align_timeframes(m1, m5, m15)
    
    if len(ft) == 0:
        return pd.DataFrame()
    
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    return ft


def load_pair_data(pair_name: str) -> dict:
    """Load M1, M5, M15 data for a pair."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        filepath = DATA_DIR / f"{pair_name}_{tf}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        data[tf] = df
    return data


def main():
    parser = argparse.ArgumentParser(description='Check for look-ahead bias')
    parser.add_argument('--pair', type=str, default='PIPPIN_USDT_USDT', help='Pair name')
    parser.add_argument('--samples', type=int, default=20, help='Number of test points')
    args = parser.parse_args()
    
    print("="*70)
    print("LOOK-AHEAD BIAS CHECK")
    print("="*70)
    print(f"Pair: {args.pair}")
    print(f"Test samples: {args.samples}")
    print()
    
    # Load models
    print("Loading models...")
    try:
        models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
            'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    print(f"Model features: {len(models['features'])}")
    
    # Load data
    print(f"Loading data for {args.pair}...")
    try:
        data = load_pair_data(args.pair)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    print(f"M1: {len(m1)} candles ({m1.index[0]} to {m1.index[-1]})")
    print(f"M5: {len(m5)} candles ({m5.index[0]} to {m5.index[-1]})")
    print(f"M15: {len(m15)} candles ({m15.index[0]} to {m15.index[-1]})")
    
    mtf_engine = MTFFeatureEngine()
    feature_cols = models['features']
    
    # Select test points (last 10% of data, evenly spaced)
    total_m5_candles = len(m5)
    test_start_idx = int(total_m5_candles * 0.9)  # Start from 90%
    test_indices = np.linspace(test_start_idx, total_m5_candles - 100, args.samples, dtype=int)
    
    print(f"\nTesting at {len(test_indices)} timestamps...")
    print("-"*70)
    
    # Compare predictions
    results = []
    
    for i, idx in enumerate(test_indices):
        test_timestamp = m5.index[idx]
        
        # METHOD 1: Full data up to idx (backtest-style, but no future)
        m1_full = m1[m1.index <= test_timestamp]
        m5_full = m5.iloc[:idx+1]
        m15_full = m15[m15.index <= test_timestamp]
        
        # Use last 2000 candles for features (like training)
        if len(m5_full) > 2000:
            start_idx = len(m5_full) - 2000
            m5_full = m5_full.iloc[start_idx:]
            m1_start = m5_full.index[0]
            m1_full = m1_full[m1_full.index >= m1_start]
            m15_full = m15_full[m15_full.index >= m1_start]
        
        ft_full = prepare_features(m1_full, m5_full, m15_full, mtf_engine)
        
        if len(ft_full) < 2:
            continue
        
        # Get features for test timestamp
        row_full = ft_full.iloc[[-2]].copy()  # Use second-to-last (closed candle)
        
        for f in feature_cols:
            if f not in row_full.columns:
                row_full[f] = 0.0
        
        X_full = row_full[feature_cols].values.astype(np.float64)
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get predictions
        proba_full = models['direction'].predict_proba(X_full)[0]
        conf_full = np.max(proba_full)
        pred_full = np.argmax(proba_full)
        
        # METHOD 2: Truncated data (simulating live - only last 1000 candles)
        live_lookback = 1000
        m5_live = m5_full.iloc[-live_lookback:] if len(m5_full) > live_lookback else m5_full
        m1_start = m5_live.index[0]
        m1_live = m1_full[m1_full.index >= m1_start]
        m15_live = m15_full[m15_full.index >= m1_start]
        
        ft_live = prepare_features(m1_live, m5_live, m15_live, mtf_engine)
        
        if len(ft_live) < 2:
            continue
        
        row_live = ft_live.iloc[[-2]].copy()
        
        for f in feature_cols:
            if f not in row_live.columns:
                row_live[f] = 0.0
        
        X_live = row_live[feature_cols].values.astype(np.float64)
        X_live = np.nan_to_num(X_live, nan=0.0, posinf=0.0, neginf=0.0)
        
        proba_live = models['direction'].predict_proba(X_live)[0]
        conf_live = np.max(proba_live)
        pred_live = np.argmax(proba_live)
        
        # Compare
        conf_diff = conf_full - conf_live
        direction = ['DOWN', 'SIDEWAYS', 'UP']
        
        results.append({
            'timestamp': test_timestamp,
            'full_pred': direction[pred_full],
            'full_conf': conf_full,
            'live_pred': direction[pred_live],
            'live_conf': conf_live,
            'conf_diff': conf_diff,
            'pred_match': pred_full == pred_live
        })
        
        status = "✅" if abs(conf_diff) < 0.05 else ("⚠️" if abs(conf_diff) < 0.10 else "❌")
        
        print(f"[{i+1:2d}] {test_timestamp} | "
              f"Full: {direction[pred_full]:8s} {conf_full:.2f} | "
              f"Live: {direction[pred_live]:8s} {conf_live:.2f} | "
              f"Diff: {conf_diff:+.2f} {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not results:
        print("No valid test samples!")
        return
    
    avg_full_conf = np.mean([r['full_conf'] for r in results])
    avg_live_conf = np.mean([r['live_conf'] for r in results])
    avg_diff = np.mean([r['conf_diff'] for r in results])
    match_rate = np.mean([r['pred_match'] for r in results]) * 100
    
    print(f"Average Full-data Confidence:  {avg_full_conf:.3f}")
    print(f"Average Live-like Confidence:  {avg_live_conf:.3f}")
    print(f"Average Difference:            {avg_diff:+.3f}")
    print(f"Prediction Match Rate:         {match_rate:.1f}%")
    
    print("\n" + "-"*70)
    
    # Interpretation
    if abs(avg_diff) < 0.03:
        print("✅ NO LOOK-AHEAD BIAS DETECTED")
        print("   Confidence is similar with full data vs truncated data.")
        print("   The issue is likely not related to data window size.")
    elif avg_diff > 0.05:
        print("⚠️ POSSIBLE LOOK-AHEAD BIAS")
        print(f"   Full-data confidence is {avg_diff:.1%} higher than live-like.")
        print("   This could be caused by:")
        print("   1. Features that depend on data window length (rolling calculations)")
        print("   2. Features that accumulate over time (cumsum-based)")
        print("   3. Normalization that uses full history")
    else:
        print("ℹ️ MINIMAL DIFFERENCE")
        print("   Some variation is normal due to rolling window effects.")
    
    # Low confidence check
    if avg_live_conf < 0.45:
        print("\n" + "-"*70)
        print("⚠️ LOW LIVE CONFIDENCE DETECTED")
        print(f"   Average live confidence: {avg_live_conf:.2f}")
        print("   This might be because:")
        print("   1. Model is uncertain about current market conditions")
        print("   2. Current market patterns differ from training data")
        print("   3. Model is overfitted to historical patterns")
        print("   4. Features have drifted from training distribution")
        print()
        print("   Possible solutions:")
        print("   1. Retrain with more recent data")
        print("   2. Reduce model complexity (fewer trees, simpler model)")
        print("   3. Use walk-forward validation")
        print("   4. Lower MIN_CONF threshold")


if __name__ == '__main__':
    main()

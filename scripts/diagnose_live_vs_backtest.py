#!/usr/bin/env python3
"""
Diagnostic Script: Live vs Backtest Feature Comparison

This script identifies WHY the model gives signals on backtest but not on live.
It compares:
1. Feature values at the same timestamp
2. Feature distributions
3. Model predictions

Usage:
    python scripts/diagnose_live_vs_backtest.py --pair BTC_USDT_USDT
"""

import sys
import argparse
import traceback
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
from src.features.feature_engine import FeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES
)

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

# Thresholds from live script
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# Timeframe multipliers for data alignment
M1_TO_M5_RATIO = 5  # M1 has 5x more candles than M5
M5_TO_M15_RATIO = 3  # M15 has 3x fewer candles than M5

# Minimum valid samples required for feature comparison
MIN_VALID_SAMPLES = 10


# ============================================================
# HELPER FUNCTIONS (same as in training/live scripts)
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


def prepare_features_backtest(m1, m5, m15, mtf_engine):
    """Prepare features EXACTLY as in train_v3_dynamic.py (backtest mode)"""
    ft = mtf_engine.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    # Backtest: simple dropna
    ft = ft.dropna()
    return ft


def prepare_features_live(m1, m5, m15, mtf_engine):
    """Prepare features EXACTLY as in live_trading_mexc_v8.py (live mode)"""
    ft = mtf_engine.align_timeframes(m1, m5, m15)
    
    if len(ft) == 0:
        return pd.DataFrame()
    
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    # Live: Match the live script's NaN handling
    critical_cols = ['close', 'atr']
    ft = ft.dropna(subset=critical_cols)
    
    # Exclude cumsum-dependent features (use global constant)
    cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
    if cols_to_drop:
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
    
    # Forward-fill non-critical columns
    non_critical = [c for c in ft.columns if c not in critical_cols]
    if non_critical:
        ft[non_critical] = ft[non_critical].ffill()
    
    ft = ft.dropna()
    return ft


def load_models():
    """Load trained models"""
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }


# ============================================================
# DIAGNOSIS FUNCTIONS
# ============================================================

def diagnose_pair(pair_name: str, lookback_live: int = 1000, lookback_backtest: int = 5000):
    """
    Run comprehensive diagnosis for a single pair.
    
    Compares:
    1. Feature values at the same timestamp using backtest vs live preparation
    2. Prediction distributions
    3. Signal generation rates
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS: {pair_name}")
    print(f"{'='*70}")
    
    # Load data
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError as e:
        print(f"Error: Could not load data - {e}")
        return None
    
    print(f"Data loaded: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
    print(f"Date range: {m5.index[0]} to {m5.index[-1]}")
    
    # Load models
    models = load_models()
    feature_cols = models['features']
    print(f"Model has {len(feature_cols)} features")
    
    mtf_engine = MTFFeatureEngine()
    
    # ================================================================
    # TEST 1: Compare features at the same timestamp
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Feature Value Comparison (Same Timestamp)")
    print("="*70)
    
    # Use last 48 hours for comparison
    comparison_end = m5.index[-1]
    comparison_start = comparison_end - timedelta(hours=48)
    
    # BACKTEST preparation: Use FULL data history (simulates training conditions)
    print(f"\nPreparing BACKTEST features (full history)...")
    ft_backtest = prepare_features_backtest(m1, m5, m15, mtf_engine)
    ft_backtest_48h = ft_backtest[ft_backtest.index >= comparison_start]
    print(f"  Backtest features: {len(ft_backtest)} total, {len(ft_backtest_48h)} in test period")
    
    # LIVE preparation: Use only last LOOKBACK candles (simulates live conditions)
    print(f"\nPreparing LIVE features (last {lookback_live} candles)...")
    m1_live = m1.iloc[-lookback_live * M1_TO_M5_RATIO:]  # M1 has 5x more candles than M5
    m5_live = m5.iloc[-lookback_live:]
    m15_live = m15.iloc[-lookback_live // M5_TO_M15_RATIO:]
    
    ft_live = prepare_features_live(m1_live, m5_live, m15_live, mtf_engine)
    ft_live_48h = ft_live[ft_live.index >= comparison_start]
    print(f"  Live features: {len(ft_live)} total, {len(ft_live_48h)} in test period")
    
    # Find common timestamps
    common_idx = ft_backtest_48h.index.intersection(ft_live_48h.index)
    print(f"\nCommon timestamps: {len(common_idx)}")
    
    if len(common_idx) < 10:
        print("ERROR: Not enough common timestamps for comparison!")
        return None
    
    # Compare feature values at common timestamps
    print(f"\n--- Feature Value Differences ---")
    print(f"{'Feature':<35} | {'Mean Diff':>12} | {'Max Diff':>12} | {'Status':<8}")
    print("-" * 75)
    
    feature_diffs = {}
    critical_diffs = []
    
    for feat in feature_cols:
        if feat not in ft_backtest_48h.columns:
            print(f"{feat:<35} | {'MISSING_BT':>12} | {'-':>12} | ❌")
            critical_diffs.append({'feature': feat, 'reason': 'missing_backtest'})
            continue
        if feat not in ft_live_48h.columns:
            print(f"{feat:<35} | {'MISSING_LV':>12} | {'-':>12} | ❌")
            critical_diffs.append({'feature': feat, 'reason': 'missing_live'})
            continue
        
        bt_vals = ft_backtest_48h.loc[common_idx, feat]
        lv_vals = ft_live_48h.loc[common_idx, feat]
        
        # Handle NaN values - skip features with too many NaN
        valid_mask = bt_vals.notna() & lv_vals.notna()
        if valid_mask.sum() < MIN_VALID_SAMPLES:
            print(f"{feat:<35} | {'TOO_FEW':>12} | {'-':>12} | ⚠️ SKIP")
            continue
        
        bt_clean = bt_vals[valid_mask]
        lv_clean = lv_vals[valid_mask]
        
        # Convert boolean to float to avoid "numpy boolean subtract" error
        if bt_clean.dtype == bool:
            bt_clean = bt_clean.astype(float)
        if lv_clean.dtype == bool:
            lv_clean = lv_clean.astype(float)
        
        diff = (bt_clean - lv_clean).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        
        # Normalize by feature range
        feat_range = bt_clean.abs().mean() + 1e-10
        mean_diff_pct = mean_diff / feat_range * 100
        
        # Handle NaN in computed values
        if np.isnan(mean_diff_pct):
            print(f"{feat:<35} | {'NaN':>12} | {'-':>12} | ⚠️ SKIP")
            continue
        
        feature_diffs[feat] = {
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'mean_diff_pct': mean_diff_pct
        }
        
        if mean_diff_pct > 50:
            status = "❌ BAD"
            critical_diffs.append({'feature': feat, 'mean_diff_pct': mean_diff_pct})
        elif mean_diff_pct > 20:
            status = "⚠️ WARN"
        else:
            status = "✅ OK"
        
        # Only print if there's a significant difference
        if mean_diff_pct > 10:
            print(f"{feat:<35} | {mean_diff:>12.4f} | {max_diff:>12.4f} | {status}")
    
    # ================================================================
    # TEST 2: Compare predictions
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Prediction Comparison")
    print("="*70)
    
    # Ensure we only use features that exist in both
    available_features = [f for f in feature_cols if f in ft_backtest_48h.columns and f in ft_live_48h.columns]
    
    if len(available_features) < len(feature_cols):
        print(f"WARNING: Only {len(available_features)}/{len(feature_cols)} features available in both")
        missing = set(feature_cols) - set(available_features)
        print(f"Missing features: {list(missing)[:10]}...")
    
    # Get predictions for backtest
    X_bt = ft_backtest_48h.loc[common_idx, available_features].values
    X_bt = np.nan_to_num(X_bt, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba_bt = models['direction'].predict_proba(X_bt)
    conf_bt = np.max(dir_proba_bt, axis=1)
    dir_pred_bt = np.argmax(dir_proba_bt, axis=1)
    timing_bt = models['timing'].predict(X_bt)
    strength_bt = models['strength'].predict(X_bt)
    
    # Get predictions for live
    X_lv = ft_live_48h.loc[common_idx, available_features].values
    X_lv = np.nan_to_num(X_lv, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba_lv = models['direction'].predict_proba(X_lv)
    conf_lv = np.max(dir_proba_lv, axis=1)
    dir_pred_lv = np.argmax(dir_proba_lv, axis=1)
    timing_lv = models['timing'].predict(X_lv)
    strength_lv = models['strength'].predict(X_lv)
    
    print(f"\n--- Prediction Statistics ---")
    print(f"{'Metric':<25} | {'Backtest':>12} | {'Live':>12} | {'Diff':>10}")
    print("-" * 65)
    print(f"{'Avg Confidence':<25} | {conf_bt.mean():>12.4f} | {conf_lv.mean():>12.4f} | {(conf_lv.mean()-conf_bt.mean()):>+10.4f}")
    print(f"{'Avg Timing':<25} | {timing_bt.mean():>12.4f} | {timing_lv.mean():>12.4f} | {(timing_lv.mean()-timing_bt.mean()):>+10.4f}")
    print(f"{'Avg Strength':<25} | {strength_bt.mean():>12.4f} | {strength_lv.mean():>12.4f} | {(strength_lv.mean()-strength_bt.mean()):>+10.4f}")
    
    # ================================================================
    # TEST 3: Signal generation rates
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Signal Generation Rates")
    print("="*70)
    
    # Backtest signals (using backtest features)
    bt_signals = (
        (dir_pred_bt != 1) &  # Not sideways
        (conf_bt >= MIN_CONF) &
        (timing_bt >= MIN_TIMING) &
        (strength_bt >= MIN_STRENGTH)
    )
    
    # Live signals (using live features)
    lv_signals = (
        (dir_pred_lv != 1) &  # Not sideways
        (conf_lv >= MIN_CONF) &
        (timing_lv >= MIN_TIMING) &
        (strength_lv >= MIN_STRENGTH)
    )
    
    print(f"\nThresholds: Conf>={MIN_CONF}, Timing>={MIN_TIMING}, Strength>={MIN_STRENGTH}")
    print(f"\n{'Source':<15} | {'Signals':>10} | {'Total':>10} | {'Rate':>10}")
    print("-" * 55)
    print(f"{'Backtest':<15} | {bt_signals.sum():>10} | {len(bt_signals):>10} | {bt_signals.mean()*100:>9.2f}%")
    print(f"{'Live':<15} | {lv_signals.sum():>10} | {len(lv_signals):>10} | {lv_signals.mean()*100:>9.2f}%")
    
    # Why are live signals rejected?
    print(f"\n--- Rejection Analysis (Live) ---")
    sideways_lv = (dir_pred_lv == 1).sum()
    low_conf_lv = ((dir_pred_lv != 1) & (conf_lv < MIN_CONF)).sum()
    low_timing_lv = ((dir_pred_lv != 1) & (conf_lv >= MIN_CONF) & (timing_lv < MIN_TIMING)).sum()
    low_strength_lv = ((dir_pred_lv != 1) & (conf_lv >= MIN_CONF) & (timing_lv >= MIN_TIMING) & (strength_lv < MIN_STRENGTH)).sum()
    
    print(f"Sideways predictions:     {sideways_lv:>6} ({sideways_lv/len(dir_pred_lv)*100:.1f}%)")
    print(f"Low confidence:           {low_conf_lv:>6} ({low_conf_lv/len(dir_pred_lv)*100:.1f}%)")
    print(f"Low timing:               {low_timing_lv:>6} ({low_timing_lv/len(dir_pred_lv)*100:.1f}%)")
    print(f"Low strength:             {low_strength_lv:>6} ({low_strength_lv/len(dir_pred_lv)*100:.1f}%)")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if len(critical_diffs) > 0:
        print(f"\n❌ CRITICAL: {len(critical_diffs)} features have significant differences:")
        for item in critical_diffs[:10]:
            if 'mean_diff_pct' in item:
                print(f"   • {item['feature']}: {item['mean_diff_pct']:.1f}% difference")
            else:
                print(f"   • {item['feature']}: {item['reason']}")
    
    if bt_signals.sum() > 0 and lv_signals.sum() == 0:
        print(f"\n❌ PROBLEM CONFIRMED: Backtest generates signals, Live does not!")
        print(f"   Backtest: {bt_signals.sum()} signals")
        print(f"   Live:     {lv_signals.sum()} signals")
        
        print(f"\n🔍 ROOT CAUSE ANALYSIS:")
        
        if timing_lv.mean() < timing_bt.mean() * 0.8:
            print(f"   • TIMING is significantly lower on live ({timing_lv.mean():.2f} vs {timing_bt.mean():.2f})")
        
        if conf_lv.mean() < conf_bt.mean() * 0.9:
            print(f"   • CONFIDENCE is lower on live ({conf_lv.mean():.2f} vs {conf_bt.mean():.2f})")
        
        if strength_lv.mean() < strength_bt.mean() * 0.8:
            print(f"   • STRENGTH is significantly lower on live ({strength_lv.mean():.2f} vs {strength_bt.mean():.2f})")
        
        # Check for feature drift
        high_drift_features = [f for f, v in feature_diffs.items() if v.get('mean_diff_pct', 0) > 30]
        if high_drift_features:
            print(f"\n   • {len(high_drift_features)} features have >30% drift:")
            for feat in high_drift_features[:5]:
                print(f"      - {feat}: {feature_diffs[feat]['mean_diff_pct']:.1f}% drift")
        
        # ✅ NEW: Check if model uses absolute price features
        absolute_in_model = [f for f in feature_cols if f in ABSOLUTE_PRICE_FEATURES]
        if absolute_in_model:
            print(f"\n   ⚠️ MODEL USES ABSOLUTE PRICE FEATURES: {absolute_in_model}")
            print(f"   These features cause SEVERE drift between training and live!")
            print(f"   ACTION: Retrain model with: python scripts/train_v3_dynamic.py --days 60 --test_days 14")
    
    elif bt_signals.sum() > 0 and lv_signals.sum() > 0:
        print(f"\n✅ Both backtest and live generate signals!")
        print(f"   Backtest: {bt_signals.sum()} signals")
        print(f"   Live:     {lv_signals.sum()} signals")
    else:
        print(f"\n⚠️ Neither backtest nor live generate signals in test period")
        print(f"   This may indicate low market activity or overly strict thresholds")
    
    return {
        'pair': pair_name,
        'bt_signals': bt_signals.sum(),
        'lv_signals': lv_signals.sum(),
        'critical_diffs': critical_diffs,
        'feature_diffs': feature_diffs
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnose Backtest vs Live discrepancy")
    parser.add_argument("--pair", type=str, default="BTC_USDT_USDT",
                       help="Pair to diagnose")
    parser.add_argument("--lookback-live", type=int, default=1000,
                       help="Lookback for live simulation (default: 1000)")
    parser.add_argument("--all-pairs", action="store_true",
                       help="Run diagnosis on all pairs")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BACKTEST vs LIVE DIAGNOSIS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return 1
    
    if args.all_pairs:
        # Find all available pairs
        pairs = set()
        for f in DATA_DIR.glob("*_5m.csv"):
            pair_name = f.name.replace("_5m.csv", "")
            pairs.add(pair_name)
        
        print(f"Found {len(pairs)} pairs")
        
        results = []
        for pair in sorted(list(pairs))[:10]:  # Limit to 10 pairs
            try:
                result = diagnose_pair(pair, args.lookback_live)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"\n❌ ERROR processing {pair}: {e}")
                traceback.print_exc()
                print(f"Skipping {pair} and continuing with next pair...\n")
                continue
        
        # Summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print("="*70)
        
        if results:
            bt_total = sum(r['bt_signals'] for r in results)
            lv_total = sum(r['lv_signals'] for r in results)
            
            print(f"Pairs analyzed: {len(results)}")
            print(f"Total Backtest signals: {bt_total}")
            print(f"Total Live signals:     {lv_total}")
            print(f"Difference:             {bt_total - lv_total}")
        else:
            print("No pairs were successfully analyzed.")
        
    else:
        try:
            diagnose_pair(args.pair, args.lookback_live)
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

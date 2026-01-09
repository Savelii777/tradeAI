#!/usr/bin/env python3
"""
Comprehensive Live vs Backtest Comparison Script for V10

This script directly compares:
1. Feature calculation in live_trading_v10_csv.py vs train_v3_dynamic.py
2. Raw data (CSV) being used
3. Model predictions on the same data
4. Signal generation rates

Run this to find WHERE and WHY live differs from backtest.

Usage:
    python scripts/compare_live_vs_backtest_v10.py --pair BTC/USDT:USDT
    python scripts/compare_live_vs_backtest_v10.py --all  # Check all pairs
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from loguru import logger

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
PAIRS_FILE = Path(__file__).parent.parent / 'config' / 'pairs_20.json'

# Thresholds from V10 live script
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


# ============================================================
# HELPER FUNCTIONS - COPIED FROM BOTH SCRIPTS FOR COMPARISON
# ============================================================

def add_volume_features_v10(df):
    """Add volume features - FROM live_trading_v10_csv.py"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def add_volume_features_train(df):
    """Add volume features - FROM train_v3_dynamic.py"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df, period=14):
    """Calculate ATR - FROM both scripts (should be identical)"""
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features_V10_STYLE(m1, m5, m15, mtf_fe):
    """
    Prepare features EXACTLY as in live_trading_v10_csv.py
    
    This is the function from V10 live script.
    """
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 200:
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        # Use MTFFeatureEngine - SAME as backtest
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        
        if len(ft) == 0:
            return pd.DataFrame()
        
        # Add OHLCV from M5
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        
        # Add volume features - SAME as backtest
        ft = add_volume_features_v10(ft)
        
        # Add ATR
        ft['atr'] = calculate_atr(ft)
        
        # NaN handling - SAME as backtest
        ft = ft.dropna(subset=['close', 'atr'])
        
        # Exclude cumsum-dependent features
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        # Exclude absolute price features  
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        # Forward fill and final dropna
        ft = ft.ffill().dropna()
        
        return ft
        
    except Exception as e:
        logger.error(f"Error preparing features (V10): {e}")
        return pd.DataFrame()


def prepare_features_TRAIN_STYLE(m1, m5, m15, mtf_fe):
    """
    Prepare features EXACTLY as in train_v3_dynamic.py (backtest mode)
    
    This is the function from training script.
    """
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features_train(ft)
    ft['atr'] = calculate_atr(ft)
    # Backtest: simple dropna
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


def get_pairs() -> list:
    """Get trading pairs."""
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            return [p['symbol'] for p in json.load(f)['pairs']][:20]
    return ['BTC/USDT:USDT', 'ETH/USDT:USDT']


def pair_to_filename(pair: str) -> str:
    """Convert pair to filename format."""
    return pair.replace('/', '_').replace(':', '_')


# ============================================================
# COMPARISON FUNCTIONS
# ============================================================

def compare_raw_data(pair: str) -> Dict:
    """Compare raw CSV data."""
    pair_file = pair_to_filename(pair)
    
    results = {'pair': pair, 'errors': []}
    
    for tf in ['1m', '5m', '15m']:
        csv_path = DATA_DIR / f"{pair_file}_{tf}.csv"
        if not csv_path.exists():
            results['errors'].append(f"Missing {tf} CSV file")
            continue
        
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        
        results[tf] = {
            'rows': len(df),
            'start': str(df.index[0]),
            'end': str(df.index[-1]),
            'has_duplicates': df.index.duplicated().any(),
            'has_nulls': df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any(),
            'price_valid': (df['high'] >= df['low']).all() and (df['close'] > 0).all()
        }
    
    return results


def compare_features(pair: str, mtf_fe: MTFFeatureEngine, feature_cols: list) -> Dict:
    """
    Compare features generated by V10 style vs TRAIN style.
    
    This is the KEY comparison - if these differ, that's the problem!
    """
    pair_file = pair_to_filename(pair)
    
    # Load data
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_file}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_file}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_file}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError as e:
        return {'pair': pair, 'error': str(e)}
    
    # Prepare features BOTH ways
    ft_v10 = prepare_features_V10_STYLE(m1.copy(), m5.copy(), m15.copy(), mtf_fe)
    ft_train = prepare_features_TRAIN_STYLE(m1.copy(), m5.copy(), m15.copy(), mtf_fe)
    
    if len(ft_v10) == 0 or len(ft_train) == 0:
        return {'pair': pair, 'error': 'Failed to generate features'}
    
    # Find common timestamps
    common_idx = ft_v10.index.intersection(ft_train.index)
    
    if len(common_idx) == 0:
        return {'pair': pair, 'error': 'No common timestamps'}
    
    # Compare features at common timestamps
    ft_v10_common = ft_v10.loc[common_idx]
    ft_train_common = ft_train.loc[common_idx]
    
    # Find common feature columns
    common_cols = [c for c in feature_cols if c in ft_v10_common.columns and c in ft_train_common.columns]
    missing_in_v10 = [c for c in feature_cols if c not in ft_v10_common.columns]
    missing_in_train = [c for c in feature_cols if c not in ft_train_common.columns]
    
    # Compare values
    differences = {}
    for col in common_cols:
        v10_vals = ft_v10_common[col].values
        train_vals = ft_train_common[col].values
        
        # Check if values are identical
        if np.allclose(v10_vals, train_vals, rtol=1e-5, equal_nan=True):
            continue
        
        # Calculate difference statistics
        diff = np.abs(v10_vals - train_vals)
        pct_diff = np.abs((v10_vals - train_vals) / (train_vals + 1e-10)) * 100
        
        differences[col] = {
            'mean_diff': float(np.nanmean(diff)),
            'max_diff': float(np.nanmax(diff)),
            'mean_pct_diff': float(np.nanmean(pct_diff)),
            'max_pct_diff': float(np.nanmax(pct_diff)),
        }
    
    return {
        'pair': pair,
        'common_timestamps': len(common_idx),
        'v10_rows': len(ft_v10),
        'train_rows': len(ft_train),
        'common_features': len(common_cols),
        'missing_in_v10': missing_in_v10,
        'missing_in_train': missing_in_train,
        'feature_differences': differences,
        'identical_features': len(common_cols) - len(differences)
    }


def compare_predictions(pair: str, mtf_fe: MTFFeatureEngine, models: Dict) -> Dict:
    """
    Compare model predictions using V10 vs TRAIN feature preparation.
    """
    pair_file = pair_to_filename(pair)
    feature_cols = models['features']
    
    # Load data
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_file}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_file}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_file}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError as e:
        return {'pair': pair, 'error': str(e)}
    
    # Prepare features BOTH ways
    ft_v10 = prepare_features_V10_STYLE(m1.copy(), m5.copy(), m15.copy(), mtf_fe)
    ft_train = prepare_features_TRAIN_STYLE(m1.copy(), m5.copy(), m15.copy(), mtf_fe)
    
    if len(ft_v10) == 0 or len(ft_train) == 0:
        return {'pair': pair, 'error': 'Failed to generate features'}
    
    # Get last N rows for comparison
    n_rows = min(100, len(ft_v10), len(ft_train))
    
    # Fill missing features
    for ft in [ft_v10, ft_train]:
        for f in feature_cols:
            if f not in ft.columns:
                ft[f] = 0.0
    
    # Get predictions
    X_v10 = ft_v10[feature_cols].iloc[-n_rows:].values.astype(np.float64)
    X_train = ft_train[feature_cols].iloc[-n_rows:].values.astype(np.float64)
    
    X_v10 = np.nan_to_num(X_v10, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Direction predictions
    dir_v10 = models['direction'].predict_proba(X_v10)
    dir_train = models['direction'].predict_proba(X_train)
    
    conf_v10 = np.max(dir_v10, axis=1)
    conf_train = np.max(dir_train, axis=1)
    
    pred_v10 = np.argmax(dir_v10, axis=1)
    pred_train = np.argmax(dir_train, axis=1)
    
    # Timing predictions
    timing_v10 = models['timing'].predict(X_v10)
    timing_train = models['timing'].predict(X_train)
    
    # Strength predictions
    strength_v10 = models['strength'].predict(X_v10)
    strength_train = models['strength'].predict(X_train)
    
    # Count signals
    signals_v10 = sum(
        1 for i in range(n_rows)
        if pred_v10[i] != 1  # Not sideways
        and conf_v10[i] >= MIN_CONF
        and timing_v10[i] >= MIN_TIMING
        and strength_v10[i] >= MIN_STRENGTH
    )
    
    signals_train = sum(
        1 for i in range(n_rows)
        if pred_train[i] != 1  # Not sideways
        and conf_train[i] >= MIN_CONF
        and timing_train[i] >= MIN_TIMING
        and strength_train[i] >= MIN_STRENGTH
    )
    
    return {
        'pair': pair,
        'n_rows': n_rows,
        'v10': {
            'mean_conf': float(np.mean(conf_v10)),
            'max_conf': float(np.max(conf_v10)),
            'mean_timing': float(np.mean(timing_v10)),
            'max_timing': float(np.max(timing_v10)),
            'mean_strength': float(np.mean(strength_v10)),
            'max_strength': float(np.max(strength_v10)),
            'signals': signals_v10,
            'sideways_pct': float(np.mean(pred_v10 == 1) * 100),
        },
        'train': {
            'mean_conf': float(np.mean(conf_train)),
            'max_conf': float(np.max(conf_train)),
            'mean_timing': float(np.mean(timing_train)),
            'max_timing': float(np.max(timing_train)),
            'mean_strength': float(np.mean(strength_train)),
            'max_strength': float(np.max(strength_train)),
            'signals': signals_train,
            'sideways_pct': float(np.mean(pred_train == 1) * 100),
        },
        'differences': {
            'conf_diff': float(np.mean(np.abs(conf_v10 - conf_train))),
            'timing_diff': float(np.mean(np.abs(timing_v10 - timing_train))),
            'strength_diff': float(np.mean(np.abs(strength_v10 - strength_train))),
            'direction_match_pct': float(np.mean(pred_v10 == pred_train) * 100),
        }
    }


def run_full_diagnosis(pair: str, verbose: bool = True):
    """Run full diagnosis for a single pair."""
    print(f"\n{'='*70}")
    print(f"FULL DIAGNOSIS: {pair}")
    print(f"{'='*70}")
    
    mtf_fe = MTFFeatureEngine()
    models = load_models()
    feature_cols = models['features']
    
    # 1. Raw data check
    print(f"\n{'─'*70}")
    print("1. RAW DATA CHECK")
    print(f"{'─'*70}")
    
    raw_data = compare_raw_data(pair)
    if raw_data.get('errors'):
        print(f"❌ ERRORS: {raw_data['errors']}")
    else:
        for tf in ['1m', '5m', '15m']:
            if tf in raw_data:
                d = raw_data[tf]
                status = "✅" if d['price_valid'] and not d['has_duplicates'] and not d['has_nulls'] else "⚠️"
                print(f"  {status} {tf}: {d['rows']} rows, {d['start']} to {d['end']}")
                if d['has_duplicates']:
                    print(f"     ⚠️ Has duplicate timestamps!")
                if d['has_nulls']:
                    print(f"     ⚠️ Has NULL values in OHLCV!")
    
    # 2. Feature comparison
    print(f"\n{'─'*70}")
    print("2. FEATURE COMPARISON (V10 vs TRAIN)")
    print(f"{'─'*70}")
    
    feat_cmp = compare_features(pair, mtf_fe, feature_cols)
    if feat_cmp.get('error'):
        print(f"❌ ERROR: {feat_cmp['error']}")
    else:
        print(f"  Common timestamps: {feat_cmp['common_timestamps']}")
        print(f"  V10 rows: {feat_cmp['v10_rows']}, TRAIN rows: {feat_cmp['train_rows']}")
        print(f"  Identical features: {feat_cmp['identical_features']}/{feat_cmp['common_features']}")
        
        if feat_cmp['missing_in_v10']:
            print(f"  ⚠️ Missing in V10: {feat_cmp['missing_in_v10'][:5]}...")
        if feat_cmp['missing_in_train']:
            print(f"  ⚠️ Missing in TRAIN: {feat_cmp['missing_in_train'][:5]}...")
        
        if feat_cmp['feature_differences']:
            print(f"\n  ⚠️ FEATURES WITH DIFFERENCES:")
            sorted_diffs = sorted(
                feat_cmp['feature_differences'].items(),
                key=lambda x: x[1]['max_pct_diff'],
                reverse=True
            )
            for name, diff in sorted_diffs[:10]:
                print(f"     {name}: max diff = {diff['max_pct_diff']:.2f}%")
        else:
            print(f"\n  ✅ ALL FEATURES ARE IDENTICAL!")
    
    # 3. Prediction comparison
    print(f"\n{'─'*70}")
    print("3. PREDICTION COMPARISON (Last 100 candles)")
    print(f"{'─'*70}")
    
    pred_cmp = compare_predictions(pair, mtf_fe, models)
    if pred_cmp.get('error'):
        print(f"❌ ERROR: {pred_cmp['error']}")
    else:
        print(f"\n  V10 STYLE:")
        v = pred_cmp['v10']
        print(f"     Mean Conf: {v['mean_conf']:.3f}, Max: {v['max_conf']:.3f}")
        print(f"     Mean Timing: {v['mean_timing']:.2f}, Max: {v['max_timing']:.2f}")
        print(f"     Mean Strength: {v['mean_strength']:.2f}, Max: {v['max_strength']:.2f}")
        print(f"     Sideways: {v['sideways_pct']:.1f}%")
        print(f"     Signals (meeting thresholds): {v['signals']}/{pred_cmp['n_rows']}")
        
        print(f"\n  TRAIN STYLE (BACKTEST):")
        t = pred_cmp['train']
        print(f"     Mean Conf: {t['mean_conf']:.3f}, Max: {t['max_conf']:.3f}")
        print(f"     Mean Timing: {t['mean_timing']:.2f}, Max: {t['max_timing']:.2f}")
        print(f"     Mean Strength: {t['mean_strength']:.2f}, Max: {t['max_strength']:.2f}")
        print(f"     Sideways: {t['sideways_pct']:.1f}%")
        print(f"     Signals (meeting thresholds): {t['signals']}/{pred_cmp['n_rows']}")
        
        print(f"\n  DIFFERENCES:")
        d = pred_cmp['differences']
        status = "✅" if d['direction_match_pct'] > 95 else "⚠️"
        print(f"     {status} Direction match: {d['direction_match_pct']:.1f}%")
        print(f"     Conf diff: {d['conf_diff']:.4f}")
        print(f"     Timing diff: {d['timing_diff']:.4f}")
        print(f"     Strength diff: {d['strength_diff']:.4f}")
    
    # 4. Summary
    print(f"\n{'─'*70}")
    print("4. SUMMARY")
    print(f"{'─'*70}")
    
    issues = []
    if feat_cmp.get('feature_differences'):
        issues.append(f"Feature differences found in {len(feat_cmp['feature_differences'])} features")
    if feat_cmp.get('missing_in_v10'):
        issues.append(f"Missing {len(feat_cmp['missing_in_v10'])} features in V10")
    if pred_cmp.get('differences') and pred_cmp['differences']['direction_match_pct'] < 95:
        issues.append(f"Direction predictions differ (only {pred_cmp['differences']['direction_match_pct']:.1f}% match)")
    
    if issues:
        print("  ⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("  ✅ NO ISSUES FOUND - V10 and TRAIN should produce same results")
        
        if pred_cmp.get('v10') and pred_cmp['v10']['signals'] == 0:
            print("\n  🔍 POSSIBLE REASONS FOR NO SIGNALS:")
            v = pred_cmp['v10']
            if v['max_conf'] < MIN_CONF:
                print(f"     - Max confidence ({v['max_conf']:.3f}) < threshold ({MIN_CONF})")
            if v['max_timing'] < MIN_TIMING:
                print(f"     - Max timing ({v['max_timing']:.2f}) < threshold ({MIN_TIMING})")
            if v['max_strength'] < MIN_STRENGTH:
                print(f"     - Max strength ({v['max_strength']:.2f}) < threshold ({MIN_STRENGTH})")
            if v['sideways_pct'] > 80:
                print(f"     - Model predicts SIDEWAYS {v['sideways_pct']:.1f}% of time")
            print("\n     This means the MODEL is predicting no good entry points,")
            print("     NOT a bug in the live script.")
    
    return {
        'pair': pair,
        'raw_data': raw_data,
        'features': feat_cmp,
        'predictions': pred_cmp,
        'issues': issues
    }


def main():
    parser = argparse.ArgumentParser(description='Compare Live V10 vs Backtest')
    parser.add_argument('--pair', type=str, help='Specific pair to check (e.g., BTC/USDT:USDT)')
    parser.add_argument('--all', action='store_true', help='Check all pairs')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVE V10 vs BACKTEST COMPARISON TOOL")
    print("=" * 70)
    print(f"Model: {MODEL_DIR}")
    print(f"Data: {DATA_DIR}")
    print(f"Thresholds: conf >= {MIN_CONF}, timing >= {MIN_TIMING}, strength >= {MIN_STRENGTH}")
    
    if args.all:
        pairs = get_pairs()
        print(f"\nChecking all {len(pairs)} pairs...")
        
        all_results = []
        for pair in pairs:
            try:
                result = run_full_diagnosis(pair, verbose=True)
                all_results.append(result)
            except Exception as e:
                print(f"Error with {pair}: {e}")
        
        # Summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        
        pairs_with_issues = [r for r in all_results if r.get('issues')]
        pairs_ok = [r for r in all_results if not r.get('issues')]
        
        print(f"\n✅ OK: {len(pairs_ok)} pairs")
        print(f"⚠️ Issues: {len(pairs_with_issues)} pairs")
        
        if pairs_with_issues:
            print("\nPairs with issues:")
            for r in pairs_with_issues:
                print(f"  - {r['pair']}: {', '.join(r['issues'])}")
    
    elif args.pair:
        run_full_diagnosis(args.pair)
    
    else:
        # Default: check BTC
        run_full_diagnosis('BTC/USDT:USDT')
        print("\nTo check all pairs: python scripts/compare_live_vs_backtest_v10.py --all")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Feature Distribution Comparison Script

Сравнивает распределение фичей между тренировочными данными (бектест)
и данными последних N часов (имитация лайва).

Цель: выявить фичи которые могут вести себя по-разному в лайве.

Использование:
    python scripts/compare_feature_distributions.py --pair BTC_USDT_USDT --hours 48
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import MTFFeatureEngine with error handling
try:
    from train_mtf import MTFFeatureEngine
except ImportError:
    print("Warning: Could not import MTFFeatureEngine from train_mtf")
    print("Make sure you're running from the scripts directory or project root")
    MTFFeatureEngine = None

# ============================================================
# CONFIG (can be overridden via command-line)
# ============================================================
DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'
DEFAULT_MAX_PAIRS = 10  # Maximum pairs to analyze when using --all-pairs

# Thresholds for warnings
MEAN_DIFF_THRESHOLD = 0.50    # 50% difference in mean
STD_DIFF_THRESHOLD = 0.50     # 50% difference in std
KS_PVALUE_THRESHOLD = 0.01    # KS test p-value for distribution difference


# ============================================================
# HELPERS
# ============================================================

def add_volume_features(df):
    """Add volume features matching train_v3_dynamic.py"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df, period=14):
    """Calculate ATR matching train_v3_dynamic.py"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_pair_data(pair_name: str, data_dir: Path) -> dict:
    """Load M1, M5, M15 data for a pair."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        filepath = data_dir / f"{pair_name}_{tf}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        data[tf] = df
    return data


def prepare_features(m1, m5, m15, mtf_engine):
    """Prepare features matching train_v3_dynamic.py"""
    ft = mtf_engine.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return ft


# ============================================================
# MAIN COMPARISON
# ============================================================

def compare_distributions(pair_name: str, recent_hours: int = 48, data_dir: Path = None, verbose: bool = True):
    """
    Compare feature distributions between old and recent data.
    
    Args:
        pair_name: Pair name (e.g., 'BTC_USDT_USDT')
        recent_hours: Number of recent hours to compare
        data_dir: Data directory path
        verbose: Print detailed output
        
    Returns:
        dict with comparison results
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    if MTFFeatureEngine is None:
        print("Error: MTFFeatureEngine not available. Cannot run comparison.")
        return None
    
    print(f"\n{'='*60}")
    print(f"Feature Distribution Comparison: {pair_name}")
    print(f"{'='*60}")
    
    # Load data
    try:
        data = load_pair_data(pair_name, data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    m1, m5, m15 = data['1m'], data['5m'], data['15m']
    
    print(f"Data range: {m5.index[0]} to {m5.index[-1]}")
    print(f"Total M5 candles: {len(m5)}")
    
    # Split into old and recent
    recent_candles = recent_hours * 12  # 12 M5 candles per hour
    split_point = len(m5) - recent_candles
    
    if split_point < recent_candles:
        print(f"Warning: Not enough data for comparison")
        return None
    
    # Prepare features for both periods
    mtf_engine = MTFFeatureEngine()
    
    print(f"\nPreparing features for OLD period (training-like)...")
    m1_old = m1.iloc[:split_point]
    m5_old = m5.iloc[:split_point]
    m15_old = m15[m15.index <= m5_old.index[-1]]
    
    # Take last portion of old data (not too much)
    warmup = 500
    m1_old = m1_old.iloc[-recent_candles*5 - warmup:]  # M1 has 5x more candles
    m5_old = m5_old.iloc[-recent_candles - warmup:]
    m15_old = m15_old.iloc[-recent_candles//3 - warmup//3:]
    
    ft_old = prepare_features(m1_old, m5_old, m15_old, mtf_engine)
    print(f"  Generated {len(ft_old)} rows, {len(ft_old.columns)} columns")
    
    print(f"\nPreparing features for RECENT period (live-like)...")
    m1_new = m1.iloc[split_point - warmup*5:]
    m5_new = m5.iloc[split_point - warmup:]
    m15_new = m15[m15.index >= m5_new.index[0]]
    
    ft_new = prepare_features(m1_new, m5_new, m15_new, mtf_engine)
    print(f"  Generated {len(ft_new)} rows, {len(ft_new.columns)} columns")
    
    # Compare distributions
    results = {
        'pair': pair_name,
        'old_period': f"{ft_old.index[0]} to {ft_old.index[-1]}",
        'new_period': f"{ft_new.index[0]} to {ft_new.index[-1]}",
        'features_analyzed': 0,
        'features_stable': [],
        'features_warning': [],
        'features_critical': [],
    }
    
    # Exclude non-numeric columns
    exclude = ['pair', 'open', 'high', 'low', 'close', 'volume', 'atr']
    
    # Cumsum patterns to flag as critical regardless of distribution
    cumsum_patterns = [
        'bars_since_swing', 'consecutive_up', 'consecutive_down',
        'obv', 'volume_delta_cumsum', 'swing_high_price', 'swing_low_price'
    ]
    
    feature_cols = [c for c in ft_old.columns 
                   if c not in exclude 
                   and ft_old[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    print(f"\n{'='*60}")
    print(f"Analyzing {len(feature_cols)} features...")
    print(f"{'='*60}")
    
    warnings_list = []
    criticals_list = []
    
    for col in feature_cols:
        old_vals = ft_old[col].dropna()
        new_vals = ft_new[col].dropna()
        
        if len(old_vals) < 100 or len(new_vals) < 100:
            continue
        
        results['features_analyzed'] += 1
        
        # Check if feature is in dangerous patterns
        is_cumsum = any(pattern in col.lower() for pattern in cumsum_patterns)
        
        # Calculate statistics
        old_mean = old_vals.mean()
        new_mean = new_vals.mean()
        old_std = old_vals.std()
        new_std = new_vals.std()
        
        # Mean difference (%)
        mean_diff = abs(new_mean - old_mean) / (abs(old_mean) + 1e-10)
        
        # Std difference (%)
        std_diff = abs(new_std - old_std) / (abs(old_std) + 1e-10)
        
        # KS test for distribution difference
        ks_stat, ks_pvalue = stats.ks_2samp(old_vals, new_vals)
        
        # Classify feature
        if is_cumsum:
            criticals_list.append({
                'feature': col,
                'reason': 'cumsum-dependent',
                'old_mean': old_mean,
                'new_mean': new_mean,
                'mean_diff': mean_diff,
            })
            results['features_critical'].append(col)
        elif mean_diff > MEAN_DIFF_THRESHOLD or ks_pvalue < KS_PVALUE_THRESHOLD:
            warnings_list.append({
                'feature': col,
                'old_mean': old_mean,
                'new_mean': new_mean,
                'mean_diff': mean_diff,
                'ks_pvalue': ks_pvalue,
            })
            results['features_warning'].append(col)
        else:
            results['features_stable'].append(col)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total features analyzed: {results['features_analyzed']}")
    print(f"Stable features: {len(results['features_stable'])}")
    print(f"Warning features: {len(results['features_warning'])}")
    print(f"Critical features: {len(results['features_critical'])}")
    
    if criticals_list:
        print(f"\n❌ CRITICAL FEATURES (cumsum-dependent - MUST be excluded):")
        for item in criticals_list:
            print(f"   {item['feature']}: {item['reason']}")
            print(f"      Old mean: {item['old_mean']:.4f}, New mean: {item['new_mean']:.4f}")
    
    if warnings_list:
        print(f"\n⚠️  WARNING FEATURES (distribution changed significantly):")
        for item in sorted(warnings_list, key=lambda x: x['mean_diff'], reverse=True)[:15]:
            print(f"   {item['feature']}:")
            print(f"      Mean: {item['old_mean']:.4f} → {item['new_mean']:.4f} (diff: {item['mean_diff']*100:.1f}%)")
            print(f"      KS p-value: {item['ks_pvalue']:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    critical_count = len(results['features_critical'])
    warning_count = len(results['features_warning'])
    
    if critical_count == 0 and warning_count < 10:
        print(f"✅ Features look stable! Model should work similarly in live.")
    elif critical_count > 0:
        print(f"❌ CRITICAL: {critical_count} cumsum-dependent features found!")
        print(f"   These MUST be excluded from the model.")
    elif warning_count >= 10:
        print(f"⚠️  WARNING: {warning_count} features have changed distributions.")
        print(f"   Model may behave differently in live. Monitor closely.")
    else:
        print(f"⚠️  CAUTION: Some features have changed ({warning_count}).")
        print(f"   Model should work but monitor for unexpected behavior.")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare feature distributions")
    parser.add_argument("--pair", type=str, default="BTC_USDT_USDT",
                       help="Pair name (e.g., BTC_USDT_USDT)")
    parser.add_argument("--hours", type=int, default=48,
                       help="Number of recent hours to compare")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                       help=f"Data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                       help=f"Maximum pairs to analyze when using --all-pairs (default: {DEFAULT_MAX_PAIRS})")
    parser.add_argument("--all-pairs", action="store_true",
                       help="Run comparison for all available pairs")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if args.all_pairs:
        # Find all available pairs
        pairs = set()
        for f in data_dir.glob("*_5m.csv"):
            pair_name = f.name.replace("_5m.csv", "")
            pairs.add(pair_name)
        
        print(f"Found {len(pairs)} pairs")
        
        all_results = []
        for pair in sorted(pairs)[:args.max_pairs]:
            result = compare_distributions(pair, args.hours, data_dir)
            if result:
                all_results.append(result)
        
        # Summary across all pairs
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY ({len(all_results)} pairs)")
        print(f"{'='*60}")
        
        total_critical = sum(len(r['features_critical']) for r in all_results)
        total_warning = sum(len(r['features_warning']) for r in all_results)
        
        print(f"Total critical features found: {total_critical}")
        print(f"Total warning features found: {total_warning}")
        
    else:
        compare_distributions(args.pair, args.hours, data_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

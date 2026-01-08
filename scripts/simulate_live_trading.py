#!/usr/bin/env python3
"""
Live Trading Simulation Test

Simulates live trading on the last N hours of data,
comparing with backtest on the same data.

Goal: ensure live trading logic is identical to backtest.

Usage:
    python scripts/simulate_live_trading.py --pair BTC_USDT_USDT --hours 48
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Try to import required modules
try:
    import joblib
    from train_mtf import MTFFeatureEngine
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please ensure you're in the correct directory")
    sys.exit(1)


# ============================================================
# CONFIGURATION (matching live_trading_mexc_v8.py)
# ============================================================

MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

# V8 Thresholds
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# Simulation parameters
LOOKBACK_CANDLES = 1000  # Same as live script


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features (matching train_v3_dynamic.py)."""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_data(pair: str, data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load M1, M5, M15 data for a pair."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        filepath = data_dir / f"{pair}_{tf}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        data[tf] = df
    return data


def prepare_features_like_live(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame,
                                mtf_engine: MTFFeatureEngine) -> pd.DataFrame:
    """
    Prepare features exactly like live trading script does.
    This simulates what happens when we get data from exchange.
    """
    # Simulate LOOKBACK limit (like in live)
    m1 = m1.tail(LOOKBACK_CANDLES * 5)  # M1 has 5x more candles
    m5 = m5.tail(LOOKBACK_CANDLES)
    m15 = m15.tail(LOOKBACK_CANDLES // 3)
    
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 50:
        return pd.DataFrame()
    
    # Generate features
    ft = mtf_engine.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    # Handle NaN (matching live script)
    ft = ft.replace([np.inf, -np.inf], np.nan)
    ft = ft.ffill()
    ft = ft.dropna()
    
    return ft


def prepare_features_like_backtest(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame,
                                    mtf_engine: MTFFeatureEngine) -> pd.DataFrame:
    """
    Prepare features like backtest does (using ALL data).
    """
    # Use ALL data (like backtest)
    ft = mtf_engine.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    # Handle NaN
    ft = ft.replace([np.inf, -np.inf], np.nan)
    ft = ft.ffill()
    ft = ft.dropna()
    
    return ft


def generate_signals(df: pd.DataFrame, feature_cols: List[str], models: Dict,
                     min_conf: float, min_timing: float, min_strength: float) -> List[Dict]:
    """Generate signals matching train_v3_dynamic.py logic."""
    signals = []
    
    # Check features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} missing features")
        return signals
    
    X = df[feature_cols].values
    
    # Predictions
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    
    timing_preds = models['timing'].predict(X)
    strength_preds = models['strength'].predict(X)
    
    # Filter signals
    for i in range(len(df)):
        if dir_preds[i] == 1:  # Sideways
            continue
        
        if dir_confs[i] < min_conf:
            continue
        if timing_preds[i] < min_timing:
            continue
        if strength_preds[i] < min_strength:
            continue
        
        signals.append({
            'timestamp': df.index[i],
            'direction': 'LONG' if dir_preds[i] == 2 else 'SHORT',
            'confidence': dir_confs[i],
            'timing': timing_preds[i],
            'strength': strength_preds[i],
            'close': df['close'].iloc[i],
            'atr': df['atr'].iloc[i],
        })
    
    return signals


# ============================================================
# MAIN SIMULATION
# ============================================================

def simulate_live_vs_backtest(pair: str, test_hours: int = 48, 
                               model_dir: Path = MODEL_DIR,
                               data_dir: Path = DATA_DIR) -> Dict:
    """
    Compare live-like feature generation with backtest feature generation.
    
    Returns:
        Dict with comparison results
    """
    print("\n" + "=" * 70)
    print(f"LIVE vs BACKTEST SIMULATION: {pair}")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    try:
        models = {
            'direction': joblib.load(model_dir / 'direction_model.joblib'),
            'timing': joblib.load(model_dir / 'timing_model.joblib'),
            'strength': joblib.load(model_dir / 'strength_model.joblib'),
        }
        feature_cols = joblib.load(model_dir / 'feature_names.joblib')
        print(f"  ✅ Loaded {len(feature_cols)} features")
    except Exception as e:
        print(f"  ❌ Error loading models: {e}")
        return {'error': str(e)}
    
    # Load data
    print("\nLoading data...")
    try:
        data = load_data(pair, data_dir)
        m1, m5, m15 = data['1m'], data['5m'], data['15m']
        print(f"  ✅ M1: {len(m1)}, M5: {len(m5)}, M15: {len(m15)} candles")
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return {'error': str(e)}
    
    # Define test period (last N hours)
    test_candles = test_hours * 12  # 12 M5 candles per hour
    end_time = m5.index.max()
    start_time = end_time - timedelta(hours=test_hours)
    
    print(f"\nTest period: {start_time} to {end_time}")
    print(f"Test candles: {test_candles}")
    
    # ============================================================
    # SIMULATE LIVE (LOOKBACK limited)
    # ============================================================
    print("\n" + "-" * 50)
    print("SIMULATING LIVE (LOOKBACK limited)...")
    print("-" * 50)
    
    mtf_engine = MTFFeatureEngine()
    
    # For each candle in test period, simulate live feature generation
    live_signals = []
    live_features_sample = None
    
    for i, timestamp in enumerate(m5.index[-test_candles:]):
        # Get data up to this point (simulating live)
        m1_up_to = m1[m1.index <= timestamp]
        m5_up_to = m5[m5.index <= timestamp]
        m15_up_to = m15[m15.index <= timestamp]
        
        # Apply LOOKBACK limit
        m1_limited = m1_up_to.tail(LOOKBACK_CANDLES * 5)
        m5_limited = m5_up_to.tail(LOOKBACK_CANDLES)
        m15_limited = m15_up_to.tail(LOOKBACK_CANDLES // 3)
        
        if len(m5_limited) < 200:
            continue
        
        # Generate features like live
        ft = mtf_engine.align_timeframes(m1_limited, m5_limited, m15_limited)
        if len(ft) == 0:
            continue
            
        ft = ft.join(m5_limited[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        # Handle NaN consistently with training
        ft = ft.replace([np.inf, -np.inf], np.nan)
        ft = ft.ffill()
        ft = ft.dropna()
        
        if len(ft) < 2:
            continue
        
        # Get last closed candle
        row = ft.iloc[[-2]]
        if row.index[0] != timestamp:
            continue  # Timestamp mismatch
        
        # Store sample
        if live_features_sample is None:
            live_features_sample = row[feature_cols].copy()
        
        # Check for signal
        missing = [f for f in feature_cols if f not in row.columns]
        if missing:
            continue
        
        X = row[feature_cols].values
        # Handle NaN/Inf the same way as training - ffill already applied above
        # For remaining NaN after ffill, use 0.0 as last resort
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        dir_proba = models['direction'].predict_proba(X)
        dir_conf = float(np.max(dir_proba))
        dir_pred = int(np.argmax(dir_proba))
        timing_pred = float(models['timing'].predict(X)[0])
        strength_pred = float(models['strength'].predict(X)[0])
        
        if dir_pred != 1 and dir_conf >= MIN_CONF and timing_pred >= MIN_TIMING and strength_pred >= MIN_STRENGTH:
            live_signals.append({
                'timestamp': timestamp,
                'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                'confidence': dir_conf,
                'timing': timing_pred,
                'strength': strength_pred,
            })
        
        if i % 100 == 0:
            print(f"  Processed {i}/{test_candles} candles...", end='\r')
    
    print(f"\n  ✅ LIVE simulation: {len(live_signals)} signals in {test_hours}h")
    
    # ============================================================
    # BACKTEST (full data)
    # ============================================================
    print("\n" + "-" * 50)
    print("RUNNING BACKTEST (full data)...")
    print("-" * 50)
    
    # Use ALL data up to end of test period
    m1_full = m1[m1.index <= end_time]
    m5_full = m5[m5.index <= end_time]
    m15_full = m15[m15.index <= end_time]
    
    ft_backtest = mtf_engine.align_timeframes(m1_full, m5_full, m15_full)
    ft_backtest = ft_backtest.join(m5_full[['open', 'high', 'low', 'close', 'volume']])
    ft_backtest = add_volume_features(ft_backtest)
    ft_backtest['atr'] = calculate_atr(ft_backtest)
    # Handle NaN consistently with training
    ft_backtest = ft_backtest.replace([np.inf, -np.inf], np.nan)
    ft_backtest = ft_backtest.ffill()
    ft_backtest = ft_backtest.dropna()
    
    # Filter to test period
    ft_test = ft_backtest[ft_backtest.index >= start_time]
    print(f"  Backtest features: {len(ft_test)} rows")
    
    # Generate signals
    backtest_signals = generate_signals(ft_test, feature_cols, models, 
                                         MIN_CONF, MIN_TIMING, MIN_STRENGTH)
    
    print(f"  ✅ BACKTEST: {len(backtest_signals)} signals in {test_hours}h")
    
    # ============================================================
    # COMPARISON
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    # Signal count comparison
    live_count = len(live_signals)
    backtest_count = len(backtest_signals)
    diff = abs(live_count - backtest_count)
    diff_pct = diff / max(backtest_count, 1) * 100
    
    print(f"\nSignal Count:")
    print(f"  LIVE:     {live_count}")
    print(f"  BACKTEST: {backtest_count}")
    print(f"  Diff:     {diff} ({diff_pct:.1f}%)")
    
    if diff_pct < 10:
        print("  ✅ Signal counts match well!")
    elif diff_pct < 30:
        print("  ⚠️ Signal counts differ moderately")
    else:
        print("  ❌ Signal counts differ significantly!")
    
    # Find matching signals
    matches = 0
    mismatches = 0
    
    live_timestamps = {s['timestamp'] for s in live_signals}
    backtest_timestamps = {s['timestamp'] for s in backtest_signals}
    
    common_timestamps = live_timestamps & backtest_timestamps
    only_live = live_timestamps - backtest_timestamps
    only_backtest = backtest_timestamps - live_timestamps
    
    print(f"\nSignal Overlap:")
    print(f"  Common:       {len(common_timestamps)}")
    print(f"  Only LIVE:    {len(only_live)}")
    print(f"  Only BACKTEST:{len(only_backtest)}")
    
    # Check direction match for common signals
    direction_matches = 0
    for ts in common_timestamps:
        live_sig = next(s for s in live_signals if s['timestamp'] == ts)
        bt_sig = next(s for s in backtest_signals if s['timestamp'] == ts)
        if live_sig['direction'] == bt_sig['direction']:
            direction_matches += 1
    
    if len(common_timestamps) > 0:
        direction_match_pct = direction_matches / len(common_timestamps) * 100
        print(f"  Direction Match: {direction_matches}/{len(common_timestamps)} ({direction_match_pct:.1f}%)")
    
    # Feature value comparison
    print(f"\nFeature Distribution Check:")
    if live_features_sample is not None:
        bt_features = ft_test[feature_cols].iloc[-1:] if len(ft_test) > 0 else None
        
        if bt_features is not None:
            diffs = []
            for col in feature_cols[:10]:  # Check first 10 features
                if col in live_features_sample.columns and col in bt_features.columns:
                    live_val = live_features_sample[col].iloc[0]
                    bt_val = bt_features[col].iloc[0]
                    if pd.notna(live_val) and pd.notna(bt_val) and abs(bt_val) > 1e-10:
                        diff_pct = abs(live_val - bt_val) / abs(bt_val) * 100
                        diffs.append((col, live_val, bt_val, diff_pct))
            
            for col, lv, bv, d in diffs[:5]:
                status = '✅' if d < 10 else ('⚠️' if d < 30 else '❌')
                print(f"  {status} {col[:25]}: LIVE={lv:.4f}, BT={bv:.4f} ({d:.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    is_consistent = diff_pct < 30 and (len(common_timestamps) > 0 or live_count + backtest_count == 0)
    
    if is_consistent:
        print("✅ LIVE and BACKTEST are CONSISTENT")
        print("   Model should perform similarly in live trading.")
    else:
        print("⚠️ LIVE and BACKTEST have DIFFERENCES")
        print("   This may indicate feature drift or data issues.")
        print("\n   Possible causes:")
        print("   - LOOKBACK limit affects indicator convergence")
        print("   - Data quality differences")
        print("   - Cumsum-dependent features (should be excluded)")
    
    return {
        'pair': pair,
        'test_hours': test_hours,
        'live_signals': live_count,
        'backtest_signals': backtest_count,
        'diff_pct': diff_pct,
        'common_signals': len(common_timestamps),
        'is_consistent': is_consistent,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulate live trading and compare with backtest"
    )
    parser.add_argument(
        "--pair", 
        type=str, 
        default="BTC_USDT_USDT",
        help="Pair name (e.g., BTC_USDT_USDT)"
    )
    parser.add_argument(
        "--hours", 
        type=int, 
        default=48,
        help="Number of hours to test"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=str(MODEL_DIR),
        help="Model directory"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=str(DATA_DIR),
        help="Data directory"
    )
    
    args = parser.parse_args()
    
    result = simulate_live_vs_backtest(
        pair=args.pair,
        test_hours=args.hours,
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir)
    )
    
    if result.get('error'):
        return 1
    
    return 0 if result.get('is_consistent', False) else 1


if __name__ == '__main__':
    sys.exit(main())

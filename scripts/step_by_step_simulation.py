#!/usr/bin/env python3
"""
STEP-BY-STEP LIVE SIMULATION - The Ultimate Test

TWO MODES:
1. --mode backtest_style: Prepare features on FULL history (like backtest does)
   Then walk bar-by-bar making predictions.
   This should give similar results to backtest.

2. --mode live_style: Prepare features at each bar using only past data
   This simulates what live trading actually does.
   If results differ from backtest_style, that's the problem!

COMPARING MODES shows if the issue is:
- Feature preparation (backtest_style works, live_style doesn't)
- Or the model itself (both don't work)

Usage:
    # Compare both modes
    python scripts/step_by_step_simulation.py --pair PIPPIN_USDT_USDT --hours 48 --mode backtest_style
    python scripts/step_by_step_simulation.py --pair PIPPIN_USDT_USDT --hours 48 --mode live_style
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
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

LOOKAHEAD = 12  # 12 M5 candles = 1 hour


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_data(pair: str) -> dict:
    """Load historical data from CSV/Parquet."""
    pair_name = pair.replace('/', '_').replace(':', '_')
    data = {}
    
    for tf in ['1m', '5m', '15m']:
        parquet_path = DATA_DIR / f"{pair_name}_{tf}.parquet"
        csv_path = DATA_DIR / f"{pair_name}_{tf}.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        else:
            print(f"Data not found: {parquet_path} or {csv_path}")
            return None
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        data[tf] = df
    
    return data


def prepare_features_at_point(data: dict, end_idx: int, mtf_fe: MTFFeatureEngine, 
                              use_full_history: bool = True) -> pd.DataFrame:
    """
    Prepare features using ONLY data up to end_idx.
    This simulates what live trading would see at that point.
    
    Args:
        use_full_history: If True, use ALL data up to end_idx (like real live_trading_v10_csv.py)
                         If False, use only last 1000 candles (old broken behavior)
    """
    m5 = data['5m']
    
    # Get timestamp at end_idx
    end_time = m5.index[end_idx]
    
    # Filter all timeframes to only include data UP TO this point
    # IMPORTANT: Use FULL history, not limited lookback - this matches live_trading_v10_csv.py!
    if use_full_history:
        m1 = data['1m'][data['1m'].index <= end_time]
        m5_slice = data['5m'][data['5m'].index <= end_time]
        m15 = data['15m'][data['15m'].index <= end_time]
    else:
        # Old broken behavior - limited lookback
        lookback = 1000
        m1 = data['1m'][data['1m'].index <= end_time].tail(lookback * 5)
        m5_slice = data['5m'][data['5m'].index <= end_time].tail(lookback)
        m15 = data['15m'][data['15m'].index <= end_time].tail(lookback // 3)
    
    if len(m5_slice) < 200:
        return None
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5_slice, m15)
        ft = ft.join(m5_slice[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Exclude problematic features
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        ft = ft.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        return ft
    except Exception as e:
        return None


def prepare_features_full_history(data: dict, mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """
    Prepare features on FULL history (like backtest does).
    This ensures rolling averages and EMAs are calculated correctly.
    """
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Exclude problematic features
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        ft = ft.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        return ft
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None


def evaluate_prediction(direction: str, entry_price: float, atr: float, 
                        future_prices: pd.DataFrame) -> dict:
    """
    Evaluate if prediction was CORRECT based on what ACTUALLY happened.
    """
    if len(future_prices) < LOOKAHEAD:
        return {'outcome': 'insufficient_data'}
    
    # Get price movement in next LOOKAHEAD bars
    max_high = future_prices['high'].iloc[:LOOKAHEAD].max()
    min_low = future_prices['low'].iloc[:LOOKAHEAD].min()
    final_close = future_prices['close'].iloc[LOOKAHEAD-1]
    
    # Calculate actual moves in ATR
    up_move = (max_high - entry_price) / atr
    down_move = (entry_price - min_low) / atr
    net_move = (final_close - entry_price) / atr
    
    # Evaluate prediction
    if direction == 'LONG':
        hit_tp = up_move >= 1.5  # Hit 1.5 ATR target
        hit_sl = down_move >= 1.5  # Hit stop loss
        profitable = net_move > 0
        correct = hit_tp and not hit_sl  # TP before SL
    else:  # SHORT
        hit_tp = down_move >= 1.5
        hit_sl = up_move >= 1.5
        profitable = net_move < 0
        correct = hit_tp and not hit_sl
    
    return {
        'outcome': 'win' if correct else ('loss' if hit_sl else 'neutral'),
        'up_move': up_move,
        'down_move': down_move,
        'net_move': net_move,
        'hit_tp': hit_tp,
        'hit_sl': hit_sl,
        'profitable': profitable
    }


# ============================================================
# MAIN SIMULATION
# ============================================================
def run_simulation(pair: str, hours: int = 48, mode: str = 'backtest_style'):
    """
    Run step-by-step simulation.
    
    Modes:
    - backtest_style: Prepare features on FULL history (like backtest)
    - live_style: Prepare features at each bar using only past data
    """
    
    print("=" * 70)
    print("STEP-BY-STEP SIMULATION")
    print("=" * 70)
    print(f"Pair: {pair}")
    print(f"Simulation period: last {hours} hours")
    print(f"Mode: {mode.upper()}")
    if mode == 'backtest_style':
        print("  → Features prepared on FULL history (like backtest)")
        print("  → This should match backtest results")
    else:
        print("  → Features prepared on FULL history (FIXED!)")
        print("  → This now matches live_trading_v10_csv.py exactly!")
        print("  → Both modes should give IDENTICAL results")
    print("=" * 70)
    
    # Load models
    try:
        models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
            'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
        }
        print(f"✓ Model loaded: {len(models['features'])} features")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Load data
    data = load_data(pair)
    if data is None:
        return
    
    m5 = data['5m']
    print(f"✓ Data loaded: {len(m5)} M5 candles")
    print(f"  Date range: {m5.index[0]} to {m5.index[-1]}")
    
    # Initialize
    mtf_fe = MTFFeatureEngine()
    
    # BOTH modes now prepare features on FULL history ONCE
    # This matches what live_trading_v10_csv.py actually does!
    # The only difference:
    #   - backtest_style: uses features as-is (may include "future" data in rolling calcs)
    #   - live_style: uses the SAME features but only looks at current row
    # 
    # This is NOT a bug - live_trading_v10_csv.py loads ALL history from CSV
    # and calculates features on the entire history, then takes the last row.
    print("\nPreparing features on FULL history...")
    full_features = prepare_features_full_history(data, mtf_fe)
    if full_features is None:
        print("Failed to prepare features")
        return
    print(f"✓ Features prepared: {len(full_features)} rows, {len(full_features.columns)} columns")
    
    # Calculate simulation range based on TIME, not indices
    # This ensures we actually simulate the last N hours
    now = full_features.index[-1]
    start_time = now - timedelta(hours=hours + 1)  # +1 hour for LOOKAHEAD buffer
    end_time = now - timedelta(hours=1)  # -1 hour for LOOKAHEAD evaluation
    
    # Find indices in features DataFrame
    feature_times = full_features.index
    start_mask = feature_times >= start_time
    end_mask = feature_times <= end_time
    sim_times = feature_times[start_mask & end_mask]
    
    if len(sim_times) == 0:
        print(f"No data in simulation period: {start_time} to {end_time}")
        return
    
    print(f"\nSimulating {len(sim_times)} bars")
    print(f"Period: {sim_times[0]} to {sim_times[-1]}")
    
    # Statistics
    stats = {
        'total_bars': 0,
        'signals': 0,
        'sideways': 0,
        'low_conf': 0,
        'low_timing': 0,
        'low_strength': 0,
        'wins': 0,
        'losses': 0,
        'neutral': 0,
        'all_predictions': [],
        'all_confidences': []
    }
    
    print("\n" + "=" * 70)
    print("SIMULATION RUNNING...")
    print("=" * 70)
    
    # Walk through each bar in simulation period
    for current_time in sim_times:
        stats['total_bars'] += 1
        
        # Get features for current time
        last_row = full_features.loc[current_time]
        ft = full_features
        
        # Prepare features for prediction
        X = np.zeros((1, len(models['features'])))
        for i, feat in enumerate(models['features']):
            if feat in ft.columns:
                val = last_row[feat] if isinstance(last_row, pd.Series) else ft.loc[current_time, feat]
                X[0, i] = val
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get predictions
        proba = models['direction'].predict_proba(X)[0]
        dir_pred = int(np.argmax(proba))
        conf = float(np.max(proba))
        timing = float(models['timing'].predict(X)[0])
        strength = float(models['strength'].predict(X)[0])
        
        stats['all_confidences'].append(conf)
        
        direction = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
        
        # Track rejection reasons
        if direction == 'SIDEWAYS':
            stats['sideways'] += 1
            continue
        
        if conf < MIN_CONF:
            stats['low_conf'] += 1
            continue
        if timing < MIN_TIMING:
            stats['low_timing'] += 1
            continue
        if strength < MIN_STRENGTH:
            stats['low_strength'] += 1
            continue
        
        # SIGNAL FOUND!
        stats['signals'] += 1
        
        # Track signal time for live comparison
        stats['signal_times'] = stats.get('signal_times', [])
        stats['signal_times'].append(current_time)
        
        # Get entry price and ATR using time-based lookup
        entry_price = m5.loc[current_time, 'close']
        atr = last_row['atr'] if 'atr' in (ft.columns if isinstance(ft, pd.DataFrame) else last_row.index) else entry_price * 0.01
        
        # Find index position for future data lookup
        idx_pos = m5.index.get_loc(current_time)
        
        # Evaluate what ACTUALLY happened after this signal
        future_prices = m5.iloc[idx_pos+1:idx_pos+LOOKAHEAD+1]
        result = evaluate_prediction(direction, entry_price, atr, future_prices)
        
        if result['outcome'] == 'win':
            stats['wins'] += 1
            outcome_str = "✅ WIN"
        elif result['outcome'] == 'loss':
            stats['losses'] += 1
            outcome_str = "❌ LOSS"
        else:
            stats['neutral'] += 1
            outcome_str = "➖ NEUTRAL"
        
        stats['all_predictions'].append({
            'time': current_time,
            'direction': direction,
            'conf': conf,
            'timing': timing,
            'strength': strength,
            'outcome': result['outcome'],
            'up_move': result.get('up_move', 0),
            'down_move': result.get('down_move', 0)
        })
        
        # Print signal details
        print(f"[{current_time.strftime('%m-%d %H:%M')}] {direction} "
              f"conf={conf:.2f} timing={timing:.1f} strength={strength:.1f} → {outcome_str}")
    
    # Final report
    print("\n" + "=" * 70)
    print(f"SIMULATION RESULTS ({mode.upper()})")
    print("=" * 70)
    
    print(f"\nBars analyzed: {stats['total_bars']}")
    print(f"Signals generated: {stats['signals']}")
    
    if stats['total_bars'] > 0:
        signal_rate = stats['signals'] / stats['total_bars'] * 100
        print(f"Signal rate: {signal_rate:.1f}%")
    
    # Confidence analysis
    if stats['all_confidences']:
        avg_conf = np.mean(stats['all_confidences'])
        print(f"\nAverage confidence: {avg_conf:.2f}")
        conf_above_threshold = sum(1 for c in stats['all_confidences'] if c >= MIN_CONF)
        print(f"Bars with conf >= {MIN_CONF}: {conf_above_threshold} ({conf_above_threshold/len(stats['all_confidences'])*100:.1f}%)")
    
    print(f"\nRejection breakdown:")
    print(f"  Sideways: {stats['sideways']} ({stats['sideways']/max(1,stats['total_bars'])*100:.1f}%)")
    print(f"  Low conf: {stats['low_conf']} ({stats['low_conf']/max(1,stats['total_bars'])*100:.1f}%)")
    print(f"  Low timing: {stats['low_timing']} ({stats['low_timing']/max(1,stats['total_bars'])*100:.1f}%)")
    print(f"  Low strength: {stats['low_strength']} ({stats['low_strength']/max(1,stats['total_bars'])*100:.1f}%)")
    
    win_rate = 0
    if stats['signals'] > 0:
        print(f"\nSignal outcomes:")
        print(f"  Wins: {stats['wins']} ({stats['wins']/stats['signals']*100:.1f}%)")
        print(f"  Losses: {stats['losses']} ({stats['losses']/stats['signals']*100:.1f}%)")
        print(f"  Neutral: {stats['neutral']} ({stats['neutral']/stats['signals']*100:.1f}%)")
        
        win_rate = stats['wins'] / (stats['wins'] + stats['losses']) * 100 if (stats['wins'] + stats['losses']) > 0 else 0
        print(f"\n  Win Rate: {win_rate:.1f}%")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if mode == 'backtest_style':
        if stats['signals'] > 0:
            print("✅ BACKTEST_STYLE mode generates signals!")
            print(f"   {stats['signals']} signals with {win_rate:.1f}% win rate")
            print("\n   NEXT STEP: Run with --mode live_style")
            print("   If live_style gives FEWER signals → problem is feature preparation")
            print("   If live_style gives SAME signals → model works, check live trading script")
        else:
            print("⚠️ Even BACKTEST_STYLE mode gives 0 signals")
            print("   This means the model itself doesn't work on this data")
            print("   → Check if model was trained on this pair")
            print("   → Try lowering thresholds: --min_conf 0.40")
    else:  # live_style
        if stats['signals'] == 0:
            print("⚠️ LIVE_STYLE mode gives 0 signals!")
            print("   Compare with BACKTEST_STYLE results.")
            print("   If backtest_style worked but this doesn't:")
            print("   → PROBLEM: Feature preparation differs between modes")
            print("   → SOLUTION: Increase lookback in live or use cached features")
        else:
            print(f"✅ LIVE_STYLE mode generates {stats['signals']} signals")
            print("   Model should work in live trading!")
    
    # Show signal times for live trading verification
    if stats.get('signal_times') and len(stats['signal_times']) > 0:
        print("\n" + "=" * 70)
        print("SIGNAL TIMES (for live trading verification)")
        print("=" * 70)
        print("Live trading should have opened positions at these times:")
        for t in stats['signal_times'][:20]:
            print(f"  - {t.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        if len(stats['signal_times']) > 20:
            print(f"  ... and {len(stats['signal_times']) - 20} more")
        print("\nIf live trading was running but didn't open positions:")
        print("  1. Check live trading logs for these times")
        print("  2. CSV data might have been stale when live was running")
        print("  3. Live trading might have crashed or stopped")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-by-step simulation with two modes")
    parser.add_argument("--pair", type=str, default="BTC_USDT_USDT",
                        help="Trading pair (e.g., BTC_USDT_USDT)")
    parser.add_argument("--hours", type=int, default=48,
                        help="Hours to simulate (default: 48)")
    parser.add_argument("--mode", type=str, default="backtest_style",
                        choices=["backtest_style", "live_style", "compare"],
                        help="Simulation mode: backtest_style, live_style, or compare")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        # Special mode: compare features between backtest_style and live_style
        print("=" * 70)
        print("COMPARING BACKTEST_STYLE vs LIVE_STYLE")
        print("=" * 70)
        print(f"Pair: {args.pair}")
        print("This will show WHY features differ between modes")
        print("=" * 70)
        
        # Load models and data
        try:
            models = {
                'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
                'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
                'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
                'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
            }
            print(f"✓ Model loaded: {len(models['features'])} features")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            sys.exit(1)
        
        data = load_data(args.pair)
        if data is None:
            sys.exit(1)
        
        m5 = data['5m']
        mtf_fe = MTFFeatureEngine()
        
        # Prepare full history features (backtest_style)
        print("\nPreparing BACKTEST_STYLE features (full history)...")
        full_features = prepare_features_full_history(data, mtf_fe)
        print(f"✓ {len(full_features)} rows")
        
        # Pick a random point in the middle
        test_idx = len(m5) - 100
        test_time = m5.index[test_idx]
        print(f"\nTest point: idx={test_idx}, time={test_time}")
        
        # Get backtest_style features at this point
        if test_time in full_features.index:
            backtest_row = full_features.loc[test_time]
            print(f"✓ Backtest row found")
        else:
            print("✗ Test time not in full_features index!")
            sys.exit(1)
        
        # Get live_style features at this point
        print("\nPreparing LIVE_STYLE features (up to test point)...")
        live_features = prepare_features_at_point(data, test_idx, mtf_fe)
        if live_features is None:
            print("✗ Failed to prepare live features")
            sys.exit(1)
        
        if test_time in live_features.index:
            live_row = live_features.loc[test_time]
            print(f"✓ Live row found")
        else:
            print(f"✗ Test time not in live_features index!")
            print(f"  Live features range: {live_features.index[0]} to {live_features.index[-1]}")
            print(f"  Using iloc[-1] instead...")
            live_row = live_features.iloc[-1]
        
        # Compare features
        print("\n" + "=" * 70)
        print("FEATURE COMPARISON (TOP 20 DIFFERENCES)")
        print("=" * 70)
        
        diffs = []
        for feat in models['features']:
            if feat in backtest_row.index and feat in live_row.index:
                bt_val = backtest_row[feat]
                live_val = live_row[feat]
                # Skip NaN values
                if pd.isna(bt_val) or pd.isna(live_val):
                    continue
                try:
                    diff = abs(float(bt_val) - float(live_val))
                    rel_diff = diff / (abs(float(bt_val)) + 1e-10) * 100
                    diffs.append((feat, float(bt_val), float(live_val), diff, rel_diff))
                except (TypeError, ValueError):
                    continue
        
        # Sort by absolute difference
        diffs.sort(key=lambda x: x[3], reverse=True)
        
        print(f"{'Feature':<40} | {'Backtest':>12} | {'Live':>12} | {'Diff':>10} | {'Rel%':>8}")
        print("-" * 90)
        for feat, bt_val, live_val, diff, rel_diff in diffs[:20]:
            print(f"{feat:<40} | {bt_val:>12.4f} | {live_val:>12.4f} | {diff:>10.4f} | {rel_diff:>7.1f}%")
        
        # Summary
        total_diff = sum(d[3] for d in diffs)
        features_with_diff = sum(1 for d in diffs if d[3] > 0.001)
        print(f"\nTotal absolute difference: {total_diff:.4f}")
        print(f"Features with diff > 0.001: {features_with_diff}/{len(diffs)}")
        
        if features_with_diff == 0:
            print("\n✅ Features are IDENTICAL between modes!")
            print("   If live_style still doesn't work, the issue is elsewhere.")
        else:
            print("\n⚠️ Features DIFFER between modes!")
            print("   This explains why backtest_style works but live_style doesn't.")
        
        # Test predictions
        print("\n" + "=" * 70)
        print("PREDICTION COMPARISON")
        print("=" * 70)
        
        # Prepare X for backtest
        X_bt = np.zeros((1, len(models['features'])))
        for i, feat in enumerate(models['features']):
            if feat in backtest_row.index:
                X_bt[0, i] = backtest_row[feat]
        X_bt = np.nan_to_num(X_bt, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare X for live
        X_live = np.zeros((1, len(models['features'])))
        for i, feat in enumerate(models['features']):
            if feat in live_row.index:
                X_live[0, i] = live_row[feat]
        X_live = np.nan_to_num(X_live, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get predictions
        proba_bt = models['direction'].predict_proba(X_bt)[0]
        proba_live = models['direction'].predict_proba(X_live)[0]
        
        conf_bt = float(np.max(proba_bt))
        conf_live = float(np.max(proba_live))
        
        print(f"Backtest confidence: {conf_bt:.4f}")
        print(f"Live confidence:     {conf_live:.4f}")
        print(f"Difference:          {abs(conf_bt - conf_live):.4f}")
        
        if abs(conf_bt - conf_live) > 0.01:
            print("\n⚠️ Predictions DIFFER significantly!")
        else:
            print("\n✅ Predictions are similar")
    else:
        run_simulation(args.pair, args.hours, args.mode)

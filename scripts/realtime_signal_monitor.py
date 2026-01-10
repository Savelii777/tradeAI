#!/usr/bin/env python3
"""
REALTIME SIGNAL MONITOR - Watch for signals on LIVE data

This script fetches LIVE data from Binance every 5 minutes and shows:
1. What the model predicts RIGHT NOW
2. Why signals are being skipped
3. Comparison with what backtest would predict

Usage:
    python scripts/realtime_signal_monitor.py
    python scripts/realtime_signal_monitor.py --pairs BTC_USDT_USDT,ETH_USDT_USDT
    python scripts/realtime_signal_monitor.py --duration 60  # Run for 60 minutes

This helps identify:
- If the model is working at all
- Why signals have low confidence
- If there's a mismatch between backtest and live
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import ccxt

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

DEFAULT_PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", 
    "DOGE/USDT:USDT", "XRP/USDT:USDT"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features."""
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
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def fetch_live_data(binance: ccxt.Exchange, pair: str) -> dict:
    """Fetch live data from Binance for all timeframes."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        try:
            candles = binance.fetch_ohlcv(pair, tf, limit=1000)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        except Exception as e:
            print(f"  Error fetching {tf}: {e}")
            return None
    return data


def prepare_features(data: dict, mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Prepare features from live data."""
    try:
        ft = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
        ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
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
        print(f"  Error preparing features: {e}")
        return None


def analyze_prediction(proba: np.ndarray, timing: float, strength: float) -> dict:
    """Analyze why a prediction passes or fails thresholds."""
    dir_pred = int(np.argmax(proba))
    conf = float(np.max(proba))
    
    reasons = []
    if dir_pred == 1:
        reasons.append("SIDEWAYS prediction")
    else:
        if conf < MIN_CONF:
            reasons.append(f"Low conf: {conf:.2f} < {MIN_CONF}")
        if timing < MIN_TIMING:
            reasons.append(f"Low timing: {timing:.2f} < {MIN_TIMING}")
        if strength < MIN_STRENGTH:
            reasons.append(f"Low strength: {strength:.2f} < {MIN_STRENGTH}")
    
    is_signal = dir_pred != 1 and conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH
    
    return {
        'direction': 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS'),
        'conf': conf,
        'timing': timing,
        'strength': strength,
        'proba': proba,
        'is_signal': is_signal,
        'reasons': reasons
    }


def load_csv_data(pair: str) -> dict:
    """Load data from CSV files (for comparison)."""
    pair_name = pair.replace('/', '_').replace(':', '_')
    data = {}
    
    for tf in ['1m', '5m', '15m']:
        csv_path = DATA_DIR / f"{pair_name}_{tf}.csv"
        parquet_path = DATA_DIR / f"{pair_name}_{tf}.parquet"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        else:
            return None
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        data[tf] = df
    
    return data


# ============================================================
# MAIN MONITORING LOOP
# ============================================================
def run_monitor(pairs: list, duration_minutes: int = 30):
    """Run the signal monitor."""
    
    print("=" * 70)
    print("REALTIME SIGNAL MONITOR")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Thresholds: Conf>={MIN_CONF}, Timing>={MIN_TIMING}, Strength>={MIN_STRENGTH}")
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
    
    # Initialize
    binance = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    iteration = 0
    
    # Statistics
    stats = {pair: {'scans': 0, 'signals': 0, 'low_conf': 0, 'low_timing': 0, 'low_strength': 0, 'sideways': 0} 
             for pair in pairs}
    
    print("\n" + "=" * 70)
    print("MONITORING STARTED - Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        while time.time() < end_time:
            iteration += 1
            now = datetime.now(timezone.utc)
            
            print(f"\n[{now.strftime('%H:%M:%S')}] === Scan #{iteration} ===")
            
            signals_found = []
            
            for pair in pairs:
                try:
                    # Fetch LIVE data
                    data = fetch_live_data(binance, pair)
                    if data is None:
                        continue
                    
                    # Prepare features
                    ft = prepare_features(data, mtf_fe)
                    if ft is None or len(ft) < 10:
                        continue
                    
                    # Get last closed candle (same as live trading)
                    last_row = ft.iloc[-2]
                    
                    # Prepare features for prediction
                    X = np.zeros((1, len(models['features'])))
                    for i, feat in enumerate(models['features']):
                        if feat in ft.columns:
                            X[0, i] = last_row[feat]
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Get predictions
                    proba = models['direction'].predict_proba(X)[0]
                    timing = float(models['timing'].predict(X)[0])
                    strength = float(models['strength'].predict(X)[0])
                    
                    # Analyze
                    result = analyze_prediction(proba, timing, strength)
                    stats[pair]['scans'] += 1
                    
                    # Update stats
                    if result['direction'] == 'SIDEWAYS':
                        stats[pair]['sideways'] += 1
                    if result['conf'] < MIN_CONF:
                        stats[pair]['low_conf'] += 1
                    if timing < MIN_TIMING:
                        stats[pair]['low_timing'] += 1
                    if strength < MIN_STRENGTH:
                        stats[pair]['low_strength'] += 1
                    
                    # Print result
                    pair_short = pair.split('/')[0]
                    if result['is_signal']:
                        stats[pair]['signals'] += 1
                        signals_found.append(pair)
                        print(f"  🔔 {pair_short}: {result['direction']} conf={result['conf']:.2f} "
                              f"timing={result['timing']:.2f} strength={result['strength']:.2f} ← SIGNAL!")
                    else:
                        print(f"  ⏸️  {pair_short}: {result['direction']} conf={result['conf']:.2f} "
                              f"timing={result['timing']:.2f} strength={result['strength']:.2f} "
                              f"[{', '.join(result['reasons'])}]")
                    
                except Exception as e:
                    print(f"  ❌ {pair}: Error - {e}")
            
            # Summary for this scan
            if signals_found:
                print(f"\n  >>> {len(signals_found)} SIGNALS: {', '.join(signals_found)}")
            else:
                print(f"\n  >>> No signals this scan")
            
            # Wait 5 minutes (M5 candle period)
            remaining = end_time - time.time()
            if remaining > 0:
                wait_time = min(300, remaining)  # 5 minutes or less
                print(f"\n  Waiting {wait_time/60:.1f} minutes for next candle...")
                time.sleep(wait_time)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    
    total_scans = 0
    total_signals = 0
    
    for pair, s in stats.items():
        if s['scans'] > 0:
            pair_short = pair.split('/')[0]
            signal_rate = s['signals'] / s['scans'] * 100
            print(f"{pair_short:10s}: {s['signals']:3d}/{s['scans']:3d} signals ({signal_rate:5.1f}%) | "
                  f"low_conf={s['low_conf']} low_timing={s['low_timing']} low_strength={s['low_strength']} "
                  f"sideways={s['sideways']}")
            total_scans += s['scans']
            total_signals += s['signals']
    
    print("-" * 70)
    if total_scans > 0:
        overall_rate = total_signals / total_scans * 100
        print(f"TOTAL: {total_signals}/{total_scans} signals ({overall_rate:.1f}%)")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    # Analyze main problem
    total_low_conf = sum(s['low_conf'] for s in stats.values())
    total_low_timing = sum(s['low_timing'] for s in stats.values())
    total_low_strength = sum(s['low_strength'] for s in stats.values())
    total_sideways = sum(s['sideways'] for s in stats.values())
    
    if total_scans > 0:
        if total_low_conf / total_scans > 0.5:
            print("⚠️  MAIN PROBLEM: Low confidence (>50% of scans)")
            print("   Model is not confident about direction.")
            print("   Possible causes:")
            print("   1. Current market conditions differ from training data")
            print("   2. Model is too conservative (try lower MIN_CONF=0.45)")
            print("   3. Market is genuinely ranging/sideways")
        
        if total_sideways / total_scans > 0.3:
            print("⚠️  Many SIDEWAYS predictions (>30%)")
            print("   Model thinks market is not trending.")
            print("   This may be correct if market is ranging.")
        
        if total_signals == 0:
            print("⚠️  NO SIGNALS during monitoring period!")
            print("   This suggests:")
            print("   1. Thresholds are too strict")
            print("   2. Current market doesn't match training conditions")
            print("   3. Model needs retraining on recent data")
        else:
            print(f"✅ {total_signals} signals generated during monitoring")
            print("   Model is working, signals are just rare")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Signal Monitor")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated list of pairs (e.g., BTC/USDT:USDT,ETH/USDT:USDT)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration in minutes (default: 30)")
    
    args = parser.parse_args()
    
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(',')]
    else:
        pairs = DEFAULT_PAIRS
    
    run_monitor(pairs, args.duration)

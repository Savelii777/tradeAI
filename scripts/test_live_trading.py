#!/usr/bin/env python3
"""
QUICK TEST: Проверка live_trading_v10_csv.py без ожидания

Этот скрипт делает ОДИН скан всех пар и показывает результаты.
Это позволяет быстро проверить работает ли live trading.

Usage:
    python scripts/test_live_trading.py
    python scripts/test_live_trading.py --pair BTC/USDT:USDT
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import joblib
import ccxt
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES


# ============================================================
# CONFIG (same as live_trading_v10_csv.py)
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_20.json"

MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


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
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_data(pair: str) -> dict:
    """Load data from CSV/Parquet files."""
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
            return None
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        data[tf] = df
    
    return data


def prepare_features(data: dict, mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Prepare features on FULL history - exactly like live_trading."""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 200:
        return None
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return None
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        ft = ft.dropna(subset=['close', 'atr'])
        
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        ft = ft.ffill().dropna()
        return ft
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None


def get_pairs() -> list:
    """Get trading pairs."""
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            return [p['symbol'] for p in json.load(f)['pairs']][:20]
    return ['BTC/USDT:USDT', 'ETH/USDT:USDT']


def test_pair(pair: str, models: dict, mtf_fe: MTFFeatureEngine) -> dict:
    """Test a single pair."""
    data = load_data(pair)
    if data is None:
        return {'pair': pair, 'error': 'No data'}
    
    # Check data freshness
    m5 = data['5m']
    last_candle_time = m5.index[-1]
    now = datetime.now(timezone.utc)
    age_minutes = (now - last_candle_time).total_seconds() / 60
    
    # Prepare features on FULL history
    df = prepare_features(data, mtf_fe)
    if df is None or len(df) < 2:
        return {'pair': pair, 'error': 'Could not prepare features'}
    
    # Get last closed candle (same as live_trading)
    row = df.iloc[[-2]].copy()
    
    # Fill missing features
    for f in models['features']:
        if f not in row.columns:
            row[f] = 0.0
    
    # Get predictions
    X = row[models['features']].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba = models['direction'].predict_proba(X)
    dir_conf = float(np.max(dir_proba))
    dir_pred = int(np.argmax(dir_proba))
    timing = float(models['timing'].predict(X)[0])
    strength = float(models['strength'].predict(X)[0])
    
    direction = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
    
    is_signal = False
    if dir_pred != 1:
        if dir_conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH:
            is_signal = True
    
    return {
        'pair': pair,
        'direction': direction,
        'conf': dir_conf,
        'timing': timing,
        'strength': strength,
        'is_signal': is_signal,
        'data_age_minutes': age_minutes,
        'last_candle': last_candle_time.strftime('%Y-%m-%d %H:%M'),
        'features_rows': len(df)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick test of live trading")
    parser.add_argument("--pair", type=str, default=None, help="Test specific pair")
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUICK LIVE TRADING TEST")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    
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
    
    mtf_fe = MTFFeatureEngine()
    
    # Get pairs
    if args.pair:
        pairs = [args.pair]
    else:
        pairs = get_pairs()
    
    print(f"✓ Testing {len(pairs)} pairs")
    print()
    
    # Test each pair
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Pair':<20} | {'Dir':>8} | {'Conf':>6} | {'Tim':>6} | {'Str':>6} | {'Age':>6} | Signal")
    print("-" * 80)
    
    signals = []
    errors = []
    
    for pair in pairs:
        result = test_pair(pair, models, mtf_fe)
        
        if 'error' in result:
            print(f"{pair:<20} | {'ERROR':>8} | {result['error']}")
            errors.append(pair)
            continue
        
        status = "✅ YES!" if result['is_signal'] else ""
        age = f"{result['data_age_minutes']:.0f}m"
        
        print(f"{pair:<20} | {result['direction']:>8} | {result['conf']:>6.2f} | "
              f"{result['timing']:>6.2f} | {result['strength']:>6.1f} | {age:>6} | {status}")
        
        if result['is_signal']:
            signals.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"Total pairs tested: {len(pairs)}")
    print(f"Errors: {len(errors)}")
    print(f"Signals found: {len(signals)}")
    
    if signals:
        print("\n🎯 SIGNALS:")
        for s in signals:
            print(f"  {s['pair']}: {s['direction']} conf={s['conf']:.2f} tim={s['timing']:.2f} str={s['strength']:.1f}")
    else:
        print("\n⚠️ No signals found!")
        print("   This is NORMAL if:")
        print("   - Model confidence is below threshold for current market conditions")
        print("   - Market is sideways (no clear trend)")
        print()
        print("   To see more signals, you can temporarily:")
        print("   - Lower MIN_CONF in live_trading_v10_csv.py")
        print("   - Or wait for stronger market moves")
    
    # Check data freshness
    print("\n" + "=" * 70)
    print("DATA FRESHNESS CHECK")
    print("=" * 70)
    
    old_data_pairs = []
    for pair in pairs:
        if pair in errors:
            continue
        result = test_pair(pair, models, mtf_fe)
        if 'error' not in result and result['data_age_minutes'] > 30:
            old_data_pairs.append((pair, result['data_age_minutes']))
    
    if old_data_pairs:
        print("⚠️ These pairs have OLD data (>30 min):")
        for pair, age in old_data_pairs:
            print(f"  {pair}: {age:.0f} minutes old")
        print("\n   To fix, either:")
        print("   1. Run the data collection script to update CSV files")
        print("   2. Let live_trading_v10_csv.py fetch new candles automatically")
    else:
        print("✅ All data is fresh!")


if __name__ == '__main__':
    main()

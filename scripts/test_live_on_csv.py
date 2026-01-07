#!/usr/bin/env python3
"""
TEST: Run Live Scanner Logic on CSV Data

This test verifies that live scanner produces the SAME signals as backtest
when running on the SAME data (CSV files).

If signals match → Problem is API vs CSV data difference
If signals differ → Problem is in live scanner code
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG (SAME AS LIVE SCANNER)
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_list.json"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

# Thresholds (SAME AS LIVE SCANNER)
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


# ============================================================
# FEATURE FUNCTIONS (COPY FROM LIVE SCANNER)
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_csv_data(pair: str) -> dict:
    """Load CSV data for a pair (same as backtest does)"""
    # BTC/USDT:USDT -> BTC_USDT_USDT
    pair_clean = pair.replace('/', '_').replace(':', '_')
    
    data = {}
    for tf in ['1m', '5m', '15m']:
        csv_path = DATA_DIR / f"{pair_clean}_{tf}.csv"
        if not csv_path.exists():
            print(f"    Missing: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        data[tf] = df
    
    return data


# ============================================================
# MAIN TEST
# ============================================================
def main():
    print("=" * 70)
    print("TEST: Live Scanner Logic on CSV Data")
    print("=" * 70)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    print(f"Model features: {len(features)}")
    
    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:20]]
    
    print(f"Testing {len(pairs)} pairs")
    print(f"Thresholds: CONF>={MIN_CONF}, TIMING>={MIN_TIMING}, STRENGTH>={MIN_STRENGTH}")
    
    mtf_fe = MTFFeatureEngine()
    
    # Test period: last 7 days of data
    all_signals = []
    
    for pair in pairs:
        print(f"\n{'='*50}")
        print(f"Pair: {pair}")
        print(f"{'='*50}")
        
        data = load_csv_data(pair)
        if data is None:
            print(f"  ✗ No CSV data found")
            continue
        
        m1 = data['1m']
        m5 = data['5m']
        m15 = data['15m']
        
        print(f"  Data: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
        print(f"  Range: {m5.index[0]} → {m5.index[-1]}")
        
        # Build features (SAME AS LIVE SCANNER)
        try:
            ft = mtf_fe.align_timeframes(m1, m5, m15)
            ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
        except Exception as e:
            print(f"  ✗ Error building features: {e}")
            continue
        
        print(f"  Features: {len(ft.columns)} cols, {len(ft)} rows")
        
        # Fill missing features with 0 (SAME AS LIVE SCANNER)
        for f in features:
            if f not in ft.columns:
                ft[f] = 0.0
        
        # Scan last 7 days (2016 5-minute candles)
        test_period = min(2016, len(ft) - 1)
        start_idx = len(ft) - test_period
        
        pair_signals = []
        direction_counts = {'LONG': 0, 'SHORT': 0, 'SIDEWAYS': 0}
        passed_filters = 0
        
        for i in range(start_idx, len(ft) - 1):
            row = ft.iloc[[i]]
            X = row[features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict (SAME AS LIVE SCANNER)
            dir_proba = models['direction'].predict_proba(X)
            dir_pred = int(np.argmax(dir_proba))
            dir_conf = float(np.max(dir_proba))
            timing = float(models['timing'].predict(X)[0])
            strength = float(models['strength'].predict(X)[0])
            
            direction = ['SHORT', 'SIDEWAYS', 'LONG'][dir_pred]
            direction_counts[direction] += 1
            
            # Check filters (SAME AS LIVE SCANNER)
            if dir_pred == 1:  # SIDEWAYS
                continue
            
            passes = (
                dir_conf >= MIN_CONF and
                timing >= MIN_TIMING and
                strength >= MIN_STRENGTH
            )
            
            if passes:
                passed_filters += 1
                ts = row.index[0]
                pair_signals.append({
                    'pair': pair,
                    'timestamp': ts,
                    'direction': direction,
                    'conf': dir_conf,
                    'timing': timing,
                    'strength': strength,
                    'price': row['close'].iloc[0]
                })
        
        print(f"\n  Predictions distribution (last 7 days):")
        total = sum(direction_counts.values())
        for d, c in direction_counts.items():
            pct = c / total * 100 if total > 0 else 0
            print(f"    {d}: {c} ({pct:.1f}%)")
        
        print(f"\n  Signals passed filters: {passed_filters}")
        
        if pair_signals:
            print(f"\n  Sample signals:")
            for sig in pair_signals[:5]:
                print(f"    {sig['timestamp']} | {sig['direction']} | "
                      f"C={sig['conf']:.3f} T={sig['timing']:.2f} S={sig['strength']:.2f}")
        
        all_signals.extend(pair_signals)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total signals found: {len(all_signals)}")
    
    if all_signals:
        # Group by day
        df_signals = pd.DataFrame(all_signals)
        df_signals['date'] = pd.to_datetime(df_signals['timestamp']).dt.date
        
        print(f"\nSignals per day:")
        daily = df_signals.groupby('date').size()
        for date, count in daily.items():
            print(f"  {date}: {count} signals")
        
        print(f"\nSignals per pair:")
        per_pair = df_signals.groupby('pair').size().sort_values(ascending=False)
        for pair, count in per_pair.head(10).items():
            print(f"  {pair}: {count} signals")
        
        print(f"\nDirection distribution:")
        dir_dist = df_signals['direction'].value_counts()
        for d, c in dir_dist.items():
            print(f"  {d}: {c} ({c/len(df_signals)*100:.1f}%)")
        
        # Compare with backtest
        print("\n" + "=" * 70)
        print("COMPARISON WITH BACKTEST")
        print("=" * 70)
        print(f"Backtest had: 408 trades in 30 days = ~14 trades/day")
        print(f"Live scanner on CSV: {len(all_signals)} signals in 7 days = {len(all_signals)/7:.1f} signals/day")
        
        if len(all_signals) / 7 < 5:
            print("\n⚠️  PROBLEM: Live scanner generates MUCH FEWER signals than backtest!")
            print("   This means the issue is in the SCANNER CODE, not in API vs CSV data.")
        else:
            print("\n✅ Signal frequency looks reasonable.")
    else:
        print("\n⚠️  NO SIGNALS FOUND!")
        print("   This confirms the problem is in the scanner/model, not API data.")


if __name__ == '__main__':
    main()

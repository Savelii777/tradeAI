#!/usr/bin/env python3
"""
TEST: Compare CSV Data vs API Data for feature differences

This test loads the SAME timestamp from CSV and API,
computes features, and compares them to find the mismatch.
"""

import sys
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

# Test pair - choose one that has good signals in CSV test
TEST_PAIR = "PIPPIN/USDT:USDT"
# TEST_PAIR = "ASTER/USDT:USDT"


# ============================================================
# FEATURE FUNCTIONS
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


def load_csv_data(pair: str, lookback: int = 3000) -> dict:
    """Load CSV data for a pair"""
    pair_clean = pair.replace('/', '_').replace(':', '_')
    
    data = {}
    for tf in ['1m', '5m', '15m']:
        csv_path = DATA_DIR / f"{pair_clean}_{tf}.csv"
        if not csv_path.exists():
            return None
        
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Take last N rows (same as live would do)
        df = df.tail(lookback)
        data[tf] = df
    
    return data


def fetch_api_data(pair: str, lookback: int = 1500) -> dict:
    """Fetch data from Binance API"""
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    
    data = {}
    for tf in ['1m', '5m', '15m']:
        all_candles = []
        limit = 1000
        
        # First request
        candles = binance.fetch_ohlcv(pair, tf, limit=min(limit, lookback))
        all_candles = candles
        
        # Get more if needed
        while len(all_candles) < lookback:
            oldest = all_candles[0][0]
            tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[tf]
            since = oldest - limit * tf_ms
            
            candles = binance.fetch_ohlcv(pair, tf, since=since, limit=limit)
            if not candles:
                break
            
            new = [c for c in candles if c[0] < oldest]
            if not new:
                break
            
            all_candles = new + all_candles
            time.sleep(0.1)
        
        # Sort and convert
        all_candles = sorted(all_candles, key=lambda x: x[0])
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        data[tf] = df
        print(f"  API {tf}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
    
    return data


def build_features(data: dict, mtf_fe) -> pd.DataFrame:
    """Build features from data"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna()
    
    return ft


def main():
    print("=" * 70)
    print(f"TEST: Compare CSV vs API Data for {TEST_PAIR}")
    print("=" * 70)
    
    # Load model
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features = joblib.load(MODEL_DIR / 'feature_names.joblib')
    mtf_fe = MTFFeatureEngine()
    
    # === LOAD CSV DATA ===
    print("\n[1] Loading CSV data...")
    csv_data = load_csv_data(TEST_PAIR, lookback=3000)
    if csv_data is None:
        print("  ✗ CSV data not found!")
        return
    
    for tf, df in csv_data.items():
        print(f"  CSV {tf}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
    
    # === FETCH API DATA ===
    print("\n[2] Fetching API data...")
    api_data = fetch_api_data(TEST_PAIR, lookback=1500)
    
    # === FIND COMMON TIME RANGE ===
    print("\n[3] Finding common time range...")
    
    # Get the overlap
    csv_end = csv_data['5m'].index[-1]
    api_start = api_data['5m'].index[0]
    api_end = api_data['5m'].index[-1]
    
    print(f"  CSV ends at: {csv_end}")
    print(f"  API range: {api_start} → {api_end}")
    
    # Common range
    common_start = max(csv_data['5m'].index[0], api_start)
    common_end = min(csv_end, api_end)
    
    print(f"  Common range: {common_start} → {common_end}")
    
    if common_start >= common_end:
        print("  ✗ No common time range!")
        return
    
    # === BUILD FEATURES FOR BOTH ===
    print("\n[4] Building features...")
    
    csv_ft = build_features(csv_data, mtf_fe)
    api_ft = build_features(api_data, mtf_fe)
    
    print(f"  CSV features: {len(csv_ft)} rows")
    print(f"  API features: {len(api_ft)} rows")
    
    # Fill missing features
    for f in features:
        if f not in csv_ft.columns:
            csv_ft[f] = 0.0
        if f not in api_ft.columns:
            api_ft[f] = 0.0
    
    # === COMPARE AT COMMON TIMESTAMPS ===
    print("\n[5] Comparing features at common timestamps...")
    
    # Get common timestamps
    common_idx = csv_ft.index.intersection(api_ft.index)
    print(f"  Common timestamps: {len(common_idx)}")
    
    if len(common_idx) == 0:
        print("  ✗ No common timestamps!")
        return
    
    # Compare last 100 common timestamps
    test_idx = common_idx[-100:]
    
    differences = []
    predictions_csv = []
    predictions_api = []
    
    for ts in test_idx:
        csv_row = csv_ft.loc[[ts]]
        api_row = api_ft.loc[[ts]]
        
        X_csv = csv_row[features].values.astype(np.float64)
        X_api = api_row[features].values.astype(np.float64)
        
        X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=0.0, neginf=0.0)
        X_api = np.nan_to_num(X_api, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check feature differences
        diff = np.abs(X_csv - X_api)
        max_diff = diff.max()
        
        # Get predictions
        csv_proba = models['direction'].predict_proba(X_csv)[0]
        api_proba = models['direction'].predict_proba(X_api)[0]
        
        csv_pred = np.argmax(csv_proba)
        api_pred = np.argmax(api_proba)
        
        predictions_csv.append(csv_pred)
        predictions_api.append(api_pred)
        
        if max_diff > 0.01 or csv_pred != api_pred:
            # Find which features differ
            diff_mask = diff[0] > 0.01
            diff_features = [(features[i], float(X_csv[0][i]), float(X_api[0][i]), float(diff[0][i])) 
                           for i in range(len(features)) if diff_mask[i]]
            diff_features.sort(key=lambda x: -x[3])
            
            differences.append({
                'timestamp': ts,
                'max_diff': max_diff,
                'csv_pred': ['SHORT', 'SIDEWAYS', 'LONG'][csv_pred],
                'api_pred': ['SHORT', 'SIDEWAYS', 'LONG'][api_pred],
                'csv_proba': csv_proba.tolist(),
                'api_proba': api_proba.tolist(),
                'top_diff_features': diff_features[:5]
            })
    
    # === RESULTS ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Prediction agreement
    csv_preds = np.array(predictions_csv)
    api_preds = np.array(predictions_api)
    
    agreement = (csv_preds == api_preds).mean() * 100
    print(f"\nPrediction agreement: {agreement:.1f}%")
    
    # Distribution
    print(f"\nCSV predictions: SHORT={sum(csv_preds==0)}, SIDEWAYS={sum(csv_preds==1)}, LONG={sum(csv_preds==2)}")
    print(f"API predictions: SHORT={sum(api_preds==0)}, SIDEWAYS={sum(api_preds==1)}, LONG={sum(api_preds==2)}")
    
    # Show examples of differences
    if differences:
        print(f"\n📊 Found {len(differences)} timestamps with significant differences:")
        for d in differences[:5]:
            print(f"\n  {d['timestamp']}")
            print(f"    CSV: {d['csv_pred']} | API: {d['api_pred']}")
            print(f"    CSV proba: {d['csv_proba']}")
            print(f"    API proba: {d['api_proba']}")
            if d['top_diff_features']:
                print(f"    Top differing features:")
                for fname, csv_val, api_val, diff in d['top_diff_features'][:3]:
                    print(f"      {fname}: CSV={csv_val:.4f}, API={api_val:.4f}, diff={diff:.4f}")
    else:
        print("\n✅ No significant differences found!")
    
    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if agreement >= 95:
        print("✅ CSV and API data produce nearly identical predictions!")
        print("   Problem might be in timing (when exactly scanner runs)")
    elif agreement >= 80:
        print("⚠️  Some difference between CSV and API data")
        print("   This could cause missed signals")
    else:
        print("❌ SIGNIFICANT DIFFERENCE between CSV and API data!")
        print("   This is likely the cause of missing signals")


if __name__ == '__main__':
    main()

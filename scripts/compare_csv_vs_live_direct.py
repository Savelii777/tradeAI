#!/usr/bin/env python3
"""
DIRECT COMPARISON: CSV (backtest) vs LIVE (Binance)
Найдём точную разницу
"""

import sys
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")
DATA_DIR = Path("data/candles")

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def main():
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    pair = 'PIPPIN/USDT:USDT'
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    print("="*80)
    print(f"COMPARING: {pair}")
    print("="*80)
    
    # ============================================================
    # 1. LOAD CSV DATA (как в бэктесте)
    # ============================================================
    print("\n[1] Loading CSV data (backtest style)...")
    
    m1_csv = pd.read_csv(DATA_DIR / f'{pair_name}_1m.csv', index_col=0, parse_dates=True)
    m5_csv = pd.read_csv(DATA_DIR / f'{pair_name}_5m.csv', index_col=0, parse_dates=True)
    m15_csv = pd.read_csv(DATA_DIR / f'{pair_name}_15m.csv', index_col=0, parse_dates=True)
    
    # FIX: Add UTC timezone to CSV data (same as Live)
    m1_csv.index = m1_csv.index.tz_localize('UTC')
    m5_csv.index = m5_csv.index.tz_localize('UTC')
    m15_csv.index = m15_csv.index.tz_localize('UTC')
    
    print(f"   CSV M1: {len(m1_csv)} rows, {m1_csv.index[0]} to {m1_csv.index[-1]}")
    print(f"   CSV M5: {len(m5_csv)} rows, {m5_csv.index[0]} to {m5_csv.index[-1]}")
    
    # Take last portion
    m1_csv = m1_csv.tail(1000)
    m5_csv = m5_csv.tail(500)
    m15_csv = m15_csv.tail(200)
    
    ft_csv = mtf_fe.align_timeframes(m1_csv, m5_csv, m15_csv)
    ft_csv = ft_csv.join(m5_csv[['open', 'high', 'low', 'close', 'volume']])
    ft_csv = add_volume_features(ft_csv)
    ft_csv['atr'] = calculate_atr(ft_csv)
    ft_csv = ft_csv.dropna()
    
    print(f"   CSV features: {len(ft_csv)} rows")
    
    # ============================================================
    # 2. LOAD LIVE DATA (как в лайве)
    # ============================================================
    print("\n[2] Loading LIVE data (Binance)...")
    
    data_live = {}
    for tf in ['1m', '5m', '15m']:
        candles = binance.fetch_ohlcv(pair, tf, limit=500)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        data_live[tf] = df
    
    print(f"   Live M1: {len(data_live['1m'])} rows")
    print(f"   Live M5: {len(data_live['5m'])} rows")
    
    ft_live = mtf_fe.align_timeframes(data_live['1m'], data_live['5m'], data_live['15m'])
    ft_live = ft_live.join(data_live['5m'][['open', 'high', 'low', 'close', 'volume']])
    ft_live = add_volume_features(ft_live)
    ft_live['atr'] = calculate_atr(ft_live)
    ft_live = ft_live.dropna()
    
    print(f"   Live features: {len(ft_live)} rows")
    
    # ============================================================
    # 3. FIND OVERLAPPING TIMESTAMPS
    # ============================================================
    print("\n[3] Finding overlapping timestamps...")
    
    common_times = ft_csv.index.intersection(ft_live.index)
    print(f"   Common timestamps: {len(common_times)}")
    
    if len(common_times) == 0:
        print("   ⚠️  NO OVERLAP! CSV data is too old.")
        print(f"   CSV ends at: {ft_csv.index[-1]}")
        print(f"   Live starts at: {ft_live.index[0]}")
        return
    
    print(f"   Overlap: {common_times[0]} to {common_times[-1]}")
    
    # ============================================================
    # 4. COMPARE FEATURES AT SAME TIMESTAMPS
    # ============================================================
    print("\n[4] Comparing features at same timestamps...")
    
    # Take a few common timestamps
    sample_times = common_times[-5:]  # Last 5
    
    for ts in sample_times:
        print(f"\n   Timestamp: {ts}")
        
        row_csv = ft_csv.loc[[ts]]
        row_live = ft_live.loc[[ts]]
        
        # Fill missing
        for f in features:
            if f not in row_csv.columns:
                row_csv[f] = 0.0
            if f not in row_live.columns:
                row_live[f] = 0.0
        
        X_csv = row_csv[features].values.astype(np.float64)
        X_live = row_live[features].values.astype(np.float64)
        
        X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=0.0, neginf=0.0)
        X_live = np.nan_to_num(X_live, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compare features
        diff = np.abs(X_csv - X_live)
        max_diff_idx = np.argmax(diff)
        max_diff_feat = features[max_diff_idx]
        max_diff_val = diff[0][max_diff_idx]
        
        print(f"      Max diff feature: {max_diff_feat} = {max_diff_val:.6f}")
        print(f"         CSV: {X_csv[0][max_diff_idx]:.6f}")
        print(f"         Live: {X_live[0][max_diff_idx]:.6f}")
        
        # Predict on BOTH
        proba_csv = models['direction'].predict_proba(X_csv)
        proba_live = models['direction'].predict_proba(X_live)
        
        pred_csv = np.argmax(proba_csv)
        pred_live = np.argmax(proba_live)
        conf_csv = np.max(proba_csv)
        conf_live = np.max(proba_live)
        
        dir_map = {0: 'SHORT', 1: 'SIDE', 2: 'LONG'}
        
        print(f"      CSV predict:  {dir_map[pred_csv]} conf={conf_csv:.3f}")
        print(f"      Live predict: {dir_map[pred_live]} conf={conf_live:.3f}")
        
        if pred_csv != pred_live:
            print(f"      ⚠️  DIFFERENT PREDICTIONS!")
    
    # ============================================================
    # 5. COUNT SIGNALS IN OVERLAP PERIOD
    # ============================================================
    print("\n[5] Counting signals in overlap period...")
    
    MIN_CONF = 0.50
    MIN_TIMING = 0.8
    MIN_STRENGTH = 1.4
    
    # CSV signals
    csv_overlap = ft_csv.loc[common_times]
    for f in features:
        if f not in csv_overlap.columns:
            csv_overlap[f] = 0.0
    
    X_csv_all = csv_overlap[features].values.astype(np.float64)
    X_csv_all = np.nan_to_num(X_csv_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    proba_csv_all = models['direction'].predict_proba(X_csv_all)
    preds_csv = np.argmax(proba_csv_all, axis=1)
    confs_csv = np.max(proba_csv_all, axis=1)
    timing_csv = models['timing'].predict(X_csv_all)
    strength_csv = models['strength'].predict(X_csv_all)
    
    signals_csv = (preds_csv != 1) & (confs_csv >= MIN_CONF) & (timing_csv >= MIN_TIMING) & (strength_csv >= MIN_STRENGTH)
    
    # Live signals
    live_overlap = ft_live.loc[common_times]
    for f in features:
        if f not in live_overlap.columns:
            live_overlap[f] = 0.0
    
    X_live_all = live_overlap[features].values.astype(np.float64)
    X_live_all = np.nan_to_num(X_live_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    proba_live_all = models['direction'].predict_proba(X_live_all)
    preds_live = np.argmax(proba_live_all, axis=1)
    confs_live = np.max(proba_live_all, axis=1)
    timing_live = models['timing'].predict(X_live_all)
    strength_live = models['strength'].predict(X_live_all)
    
    signals_live = (preds_live != 1) & (confs_live >= MIN_CONF) & (timing_live >= MIN_TIMING) & (strength_live >= MIN_STRENGTH)
    
    print(f"   CSV signals in overlap: {signals_csv.sum()}")
    print(f"   Live signals in overlap: {signals_live.sum()}")
    
    # Show where they differ
    diff_mask = signals_csv != signals_live
    if diff_mask.any():
        print(f"\n   ⚠️  {diff_mask.sum()} timestamps have different signal status!")
        diff_times = common_times[diff_mask][:5]
        for t in diff_times:
            i = list(common_times).index(t)
            print(f"      {t}: CSV={signals_csv[i]}, Live={signals_live[i]}")
            print(f"         CSV: conf={confs_csv[i]:.3f}, timing={timing_csv[i]:.2f}, strength={strength_csv[i]:.2f}")
            print(f"         Live: conf={confs_live[i]:.3f}, timing={timing_live[i]:.2f}, strength={strength_live[i]:.2f}")


if __name__ == '__main__':
    main()

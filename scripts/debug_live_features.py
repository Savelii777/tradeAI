#!/usr/bin/env python3
"""Debug why live confidence is so low."""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features_like_live(data, mtf_fe):
    """Same as live_trading_v10_csv.py prepare_features()"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna(subset=['close', 'atr'])
    
    # Exclude cumsum
    cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
    ft = ft.drop(columns=cols_to_drop, errors='ignore')
    
    # Exclude absolute price
    absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
    ft = ft.drop(columns=absolute_cols, errors='ignore')
    
    ft = ft.ffill().dropna()
    return ft

def main():
    model_dir = Path(__file__).parent.parent / "models" / "v8_improved"
    dir_model = joblib.load(model_dir / 'direction_model.joblib')
    features = joblib.load(model_dir / 'feature_names.joblib')
    
    data_dir = Path(__file__).parent.parent / "data" / "candles"
    
    # Test with PIPPIN (one of the pairs with low confidence)
    pair = 'PIPPIN_USDT_USDT'
    
    print(f"Testing pair: {pair}")
    print(f"Model expects {len(features)} features")
    
    m1 = pd.read_parquet(data_dir / f'{pair}_1m.parquet')
    m5 = pd.read_parquet(data_dir / f'{pair}_5m.parquet')
    m15 = pd.read_parquet(data_dir / f'{pair}_15m.parquet')
    
    print(f"Loaded data: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
    
    mtf_fe = MTFFeatureEngine()
    data = {'1m': m1, '5m': m5, '15m': m15}
    ft = prepare_features_like_live(data, mtf_fe)
    
    print(f"Features calculated: {len(ft)} rows, {len(ft.columns)} columns")
    
    # Check which model features are present
    present = [f for f in features if f in ft.columns]
    missing = [f for f in features if f not in ft.columns]
    
    print(f"\nModel features present: {len(present)}/{len(features)}")
    if missing:
        print(f"Missing: {missing}")
    
    # Get last row prediction
    row = ft.iloc[[-1]].copy()
    
    # Fill missing with 0
    for f in features:
        if f not in row.columns:
            row[f] = 0.0
    
    X = row[features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    
    probas = dir_model.predict_proba(X)
    print(f"\nLast row prediction:")
    print(f"  Probas: {probas}")
    print(f"  Max conf: {np.max(probas):.3f}")
    
    # Check feature values vs BTC
    print("\n=== Comparing feature values PIPPIN vs BTC ===")
    
    # Load BTC for comparison
    btc_m1 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_1m.parquet')
    btc_m5 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_5m.parquet')
    btc_m15 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_15m.parquet')
    
    btc_data = {'1m': btc_m1, '5m': btc_m5, '15m': btc_m15}
    btc_ft = prepare_features_like_live(btc_data, mtf_fe)
    
    btc_row = btc_ft.iloc[[-1]].copy()
    for f in features:
        if f not in btc_row.columns:
            btc_row[f] = 0.0
    
    btc_X = btc_row[features].values.astype(np.float64)
    btc_X = np.nan_to_num(btc_X, nan=0.0)
    
    btc_probas = dir_model.predict_proba(btc_X)
    print(f"BTC last row: {np.max(btc_probas):.3f}")
    print(f"PIPPIN last row: {np.max(probas):.3f}")
    
    # Compare individual feature values
    print("\nFeature comparison (PIPPIN vs BTC):")
    for i, f in enumerate(features[:10]):  # First 10
        pippin_val = X[0, i] if i < X.shape[1] else 0
        btc_val = btc_X[0, i] if i < btc_X.shape[1] else 0
        diff = abs(pippin_val - btc_val)
        print(f"  {f}: PIPPIN={pippin_val:.4f}, BTC={btc_val:.4f}, diff={diff:.4f}")

if __name__ == "__main__":
    main()

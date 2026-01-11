#!/usr/bin/env python3
"""Test confidence across different pairs."""

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
    return df

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe):
    m1, m5, m15 = data['1m'], data['5m'], data['15m']
    
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna(subset=['close', 'atr'])
    
    cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
    ft = ft.drop(columns=cols_to_drop, errors='ignore')
    
    absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
    ft = ft.drop(columns=absolute_cols, errors='ignore')
    
    return ft.ffill().dropna()

def main():
    model_dir = Path(__file__).parent.parent / "models" / "v8_improved"
    dir_model = joblib.load(model_dir / 'direction_model.joblib')
    timing_model = joblib.load(model_dir / 'timing_model.joblib')
    strength_model = joblib.load(model_dir / 'strength_model.joblib')
    features = joblib.load(model_dir / 'feature_names.joblib')
    
    print(f"Model uses {len(features)} features")
    
    data_dir = Path(__file__).parent.parent / "data" / "candles"
    mtf_fe = MTFFeatureEngine()
    
    # Test pairs from training set
    test_pairs = ['BTC_USDT_USDT', 'ETH_USDT_USDT', 'SOL_USDT_USDT', 'ZEC_USDT_USDT', 'DOGE_USDT_USDT']
    
    print("\n=== Pairs from TRAINING set ===")
    for pair in test_pairs:
        try:
            m1 = pd.read_parquet(data_dir / f'{pair}_1m.parquet')
            m5 = pd.read_parquet(data_dir / f'{pair}_5m.parquet')
            m15 = pd.read_parquet(data_dir / f'{pair}_15m.parquet')
            
            ft = prepare_features({'1m': m1, '5m': m5, '15m': m15}, mtf_fe)
            row = ft.iloc[[-1]].copy()
            
            for f in features:
                if f not in row.columns:
                    row[f] = 0.0
            
            X = row[features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0)
            
            probas = dir_model.predict_proba(X)
            timing = timing_model.predict(X)[0]
            strength = strength_model.predict(X)[0]
            atr_pct = row['m5_atr_14_pct'].values[0] if 'm5_atr_14_pct' in row.columns else 0
            
            print(f"  {pair:20s}: Conf={np.max(probas):.3f} Tim={timing:.2f} Str={strength:.2f} ATR%={atr_pct:.4f}")
        except Exception as e:
            print(f"  {pair:20s}: Error - {e}")
    
    # Test memecoins not in training
    print("\n=== Memecoins (NOT in training) ===")
    meme_pairs = ['PIPPIN_USDT_USDT', 'ASTER_USDT_USDT']
    
    for pair in meme_pairs:
        try:
            m1 = pd.read_parquet(data_dir / f'{pair}_1m.parquet')
            m5 = pd.read_parquet(data_dir / f'{pair}_5m.parquet')
            m15 = pd.read_parquet(data_dir / f'{pair}_15m.parquet')
            
            ft = prepare_features({'1m': m1, '5m': m5, '15m': m15}, mtf_fe)
            row = ft.iloc[[-1]].copy()
            
            for f in features:
                if f not in row.columns:
                    row[f] = 0.0
            
            X = row[features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0)
            
            probas = dir_model.predict_proba(X)
            timing = timing_model.predict(X)[0]
            strength = strength_model.predict(X)[0]
            atr_pct = row['m5_atr_14_pct'].values[0] if 'm5_atr_14_pct' in row.columns else 0
            
            print(f"  {pair:20s}: Conf={np.max(probas):.3f} Tim={timing:.2f} Str={strength:.2f} ATR%={atr_pct:.4f}")
        except Exception as e:
            print(f"  {pair:20s}: Error - {e}")

if __name__ == "__main__":
    main()

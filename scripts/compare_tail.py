#!/usr/bin/env python3
"""Compare predictions with all data vs tail 1500."""

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

def prepare(m1, m5, m15, mtf_fe, use_tail=False):
    if use_tail:
        m1 = m1.tail(7500)
        m5 = m5.tail(1500)
        m15 = m15.tail(500)
    
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
    model_dir = Path(__file__).parent.parent / 'models' / 'v8_improved'
    dir_model = joblib.load(model_dir / 'direction_model.joblib')
    timing_model = joblib.load(model_dir / 'timing_model.joblib')
    strength_model = joblib.load(model_dir / 'strength_model.joblib')
    features = joblib.load(model_dir / 'feature_names.joblib')
    
    data_dir = Path(__file__).parent.parent / 'data' / 'candles'
    mtf_fe = MTFFeatureEngine()
    
    print('Pair               | ALL DATA        | TAIL 1500       | DIFF')
    print('-' * 70)
    
    for pair in ['BTC_USDT_USDT', 'ETH_USDT_USDT', 'SOL_USDT_USDT', 'ZEC_USDT_USDT', 'PIPPIN_USDT_USDT']:
        try:
            m1 = pd.read_parquet(data_dir / f'{pair}_1m.parquet')
            m5 = pd.read_parquet(data_dir / f'{pair}_5m.parquet')
            m15 = pd.read_parquet(data_dir / f'{pair}_15m.parquet')
            
            # All data
            ft_all = prepare(m1.copy(), m5.copy(), m15.copy(), mtf_fe, use_tail=False)
            row_all = ft_all.iloc[[-1]].copy()
            for f in features:
                if f not in row_all.columns:
                    row_all[f] = 0.0
            X_all = row_all[features].values.astype(np.float64)
            X_all = np.nan_to_num(X_all, nan=0.0)
            conf_all = np.max(dir_model.predict_proba(X_all))
            str_all = strength_model.predict(X_all)[0]
            
            # Tail only
            ft_tail = prepare(m1.copy(), m5.copy(), m15.copy(), mtf_fe, use_tail=True)
            row_tail = ft_tail.iloc[[-1]].copy()
            for f in features:
                if f not in row_tail.columns:
                    row_tail[f] = 0.0
            X_tail = row_tail[features].values.astype(np.float64)
            X_tail = np.nan_to_num(X_tail, nan=0.0)
            conf_tail = np.max(dir_model.predict_proba(X_tail))
            str_tail = strength_model.predict(X_tail)[0]
            
            diff_c = conf_tail - conf_all
            diff_s = str_tail - str_all
            
            print(f'{pair:18s} | C={conf_all:.2f} S={str_all:.2f} | C={conf_tail:.2f} S={str_tail:.2f} | ΔC={diff_c:+.2f} ΔS={diff_s:+.2f}')
            
        except Exception as e:
            print(f'{pair:18s} | Error: {e}')
    
    print('\nC=Confidence, S=Strength')

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick test to find missing feature in live."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import pandas as pd
import joblib
from train_mtf import MTFFeatureEngine


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same as in live_trading_v10_csv.py"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def main():
    # Load model features
    model_features = joblib.load('models/v8_improved/feature_names.joblib')
    print(f'Model expects {len(model_features)} features')
    
    # Load sample data like in live (LOOKBACK limited)
    pair = 'BTC_USDT_USDT'
    LOOKBACK_M1 = 7500
    LOOKBACK_M5 = 1500
    LOOKBACK_M15 = 500
    
    m1 = pd.read_csv(f'data/candles/{pair}_1m.csv', index_col=0, parse_dates=True).tail(LOOKBACK_M1)
    m5 = pd.read_csv(f'data/candles/{pair}_5m.csv', index_col=0, parse_dates=True).tail(LOOKBACK_M5)
    m15 = pd.read_csv(f'data/candles/{pair}_15m.csv', index_col=0, parse_dates=True).tail(LOOKBACK_M15)
    
    print(f'Loaded {pair}')
    print(f'1m: {len(m1)} candles (lookback {LOOKBACK_M1})')
    print(f'5m: {len(m5)} candles (lookback {LOOKBACK_M5})')  
    print(f'15m: {len(m15)} candles (lookback {LOOKBACK_M15})')

    # Generate features using align_timeframes (same as training)
    fe = MTFFeatureEngine()
    features = fe.align_timeframes(m1, m5, m15)
    
    # Add OHLCV from M5
    features = features.join(m5[['open', 'high', 'low', 'close', 'volume']])
    
    # Add volume features - SAME as live
    features = add_volume_features(features)
    
    # Add ATR
    features['atr'] = calculate_atr(features)
    
    print(f'\nGenerated features: {len(features)} rows, {len(features.columns)} columns')
    
    # Check which model features are missing
    model_set = set(model_features)
    available_set = set(features.columns)
    
    missing = model_set - available_set
    present = model_set.intersection(available_set)
    
    print(f'\nModel features: {len(model_features)}')
    print(f'Present: {len(present)}')
    print(f'Missing: {len(missing)}')
    
    if missing:
        print(f'\n❌ MISSING FEATURES:')
        for m in sorted(missing):
            print(f'  - {m}')
            
    # Match percentage
    match_pct = len(present) / len(model_features) * 100
    print(f'\nMatch percentage: {match_pct:.1f}%')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare training scaler parameters with live data distribution."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import joblib
import pandas as pd
import numpy as np
from train_mtf import MTFFeatureEngine

def main():
    # Load scaler from training
    scaler = joblib.load('models/v8_improved/scaler.joblib')
    features_list = joblib.load('models/v8_improved/feature_names.joblib')
    
    print('=' * 80)
    print('COMPARISON: TRAINING vs LIVE DATA')
    print('=' * 80)
    
    # Calculate features on recent data (like live)
    pair = 'BTC_USDT_USDT'
    data = {
        '1m': pd.read_csv(f'data/candles/{pair}_1m.csv', index_col=0, parse_dates=True).tail(7500),
        '5m': pd.read_csv(f'data/candles/{pair}_5m.csv', index_col=0, parse_dates=True).tail(1500),
        '15m': pd.read_csv(f'data/candles/{pair}_15m.csv', index_col=0, parse_dates=True).tail(500),
    }
    
    # Generate features
    fe = MTFFeatureEngine()
    ft = fe.align_timeframes(data['1m'], data['5m'], data['15m'])
    ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
    
    # Add volume features
    ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
    ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
    ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
    ft = ft.dropna()
    
    print(f'\nLast 5 days BTC data: {len(ft)} rows')
    print(f'Training used: ~20 pairs x ~40000 rows = ~800K samples\n')
    
    print(f'{"Feature":<20} | {"Train Mean":>12} | {"Live Mean":>12} | {"Train Std":>12} | {"Live Std":>12} | Match?')
    print('-' * 90)
    
    all_ok = True
    for f in features_list:
        if f in ft.columns:
            idx = list(features_list).index(f)
            train_mean = scaler.mean_[idx]
            train_std = scaler.scale_[idx]
            live_mean = ft[f].mean()
            live_std = ft[f].std()
            
            # Check if similar
            mean_ok = abs(train_mean - live_mean) < 2 * train_std if train_std > 0 else True
            std_ok = 0.2 < live_std / train_std < 5.0 if train_std > 0 else True
            match = '✅' if mean_ok and std_ok else '⚠️'
            if not (mean_ok and std_ok):
                all_ok = False
            
            print(f'{f:<20} | {train_mean:>12.4f} | {live_mean:>12.4f} | {train_std:>12.4f} | {live_std:>12.4f} | {match}')
    
    print('\n' + '=' * 80)
    print('INTERPRETATION:')
    print('=' * 80)
    print('Train Mean/Std = parameters saved in scaler.joblib during training')
    print('Live Mean/Std = calculated from last 5 days of BTC data')
    print('')
    print('✅ = Distributions match - model predictions should be accurate')
    print('⚠️ = Distribution shift - model may have lower accuracy')
    print('')
    
    if all_ok:
        print('🎉 ALL FEATURES MATCH - Model trained on same normalized data!')
    else:
        print('⚠️ Some features have distribution shift - monitor predictions carefully')
    
    # Show example transformation
    print('\n' + '=' * 80)
    print('EXAMPLE: How scaler normalizes live values')
    print('=' * 80)
    last_row = ft.iloc[-1][features_list].values.astype(np.float64).reshape(1, -1)
    last_row = np.nan_to_num(last_row, nan=0.0)
    scaled = scaler.transform(last_row)
    
    examples = ['m5_rsi_14', 'm5_atr_14_pct', 'vol_ratio', 'm5_return_5']
    for f in examples:
        if f in features_list:
            idx = list(features_list).index(f)
            raw = last_row[0][idx]
            scl = scaled[0][idx]
            mean = scaler.mean_[idx]
            std = scaler.scale_[idx]
            print(f'{f}: raw={raw:.4f} → (raw - {mean:.4f}) / {std:.4f} = {scl:.4f}')

if __name__ == '__main__':
    main()

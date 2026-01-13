#!/usr/bin/env python3
"""Check which CORE_20 features are missing or have NaN values."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import pandas as pd
import numpy as np
from train_mtf import MTFFeatureEngine
from src.utils.constants import CORE_20_FEATURES


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same as in live_trading_v10_csv.py"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    return df


def main():
    # Test multiple pairs
    pairs_to_check = ['BTC_USDT_USDT', 'ETH_USDT_USDT', 'SOL_USDT_USDT', 'DOGE_USDT_USDT', 'XRP_USDT_USDT']
    
    all_ok = True
    
    for pair in pairs_to_check:
        print(f'\n{"="*60}')
        print(f'Checking {pair}')
        print("="*60)
        
        try:
            # Load sample data
            data = {
                '1m': pd.read_csv(f'data/candles/{pair}_1m.csv', index_col=0, parse_dates=True),
                '5m': pd.read_csv(f'data/candles/{pair}_5m.csv', index_col=0, parse_dates=True),
                '15m': pd.read_csv(f'data/candles/{pair}_15m.csv', index_col=0, parse_dates=True),
            }
            print(f'  1m: {len(data["1m"])} | 5m: {len(data["5m"])} | 15m: {len(data["15m"])}')

            # Generate features using align_timeframes (same as training)
            fe = MTFFeatureEngine()
            features = fe.align_timeframes(data['1m'], data['5m'], data['15m'])
            
            # Add OHLCV from M5
            features = features.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
            
            # Add volume features - SAME as live
            features = add_volume_features(features)
            
            # Check which CORE_20 are missing
            available = set(features.columns)
            core_set = set(CORE_20_FEATURES)
            missing = core_set - available
            present = core_set.intersection(available)
            
            match_pct = len(present) / len(CORE_20_FEATURES) * 100
            
            if missing:
                print(f'  ❌ Missing {len(missing)} features: {sorted(missing)}')
                all_ok = False
            else:
                print(f'  ✅ All 30 features present ({match_pct:.0f}%)')
                
            # Check last row NaN
            last_row = features.iloc[-1]
            nan_features = [f for f in CORE_20_FEATURES if f in features.columns and pd.isna(last_row[f])]
            if nan_features:
                print(f'  ⚠️ NaN in last row: {nan_features}')
                all_ok = False
            
            # Check gaps
            for tf, df in data.items():
                expected_gap = {'1m': pd.Timedelta(minutes=1), '5m': pd.Timedelta(minutes=5), '15m': pd.Timedelta(minutes=15)}[tf]
                gaps = df.index.to_series().diff()
                big_gaps = gaps[gaps > expected_gap * 10]  # More than 10x expected
                if len(big_gaps) > 0:
                    print(f'  ⚠️ {tf} has {len(big_gaps)} large gaps')
                    
        except FileNotFoundError:
            print(f'  ⚠️ Data files not found')
        except Exception as e:
            print(f'  ❌ Error: {e}')
            all_ok = False
    
    print(f'\n{"="*60}')
    if all_ok:
        print('✅ ALL PAIRS OK - Data is ready for trading!')
    else:
        print('⚠️ Some issues found - check above')
    print("="*60)

    # Check which CORE_20 are missing
    available = set(features.columns)
    core_set = set(CORE_20_FEATURES)
    missing = core_set - available
    present = core_set.intersection(available)

    print(f'\nCORE_20_FEATURES: {len(CORE_20_FEATURES)}')
    print(f'Present: {len(present)}')
    print(f'Missing: {len(missing)}')
    if missing:
        print('Missing features:')
        for m in sorted(missing):
            print(f'  - {m}')
            
    # Check for NaN in last row for present features
    last_row = features.iloc[-1]
    nan_features = []
    for f in CORE_20_FEATURES:
        if f in features.columns and pd.isna(last_row[f]):
            nan_features.append(f)
            
    print(f'\nNaN in last row: {len(nan_features)}')
    if nan_features:
        for f in nan_features:
            print(f'  - {f}')
            
    # Match percentage
    match_pct = len(present) / len(CORE_20_FEATURES) * 100
    print(f'\nMatch percentage: {match_pct:.1f}%')    
    # === Additional Data Quality Checks ===
    print('\n' + '='*60)
    print('DATA QUALITY CHECKS')
    print('='*60)
    
    # Check for gaps in timestamps
    print('\n📊 Timestamp gaps:')
    for tf, df in data.items():
        expected_gap = {'1m': pd.Timedelta(minutes=1), '5m': pd.Timedelta(minutes=5), '15m': pd.Timedelta(minutes=15)}[tf]
        gaps = df.index.to_series().diff()
        big_gaps = gaps[gaps > expected_gap * 2]
        if len(big_gaps) > 0:
            print(f'  {tf}: {len(big_gaps)} gaps found')
            print(f'      Largest: {big_gaps.max()}, at {big_gaps.idxmax()}')
        else:
            print(f'  {tf}: ✅ No significant gaps')
    
    # Check for zero/nan values in OHLCV
    print('\n📊 OHLCV integrity:')
    for tf, df in data.items():
        issues = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            nan_count = df[col].isna().sum()
            zero_count = (df[col] == 0).sum() if col != 'volume' else 0  # zero volume is ok
            if nan_count > 0:
                issues.append(f'{col}: {nan_count} NaN')
            if zero_count > 0:
                issues.append(f'{col}: {zero_count} zeros')
        if issues:
            print(f'  {tf}: ⚠️ {", ".join(issues)}')
        else:
            print(f'  {tf}: ✅ Clean')
    
    # Check for impossible prices (low > high, etc)
    print('\n📊 Price logic:')
    for tf, df in data.items():
        bad_hl = (df['low'] > df['high']).sum()
        bad_oc_high = ((df['open'] > df['high']) | (df['close'] > df['high'])).sum()
        bad_oc_low = ((df['open'] < df['low']) | (df['close'] < df['low'])).sum()
        if bad_hl > 0 or bad_oc_high > 0 or bad_oc_low > 0:
            print(f'  {tf}: ⚠️ low>high: {bad_hl}, O/C>high: {bad_oc_high}, O/C<low: {bad_oc_low}')
        else:
            print(f'  {tf}: ✅ Logical')
    
    # Feature distribution check
    print('\n📊 Feature distributions (last 100 rows):')
    recent = features.tail(100)
    for f in CORE_20_FEATURES[:10]:  # Check first 10
        if f in recent.columns:
            vals = recent[f].dropna()
            print(f'  {f}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}')
    
    print('\n✅ Data quality check complete!')
if __name__ == '__main__':
    main()

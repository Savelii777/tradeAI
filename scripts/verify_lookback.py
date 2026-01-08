#!/usr/bin/env python3
"""
Verify that increasing LOOKBACK fixes the feature discrepancy.

This script compares volume_ratio features at different LOOKBACK values
and shows how close they are to backtest values.

Usage:
    python scripts/verify_lookback.py --pair BTC/USDT:USDT
    
Expected result:
    With LOOKBACK=10000, features should be within 50% of backtest values
    (vs 500-2500% difference with LOOKBACK=3000)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

# Key features that showed >500% difference at LOOKBACK=3000
KEY_FEATURES = [
    'm5_volume_ratio_20',
    'm5_volume_ratio_10', 
    'm5_volume_ratio_5',
    'm15_volume_ratio',
    'vol_ratio',
    'm5_ema_9_slope',
    'm5_ema_21_slope',
]


def add_volume_features(df):
    """Add volume features (same as train_v3_dynamic.py)"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def atr(df, period=14):
    """Calculate ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def fetch_binance_data(pair, timeframe, limit):
    """Fetch data from Binance"""
    exchange = ccxt.binance({'enableRateLimit': True})
    
    symbol = pair.replace('_', '/').replace(':USDT', '')
    if not symbol.endswith('/USDT'):
        symbol = symbol + '/USDT'
    
    all_ohlcv = []
    remaining = limit
    since = None
    
    while remaining > 0:
        try:
            batch_size = min(remaining, 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            remaining -= len(ohlcv)
            
            if len(ohlcv) < batch_size:
                break
                
            since = ohlcv[-1][0] + 1
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if not all_ohlcv:
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df = df.sort_index()
    df = df.iloc[-limit:] if len(df) > limit else df
    
    return df


def prepare_features(m1, m5, m15, mtf_fe):
    """Prepare features (same as live script)"""
    m5 = add_volume_features(m5.copy())
    m5['atr'] = atr(m5, period=14)
    
    # Prepare M15 features
    m15_prepared = m15.copy()
    m15_prepared['m15_close'] = m15_prepared['close']
    m15_prepared['m15_volume'] = m15_prepared['volume']
    m15_prepared['m15_atr'] = atr(m15_prepared, period=14)
    m15_prepared['m15_volume_ma'] = m15_prepared['volume'].rolling(20).mean()
    m15_prepared['m15_volume_ratio'] = m15_prepared['volume'] / m15_prepared['m15_volume_ma']
    m15_prepared['m15_momentum'] = m15_prepared['close'].pct_change(5).clip(-1, 1) * 100
    m15_prepared['m15_trend'] = (m15_prepared['close'] - m15_prepared['close'].rolling(50).mean()) / m15_prepared['close'].rolling(50).mean() * 100
    
    # Generate M5 features
    ft_m5 = mtf_fe.generate_m5_features(m5)
    
    # Generate M1 features
    ft_m1 = mtf_fe.generate_m1_features(m1, prefix='m1_')
    if 'm1_ts' in ft_m1.columns:
        ft_m1 = ft_m1.drop(columns=['m1_ts'])
    if 'ts' in ft_m1.columns:
        ft_m1 = ft_m1.drop(columns=['ts'])
    ft_m1.columns = [f'{c}_last' if not c.startswith('m1_') else c for c in ft_m1.columns]
    
    # Merge all
    combined = ft_m5.copy()
    
    # Merge M15
    combined = combined.merge(
        m15_prepared[['m15_close', 'm15_volume', 'm15_atr', 'm15_volume_ma', 'm15_volume_ratio', 'm15_momentum', 'm15_trend']],
        left_index=True, right_index=True, how='left'
    )
    
    # Merge M1
    combined = combined.merge(ft_m1, left_index=True, right_index=True, how='left')
    
    # Add volume features from m5
    for col in ['vol_sma_20', 'vol_ratio', 'vol_zscore', 'vwap', 'price_vs_vwap', 'vol_momentum']:
        if col in m5.columns:
            combined[col] = m5[col]
    
    combined['atr'] = m5['atr'] if 'atr' in m5.columns else atr(m5)
    
    combined = combined.dropna()
    
    return combined


def load_csv_data(pair_csv):
    """Load CSV data for comparison"""
    m1_path = DATA_DIR / f"{pair_csv}_1m.csv"
    m5_path = DATA_DIR / f"{pair_csv}_5m.csv"
    m15_path = DATA_DIR / f"{pair_csv}_15m.csv"
    
    if not all(p.exists() for p in [m1_path, m5_path, m15_path]):
        return None, None, None
    
    m1 = pd.read_csv(m1_path, parse_dates=['timestamp'], index_col='timestamp')
    m5 = pd.read_csv(m5_path, parse_dates=['timestamp'], index_col='timestamp')
    m15 = pd.read_csv(m15_path, parse_dates=['timestamp'], index_col='timestamp')
    
    return m1, m5, m15


def main():
    parser = argparse.ArgumentParser(description='Verify LOOKBACK fix')
    parser.add_argument('--pair', type=str, default='BTC/USDT:USDT', help='Pair to analyze')
    args = parser.parse_args()
    
    pair = args.pair
    pair_csv = pair.replace('/', '_').replace(':', '_')
    
    print("="*70)
    print(f"VERIFY LOOKBACK FIX: {pair}")
    print("="*70)
    
    try:
        mtf_fe = MTFFeatureEngine()
    except Exception as e:
        print(f"❌ Error initializing MTFFeatureEngine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 1. Load CSV data for backtest reference
    print("\n📊 Loading CSV data for backtest reference...")
    try:
        m1_csv, m5_csv, m15_csv = load_csv_data(pair_csv)
        if m1_csv is None:
            print(f"❌ CSV data not found for {pair_csv}")
            print(f"   Expected files in: {DATA_DIR}")
            print(f"   Looking for: {pair_csv}_1m.csv, {pair_csv}_5m.csv, {pair_csv}_15m.csv")
            return
        
        print(f"  M1: {len(m1_csv)} candles")
        print(f"  M5: {len(m5_csv)} candles")
        print(f"  M15: {len(m15_csv)} candles")
        
        ft_csv = prepare_features(m1_csv, m5_csv, m15_csv, mtf_fe)
        csv_last_row = ft_csv.iloc[-1]
        print(f"  CSV range: {ft_csv.index[0]} to {ft_csv.index[-1]}")
        
    except Exception as e:
        print(f"❌ Error loading/preparing CSV data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Test different LOOKBACK values
    lookback_values = [3000, 5000, 10000]
    
    print("\n" + "="*70)
    print("COMPARING LOOKBACK VALUES")
    print("="*70)
    
    results = {}
    
    for lookback in lookback_values:
        print(f"\n📊 Testing LOOKBACK={lookback}...")
        
        try:
            m1_live = fetch_binance_data(pair, '1m', lookback)
            m5_live = fetch_binance_data(pair, '5m', lookback)
            m15_live = fetch_binance_data(pair, '15m', lookback)
            
            if m1_live is None or m5_live is None or m15_live is None:
                print(f"  ❌ Failed to fetch data")
                continue
            
            ft_live = prepare_features(m1_live, m5_live, m15_live, mtf_fe)
            live_last_row = ft_live.iloc[-1]
            
            # Calculate differences for key features
            diffs = {}
            for feat in KEY_FEATURES:
                if feat in live_last_row and feat in csv_last_row:
                    live_val = live_last_row[feat]
                    csv_val = csv_last_row[feat]
                    
                    if abs(csv_val) > 1e-6:
                        diff_pct = abs((live_val - csv_val) / csv_val) * 100
                    else:
                        diff_pct = abs(live_val - csv_val) * 100 if abs(live_val) > 1e-6 else 0
                    
                    diffs[feat] = {
                        'live': live_val,
                        'csv': csv_val,
                        'diff_pct': diff_pct
                    }
            
            results[lookback] = diffs
            
            # Print summary
            avg_diff = np.mean([d['diff_pct'] for d in diffs.values()])
            max_diff = max(d['diff_pct'] for d in diffs.values())
            print(f"  📊 Avg diff: {avg_diff:.1f}% | Max diff: {max_diff:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 3. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'LOOKBACK':<10} | {'Avg Diff':<12} | {'Max Diff':<12} | Status")
    print("-"*55)
    
    for lookback, diffs in results.items():
        if diffs:
            avg_diff = np.mean([d['diff_pct'] for d in diffs.values()])
            max_diff = max(d['diff_pct'] for d in diffs.values())
            status = "✅ GOOD" if max_diff < 100 else ("⚠️ OKAY" if max_diff < 300 else "❌ BAD")
            print(f"{lookback:<10} | {avg_diff:>10.1f}% | {max_diff:>10.1f}% | {status}")
    
    print("\n" + "="*70)
    print("DETAILED FEATURE COMPARISON (LOOKBACK=10000 vs 3000)")
    print("="*70)
    
    if 10000 in results and 3000 in results:
        print(f"\n{'Feature':<30} | {'3000':<12} | {'10000':<12} | {'Improvement':<12}")
        print("-"*75)
        
        for feat in KEY_FEATURES:
            if feat in results[3000] and feat in results[10000]:
                diff_3k = results[3000][feat]['diff_pct']
                diff_10k = results[10000][feat]['diff_pct']
                improvement = (diff_3k - diff_10k) / diff_3k * 100 if diff_3k > 0 else 0
                
                status = "✅" if improvement > 50 else ("⚠️" if improvement > 0 else "❌")
                print(f"{feat:<30} | {diff_3k:>10.1f}% | {diff_10k:>10.1f}% | {improvement:>+10.1f}% {status}")
    
    print("\n💡 CONCLUSION:")
    if 10000 in results:
        max_diff_10k = max(d['diff_pct'] for d in results[10000].values()) if results[10000] else 0
        if max_diff_10k < 100:
            print("   ✅ LOOKBACK=10000 brings features within 100% of backtest values")
            print("   ✅ Model should now produce similar confidence on live")
        elif max_diff_10k < 300:
            print("   ⚠️ LOOKBACK=10000 improves but some features still differ >100%")
            print("   ⚠️ Consider using even larger LOOKBACK or investigating specific features")
        else:
            print("   ❌ LOOKBACK=10000 not enough - features still differ significantly")
            print("   ❌ Try LOOKBACK=15000 or investigate data quality")


if __name__ == '__main__':
    main()

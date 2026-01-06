#!/usr/bin/env python3
"""
Debug script to compare live features with backtest features.
This helps identify why model predictions differ.
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

# Load models
MODEL_DIR = Path("models/v8_improved")
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
    'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

# Load backtest data for comparison
def load_backtest_data(pair, data_dir):
    """Load backtest data from CSV"""
    pair_name = pair.replace('/', '_').replace(':', '_')
    try:
        m1 = pd.read_csv(data_dir / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(data_dir / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(data_dir / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
        return m1, m5, m15
    except Exception as e:
        print(f"Error loading backtest data: {e}")
        return None, None, None

def add_volume_features(df):
    """Same as in live_trading_mexc_v8.py"""
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features_live(m1, m5, m15, mtf_fe):
    """Prepare features from live data (same as live_trading_mexc_v8.py)"""
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    if len(ft) == 0:
        return pd.DataFrame()
    
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    critical_cols = ['close', 'atr']
    ft = ft.dropna(subset=critical_cols)
    
    cumsum_patterns = ['bars_since_swing', 'consecutive_up', 'consecutive_down']
    cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in cumsum_patterns)]
    if cols_to_drop:
        ft = ft.drop(columns=cols_to_drop)
    
    non_critical = [c for c in ft.columns if c not in critical_cols]
    if non_critical:
        ft[non_critical] = ft[non_critical].ffill()
    
    ft = ft.dropna(subset=critical_cols)
    return ft

def prepare_features_backtest(m1, m5, m15, mtf_fe):
    """Prepare features from backtest data (same as train_v3_dynamic.py)"""
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    return ft

def compare_features(pair, lookback=3000):
    """Compare live vs backtest features"""
    print(f"\n{'='*70}")
    print(f"Comparing features for {pair}")
    print(f"{'='*70}")
    
    # Load backtest data
    data_dir = Path("data/candles")
    m1_bt, m5_bt, m15_bt = load_backtest_data(pair, data_dir)
    if m1_bt is None:
        print(f"❌ Could not load backtest data for {pair}")
        return
    
    # Get recent backtest data (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    m1_bt = m1_bt[(m1_bt.index >= start_date) & (m1_bt.index <= end_date)]
    m5_bt = m5_bt[(m5_bt.index >= start_date) & (m5_bt.index <= end_date)]
    m15_bt = m15_bt[(m15_bt.index >= start_date) & (m15_bt.index <= end_date)]
    
    if len(m5_bt) < 100:
        print(f"❌ Insufficient backtest data: {len(m5_bt)} candles")
        return
    
    # Fetch live data
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    print(f"📥 Fetching live data from Binance...")
    data_live = {}
    for tf in ['1m', '5m', '15m']:
        candles = binance.fetch_ohlcv(pair, tf, limit=lookback)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        data_live[tf] = df
    
    # Prepare features
    mtf_fe = MTFFeatureEngine()
    
    print(f"🔧 Preparing backtest features...")
    ft_bt = prepare_features_backtest(m1_bt, m5_bt, m15_bt, mtf_fe)
    
    print(f"🔧 Preparing live features...")
    ft_live = prepare_features_live(data_live['1m'], data_live['5m'], data_live['15m'], mtf_fe)
    
    if len(ft_bt) == 0 or len(ft_live) == 0:
        print(f"❌ Could not prepare features")
        return
    
    # Get last row from each
    row_bt = ft_bt.iloc[[-1]]
    row_live = ft_live.iloc[[-2]]  # -2 because last candle might not be closed
    
    print(f"\n📊 Backtest data: {len(ft_bt)} rows, using row @ {row_bt.index[0]}")
    print(f"📊 Live data: {len(ft_live)} rows, using row @ {row_live.index[0]}")
    
    # Extract features
    cumsum_patterns = ['bars_since_swing', 'consecutive_up', 'consecutive_down']
    features_to_use = [f for f in models['features'] 
                      if not any(p in f.lower() for p in cumsum_patterns)]
    
    X_bt = row_bt[features_to_use].values
    X_live = row_live[features_to_use].values
    
    # Predictions
    dir_proba_bt = models['direction'].predict_proba(X_bt)[0]
    dir_proba_live = models['direction'].predict_proba(X_live)[0]
    
    print(f"\n🎯 PREDICTIONS:")
    print(f"   Backtest: DOWN={dir_proba_bt[0]:.3f}, SIDEWAYS={dir_proba_bt[1]:.3f}, UP={dir_proba_bt[2]:.3f}")
    print(f"   Live:     DOWN={dir_proba_live[0]:.3f}, SIDEWAYS={dir_proba_live[1]:.3f}, UP={dir_proba_live[2]:.3f}")
    
    # Compare feature values
    print(f"\n📈 FEATURE COMPARISON (top 10 differences):")
    diffs = []
    for i, feat in enumerate(features_to_use):
        if feat in row_bt.columns and feat in row_live.columns:
            val_bt = float(X_bt[0][i])
            val_live = float(X_live[0][i])
            diff = abs(val_bt - val_live)
            if not (np.isnan(val_bt) or np.isnan(val_live) or np.isinf(val_bt) or np.isinf(val_live)):
                diffs.append((feat, val_bt, val_live, diff))
    
    diffs.sort(key=lambda x: x[3], reverse=True)
    for feat, val_bt, val_live, diff in diffs[:10]:
        print(f"   {feat:40s} | BT: {val_bt:12.6f} | Live: {val_live:12.6f} | Diff: {diff:12.6f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', type=str, default='BTC/USDT:USDT', help='Pair to compare')
    parser.add_argument('--lookback', type=int, default=3000, help='Lookback candles')
    args = parser.parse_args()
    
    compare_features(args.pair, args.lookback)


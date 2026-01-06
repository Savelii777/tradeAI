#!/usr/bin/env python3
"""
Debug script to compare feature distributions between backtest and live data.
This will help identify why model confidence is low on live data.
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
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine

# Config
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
DATA_DIR = Path("data/candles")


def add_volume_features(df):
    """Add volume features (same as training)"""
    df = df.copy()
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


def load_backtest_data(pair, mtf_fe):
    """Load data from CSV (same as backtest training)"""
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    m1 = pd.read_csv(DATA_DIR / f'{pair_name}_1m.csv', index_col=0, parse_dates=True)
    m5 = pd.read_csv(DATA_DIR / f'{pair_name}_5m.csv', index_col=0, parse_dates=True)
    m15 = pd.read_csv(DATA_DIR / f'{pair_name}_15m.csv', index_col=0, parse_dates=True)
    
    # Use last 14 days (like test period)
    now = datetime.now()
    start = now - timedelta(days=14)
    
    m1 = m1[m1.index >= str(start)]
    m5 = m5[m5.index >= str(start)]
    m15 = m15[m15.index >= str(start)]
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    return ft.dropna()


def fetch_live_data(pair, binance, mtf_fe):
    """Fetch live data from Binance (same as live script)"""
    data = {}
    
    for tf in ['1m', '5m', '15m']:
        candles = binance.fetch_ohlcv(pair, tf, limit=1000)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        data[tf] = df
    
    ft = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
    ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    return ft.dropna()


def compare_features(backtest_df, live_df, feature_names):
    """Compare feature distributions between backtest and live"""
    
    # Exclude cumsum features
    cumsum_patterns = [
        'bars_since_swing', 'consecutive_up', 'consecutive_down',
        'obv', 'volume_delta_cumsum', 'swing_high_price', 'swing_low_price'
    ]
    features = [f for f in feature_names if not any(p in f.lower() for p in cumsum_patterns)]
    
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION COMPARISON: BACKTEST vs LIVE")
    print("="*80)
    
    problems = []
    
    for feat in features:
        if feat not in backtest_df.columns or feat not in live_df.columns:
            print(f"⚠️  {feat}: MISSING in one of the datasets!")
            problems.append((feat, "MISSING", 0, 0))
            continue
        
        bt_vals = backtest_df[feat].dropna()
        live_vals = live_df[feat].dropna()
        
        if len(bt_vals) == 0 or len(live_vals) == 0:
            continue
        
        bt_mean = bt_vals.mean()
        bt_std = bt_vals.std()
        live_mean = live_vals.mean()
        live_std = live_vals.std()
        
        # Calculate z-score of live mean relative to backtest distribution
        if bt_std > 0:
            z_score = abs(live_mean - bt_mean) / bt_std
        else:
            z_score = 0
        
        # Check for significant differences
        if z_score > 2:  # More than 2 standard deviations
            problems.append((feat, z_score, bt_mean, live_mean))
    
    # Sort by z-score (biggest problems first)
    problems.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else 999, reverse=True)
    
    print("\n🚨 TOP PROBLEMATIC FEATURES (z-score > 2):")
    print("-"*80)
    print(f"{'Feature':<40} {'Z-Score':<10} {'BT Mean':<15} {'Live Mean':<15}")
    print("-"*80)
    
    for feat, z, bt_m, live_m in problems[:20]:
        if isinstance(z, str):
            print(f"{feat:<40} {z:<10} {'-':<15} {'-':<15}")
        else:
            print(f"{feat:<40} {z:<10.2f} {bt_m:<15.4f} {live_m:<15.4f}")
    
    return problems


def analyze_predictions(backtest_df, live_df, models, feature_names):
    """Analyze model predictions on both datasets"""
    
    cumsum_patterns = [
        'bars_since_swing', 'consecutive_up', 'consecutive_down',
        'obv', 'volume_delta_cumsum', 'swing_high_price', 'swing_low_price'
    ]
    features = [f for f in feature_names if not any(p in f.lower() for p in cumsum_patterns)]
    
    print("\n" + "="*80)
    print("PREDICTION DISTRIBUTION COMPARISON")
    print("="*80)
    
    # Backtest predictions (last 100 rows)
    bt_sample = backtest_df.tail(500)
    missing_bt = [f for f in features if f not in bt_sample.columns]
    for mf in missing_bt:
        bt_sample[mf] = 0.0
    
    X_bt = bt_sample[features].values.astype(np.float64)
    X_bt = np.nan_to_num(X_bt, nan=0.0, posinf=0.0, neginf=0.0)
    
    bt_proba = models['direction'].predict_proba(X_bt)
    bt_preds = np.argmax(bt_proba, axis=1)
    bt_confs = np.max(bt_proba, axis=1)
    
    # Live predictions (last 100 rows)
    live_sample = live_df.tail(500)
    missing_live = [f for f in features if f not in live_sample.columns]
    for mf in missing_live:
        live_sample[mf] = 0.0
    
    X_live = live_sample[features].values.astype(np.float64)
    X_live = np.nan_to_num(X_live, nan=0.0, posinf=0.0, neginf=0.0)
    
    live_proba = models['direction'].predict_proba(X_live)
    live_preds = np.argmax(live_proba, axis=1)
    live_confs = np.max(live_proba, axis=1)
    
    print("\n📊 BACKTEST (last 500 candles):")
    print(f"   Direction distribution: DOWN={np.mean(bt_preds==0)*100:.1f}%, SIDEWAYS={np.mean(bt_preds==1)*100:.1f}%, UP={np.mean(bt_preds==2)*100:.1f}%")
    print(f"   Avg confidence: {bt_confs.mean():.3f}")
    print(f"   Confidence when LONG/SHORT: {bt_confs[bt_preds != 1].mean():.3f}")
    print(f"   High conf signals (>0.5, not sideways): {np.sum((bt_confs > 0.5) & (bt_preds != 1))}")
    
    # Distribution of probabilities
    bt_up_probs = bt_proba[:, 2]
    bt_down_probs = bt_proba[:, 0]
    bt_side_probs = bt_proba[:, 1]
    print(f"   Avg P(UP): {bt_up_probs.mean():.3f}, Avg P(DOWN): {bt_down_probs.mean():.3f}, Avg P(SIDEWAYS): {bt_side_probs.mean():.3f}")
    
    print("\n📊 LIVE (last 500 candles):")
    print(f"   Direction distribution: DOWN={np.mean(live_preds==0)*100:.1f}%, SIDEWAYS={np.mean(live_preds==1)*100:.1f}%, UP={np.mean(live_preds==2)*100:.1f}%")
    print(f"   Avg confidence: {live_confs.mean():.3f}")
    print(f"   Confidence when LONG/SHORT: {live_confs[live_preds != 1].mean():.3f}")
    print(f"   High conf signals (>0.5, not sideways): {np.sum((live_confs > 0.5) & (live_preds != 1))}")
    
    live_up_probs = live_proba[:, 2]
    live_down_probs = live_proba[:, 0]
    live_side_probs = live_proba[:, 1]
    print(f"   Avg P(UP): {live_up_probs.mean():.3f}, Avg P(DOWN): {live_down_probs.mean():.3f}, Avg P(SIDEWAYS): {live_side_probs.mean():.3f}")
    
    # Check if problem is bias toward sideways
    print("\n🔍 DIAGNOSIS:")
    if live_side_probs.mean() > bt_side_probs.mean() + 0.1:
        print("   ⚠️  Live data has HIGHER P(SIDEWAYS) than backtest!")
        print("      This means features are in a range the model associates with sideways movement.")
    
    if live_confs[live_preds != 1].mean() < bt_confs[bt_preds != 1].mean() - 0.1:
        print("   ⚠️  Live LONG/SHORT confidence is LOWER than backtest!")
        print("      Model is less certain about directional moves on live data.")


def main():
    print("="*80)
    print("DEBUG: BACKTEST vs LIVE FEATURE COMPARISON")
    print("="*80)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    print(f"Loaded {len(models['features'])} features from model")
    
    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:5]]  # Test on first 5
    
    # Init
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    for pair in pairs:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {pair}")
        print(f"{'='*80}")
        
        try:
            # Load both datasets
            print("Loading backtest data from CSV...")
            bt_df = load_backtest_data(pair, mtf_fe)
            print(f"   Backtest: {len(bt_df)} rows")
            
            print("Fetching live data from Binance...")
            live_df = fetch_live_data(pair, binance, mtf_fe)
            print(f"   Live: {len(live_df)} rows")
            
            # Compare
            problems = compare_features(bt_df, live_df, models['features'])
            analyze_predictions(bt_df, live_df, models, models['features'])
            
        except Exception as e:
            print(f"Error analyzing {pair}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("""
Possible causes of low confidence on live:
1. Feature scaling differs (backtest has more history for rolling calcs)
2. Market regime changed since training period
3. Some features have different distributions live vs backtest

Fixes to try:
1. Retrain model on more recent data
2. Lower MIN_CONF threshold temporarily to see if signals are valid
3. Use feature normalization (z-score) before prediction
4. Check if CSV data timestamps match live data timestamps
""")


if __name__ == '__main__':
    main()

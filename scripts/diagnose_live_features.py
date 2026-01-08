#!/usr/bin/env python3
"""
Diagnose Live Features: Compare features between backtest (CSV) and live (API) data preparation.

This script identifies EXACTLY which features differ between backtest and live,
and whether missing features (filled with 0) are causing low confidence.

Usage:
    python scripts/diagnose_live_features.py --pair BTC/USDT:USDT
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES, DEFAULT_EXCLUDE_FEATURES
)

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'
LOOKBACK = 3000


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


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def fetch_binance_data(pair, timeframe, limit=LOOKBACK):
    """Fetch data from Binance (same as live script)"""
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Convert pair format
    symbol = pair.replace('_', '/').replace(':USDT', '')
    if not symbol.endswith('/USDT'):
        symbol = symbol + '/USDT'
    
    print(f"  Fetching {symbol} {timeframe} from Binance...")
    
    all_ohlcv = []
    remaining = limit
    since = None
    
    while remaining > 0:
        batch_limit = min(remaining, 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
        if not ohlcv:
            break
        all_ohlcv = ohlcv + all_ohlcv
        remaining -= len(ohlcv)
        since = ohlcv[0][0] - 1
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    
    # Trim to limit
    if len(df) > limit:
        df = df.tail(limit)
    
    print(f"    Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")
    return df


def load_models():
    """Load trained models"""
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }


def prepare_features_live_style(m1, m5, m15, mtf_fe):
    """Prepare features EXACTLY like live script does"""
    
    # Pre-process data (same as live script)
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    # Remove duplicates
    m1 = m1[~m1.index.duplicated(keep='first')]
    m5 = m5[~m5.index.duplicated(keep='first')]
    m15 = m15[~m15.index.duplicated(keep='first')]
    
    print(f"  M1: {len(m1)} candles, {m1.index[0]} to {m1.index[-1]}")
    print(f"  M5: {len(m5)} candles, {m5.index[0]} to {m5.index[-1]}")
    print(f"  M15: {len(m15)} candles, {m15.index[0]} to {m15.index[-1]}")
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
    except Exception as e:
        print(f"  Error in align_timeframes: {e}")
        import traceback
        traceback.print_exc()
        return None
    if len(ft) == 0:
        return None
    
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    # Drop critical NaN
    ft = ft.dropna(subset=['close', 'atr'])
    
    # Exclude cumsum features (same as live)
    cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
    if cols_to_drop:
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
    
    # Exclude absolute features (same as live)
    cols_to_drop = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
    if cols_to_drop:
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
    
    # Forward fill non-critical
    for col in ft.columns:
        if col not in ['close', 'atr']:
            ft[col] = ft[col].ffill()
    
    ft = ft.dropna()
    return ft


def main():
    parser = argparse.ArgumentParser(description='Diagnose live features')
    parser.add_argument('--pair', type=str, default='BTC/USDT:USDT', help='Pair to analyze')
    args = parser.parse_args()
    
    pair = args.pair
    pair_csv = pair.replace('/', '_').replace(':', '_')
    
    print("="*70)
    print(f"LIVE FEATURES DIAGNOSTIC: {pair}")
    print("="*70)
    
    # Load models
    models = load_models()
    model_features = models['features']
    print(f"\nModel expects {len(model_features)} features")
    
    mtf_fe = MTFFeatureEngine()
    
    # ================================================================
    # 1. FETCH LIVE DATA (same as live script)
    # ================================================================
    print("\n" + "="*70)
    print("STEP 1: Fetching LIVE data from Binance API")
    print("="*70)
    
    try:
        m1_live = fetch_binance_data(pair, '1m', LOOKBACK)
        m5_live = fetch_binance_data(pair, '5m', LOOKBACK)
        m15_live = fetch_binance_data(pair, '15m', LOOKBACK)
    except Exception as e:
        print(f"Error fetching live data: {e}")
        print("Make sure you have internet connection")
        return
    
    # ================================================================
    # 2. PREPARE FEATURES (live style)
    # ================================================================
    print("\n" + "="*70)
    print("STEP 2: Preparing features (LIVE style)")
    print("="*70)
    
    ft_live = prepare_features_live_style(m1_live, m5_live, m15_live, mtf_fe)
    if ft_live is None or len(ft_live) == 0:
        print("Error: Could not prepare features")
        return
    
    print(f"Features prepared: {ft_live.shape}")
    print(f"Feature columns: {len(ft_live.columns)}")
    
    # ================================================================
    # 3. CHECK MISSING FEATURES
    # ================================================================
    print("\n" + "="*70)
    print("STEP 3: Checking for MISSING features")
    print("="*70)
    
    missing_features = [f for f in model_features if f not in ft_live.columns]
    extra_features = [f for f in ft_live.columns if f not in model_features]
    
    print(f"\n📊 Feature Count:")
    print(f"   Model expects: {len(model_features)}")
    print(f"   Live generated: {len(ft_live.columns)}")
    print(f"   Missing: {len(missing_features)}")
    print(f"   Extra: {len(extra_features)}")
    
    if missing_features:
        print(f"\n⚠️ MISSING FEATURES ({len(missing_features)}):")
        print("   These features are FILLED WITH 0.0 during live prediction!")
        for mf in missing_features:
            print(f"   - {mf}")
        
        print("\n   🔴 THIS IS THE PROBLEM!")
        print("   Features filled with 0 will distort model predictions.")
    else:
        print("\n✅ No missing features - all model features are present in live data")
    
    if extra_features:
        print(f"\n📝 Extra features ({len(extra_features)}) - ignored by model:")
        for ef in extra_features[:10]:
            print(f"   - {ef}")
        if len(extra_features) > 10:
            print(f"   ... and {len(extra_features)-10} more")
    
    # ================================================================
    # 4. MAKE PREDICTIONS
    # ================================================================
    print("\n" + "="*70)
    print("STEP 4: Making predictions on LIVE data")
    print("="*70)
    
    # Get last row (most recent candle)
    row = ft_live.iloc[-2:-1].copy()  # Second to last (last closed candle)
    
    # Fill missing features with 0 (same as live script)
    for mf in missing_features:
        row[mf] = 0.0
    
    # Extract features in model order
    X = row[model_features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Predict
    dir_proba = models['direction'].predict_proba(X)
    dir_pred = np.argmax(dir_proba[0])
    dir_conf = np.max(dir_proba[0])
    
    timing_pred = models['timing'].predict(X)[0]
    strength_pred = models['strength'].predict(X)[0]
    
    direction = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
    
    print(f"\nPrediction on last closed candle:")
    print(f"   Timestamp: {ft_live.index[-2]}")
    print(f"   Close: {row['close'].values[0]:.2f}")
    print(f"   Direction: {direction}")
    print(f"   Confidence: {dir_conf:.4f}")
    print(f"   Timing: {timing_pred:.2f}")
    print(f"   Strength: {strength_pred:.2f}")
    print(f"   Probabilities: DOWN={dir_proba[0][0]:.3f}, SIDE={dir_proba[0][1]:.3f}, UP={dir_proba[0][2]:.3f}")
    
    # ================================================================
    # 5. COMPARE WITH/WITHOUT ZERO-FILLED FEATURES
    # ================================================================
    if missing_features:
        print("\n" + "="*70)
        print("STEP 5: Impact of ZERO-FILLED features on prediction")
        print("="*70)
        
        print("\n📊 Comparison:")
        print(f"   With 0-filled missing features: {direction} @ {dir_conf:.4f}")
        
        # Show which features are 0 and their typical ranges
        print(f"\n   Missing features that are filled with 0:")
        for mf in missing_features:
            print(f"     {mf} = 0.0 (should be non-zero!)")
    
    # ================================================================
    # 6. CHECK FEATURE VALUES
    # ================================================================
    print("\n" + "="*70)
    print("STEP 6: Feature value sanity check")
    print("="*70)
    
    # Check for suspicious values
    feature_vals = {model_features[i]: float(X[0][i]) for i in range(len(model_features))}
    
    zero_features = [f for f, v in feature_vals.items() if v == 0.0]
    nan_features = [f for f, v in feature_vals.items() if np.isnan(v)]
    inf_features = [f for f, v in feature_vals.items() if np.isinf(v)]
    
    print(f"\n   Features with value = 0: {len(zero_features)}")
    if zero_features:
        print(f"     (First 10): {zero_features[:10]}")
    
    print(f"   Features with NaN: {len(nan_features)}")
    print(f"   Features with Inf: {len(inf_features)}")
    
    # ================================================================
    # 7. CONCLUSION
    # ================================================================
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if missing_features:
        print(f"""
🔴 PROBLEM IDENTIFIED: {len(missing_features)} missing features!

These features are expected by the model but NOT generated in live:
{missing_features}

They are filled with 0.0, which distorts predictions and causes
low confidence (e.g., 41% instead of 65%).

SOLUTION:
1. Check why these features are not generated in MTFFeatureEngine
2. Or retrain model WITHOUT these features
3. Run: python scripts/train_v3_dynamic.py --days 60 --test_days 14 --walk-forward
""")
    else:
        print(f"""
✅ All features present - no missing features problem.

If confidence is still low, the issue may be:
1. Current market conditions differ from training period
2. Feature values are in different ranges (check feature scaling)
3. Model is correctly predicting SIDEWAYS for this pair/period
""")


if __name__ == '__main__':
    main()

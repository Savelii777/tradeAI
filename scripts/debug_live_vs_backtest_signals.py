#!/usr/bin/env python3
"""
Debug script to compare live vs backtest signal generation
Fetches the same data and compares predictions
"""

import sys
import pandas as pd
import numpy as np
import ccxt
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

# Config
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 1500

# Thresholds
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


def load_models():
    """Load trained models"""
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    logger.info(f"✅ Loaded models from {MODEL_DIR}")
    logger.info(f"   Features: {len(models['features'])}")
    return models


def add_volume_features(df):
    """Add volume features"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['obv'] = (df['price_change'].apply(np.sign) * df['volume']).cumsum()
    df['obv_sma'] = df['obv'].rolling(20).mean()
    return df


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def prepare_features(data, mtf_fe):
    """Prepare features from multi-timeframe data (same as live)"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Fill NaN
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        ft = ft.ffill().bfill()
        
        if ft.isna().any().any():
            ft = ft.fillna(0)
        
        return ft
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()


def generate_signals_backtest(df, feature_cols, models, pair_name):
    """Generate signals like in backtest (all rows)"""
    signals = []
    
    X = df[feature_cols].values
    
    # Predictions
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    timing_preds = models['timing'].predict(X)
    strength_preds = models['strength'].predict(X)
    
    # Filter
    for i in range(len(df)):
        if dir_preds[i] == 1:  # Sideways
            continue
        
        if dir_confs[i] < MIN_CONF:
            continue
        if timing_preds[i] < MIN_TIMING:
            continue
        if strength_preds[i] < MIN_STRENGTH:
            continue
        
        signals.append({
            'timestamp': df.index[i],
            'pair': pair_name,
            'direction': 'LONG' if dir_preds[i] == 2 else 'SHORT',
            'entry_price': df['close'].iloc[i],
            'atr': df['atr'].iloc[i],
            'conf': dir_confs[i],
            'timing': timing_preds[i],
            'strength': strength_preds[i]
        })
    
    return signals


def check_live_signal(df, models):
    """Check signal like in live (only last closed candle)"""
    if len(df) < 2:
        return None
    
    row = df.iloc[[-2]]
    
    # Validate features
    missing_features = [f for f in models['features'] if f not in row.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features[:5]}...")
        return None
    
    X = row[models['features']].values
    
    if pd.isna(X).any():
        return None
    
    # Predictions
    dir_proba = models['direction'].predict_proba(X)
    dir_conf = float(np.max(dir_proba))
    dir_pred = int(np.argmax(dir_proba))
    timing_pred = float(models['timing'].predict(X)[0])
    strength_pred = float(models['strength'].predict(X)[0])
    
    direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
    
    # Check filters
    passed = True
    reasons = []
    
    if dir_pred == 1:
        passed = False
        reasons.append("SIDEWAYS")
    if dir_conf < MIN_CONF:
        passed = False
        reasons.append(f"Conf({dir_conf:.2f}<{MIN_CONF})")
    if timing_pred < MIN_TIMING:
        passed = False
        reasons.append(f"Timing({timing_pred:.2f}<{MIN_TIMING})")
    if strength_pred < MIN_STRENGTH:
        passed = False
        reasons.append(f"Strength({strength_pred:.1f}<{MIN_STRENGTH})")
    
    return {
        'timestamp': row.index[0],
        'direction': direction_str,
        'conf': dir_conf,
        'timing': timing_pred,
        'strength': strength_pred,
        'passed': passed,
        'reasons': reasons
    }


def main():
    import json
    
    logger.info("=" * 70)
    logger.info("🔍 Debug: Live vs Backtest Signal Comparison")
    logger.info("=" * 70)
    
    # Load models
    models = load_models()
    
    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs_data = json.load(f)
    pairs = [p['symbol'] for p in pairs_data['pairs'][:5]]  # Test first 5 pairs
    
    logger.info(f"Testing {len(pairs)} pairs")
    
    # Initialize exchange
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    mtf_fe = MTFFeatureEngine()
    
    # Test period: last 7 days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    results = []
    
    for pair in pairs:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {pair}")
        logger.info(f"{'='*70}")
        
        try:
            # Fetch data
            data = {}
            for tf in TIMEFRAMES:
                since = int((start_date - timedelta(days=1)).timestamp() * 1000)
                candles = binance.fetch_ohlcv(pair, tf, since=since, limit=LOOKBACK)
                
                if not candles or len(candles) < 50:
                    logger.warning(f"  Not enough {tf} data")
                    break
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                data[tf] = df
            
            if len(data) < 3:
                continue
            
            # Prepare features
            df_features = prepare_features(data, mtf_fe)
            if df_features is None or len(df_features) < 2:
                logger.warning(f"  Failed to prepare features")
                continue
            
            # Filter to test period
            df_test = df_features[(df_features.index >= start_date) & (df_features.index <= end_date)]
            
            if len(df_test) == 0:
                logger.warning(f"  No data in test period")
                continue
            
            logger.info(f"  Features prepared: {len(df_test)} bars")
            
            # Check live signal (last closed candle)
            live_result = check_live_signal(df_features, models)
            
            if live_result:
                logger.info(f"\n  📊 LIVE SIGNAL CHECK (last closed candle):")
                logger.info(f"     Timestamp: {live_result['timestamp']}")
                logger.info(f"     Direction: {live_result['direction']}")
                logger.info(f"     Conf: {live_result['conf']:.2f}")
                logger.info(f"     Timing: {live_result['timing']:.2f} ATR")
                logger.info(f"     Strength: {live_result['strength']:.1f}")
                logger.info(f"     Passed: {'✅ YES' if live_result['passed'] else '❌ NO'}")
                if live_result['reasons']:
                    logger.info(f"     Reasons: {', '.join(live_result['reasons'])}")
            
            # Generate backtest signals (all rows in test period)
            feature_cols = [f for f in models['features'] if f in df_test.columns]
            missing = [f for f in models['features'] if f not in df_test.columns]
            
            if missing:
                logger.warning(f"  Missing {len(missing)} features in test data")
                # Add missing features as zeros
                for f in missing:
                    df_test[f] = 0
            
            backtest_signals = generate_signals_backtest(df_test, models['features'], models, pair)
            
            logger.info(f"\n  📊 BACKTEST SIGNALS (all bars in test period):")
            logger.info(f"     Total signals: {len(backtest_signals)}")
            
            if len(backtest_signals) > 0:
                logger.info(f"     First signal: {backtest_signals[0]['timestamp']} | {backtest_signals[0]['direction']} | Conf: {backtest_signals[0]['conf']:.2f}")
                logger.info(f"     Last signal: {backtest_signals[-1]['timestamp']} | {backtest_signals[-1]['direction']} | Conf: {backtest_signals[-1]['conf']:.2f}")
            
            # Compare
            logger.info(f"\n  🔄 COMPARISON:")
            if live_result and live_result['passed']:
                logger.info(f"     ✅ Live: SIGNAL FOUND")
            else:
                logger.info(f"     ❌ Live: NO SIGNAL")
            
            if len(backtest_signals) > 0:
                logger.info(f"     ✅ Backtest: {len(backtest_signals)} SIGNALS FOUND")
            else:
                logger.info(f"     ❌ Backtest: NO SIGNALS")
            
            # Check if live timestamp matches any backtest signal
            if live_result and len(backtest_signals) > 0:
                live_ts = live_result['timestamp']
                matching = [s for s in backtest_signals if abs((s['timestamp'] - live_ts).total_seconds()) < 300]
                if matching:
                    logger.info(f"     ✅ Live timestamp matches {len(matching)} backtest signal(s)")
                else:
                    logger.info(f"     ⚠️  Live timestamp does NOT match any backtest signal")
                    logger.info(f"        Live: {live_ts}")
                    logger.info(f"        Nearest backtest: {backtest_signals[0]['timestamp']}")
            
            results.append({
                'pair': pair,
                'live_signal': live_result['passed'] if live_result else False,
                'backtest_signals': len(backtest_signals),
                'live_conf': live_result['conf'] if live_result else 0,
                'live_timing': live_result['timing'] if live_result else 0,
                'live_strength': live_result['strength'] if live_result else 0
            })
            
        except Exception as e:
            logger.error(f"  Error processing {pair}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("📊 SUMMARY")
    logger.info(f"{'='*70}")
    
    live_signals = sum(1 for r in results if r['live_signal'])
    backtest_total = sum(r['backtest_signals'] for r in results)
    
    logger.info(f"Pairs tested: {len(results)}")
    logger.info(f"Live signals: {live_signals}/{len(results)}")
    logger.info(f"Backtest signals: {backtest_total} total")
    
    logger.info(f"\nDetailed results:")
    for r in results:
        logger.info(f"  {r['pair']}: Live={r['live_signal']}, Backtest={r['backtest_signals']}, "
                   f"Conf={r['live_conf']:.2f}, Timing={r['live_timing']:.2f}, Strength={r['live_strength']:.1f}")


if __name__ == '__main__':
    main()


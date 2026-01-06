#!/usr/bin/env python3
"""
Диагностика проблемы с confidence для LONG/SHORT сигналов.
Проверяет почему confidence всегда < 0.40 для LONG/SHORT.
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'


def prepare_features(data, mtf_fe):
    """Prepare features from multi-timeframe data (same as live_trading_mexc_v8.py)"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    print(f"   Checking data lengths: 1m={len(m1)}, 5m={len(m5)}, 15m={len(m15)}")
    
    # Check minimum requirements (same as live_trading_mexc_v8.py)
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        print(f"   ❌ Insufficient data (need at least 50 candles for each timeframe)")
        print(f"      Current: 1m={len(m1)}, 5m={len(m5)}, 15m={len(m15)}")
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        print(f"   Aligning timeframes...")
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        print(f"   After align_timeframes: {len(ft)} rows")
        
        if len(ft) == 0:
            print(f"   ❌ align_timeframes returned empty DataFrame")
            return pd.DataFrame()
        
        print(f"   Joining 5m data...")
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        print(f"   After join: {len(ft)} rows")
        
        print(f"   Adding volume features...")
        ft = add_volume_features(ft)
        print(f"   After volume features: {len(ft)} rows")
        
        print(f"   Calculating ATR...")
        ft['atr'] = calculate_atr(ft)
        print(f"   After ATR: {len(ft)} rows")
        
        # Fill NaN (same as live_trading_mexc_v8.py)
        print(f"   Handling NaN...")
        critical_cols = ['close', 'atr']
        before_dropna = len(ft)
        ft = ft.dropna(subset=critical_cols)  # Only drop rows missing critical columns
        after_dropna = len(ft)
        print(f"   After dropna(subset={critical_cols}): {before_dropna} -> {after_dropna} rows")
        
        ft = ft.ffill().bfill()  # Forward/backward fill
        
        if ft.isna().any().any():
            print(f"   Filling remaining NaN with 0...")
            ft = ft.fillna(0)
        
        print(f"   Final: {len(ft)} rows")
        
        return ft
    except Exception as e:
        print(f"   ❌ Error in prepare_features: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features (same as live_trading_mexc_v8.py)"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df, period=14):
    """Calculate ATR (same as live_trading_mexc_v8.py)"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_models():
    """Load trained models."""
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    return models

def analyze_predictions(models, pair='BTC/USDT', limit=100):
    """Анализирует предсказания модели на последних данных."""
    print(f"\n{'='*70}")
    print(f"АНАЛИЗ ПРЕДСКАЗАНИЙ ДЛЯ {pair}")
    print(f"{'='*70}\n")
    
    # Fetch live data
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Get 5m data
    print(f"📥 Fetching data for {pair}...")
    ohlcv = binance.fetch_ohlcv(pair, '5m', limit=limit)
    df_5m = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='ms', utc=True)
    df_5m.set_index('timestamp', inplace=True)
    print(f"   ✅ 5m: {len(df_5m)} candles")
    
    # Get 1m and 15m
    ohlcv_1m = binance.fetch_ohlcv(pair, '1m', limit=limit*5)
    df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms', utc=True)
    df_1m.set_index('timestamp', inplace=True)
    print(f"   ✅ 1m: {len(df_1m)} candles")
    
    # For 15m, we need at least 50 candles, so use max(limit//3, 50)
    ohlcv_15m = binance.fetch_ohlcv(pair, '15m', limit=max(limit//3, 50))
    df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], unit='ms', utc=True)
    df_15m.set_index('timestamp', inplace=True)
    print(f"   ✅ 15m: {len(df_15m)} candles")
    
    # Feature engineering
    print(f"\n🔧 Preparing features...")
    mtf_fe = MTFFeatureEngine()
    
    # Prepare features same way as in live trading
    data = {'1m': df_1m, '5m': df_5m, '15m': df_15m}
    features = prepare_features(data, mtf_fe)
    
    if features is None or len(features) == 0:
        print("❌ ERROR: prepare_features returned empty DataFrame!")
        print("   This might be due to insufficient data or feature engineering issues.")
        return
    
    if features is None or len(features) == 0:
        print("❌ ERROR: No features generated!")
        return
    
    print(f"✅ Generated {len(features)} rows of features")
    
    # Add missing features
    for feat in models['features']:
        if feat not in features.columns:
            features[feat] = 0
    
    # Analyze last 20 predictions
    X_df = features[models['features']].tail(20)
    
    if len(X_df) == 0:
        print("❌ ERROR: No data after filtering!")
        return
    
    # Save index before converting to numpy
    X_index = X_df.index
    
    X = X_df.fillna(0).replace([np.inf, -np.inf], 0).values
    
    if X.shape[0] == 0:
        print("❌ ERROR: Empty X array!")
        return
    
    print(f"✅ Preparing to analyze {X.shape[0]} predictions")
    
    dir_proba_all = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba_all, axis=1)
    dir_confs = np.max(dir_proba_all, axis=1)
    
    print("Последние 20 предсказаний:\n")
    print(f"{'Time':<20} {'Direction':<10} {'Conf':<6} {'SHORT':<8} {'SIDEWAYS':<10} {'LONG':<8}")
    print("-" * 70)
    
    long_count = 0
    short_count = 0
    sideways_count = 0
    
    long_confs = []
    short_confs = []
    sideways_confs = []
    
    for i in range(len(X)):
        idx = X_index[i]  # Use saved index
        dir_proba = dir_proba_all[i]
        dir_pred = dir_preds[i]
        dir_conf = dir_confs[i]
        
        direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
        
        print(f"{str(idx):<20} {direction_str:<10} {dir_conf:.2f}   {dir_proba[0]:.3f}    {dir_proba[1]:.3f}      {dir_proba[2]:.3f}")
        
        if dir_pred == 2:  # LONG
            long_count += 1
            long_confs.append(dir_conf)
        elif dir_pred == 0:  # SHORT
            short_count += 1
            short_confs.append(dir_conf)
        else:  # SIDEWAYS
            sideways_count += 1
            sideways_confs.append(dir_conf)
    
    print("\n" + "="*70)
    print("СТАТИСТИКА:")
    print("="*70)
    print(f"LONG сигналов: {long_count}")
    if long_confs:
        print(f"  Средний confidence: {np.mean(long_confs):.3f}")
        print(f"  Мин confidence: {np.min(long_confs):.3f}")
        print(f"  Макс confidence: {np.max(long_confs):.3f}")
        print(f"  Confidence > 0.50: {sum(1 for c in long_confs if c > 0.50)}")
    
    print(f"\nSHORT сигналов: {short_count}")
    if short_confs:
        print(f"  Средний confidence: {np.mean(short_confs):.3f}")
        print(f"  Мин confidence: {np.min(short_confs):.3f}")
        print(f"  Макс confidence: {np.max(short_confs):.3f}")
        print(f"  Confidence > 0.50: {sum(1 for c in short_confs if c > 0.50)}")
    
    print(f"\nSIDEWAYS сигналов: {sideways_count}")
    if sideways_confs:
        print(f"  Средний confidence: {np.mean(sideways_confs):.3f}")
        print(f"  Мин confidence: {np.min(sideways_confs):.3f}")
        print(f"  Макс confidence: {np.max(sideways_confs):.3f}")
    
    print("\n" + "="*70)
    print("ПРОБЛЕМА:")
    print("="*70)
    if long_confs and max(long_confs) < 0.50:
        print(f"❌ LONG сигналы имеют confidence < 0.50 (макс: {max(long_confs):.3f})")
    if short_confs and max(short_confs) < 0.50:
        print(f"❌ SHORT сигналы имеют confidence < 0.50 (макс: {max(short_confs):.3f})")
    if sideways_confs and min(sideways_confs) > 0.50:
        print(f"⚠️  SIDEWAYS сигналы имеют высокий confidence (мин: {min(sideways_confs):.3f})")
        print("   Модель слишком уверена в SIDEWAYS, что подавляет LONG/SHORT!")
    
    # Проверяем распределение вероятностей
    print("\n" + "="*70)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ВЕРОЯТНОСТЕЙ:")
    print("="*70)
    
    all_proba = dir_proba_all.reshape(-1)
    print(f"Всего предсказаний: {len(dir_proba_all)}")
    print(f"\nРаспределение вероятностей:")
    print(f"  SHORT (класс 0):   мин={dir_proba_all[:, 0].min():.3f}, макс={dir_proba_all[:, 0].max():.3f}, среднее={dir_proba_all[:, 0].mean():.3f}")
    print(f"  SIDEWAYS (класс 1): мин={dir_proba_all[:, 1].min():.3f}, макс={dir_proba_all[:, 1].max():.3f}, среднее={dir_proba_all[:, 1].mean():.3f}")
    print(f"  LONG (класс 2):     мин={dir_proba_all[:, 2].min():.3f}, макс={dir_proba_all[:, 2].max():.3f}, среднее={dir_proba_all[:, 2].mean():.3f}")
    
    # Проверяем случаи когда LONG/SHORT предсказан, но confidence низкий
    print("\n" + "="*70)
    print("СЛУЧАИ КОГДА LONG/SHORT ПРЕДСКАЗАН С НИЗКИМ CONFIDENCE:")
    print("="*70)
    
    for i in range(len(X)):
        dir_proba = dir_proba_all[i]
        dir_pred = dir_preds[i]
        dir_conf = dir_confs[i]
        
        if dir_pred in [0, 2] and dir_conf < 0.50:  # LONG или SHORT с низким confidence
            direction_str = 'LONG' if dir_pred == 2 else 'SHORT'
            print(f"\n{direction_str} предсказан с confidence {dir_conf:.3f}:")
            print(f"  SHORT: {dir_proba[0]:.3f}")
            print(f"  SIDEWAYS: {dir_proba[1]:.3f}")
            print(f"  LONG: {dir_proba[2]:.3f}")
            print(f"  Разница между max и вторым: {dir_conf - sorted(dir_proba)[1]:.3f}")

if __name__ == '__main__':
    models = load_models()
    # Use larger limit to ensure we have enough 15m candles
    analyze_predictions(models, limit=150)  # Increased from 100 to ensure 15m has >= 50


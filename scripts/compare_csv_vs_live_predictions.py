#!/usr/bin/env python3
"""
Сравнение предсказаний модели на CSV данных (как в бектесте) 
и на данных с биржи (как в лайве) за тот же период.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import ccxt
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

def load_models():
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    return models

def load_csv_data(pair, days=7):
    """Load data from CSV (as in backtest)"""
    # Format: BTC/USDT -> BTC_USDT_USDT (as in train_v3_dynamic.py)
    pair_name = pair.replace('/', '_').replace(':', '_')
    if not pair_name.endswith('_USDT'):
        pair_name = pair_name + '_USDT'
    
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError as e:
        print(f"   FileNotFoundError: {e}")
        return None
    
    # Get last N days
    end_time = m5.index[-1]
    start_time = end_time - timedelta(days=days)
    
    m1 = m1[(m1.index >= start_time) & (m1.index <= end_time)]
    m5 = m5[(m5.index >= start_time) & (m5.index <= end_time)]
    m15 = m15[(m15.index >= start_time) & (m15.index <= end_time)]
    
    return {'1m': m1, '5m': m5, '15m': m15}

def fetch_live_data(pair, days=7):
    """Fetch data from exchange (as in live trading)"""
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Calculate start time
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)
    
    data = {}
    for tf in ['1m', '5m', '15m']:
        limit_map = {'1m': 10000, '5m': 2000, '15m': 1000}
        candles = binance.fetch_ohlcv(pair, tf, since=since, limit=limit_map[tf])
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates and sort (same as live_trading_mexc_v8.py)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        data[tf] = df
    
    return data

def prepare_features(data, mtf_fe):
    """Prepare features (same as live_trading_mexc_v8.py)"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        return pd.DataFrame()
    
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        
        # Add volume features
        ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
        ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
        ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
        ft['price_change'] = ft['close'].diff()
        ft['vwap'] = (ft['close'] * ft['volume']).rolling(20).sum() / ft['volume'].rolling(20).sum()
        ft['price_vs_vwap'] = ft['close'] / ft['vwap'] - 1
        ft['vol_momentum'] = ft['volume'].pct_change(5)
        
        # ATR
        high = ft['high']
        low = ft['low']
        close = ft['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        ft['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        # Fill NaN
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        ft = ft.ffill().bfill()
        if ft.isna().any().any():
            ft = ft.fillna(0)
        
        return ft
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def analyze_predictions(features, models, source_name):
    """Analyze model predictions"""
    if len(features) == 0:
        print(f"❌ {source_name}: No features")
        return
    
    # Add missing features
    for feat in models['features']:
        if feat not in features.columns:
            features[feat] = 0
    
    X = features[models['features']].tail(50).fillna(0).replace([np.inf, -np.inf], 0).values
    
    if X.shape[0] == 0:
        print(f"❌ {source_name}: Empty X")
        return
    
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    
    long_count = sum(1 for p in dir_preds if p == 2)
    short_count = sum(1 for p in dir_preds if p == 0)
    sideways_count = sum(1 for p in dir_preds if p == 1)
    
    long_confs = [dir_confs[i] for i in range(len(dir_preds)) if dir_preds[i] == 2]
    short_confs = [dir_confs[i] for i in range(len(dir_preds)) if dir_preds[i] == 0]
    
    print(f"\n{source_name}:")
    print(f"  Всего предсказаний: {len(dir_preds)}")
    print(f"  LONG: {long_count} (conf: {np.mean(long_confs) if long_confs else 0:.3f})")
    print(f"  SHORT: {short_count} (conf: {np.mean(short_confs) if short_confs else 0:.3f})")
    print(f"  SIDEWAYS: {sideways_count}")
    print(f"  LONG/SHORT с conf > 0.50: {sum(1 for c in long_confs + short_confs if c > 0.50)}")

if __name__ == '__main__':
    pair = 'BTC/USDT'
    models = load_models()
    mtf_fe = MTFFeatureEngine()
    
    print("="*70)
    print("СРАВНЕНИЕ CSV vs LIVE ДАННЫХ")
    print("="*70)
    
    # CSV data
    print("\n📂 Загрузка CSV данных...")
    csv_data = load_csv_data(pair, days=7)
    if csv_data:
        csv_features = prepare_features(csv_data, mtf_fe)
        analyze_predictions(csv_features, models, "CSV (как в бектесте)")
    else:
        print("❌ CSV данные не найдены")
    
    # Live data
    print("\n🌐 Загрузка данных с биржи...")
    live_data = fetch_live_data(pair, days=7)
    if live_data:
        live_features = prepare_features(live_data, mtf_fe)
        analyze_predictions(live_features, models, "LIVE (как в лайве)")
    else:
        print("❌ Не удалось загрузить данные с биржи")


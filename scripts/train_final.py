#!/usr/bin/env python3
"""
Train Final Model (V7 Sniper)
Trains on the last 30 days of available data (2025-11-24 to 2025-12-24).
Saves the models and feature list for Paper Trading.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
LOOKAHEAD = 12       # 1 hour on M5
MODEL_DIR = Path("models/v7_sniper_final")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA LOADING (Local)
# ============================================================
def load_local_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    """Load data from local CSVs."""
    clean_symbol = symbol.replace('/', '_').replace(':', '_')
    file_path = Path(f"data/candles/{clean_symbol}_{timeframe}.csv")
    
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            df = df.sort_index()
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df[mask]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# ============================================================
# FEATURE ENGINEERING (Copied from train_v3_dynamic.py)
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def create_targets_v1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = calculate_atr(df)
    rolling_vol = df['close'].pct_change().rolling(window=100).std()
    threshold = np.maximum(rolling_vol, 0.005)
    future_return = df['close'].pct_change(LOOKAHEAD).shift(-LOOKAHEAD)
    
    df['target_dir'] = np.where(future_return > threshold, 2, np.where(future_return < -threshold, 0, 1))
    
    future_lows = df['low'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
    future_highs = df['high'].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
    
    adv_long = (df['close'] - future_lows) / df['atr']
    fav_long = (future_highs - df['close']) / df['atr']
    adv_short = (future_highs - df['close']) / df['atr']
    fav_short = (df['close'] - future_lows) / df['atr']
    
    is_good_long = (fav_long > adv_long) & (fav_long > 1.0)
    is_good_short = (fav_short > adv_short) & (fav_short > 1.0)
    
    df['target_timing'] = 0
    df.loc[(df['target_dir'] == 2) & is_good_long, 'target_timing'] = 1
    df.loc[(df['target_dir'] == 0) & is_good_short, 'target_timing'] = 1
    
    move_long = (future_highs - df['close']) / df['atr']
    move_short = (df['close'] - future_lows) / df['atr']
    df['target_strength'] = np.where(df['target_dir'] == 2, move_long, np.where(df['target_dir'] == 0, move_short, 0))
    df['target_strength'] = df['target_strength'].clip(0, 10)
    return df

# ============================================================
# TRAINING
# ============================================================
def train_models(X_train, y_train):
    print("   Training Direction Model...")
    dir_model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        n_estimators=600, max_depth=5, num_leaves=20,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    dir_model.fit(X_train, y_train['target_dir'])
    
    print("   Training Timing Model...")
    timing_model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        n_estimators=400, max_depth=5, num_leaves=15,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    timing_model.fit(X_train, y_train['target_timing'])
    
    print("   Training Strength Model...")
    strength_model = lgb.LGBMRegressor(
        objective='regression', metric='mse',
        n_estimators=400, max_depth=5, num_leaves=15,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    strength_model.fit(X_train, y_train['target_strength'])
    
    return {'direction': dir_model, 'timing': timing_model, 'strength': strength_model}

def main():
    # Define Training Window (Last 30 Days)
    # Based on data check: End is 2025-12-24
    end_date = datetime(2025, 12, 24, 22, 0, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=30)
    
    print(f"Training Final Model")
    print(f"Window: {start_date} to {end_date}")
    print("-" * 50)
    
    # Load Pairs
    import json
    with open("config/pairs_list.json", "r") as f:
        data = json.load(f)
        all_pairs = [p['symbol'] for p in data['pairs']]
    
    # Use top 20 pairs
    pairs = all_pairs[:20]
    
    mtf_fe = MTFFeatureEngine()
    all_train_dfs = []
    
    print("Loading and processing data...")
    for pair in pairs:
        print(f"Processing {pair}...", end='\r')
        m1 = load_local_data(pair, '1m', start_date, end_date)
        m5 = load_local_data(pair, '5m', start_date, end_date)
        m15 = load_local_data(pair, '15m', start_date, end_date)
        
        if len(m5) < 500: continue
        
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft = create_targets_v1(ft)
        ft['pair'] = pair
        
        all_train_dfs.append(ft.dropna())
        
    print("\nData loaded.")
    
    # Combine
    train_df = pd.concat(all_train_dfs)
    
    # Define Features
    exclude = ['pair', 'target_dir', 'target_timing', 'target_strength', 
               'open', 'high', 'low', 'close', 'volume', 'atr', 'price_change', 'obv', 'obv_sma']
    features = [c for c in train_df.columns if c not in exclude]
    
    print(f"Training on {len(train_df)} samples with {len(features)} features.")
    
    X_train = train_df[features]
    y_train = {
        'target_dir': train_df['target_dir'],
        'target_timing': train_df['target_timing'],
        'target_strength': train_df['target_strength']
    }
    
    # Train
    models = train_models(X_train, y_train)
    
    # Save
    print(f"Saving models to {MODEL_DIR}...")
    joblib.dump(models['direction'], MODEL_DIR / 'direction_model.pkl')
    joblib.dump(models['timing'], MODEL_DIR / 'timing_model.pkl')
    joblib.dump(models['strength'], MODEL_DIR / 'strength_model.pkl')
    joblib.dump(features, MODEL_DIR / 'features.pkl')
    
    print("Done!")

if __name__ == '__main__':
    main()

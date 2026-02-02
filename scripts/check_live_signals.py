#!/usr/bin/env python3
"""
Проверка сигналов на live данных.
Загружает модель и данные, рассчитывает фичи и проверяет сигналы.

СИНХРОНИЗИРОВАНО С train_v3_dynamic.py:
- Пороги: MIN_CONFIDENCE=0.60, MIN_TIMING=1.7, MIN_STRENGTH=2.3
- Логика фильтрации полностью совпадает с generate_signals()
- Используется для сравнения лайв данных с бэктестом
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

from train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES


# Ensemble model class (needed for unpickling)
class EnsembleDirectionModel(ClassifierMixin, BaseEstimator):
    """Ensemble of LightGBM + CatBoost."""
    _estimator_type = "classifier"
    
    def __init__(self, lgb_params=None, use_catboost=True):
        self.lgb_params = lgb_params
        self.use_catboost = use_catboost
        self.lgb_model = None
        self.catboost_model = None
        self.weights = {'lgb': 0.5, 'catboost': 0.5}
        self._classes = np.array([0, 1, 2])
    
    @property
    def classes_(self):
        return self._classes
        
    def fit(self, X, y, sample_weight=None):
        lgb_default = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 50,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1
        }
        lgb_default.update(self.lgb_params or {})
        
        self.lgb_model = lgb.LGBMClassifier(**lgb_default)
        self.lgb_model.fit(X, y, sample_weight=sample_weight)
        
        if self.use_catboost:
            self.catboost_model = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.03,
                l2_leaf_reg=3, loss_function='MultiClass',
                classes_count=3, random_seed=42, verbose=False
            )
            self.catboost_model.fit(X, y, sample_weight=sample_weight)
        
        self._classes = np.unique(y)
        return self
    
    def predict_proba(self, X):
        lgb_proba = self.lgb_model.predict_proba(X)
        if self.use_catboost and self.catboost_model is not None:
            cat_proba = self.catboost_model.predict_proba(X)
            return self.weights['lgb'] * lgb_proba + self.weights['catboost'] * cat_proba
        return lgb_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    @property
    def feature_importances_(self):
        if self.lgb_model is not None:
            return self.lgb_model.feature_importances_
        return None

# Настройки
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_20.json"

# Пороги из модели (СИНХРОНИЗИРОВАНО С TRAIN_V3_DYNAMIC.PY)
MIN_CONFIDENCE = 0.60
MIN_TIMING = 1.7
MIN_STRENGTH = 2.3

LOOKBACK_M1 = 7500
LOOKBACK_M5 = 1500
LOOKBACK_M15 = 500


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features."""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def prepare_features(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, mtf_fe: MTFFeatureEngine, end_date=None, remove_last=True) -> pd.DataFrame:
    """Prepare features exactly like backtest.
    
    Args:
        remove_last: If True, removes last candle (for live mode to avoid forming candle).
                     If False, keeps all candles (for historical backtest mode).
    """
    # Filter data up to specific date if provided
    if end_date is not None:
        m1 = m1[m1.index <= end_date]
        m5 = m5[m5.index <= end_date]
        m15 = m15[m15.index <= end_date]
    
    # Remove last candle ONLY in live mode to prevent lookahead bias
    # In backtest/historical mode, all candles are already closed
    if remove_last:
        m1 = m1.iloc[:-1]
        m5 = m5.iloc[:-1]
        m15 = m15.iloc[:-1]
    
    # Use lookback windows
    m1 = m1.tail(LOOKBACK_M1)
    m5 = m5.tail(LOOKBACK_M5)
    m15 = m15.tail(LOOKBACK_M15)
    
    # Align timeframes
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    
    if len(ft) == 0:
        return pd.DataFrame()
    
    # Add OHLCV from M5
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    
    # Add volume features
    ft = add_volume_features(ft)
    
    # Add ATR
    ft['atr'] = calculate_atr(ft)
    
    # NaN handling
    ft = ft.dropna(subset=['close', 'atr'])
    
    # Exclude cumsum-dependent features
    cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
    ft = ft.drop(columns=cols_to_drop, errors='ignore')
    
    # Exclude absolute price features
    absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
    ft = ft.drop(columns=absolute_cols, errors='ignore')
    
    # Forward fill and final dropna
    ft = ft.ffill().dropna()
    
    return ft


def load_parquet(pair: str, timeframe: str) -> pd.DataFrame:
    """Load parquet data."""
    safe_symbol = pair.replace('/', '_').replace(':', '_')
    filepath = DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"
    
    if not filepath.exists():
        return None
    
    df = pd.read_parquet(filepath)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    return df


def main(check_date=None, target_time=None, show_all=False):
    """Check signals on live data.
    
    Args:
        check_date: Date to check (e.g., '2026-01-25'). If None, uses latest data.
        target_time: Specific time to check (e.g., '17:45'). Only used with check_date.
        show_all: If True, show all predictions regardless of threshold.
    """
    import json
    
    if check_date:
        print(f"📅 Checking signals for date: {check_date}")
        if target_time:
            print(f"⏰ Specific time: {target_time}")
        if show_all:
            print(f"📊 Mode: Show ALL pairs (including below threshold)")
        check_datetime = pd.to_datetime(check_date, utc=True)
        # Check entire day - all M5 candles from 00:00 to 23:55
        check_mode = 'intraday' if not target_time else 'specific_time'
        start_time = check_datetime
        end_time = check_datetime + pd.Timedelta(days=1)
        
        # Parse specific time if provided
        specific_timestamp = None
        if target_time:
            try:
                hour, minute = map(int, target_time.split(':'))
                specific_timestamp = start_time.replace(hour=hour, minute=minute)
                print(f"🎯 Target timestamp: {specific_timestamp}")
            except:
                print(f"❌ Invalid time format: {target_time}. Use HH:MM format.")
                return
    else:
        print(f"📅 Checking latest signals")
        check_datetime = None
        check_mode = 'latest'
        start_time = None
        end_time = None
        specific_timestamp = None
    print()
    
    # Load pairs
    with open(PAIRS_FILE, 'r') as f:
        pairs_data = json.load(f)
    
    # Load all pairs from config
    if isinstance(pairs_data, dict) and 'pairs' in pairs_data:
        pairs = [p['symbol'] for p in pairs_data['pairs']]
    elif isinstance(pairs_data, list):
        pairs = pairs_data
    else:
        print("❌ Could not parse pairs file")
        return
    
    # Load model
    print(f"📦 Loading model from {MODEL_DIR}...")
    direction_model = joblib.load(MODEL_DIR / "direction_model.joblib")
    timing_model = joblib.load(MODEL_DIR / "timing_model.joblib")
    strength_model = joblib.load(MODEL_DIR / "strength_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    
    print(f"✅ Model loaded: {len(feature_names)} features")
    print(f"🎯 Thresholds (SAME AS BACKTEST): conf≥{MIN_CONFIDENCE}, timing≥{MIN_TIMING}, strength≥{MIN_STRENGTH}")
    print()
    
    # Initialize feature engine
    mtf_fe = MTFFeatureEngine()
    
    # Check signals for each pair
    signals_found = []
    all_predictions = []  # Store all predictions for show_all mode
    all_confidences = []
    
    print(f"🔍 Checking signals for {len(pairs)} pairs...", flush=True)
    if check_mode == 'intraday':
        print(f"⏰ Scanning all M5 candles from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}", flush=True)
    elif check_mode == 'specific_time':
        print(f"⏰ Checking timestamp: {specific_timestamp.strftime('%Y-%m-%d %H:%M')}", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    for pair in pairs:
        if check_mode == 'intraday':
            print(f"📊 Checking {pair}...", flush=True)
        
        # Load data
        m1 = load_parquet(pair, '1m')
        m5 = load_parquet(pair, '5m')
        m15 = load_parquet(pair, '15m')
        
        if m1 is None or m5 is None or m15 is None:
            print(f"⚠️  {pair}: Data not found", flush=True)
            continue
        
        if check_mode == 'intraday':
            # Get all M5 timestamps for the day
            m5_day = m5[(m5.index >= start_time) & (m5.index < end_time)]
            timestamps_to_check = m5_day.index.tolist()
            print(f"   Found {len(timestamps_to_check)} M5 candles to check", flush=True)
        elif check_mode == 'specific_time':
            # Check specific timestamp only
            if specific_timestamp in m5.index:
                timestamps_to_check = [specific_timestamp]
                if not show_all:
                    print(f"   Checking {specific_timestamp.strftime('%H:%M')}", flush=True)
            else:
                print(f"   ⚠️  Timestamp {specific_timestamp} not found in data", flush=True)
                continue
        else:
            # Latest mode - single timestamp check
            timestamps_to_check = [None]
        
        # Prepare features ONCE for all data (much faster)
        print(f"   Preparing features for full dataset...", flush=True)
        # For historical mode: keep all closed candles (remove_last=False)
        # For latest mode: remove forming candle (remove_last=True)
        remove_last = (check_mode == 'latest')
        features_full = prepare_features(m1, m5, m15, mtf_fe, None, remove_last=remove_last)
        
        if len(features_full) == 0:
            print(f"⚠️  {pair}: No features generated", flush=True)
            continue
        
        print(f"   Features ready. Shape: {features_full.shape}, Columns: {len(features_full.columns)}", flush=True)
        print(f"   Model expects {len(feature_names)} features: {feature_names[:5]}...", flush=True)
        
        # Check if we have the required features
        missing = [f for f in feature_names if f not in features_full.columns]
        if missing:
            print(f"   ⚠️  Missing features: {missing}", flush=True)
            continue
        
        checked = 0
        for timestamp in timestamps_to_check:
            checked += 1
            if checked % 50 == 0:
                print(f"   Progress: {checked}/{len(timestamps_to_check)} candles checked", flush=True)
            
            # Get features for specific timestamp
            if timestamp is None:
                features = features_full.iloc[-1:]
            else:
                if timestamp not in features_full.index:
                    continue
                features = features_full.loc[[timestamp]]
            
            if len(features) == 0:
                continue
            
            # Get last row (current signal)
            last_row = features.iloc[-1:]
            actual_timestamp = features.index[-1]
            
            # Ensure feature order matches model
            X = last_row[feature_names].values
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Predict
            dir_proba = direction_model.predict_proba(X_scaled)[0]
            timing_pred = timing_model.predict(X_scaled)[0]
            strength_pred = strength_model.predict(X_scaled)[0]
            
            # Get direction and confidence
            # 0=SHORT, 1=NEUTRAL, 2=LONG (same as train_v3_dynamic.py)
            dir_class = np.argmax(dir_proba)
            direction = ['SHORT', 'NEUTRAL', 'LONG'][dir_class]
            confidence = dir_proba[dir_class]
            
            all_confidences.append(confidence)
            
            # Store prediction for show_all mode
            if show_all:
                all_predictions.append({
                    'pair': pair,
                    'direction': direction,
                    'confidence': confidence,
                    'timing': timing_pred,
                    'strength': strength_pred,
                    'timestamp': actual_timestamp
                })
            
            # Check if signal passes thresholds (SAME AS train_v3_dynamic.py generate_signals)
            # 1. Skip sideways (dir_class == 1)
            # 2. Check confidence >= MIN_CONFIDENCE
            # 3. Check timing >= MIN_TIMING
            # 4. Check strength >= MIN_STRENGTH
            passes = (dir_class != 1 and  # Not sideways
                     confidence >= MIN_CONFIDENCE and 
                     timing_pred >= MIN_TIMING and 
                     strength_pred >= MIN_STRENGTH)
            
            if check_mode == 'latest':
                status = "✅ SIGNAL" if passes else ""
                print(f"{pair:20s} {direction:5s} | Conf: {confidence:.2f} | "
                      f"Tim: {timing_pred:.1f} | Str: {strength_pred:.1f} {status}")
            
            # Log strong signals in intraday mode  
            if check_mode == 'intraday' and confidence >= 0.55:
                ts_str = actual_timestamp.strftime('%H:%M')
                status = "✅" if passes else "❌"
                print(f"   {status} {ts_str} {direction:5s} conf={confidence:.3f} tim={timing_pred:.2f} str={strength_pred:.2f}", flush=True)
            
            if passes:
                signals_found.append({
                    'pair': pair,
                    'direction': direction,
                    'confidence': confidence,
                    'timing': timing_pred,
                    'strength': strength_pred,
                    'timestamp': actual_timestamp
                })
                
                # Print immediately when signal found
                if check_mode == 'intraday':
                    print(f"   ✅ {actual_timestamp.strftime('%H:%M')} | {direction:5s} | "
                          f"Conf: {confidence:.2f} | Tim: {timing_pred:.1f} | Str: {strength_pred:.1f}")
        
        if check_mode == 'intraday':
            print()  # Empty line between pairs
    
    print("=" * 80)
    print()
    
    # Statistics
    if all_confidences:
        avg_conf = np.mean(all_confidences)
        min_conf = np.min(all_confidences)
        max_conf = np.max(all_confidences)
        above_threshold = sum(1 for c in all_confidences if c >= MIN_CONFIDENCE)
        
        print(f"📊 Confidence Stats:")
        print(f"   Avg: {avg_conf:.2f} | Min: {min_conf:.2f} | Max: {max_conf:.2f}")
        print(f"   Above {MIN_CONFIDENCE}: {above_threshold}/{len(all_confidences)}")
        print()
    
    # Results
    if show_all and all_predictions:
        print()
        print("=" * 80)
        print(f"📊 ALL PREDICTIONS (threshold check disabled):")
        print()
        for pred in sorted(all_predictions, key=lambda x: x['pair']):
            print(f"  {pred['pair']:20s}: {pred['direction']:5s} conf={pred['confidence']:.2f} tim={pred['timing']:.2f} str={pred['strength']:.1f}")
        print()
        print(f"📈 Confidence Stats: Avg={np.mean([p['confidence'] for p in all_predictions]):.2f} | "
              f"Min={min([p['confidence'] for p in all_predictions]):.2f} | "
              f"Max={max([p['confidence'] for p in all_predictions]):.2f} | "
              f"Above {MIN_CONFIDENCE}={sum(1 for p in all_predictions if p['confidence'] >= MIN_CONFIDENCE)}/{len(all_predictions)}")
    else:
        print(f"🎯 Signals found: {len(signals_found)}")
        
        if signals_found:
            print()
            print("Valid signals:")
            for sig in signals_found:
                print(f"  • {sig['pair']} {sig['direction']} | "
                      f"Conf: {sig['confidence']:.2f} | "
                      f"Tim: {sig['timing']:.1f} | "
                      f"Str: {sig['strength']:.1f} | "
                      f"Time: {sig['timestamp']}")
        else:
            print()
            print("❌ No signals pass thresholds - market is too weak today.")
            if all_confidences:
                print(f"   Max confidence was {max(all_confidences):.2f}, need {MIN_CONFIDENCE}")
            else:
                print(f"   No data processed")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check live signals for a specific date')
    parser.add_argument('--date', type=str, default=None, help='Date to check (YYYY-MM-DD)')
    parser.add_argument('--time', type=str, default=None, help='Specific time to check (HH:MM, e.g. 17:45)')
    parser.add_argument('--show-all', action='store_true', help='Show all pairs including below threshold')
    args = parser.parse_args()
    
    main(args.date, args.time, args.show_all)

#!/usr/bin/env python3
"""
Train V2 Improved - Модель с RR-based таргетами

Улучшения по сравнению с V1:
1. RR-based Direction Target - учит предсказывать реальный исход сделки
2. Sample Weights - больший вес успешным сделкам
3. Увеличенный lookahead (10 баров = 50 мин)
4. Более строгий threshold (0.5% минимум)
5. Time Decay Weights - недавние данные важнее
6. Improved Entry Quality - предсказывает достижение RR 1:3

Гиперпараметры V1 сохранены!
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
import joblib
from tqdm import tqdm
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_mtf import (
    TOP_20_PAIRS, 
    load_mtf_data, 
    MTFFeatureEngine
)


# ============================================================
# IMPROVED TARGET CREATION (RR-BASED!)
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    return tr.ewm(span=period, adjust=False).mean()


def create_rr_based_direction_target(
    df: pd.DataFrame,
    sl_atr: float = 1.5,
    tp_rr: float = 2.0,
    max_bars: int = 50
) -> Tuple[pd.Series, pd.Series]:
    """
    RR-Based Direction Target.
    
    Вместо "вырастет ли цена" учим модель:
    "Достигнет ли LONG позиция TP раньше SL?"
    "Достигнет ли SHORT позиция TP раньше SL?"
    
    Returns:
        direction: 0=short_wins, 1=sideways, 2=long_wins
        trade_outcome: 1=TP reached, 0=SL reached, -1=timeout
    """
    atr = calculate_atr(df, 14)
    
    direction = np.full(len(df), 1)  # Default: sideways
    trade_outcome = np.full(len(df), -1.0)  # Default: timeout
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atr_values = atr.values
    
    for i in range(len(df) - max_bars):
        entry_price = closes[i]
        current_atr = atr_values[i]
        
        if np.isnan(current_atr) or current_atr <= 0:
            continue
        
        sl_distance = current_atr * sl_atr
        tp_distance = sl_distance * tp_rr
        
        # LONG scenario
        long_sl = entry_price - sl_distance
        long_tp = entry_price + tp_distance
        
        # SHORT scenario
        short_sl = entry_price + sl_distance
        short_tp = entry_price - tp_distance
        
        long_result = 0  # 0=SL, 1=TP, -1=timeout
        short_result = 0
        
        # Check LONG
        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            if lows[j] <= long_sl:
                long_result = 0  # SL hit
                break
            elif highs[j] >= long_tp:
                long_result = 1  # TP hit
                break
        else:
            long_result = -1  # Timeout
        
        # Check SHORT
        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            if highs[j] >= short_sl:
                short_result = 0  # SL hit
                break
            elif lows[j] <= short_tp:
                short_result = 1  # TP hit
                break
        else:
            short_result = -1  # Timeout
        
        # Determine direction
        if long_result == 1 and short_result != 1:
            direction[i] = 2  # Long wins
            trade_outcome[i] = 1
        elif short_result == 1 and long_result != 1:
            direction[i] = 0  # Short wins
            trade_outcome[i] = 1
        elif long_result == 1 and short_result == 1:
            # Both would win - use simple return
            future_return = closes[min(i + 10, len(df) - 1)] - closes[i]
            direction[i] = 2 if future_return > 0 else 0
            trade_outcome[i] = 1
        else:
            direction[i] = 1  # Sideways
            trade_outcome[i] = 0 if (long_result == 0 or short_result == 0) else -1
    
    return (
        pd.Series(direction, index=df.index, name='direction'),
        pd.Series(trade_outcome, index=df.index, name='trade_outcome')
    )


def create_sample_weights(
    df: pd.DataFrame,
    trade_outcome: pd.Series,
    time_decay: bool = True
) -> pd.Series:
    """
    Create sample weights.
    
    - TP reached (outcome=1): weight = 2.0 (учимся на успехах)
    - SL reached (outcome=0): weight = 1.5 (учимся на ошибках)
    - Timeout (outcome=-1): weight = 0.5 (менее важные)
    
    + Time decay: недавние данные важнее
    """
    weights = np.ones(len(df))
    
    # Outcome-based weights
    weights[trade_outcome == 1] = 2.0   # TP reached - very important
    weights[trade_outcome == 0] = 1.5   # SL reached - important  
    weights[trade_outcome == -1] = 0.5  # Timeout - less important
    
    # Time decay
    if time_decay:
        n = len(df)
        # Exponential decay: last sample = 1.0, first = ~0.3
        decay = np.exp(-0.001 * np.arange(n)[::-1])
        weights = weights * decay
    
    # Normalize
    weights = weights / weights.mean()
    
    return pd.Series(weights, index=df.index, name='sample_weight')


def create_timing_target_improved(
    df: pd.DataFrame,
    lookahead: int = 10
) -> pd.Series:
    """
    Improved timing target.
    
    Good entry = цена прошла в нужную сторону > 1.5 ATR 
    БЕЗ значительного adverse movement (< 0.5 ATR)
    """
    atr = calculate_atr(df, 14)
    
    future_highs = df['high'].rolling(lookahead).max().shift(-lookahead)
    future_lows = df['low'].rolling(lookahead).min().shift(-lookahead)
    
    # For long
    favorable_long = (future_highs - df['close']) / atr
    adverse_long = (df['close'] - future_lows) / atr
    
    # Good long entry: moved up > 1.5 ATR with adverse < 0.5 ATR
    good_long = (favorable_long > 1.5) & (adverse_long < 0.5)
    
    # Good short entry: moved down > 1.5 ATR with adverse < 0.5 ATR
    favorable_short = (df['close'] - future_lows) / atr
    adverse_short = (future_highs - df['close']) / atr
    good_short = (favorable_short > 1.5) & (adverse_short < 0.5)
    
    # Good entry = good for either direction
    good_entry = (good_long | good_short).astype(int)
    
    return pd.Series(good_entry, index=df.index, name='timing')


def create_entry_quality_rr_target(
    df: pd.DataFrame,
    sl_atr: float = 0.5,
    tp_rr: float = 3.0,
    max_bars: int = 50
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Entry Quality for RR 1:3.
    
    Предсказывает: "Достигнет ли сделка с RR 1:3 своего TP?"
    
    Returns:
        entry_quality: общий (любое направление)
        long_quality: для LONG
        short_quality: для SHORT
    """
    atr = calculate_atr(df, 14)
    
    long_quality = np.zeros(len(df))
    short_quality = np.zeros(len(df))
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atr_values = atr.values
    
    for i in range(len(df) - max_bars):
        entry_price = closes[i]
        current_atr = atr_values[i]
        
        if np.isnan(current_atr) or current_atr <= 0:
            continue
        
        sl_distance = current_atr * sl_atr
        tp_distance = sl_distance * tp_rr
        
        # LONG: SL = -0.5 ATR, TP = +1.5 ATR (RR 1:3)
        long_sl = entry_price - sl_distance
        long_tp = entry_price + tp_distance
        
        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            if lows[j] <= long_sl:
                long_quality[i] = 0  # SL hit first
                break
            elif highs[j] >= long_tp:
                long_quality[i] = 1  # TP hit first!
                break
        
        # SHORT: SL = +0.5 ATR, TP = -1.5 ATR (RR 1:3)
        short_sl = entry_price + sl_distance
        short_tp = entry_price - tp_distance
        
        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            if highs[j] >= short_sl:
                short_quality[i] = 0  # SL hit first
                break
            elif lows[j] <= short_tp:
                short_quality[i] = 1  # TP hit first!
                break
    
    # Entry quality = either long or short would work
    entry_quality = np.maximum(long_quality, short_quality)
    
    return (
        pd.Series(entry_quality, index=df.index, name='entry_quality'),
        pd.Series(long_quality, index=df.index, name='long_quality'),
        pd.Series(short_quality, index=df.index, name='short_quality')
    )


def create_strength_target(df: pd.DataFrame, lookahead: int = 10) -> pd.Series:
    """Expected movement in ATRs."""
    atr = calculate_atr(df, 14)
    future_return = abs(df['close'].pct_change(lookahead).shift(-lookahead))
    strength = future_return * df['close'] / atr
    return pd.Series(strength.clip(0, 5), index=df.index, name='strength')


def create_volatility_target(df: pd.DataFrame, lookahead: int = 10) -> pd.Series:
    """Future volatility in ATRs."""
    atr = calculate_atr(df, 14)
    future_highs = df['high'].rolling(lookahead).max().shift(-lookahead)
    future_lows = df['low'].rolling(lookahead).min().shift(-lookahead)
    future_range = (future_highs - future_lows) / atr
    return pd.Series(future_range.clip(0, 5), index=df.index, name='volatility')


# ============================================================
# MODEL TRAINING
# ============================================================

def train_v2_models(
    X_train: pd.DataFrame,
    y_train: Dict[str, pd.Series],
    X_val: pd.DataFrame,
    y_val: Dict[str, pd.Series],
    sample_weights_train: pd.Series = None
) -> Dict:
    """
    Train V2 models with V1 hyperparameters but improved targets.
    """
    import lightgbm as lgb
    
    results = {}
    
    # ============================================================
    # Direction Model (V1 params, RR-based target)
    # ============================================================
    direction_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'max_depth': 4,
        'num_leaves': 15,
        'min_child_samples': 200,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'subsample_freq': 3,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': -1
    }
    
    logger.info("Training V2 Direction Model (RR-based target)...")
    direction_model = lgb.LGBMClassifier(**direction_params)
    
    fit_params = {
        'eval_set': [(X_val, y_val['direction'])]
    }
    if sample_weights_train is not None:
        fit_params['sample_weight'] = sample_weights_train.values
    
    direction_model.fit(X_train, y_train['direction'], **fit_params)
    
    train_pred = direction_model.predict(X_train)
    val_pred = direction_model.predict(X_val)
    
    train_acc = (train_pred == y_train['direction']).mean()
    val_acc = (val_pred == y_val['direction']).mean()
    
    logger.info(f"Direction Model: Train Acc={train_acc:.1%}, Val Acc={val_acc:.1%}")
    results['direction'] = {
        'model': direction_model,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    }
    
    val_pred_dist = pd.Series(val_pred).value_counts(normalize=True)
    logger.info(f"Prediction distribution: Down={val_pred_dist.get(0, 0):.1%}, "
                f"Sideways={val_pred_dist.get(1, 0):.1%}, Up={val_pred_dist.get(2, 0):.1%}")
    
    # ============================================================
    # Timing Model
    # ============================================================
    timing_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'max_depth': 4,
        'num_leaves': 12,
        'min_child_samples': 200,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': -1
    }
    
    logger.info("Training V2 Timing Model...")
    timing_model = lgb.LGBMClassifier(**timing_params)
    timing_model.fit(
        X_train, y_train['timing'],
        eval_set=[(X_val, y_val['timing'])]
    )
    
    train_timing_acc = (timing_model.predict(X_train) == y_train['timing']).mean()
    val_timing_acc = (timing_model.predict(X_val) == y_val['timing']).mean()
    logger.info(f"Timing Model: Train Acc={train_timing_acc:.1%}, Val Acc={val_timing_acc:.1%}")
    results['timing'] = {
        'model': timing_model,
        'train_accuracy': train_timing_acc,
        'val_accuracy': val_timing_acc
    }
    
    # ============================================================
    # Regression Models
    # ============================================================
    regression_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'max_depth': 4,
        'num_leaves': 12,
        'min_child_samples': 200,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': -1
    }
    
    logger.info("Training V2 Strength Model...")
    strength_model = lgb.LGBMRegressor(**regression_params)
    strength_model.fit(X_train, y_train['strength'], eval_set=[(X_val, y_val['strength'])])
    results['strength'] = {'model': strength_model}
    
    logger.info("Training V2 Volatility Model...")
    volatility_model = lgb.LGBMRegressor(**regression_params)
    volatility_model.fit(X_train, y_train['volatility'], eval_set=[(X_val, y_val['volatility'])])
    results['volatility'] = {'model': volatility_model}
    
    # ============================================================
    # Entry Quality Models (RR 1:3 based)
    # ============================================================
    if 'entry_quality' in y_train:
        entry_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'max_depth': 5,
            'num_leaves': 20,
            'min_child_samples': 100,
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': 42,
            'verbosity': -1,
            'class_weight': 'balanced'
        }
        
        logger.info("Training Entry Quality Model (RR 1:3)...")
        entry_model = lgb.LGBMClassifier(**entry_params)
        entry_model.fit(X_train, y_train['entry_quality'], eval_set=[(X_val, y_val['entry_quality'])])
        
        train_entry_acc = (entry_model.predict(X_train) == y_train['entry_quality']).mean()
        val_entry_acc = (entry_model.predict(X_val) == y_val['entry_quality']).mean()
        logger.info(f"Entry Quality: Train Acc={train_entry_acc:.1%}, Val Acc={val_entry_acc:.1%}")
        
        good_entries = y_train['entry_quality'].sum()
        total = len(y_train['entry_quality'])
        logger.info(f"Good entries (RR 1:3): {good_entries}/{total} ({good_entries/total*100:.1f}%)")
        
        results['entry_quality'] = {
            'model': entry_model,
            'train_accuracy': train_entry_acc,
            'val_accuracy': val_entry_acc
        }
        
        logger.info("Training Long Quality Model (RR 1:3)...")
        long_model = lgb.LGBMClassifier(**entry_params)
        long_model.fit(X_train, y_train['long_quality'], eval_set=[(X_val, y_val['long_quality'])])
        results['long_quality'] = {'model': long_model}
        
        logger.info("Training Short Quality Model (RR 1:3)...")
        short_model = lgb.LGBMClassifier(**entry_params)
        short_model.fit(X_train, y_train['short_quality'], eval_set=[(X_val, y_val['short_quality'])])
        results['short_quality'] = {'model': short_model}
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train V2 Improved Model")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--output", type=str, default="./models/v2_improved")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--sl-atr", type=float, default=1.5, help="SL in ATR units")
    parser.add_argument("--tp-rr", type=float, default=2.0, help="RR ratio for TP")
    parser.add_argument("--entry-sl-atr", type=float, default=0.5, help="SL ATR for entry quality")
    parser.add_argument("--entry-tp-rr", type=float, default=3.0, help="RR for entry quality (1:3)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("V2 IMPROVED MODEL TRAINING")
    print("="*70)
    print(f"Pairs: {args.pairs}")
    print(f"Days: {args.days}")
    print(f"Output: {args.output}")
    print()
    print("🚀 IMPROVEMENTS:")
    print(f"   • RR-based Direction Target (SL={args.sl_atr} ATR, RR=1:{args.tp_rr})")
    print(f"   • Sample Weights (TP=2.0, SL=1.5, Timeout=0.5)")
    print(f"   • Time Decay Weights")
    print(f"   • Entry Quality for RR 1:{args.entry_tp_rr}")
    print(f"   • Improved Timing Target")
    print("="*70)
    
    pairs = TOP_20_PAIRS[:args.pairs]
    
    all_features = []
    all_targets = {
        'direction': [],
        'timing': [],
        'strength': [],
        'volatility': [],
        'entry_quality': [],
        'long_quality': [],
        'short_quality': []
    }
    all_sample_weights = []
    
    mtf_engine = MTFFeatureEngine()
    
    for pair in tqdm(pairs, desc="Processing pairs", unit="pair"):
        logger.info(f"Processing {pair}...")
        
        mtf_data = load_mtf_data(pair, args.data_dir)
        if mtf_data is None:
            continue
        
        m1_df = mtf_data['m1']
        m5_df = mtf_data['m5']
        m15_df = mtf_data['m15']
        
        # Limit to last N days
        if args.days and len(m5_df) > 0:
            last_ts = m5_df.index[-1]
            cutoff = last_ts - pd.Timedelta(days=args.days)
            m5_df = m5_df[m5_df.index >= cutoff]
            m1_df = m1_df[m1_df.index >= cutoff]
            m15_df = m15_df[m15_df.index >= cutoff]
        
        if len(m5_df) < 500:
            logger.warning(f"  Skipping {pair}: only {len(m5_df)} M5 bars")
            continue
        
        # Generate MTF features
        features = mtf_engine.align_timeframes(m1_df, m5_df, m15_df)
        
        if features.empty:
            continue
        
        # Create RR-based targets
        direction, trade_outcome = create_rr_based_direction_target(
            m5_df, sl_atr=args.sl_atr, tp_rr=args.tp_rr
        )
        
        # Sample weights
        sample_weights = create_sample_weights(m5_df, trade_outcome, time_decay=True)
        
        # Other targets
        timing = create_timing_target_improved(m5_df, lookahead=10)
        strength = create_strength_target(m5_df, lookahead=10)
        volatility = create_volatility_target(m5_df, lookahead=10)
        
        # Entry quality for RR 1:3
        entry_quality, long_quality, short_quality = create_entry_quality_rr_target(
            m5_df, sl_atr=args.entry_sl_atr, tp_rr=args.entry_tp_rr
        )
        
        # Align indices
        common_idx = features.index.intersection(direction.index)
        common_idx = common_idx.intersection(timing.index)
        
        if len(common_idx) < 100:
            continue
        
        features = features.loc[common_idx]
        direction = direction.loc[common_idx]
        timing = timing.loc[common_idx]
        strength = strength.loc[common_idx]
        volatility = volatility.loc[common_idx]
        entry_quality = entry_quality.loc[common_idx]
        long_quality = long_quality.loc[common_idx]
        short_quality = short_quality.loc[common_idx]
        sample_weights = sample_weights.loc[common_idx]
        
        # Drop NaN
        valid_mask = ~(direction.isna() | timing.isna() | strength.isna() | volatility.isna())
        features = features[valid_mask]
        direction = direction[valid_mask]
        timing = timing[valid_mask]
        strength = strength[valid_mask]
        volatility = volatility[valid_mask]
        entry_quality = entry_quality[valid_mask]
        long_quality = long_quality[valid_mask]
        short_quality = short_quality[valid_mask]
        sample_weights = sample_weights[valid_mask]
        
        all_features.append(features)
        all_targets['direction'].append(direction)
        all_targets['timing'].append(timing)
        all_targets['strength'].append(strength)
        all_targets['volatility'].append(volatility)
        all_targets['entry_quality'].append(entry_quality)
        all_targets['long_quality'].append(long_quality)
        all_targets['short_quality'].append(short_quality)
        all_sample_weights.append(sample_weights)
        
        logger.info(f"  Added {len(features)} samples")
        
        del m1_df, m5_df, m15_df, mtf_data, features
        gc.collect()
    
    if not all_features:
        logger.error("No data collected!")
        return 1
    
    X = pd.concat(all_features, ignore_index=True)
    y = {k: pd.concat(v, ignore_index=True) for k, v in all_targets.items()}
    sample_weights = pd.concat(all_sample_weights, ignore_index=True)
    
    # Convert object columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0)
    
    feature_names = list(X.columns)
    
    logger.info(f"\nTotal samples: {len(X)}")
    logger.info(f"Features: {X.shape[1]}")
    
    # Distribution analysis
    dir_dist = y['direction'].value_counts(normalize=True)
    logger.info(f"Direction (RR-based): Short={dir_dist.get(0, 0):.1%}, "
                f"Sideways={dir_dist.get(1, 0):.1%}, Long={dir_dist.get(2, 0):.1%}")
    
    entry_dist = y['entry_quality'].value_counts(normalize=True)
    logger.info(f"Entry Quality (RR 1:3): Bad={entry_dist.get(0, 0):.1%}, Good={entry_dist.get(1, 0):.1%}")
    
    # Split
    split_idx = int(len(X) * (1 - args.val_split))
    
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = {k: v.iloc[:split_idx] for k, v in y.items()}
    y_val = {k: v.iloc[split_idx:] for k, v in y.items()}
    sample_weights_train = sample_weights.iloc[:split_idx]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train
    print("\n" + "="*70)
    print("TRAINING V2 IMPROVED MODELS")
    print("="*70)
    
    results = train_v2_models(X_train, y_train, X_val, y_val, sample_weights_train)
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving models to {output_dir}")
    
    joblib.dump(results['direction']['model'], output_dir / 'direction_model.joblib')
    joblib.dump(results['timing']['model'], output_dir / 'timing_model.joblib')
    joblib.dump(results['strength']['model'], output_dir / 'strength_model.joblib')
    joblib.dump(results['volatility']['model'], output_dir / 'volatility_model.joblib')
    
    if 'entry_quality' in results:
        joblib.dump(results['entry_quality']['model'], output_dir / 'entry_quality_model.joblib')
        joblib.dump(results['long_quality']['model'], output_dir / 'long_quality_model.joblib')
        joblib.dump(results['short_quality']['model'], output_dir / 'short_quality_model.joblib')
    
    joblib.dump(feature_names, output_dir / 'feature_names.joblib')
    
    ensemble_meta = {
        'version': 'v2_improved',
        'improvements': [
            'RR-based direction target',
            'Sample weights',
            'Time decay',
            'Entry quality RR 1:3'
        ],
        'params': {
            'sl_atr': args.sl_atr,
            'tp_rr': args.tp_rr,
            'entry_sl_atr': args.entry_sl_atr,
            'entry_tp_rr': args.entry_tp_rr
        },
        'n_features': X.shape[1],
        'trained_at': datetime.now().isoformat()
    }
    joblib.dump(ensemble_meta, output_dir / 'ensemble_meta.joblib')
    
    # Summary
    print("\n" + "="*70)
    print("V2 IMPROVED TRAINING COMPLETE")
    print("="*70)
    
    entry_msg = ""
    if 'entry_quality' in results:
        entry_msg = f"""
Entry Quality (RR 1:3):
  Train Accuracy: {results['entry_quality']['train_accuracy']:.1%}
  Val Accuracy:   {results['entry_quality']['val_accuracy']:.1%}
"""
    
    print(f"""
Model saved to: {output_dir}

Direction Model (RR-based):
  Train Accuracy: {results['direction']['train_accuracy']:.1%}
  Val Accuracy:   {results['direction']['val_accuracy']:.1%}

Timing Model:
  Train Accuracy: {results['timing']['train_accuracy']:.1%}
  Val Accuracy:   {results['timing']['val_accuracy']:.1%}
{entry_msg}
Features: {X.shape[1]}
Samples: {len(X_train)} train, {len(X_val)} val

IMPROVEMENTS APPLIED:
✅ RR-based targets (SL={args.sl_atr} ATR, RR=1:{args.tp_rr})
✅ Sample weights (success=2.0, fail=1.5, timeout=0.5)
✅ Time decay (recent data weighted more)
✅ Entry quality for RR 1:{args.entry_tp_rr}

To compare with V1:
  python scripts/fetch_and_backtest.py --date 2025-12-25 --model-path {output_dir}
""")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

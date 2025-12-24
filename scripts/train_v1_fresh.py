#!/usr/bin/env python3
"""
V1 Model Retraining Script

Retrains the V1 model with CURRENT feature engineering,
but uses V1 hyperparameters (no aggressive class weights).

This is needed because:
1. Original V1 model was trained with 167 features
2. Current feature engine generates 166 features
3. We need to retrain with same V1 params to restore performance

V1 Params that WORKED:
- No class weights or balanced=1.0
- max_depth=4, num_leaves=15
- n_estimators=500
- Simple threshold-based targets

Usage:
    docker-compose -f docker/docker-compose.yml run --rm trading-bot \
        python scripts/train_v1_fresh.py --pairs 20 --days 30
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing training infrastructure
from train_mtf import (
    TOP_20_PAIRS, 
    load_mtf_data, 
    MTFFeatureEngine
)


# ============================================================
# V1 TARGET CREATION (SIMPLE - NOT RR BASED!)
# ============================================================

def create_direction_target_v1(
    df: pd.DataFrame,
    lookahead: int = 5
) -> pd.Series:
    """
    Simple V1 direction target based on threshold.
    
    V1 Logic (which worked!):
    - Calculate rolling volatility as adaptive threshold
    - If future return > threshold: UP (2)
    - If future return < -threshold: DOWN (0)
    - Else: SIDEWAYS (1)
    
    This is SIMPLER than V2's RR-based target.
    """
    # Adaptive threshold based on rolling volatility
    rolling_vol = df['close'].pct_change().rolling(window=100).std()
    threshold = np.maximum(rolling_vol, 0.003)  # Minimum 0.3%
    
    # Future returns
    future_return = df['close'].pct_change(lookahead).shift(-lookahead)
    
    # Create direction target
    direction = np.where(
        future_return > threshold, 2,       # Up
        np.where(future_return < -threshold, 0, 1)  # Down / Sideways
    )
    
    return pd.Series(direction, index=df.index, name='direction')


def create_timing_target_v1(
    df: pd.DataFrame,
    lookahead: int = 5
) -> pd.Series:
    """
    Simple V1 timing target.
    
    V1 Logic:
    - Good entry = favorable movement > adverse movement
    - Returns binary 0/1 (not continuous)
    """
    atr = calculate_atr(df, 14)
    
    future_lows = df['low'].rolling(lookahead).min().shift(-lookahead)
    future_highs = df['high'].rolling(lookahead).max().shift(-lookahead)
    
    # For long entries
    adverse_long = (df['close'] - future_lows) / atr
    favorable_long = (future_highs - df['close']) / atr
    
    # Good entry: favorable > adverse AND favorable > 1 ATR
    good_entry = ((favorable_long > adverse_long) & (favorable_long > 1.0)).astype(int)
    
    return pd.Series(good_entry, index=df.index, name='timing')


def create_strength_target_v1(
    df: pd.DataFrame,
    lookahead: int = 5
) -> pd.Series:
    """
    V1 strength target - expected movement in ATRs.
    """
    atr = calculate_atr(df, 14)
    
    future_return = abs(df['close'].pct_change(lookahead).shift(-lookahead))
    strength = future_return * df['close'] / atr
    
    # Clip to reasonable range
    strength = strength.clip(0, 5)
    
    return pd.Series(strength, index=df.index, name='strength')


def create_volatility_target_v1(
    df: pd.DataFrame,
    lookahead: int = 5
) -> pd.Series:
    """
    V1 volatility target - future volatility.
    """
    atr = calculate_atr(df, 14)
    
    future_highs = df['high'].rolling(lookahead).max().shift(-lookahead)
    future_lows = df['low'].rolling(lookahead).min().shift(-lookahead)
    
    future_range = (future_highs - future_lows) / atr
    
    return pd.Series(future_range.clip(0, 5), index=df.index, name='volatility')


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


# ============================================================
# V1 MODEL TRAINING (NO AGGRESSIVE WEIGHTS!)
# ============================================================

def train_v1_models(
    X_train: pd.DataFrame,
    y_train: Dict[str, pd.Series],
    X_val: pd.DataFrame,
    y_val: Dict[str, pd.Series]
) -> Dict:
    """
    Train V1 models with V1 hyperparameters.
    
    Key differences from V2:
    - NO class_weight or balanced weights
    - Conservative hyperparameters
    - Simple targets
    """
    import lightgbm as lgb
    
    results = {}
    
    # V1 Direction Model Parameters (WORKING!)
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
    
    # Train Direction Model (NO CLASS WEIGHTS!)
    logger.info("Training V1 Direction Model (no class weights)...")
    direction_model = lgb.LGBMClassifier(**direction_params)
    direction_model.fit(
        X_train, 
        y_train['direction'],
        eval_set=[(X_val, y_val['direction'])],
    )
    
    # Evaluate
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
    
    # Class distribution
    val_pred_dist = pd.Series(val_pred).value_counts(normalize=True)
    logger.info(f"Prediction distribution: Down={val_pred_dist.get(0, 0):.1%}, "
                f"Sideways={val_pred_dist.get(1, 0):.1%}, Up={val_pred_dist.get(2, 0):.1%}")
    
    # V1 Timing Model Parameters
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
    
    logger.info("Training V1 Timing Model...")
    timing_model = lgb.LGBMClassifier(**timing_params)
    timing_model.fit(
        X_train,
        y_train['timing'],
        eval_set=[(X_val, y_val['timing'])],
    )
    
    train_timing_acc = (timing_model.predict(X_train) == y_train['timing']).mean()
    val_timing_acc = (timing_model.predict(X_val) == y_val['timing']).mean()
    logger.info(f"Timing Model: Train Acc={train_timing_acc:.1%}, Val Acc={val_timing_acc:.1%}")
    results['timing'] = {
        'model': timing_model,
        'train_accuracy': train_timing_acc,
        'val_accuracy': val_timing_acc
    }
    
    # V1 Strength Model Parameters
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
    
    logger.info("Training V1 Strength Model...")
    strength_model = lgb.LGBMRegressor(**regression_params)
    strength_model.fit(
        X_train,
        y_train['strength'],
        eval_set=[(X_val, y_val['strength'])],
    )
    
    results['strength'] = {'model': strength_model}
    
    logger.info("Training V1 Volatility Model...")
    volatility_model = lgb.LGBMRegressor(**regression_params)
    volatility_model.fit(
        X_train,
        y_train['volatility'],
        eval_set=[(X_val, y_val['volatility'])],
    )
    
    results['volatility'] = {'model': volatility_model}
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Retrain V1 Model with Fresh Features")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs")
    parser.add_argument("--days", type=int, default=30, help="Days of data")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--output", type=str, default="./models/v1_fresh",
                       help="Output directory for retrained V1 model")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Validation split ratio")
    
    args = parser.parse_args()
    
    print("="*70)
    print("V1 MODEL RETRAINING WITH CURRENT FEATURES")
    print("="*70)
    print(f"Pairs: {args.pairs}")
    print(f"Days: {args.days}")
    print(f"Output: {args.output}")
    print()
    print("⚠️  Using V1 parameters (NO aggressive class weights)")
    print("="*70)
    
    # Load data for all pairs
    pairs = TOP_20_PAIRS[:args.pairs]
    
    all_features = []
    all_targets = {
        'direction': [],
        'timing': [],
        'strength': [],
        'volatility': []
    }
    
    mtf_engine = MTFFeatureEngine()
    
    for pair in pairs:
        logger.info(f"Processing {pair}...")
        
        mtf_data = load_mtf_data(pair, args.data_dir)
        if mtf_data is None:
            logger.warning(f"No data for {pair}")
            continue
        
        m1_df = mtf_data['m1']
        m5_df = mtf_data['m5']
        m15_df = mtf_data['m15']
        
        # Filter to last N days
        if args.days > 0:
            end_time = m5_df.index[-1]
            start_time = end_time - pd.Timedelta(days=args.days)
            
            m1_df = m1_df[m1_df.index >= start_time]
            m5_df = m5_df[m5_df.index >= start_time]
            m15_df = m15_df[m15_df.index >= start_time]
        
        logger.info(f"  M5 data: {len(m5_df)} bars")
        
        # Generate features
        try:
            features = mtf_engine.align_timeframes(m1_df, m5_df, m15_df)
        except Exception as e:
            logger.error(f"Feature generation failed for {pair}: {e}")
            continue
        
        # Convert object columns
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.Categorical(features[col]).codes
        
        features = features.fillna(0)
        
        # Create V1 targets (simple threshold-based!)
        direction = create_direction_target_v1(m5_df)
        timing = create_timing_target_v1(m5_df)
        strength = create_strength_target_v1(m5_df)
        volatility = create_volatility_target_v1(m5_df)
        
        # Align targets to features
        common_idx = features.index.intersection(direction.index)
        common_idx = common_idx.intersection(timing.index)
        
        if len(common_idx) < 100:
            logger.warning(f"Not enough aligned data for {pair}")
            continue
        
        features = features.loc[common_idx]
        direction = direction.loc[common_idx]
        timing = timing.loc[common_idx]
        strength = strength.loc[common_idx]
        volatility = volatility.loc[common_idx]
        
        # Drop rows with NaN targets
        valid_mask = ~(direction.isna() | timing.isna() | strength.isna() | volatility.isna())
        features = features[valid_mask]
        direction = direction[valid_mask]
        timing = timing[valid_mask]
        strength = strength[valid_mask]
        volatility = volatility[valid_mask]
        
        all_features.append(features)
        all_targets['direction'].append(direction)
        all_targets['timing'].append(timing)
        all_targets['strength'].append(strength)
        all_targets['volatility'].append(volatility)
        
        logger.info(f"  Added {len(features)} samples")
    
    # Combine all data
    if not all_features:
        logger.error("No data collected!")
        return 1
    
    X = pd.concat(all_features, ignore_index=True)
    y = {
        'direction': pd.concat(all_targets['direction'], ignore_index=True),
        'timing': pd.concat(all_targets['timing'], ignore_index=True),
        'strength': pd.concat(all_targets['strength'], ignore_index=True),
        'volatility': pd.concat(all_targets['volatility'], ignore_index=True)
    }
    
    logger.info(f"\nTotal samples: {len(X)}")
    logger.info(f"Features: {X.shape[1]}")
    
    # Target distribution
    dir_dist = y['direction'].value_counts(normalize=True)
    logger.info(f"Direction distribution: Down={dir_dist.get(0, 0):.1%}, "
                f"Sideways={dir_dist.get(1, 0):.1%}, Up={dir_dist.get(2, 0):.1%}")
    
    # Split train/val
    split_idx = int(len(X) * (1 - args.val_split))
    
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    
    y_train = {k: v.iloc[:split_idx] for k, v in y.items()}
    y_val = {k: v.iloc[split_idx:] for k, v in y.items()}
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train V1 models
    print("\n" + "="*70)
    print("TRAINING V1 MODELS")
    print("="*70)
    
    results = train_v1_models(X_train, y_train, X_val, y_val)
    
    # Save models
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving models to {output_dir}")
    
    # Save individual models
    joblib.dump(results['direction']['model'], output_dir / 'direction_model.joblib')
    joblib.dump(results['timing']['model'], output_dir / 'timing_model.joblib')
    joblib.dump(results['strength']['model'], output_dir / 'strength_model.joblib')
    joblib.dump(results['volatility']['model'], output_dir / 'volatility_model.joblib')
    
    # Save ensemble metadata
    ensemble_meta = {
        'weights': {
            'direction': 0.4,
            'strength': 0.2,
            'timing': 0.2,
            'volatility': 0.2
        },
        'config': {},
        'use_meta_model': False,
        'is_trained': True,
        'version': 'v1_fresh',
        'n_features': X.shape[1],
        'trained_at': datetime.now().isoformat()
    }
    joblib.dump(ensemble_meta, output_dir / 'ensemble_meta.joblib')
    
    # Print results summary
    print("\n" + "="*70)
    print("V1 RETRAINING COMPLETE")
    print("="*70)
    print(f"""
Model saved to: {output_dir}

Direction Model:
  Train Accuracy: {results['direction']['train_accuracy']:.1%}
  Val Accuracy:   {results['direction']['val_accuracy']:.1%}

Timing Model:
  Train Accuracy: {results['timing']['train_accuracy']:.1%}
  Val Accuracy:   {results['timing']['val_accuracy']:.1%}

Features: {X.shape[1]}
Training samples: {len(X_train)}
Validation samples: {len(X_val)}

To run backtest:
  docker-compose -f docker/docker-compose.yml run --rm trading-bot \\
      python scripts/backtest_v1_risk.py --model-path {output_dir} --pairs 19 --days 14
""")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

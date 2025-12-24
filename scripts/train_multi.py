#!/usr/bin/env python3
"""
Train ML model on multiple trading pairs.
Combines data from all pairs, maintains chronological order within each pair.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from src.models.ensemble import EnsembleModel
from src.models.training import ModelTrainer
from src.data.splitter import DataSplitter
from src.models.validator import ModelValidator

warnings.filterwarnings('ignore')


def load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs_list(path: str = 'config/pairs_list.json') -> List[str]:
    """Load pairs list from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get('symbols', [])


def load_pair_data(symbol: str, data_dir: str = 'data/candles', timeframe: str = '5m') -> Optional[pd.DataFrame]:
    """Load OHLCV data for a single pair."""
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    filepath = Path(data_dir) / f"{safe_symbol}_{timeframe}.csv"
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def prepare_multi_pair_data(
    symbols: List[str],
    data_dir: str = 'data/candles',
    timeframe: str = '5m',
    min_samples: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load and prepare data for all pairs.
    
    Args:
        symbols: List of trading pair symbols
        data_dir: Directory with CSV files
        timeframe: Timeframe to load
        min_samples: Minimum samples required per pair
    
    Returns:
        Tuple of (features_df, labels_df, pair_stats)
    """
    config = load_yaml_config('./config/trading_params.yaml')
    feature_engine = FeatureEngine(config.get('features', {}))
    trainer = ModelTrainer()
    
    all_features = []
    all_labels = []
    pair_stats = {}
    
    loaded_count = 0
    skipped_count = 0
    
    for symbol in symbols:
        df = load_pair_data(symbol, data_dir, timeframe)
        
        if df is None:
            logger.warning(f"No data for {symbol}")
            skipped_count += 1
            continue
        
        if len(df) < min_samples:
            logger.warning(f"Skipping {symbol}: only {len(df)} samples (min: {min_samples})")
            skipped_count += 1
            continue
        
        try:
            # Generate features
            df_clean = df.dropna()
            features = feature_engine.generate_all_features(df_clean, normalize=True)
            
            # Encode categorical columns
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = pd.Categorical(features[col]).codes
            
            # Generate labels (returns dict)
            labels_dict = trainer.prepare_labels(df_clean)
            
            # Convert labels dict to DataFrame
            labels = pd.DataFrame(labels_dict)
            
            # Align features and labels
            common_idx = features.index.intersection(labels.index)
            features = features.loc[common_idx].copy()
            labels = labels.loc[common_idx].copy()
            
            # Add symbol column
            features['symbol'] = symbol
            labels['symbol'] = symbol
            
            all_features.append(features)
            all_labels.append(labels)
            
            pair_stats[symbol] = {
                'samples': len(features),
                'start': features.index[0].isoformat(),
                'end': features.index[-1].isoformat()
            }
            
            loaded_count += 1
            logger.info(f"✓ {symbol}: {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            skipped_count += 1
    
    if not all_features:
        raise ValueError("No valid data loaded")
    
    # Combine all data
    combined_features = pd.concat(all_features, axis=0)
    combined_labels = pd.concat(all_labels, axis=0)
    
    logger.info(f"\nLoaded {loaded_count} pairs, skipped {skipped_count}")
    logger.info(f"Total samples: {len(combined_features):,}")
    
    return combined_features, combined_labels, pair_stats


def chronological_multi_pair_split(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically within each pair.
    
    First 70% of each pair -> train
    Next 15% of each pair -> validation  
    Last 15% of each pair -> test
    """
    train_features_list = []
    train_labels_list = []
    val_features_list = []
    val_labels_list = []
    test_features_list = []
    test_labels_list = []
    
    for symbol in features['symbol'].unique():
        # Get data for this pair
        mask = features['symbol'] == symbol
        pair_features = features[mask].copy()
        pair_labels = labels[mask].copy()
        
        # Sort by time
        pair_features.sort_index(inplace=True)
        pair_labels = pair_labels.loc[pair_features.index]
        
        n = len(pair_features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split
        train_features_list.append(pair_features.iloc[:train_end])
        train_labels_list.append(pair_labels.iloc[:train_end])
        
        val_features_list.append(pair_features.iloc[train_end:val_end])
        val_labels_list.append(pair_labels.iloc[train_end:val_end])
        
        test_features_list.append(pair_features.iloc[val_end:])
        test_labels_list.append(pair_labels.iloc[val_end:])
    
    # Combine
    X_train = pd.concat(train_features_list, axis=0)
    y_train = pd.concat(train_labels_list, axis=0)
    X_val = pd.concat(val_features_list, axis=0)
    y_val = pd.concat(val_labels_list, axis=0)
    X_test = pd.concat(test_features_list, axis=0)
    y_test = pd.concat(test_labels_list, axis=0)
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_multi_pair_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    pair_stats: Dict,
    save_dir: str = 'models/saved_multi',
    config_path: str = './config/trading_params.yaml'
) -> Dict:
    """
    Train model on multi-pair data.
    
    Args:
        features: Combined features DataFrame
        labels: Combined labels DataFrame
        pair_stats: Statistics per pair
        save_dir: Directory to save model
        config_path: Path to config file
    
    Returns:
        Training metrics
    """
    # Load config
    config = load_yaml_config(config_path)
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_multi_pair_split(
        features, labels
    )
    
    # Remove symbol column for training
    symbol_train = X_train['symbol'].copy()
    symbol_val = X_val['symbol'].copy()
    symbol_test = X_test['symbol'].copy()
    
    X_train = X_train.drop('symbol', axis=1)
    X_val = X_val.drop('symbol', axis=1)
    X_test = X_test.drop('symbol', axis=1)
    
    y_train = y_train.drop('symbol', axis=1)
    y_val = y_val.drop('symbol', axis=1)
    y_test = y_test.drop('symbol', axis=1)
    
    # Handle NaN/Inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = y_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = y_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Create and train ensemble
    logger.info("Training ensemble model...")
    ensemble = EnsembleModel(config)
    
    train_metrics = ensemble.train_all(
        X_train, y_train,
        X_val, y_val
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = ensemble.predict(X_test)
    
    # Convert dict predictions to arrays for easier indexing
    pred_direction = test_predictions.get('direction', np.array([]))
    true_direction = y_test['direction'].values if 'direction' in y_test.columns else np.array([])
    
    # Calculate test metrics per pair
    pair_metrics = {}
    for symbol in symbol_test.unique():
        mask = (symbol_test == symbol).values
        if mask.sum() < 10:
            continue
        
        # Direction accuracy per pair
        if len(pred_direction) > 0 and len(true_direction) > 0:
            pair_pred = pred_direction[mask]
            pair_true = true_direction[mask]
            direction_acc = (pair_pred == pair_true).mean()
            pair_metrics[symbol] = {
                'test_samples': int(mask.sum()),
                'direction_accuracy': float(direction_acc)
            }
    
    # Overall test metrics
    if len(pred_direction) > 0 and len(true_direction) > 0:
        overall_direction_acc = (pred_direction == true_direction).mean()
    else:
        overall_direction_acc = 0.0
    
    # Check for overfitting
    validator = ModelValidator()
    
    # Get train/val accuracy from metrics for overfitting check
    train_acc = train_metrics.get('direction', {}).get('train_accuracy', 0.0)
    val_acc = train_metrics.get('direction', {}).get('val_accuracy', 0.0)
    
    # Build metrics dicts expected by check_overfitting
    train_metrics_dict = {'accuracy': train_acc}
    val_metrics_dict = {'accuracy': val_acc}
    
    is_overfit, overfit_metrics = validator.check_overfitting(
        train_metrics=train_metrics_dict,
        val_metrics=val_metrics_dict
    )
    
    # Save model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save(save_dir)
    logger.info(f"Model saved to {save_dir}")
    
    # Save pair stats
    with open(Path(save_dir) / 'pair_stats.json', 'w') as f:
        json.dump({
            'training_pairs': len(pair_stats),
            'pair_stats': pair_stats,
            'pair_test_metrics': pair_metrics
        }, f, indent=2)
    
    return {
        'train_metrics': train_metrics,
        'test_direction_accuracy': overall_direction_acc,
        'pair_metrics': pair_metrics,
        'is_overfitting': is_overfit,
        'overfit_metrics': overfit_metrics
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train model on multiple pairs')
    parser.add_argument('--pairs-file', type=str, default='config/pairs_list.json',
                        help='Path to pairs list JSON')
    parser.add_argument('--data-dir', type=str, default='data/candles',
                        help='Directory with candle data')
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Timeframe to use (default: 5m)')
    parser.add_argument('--min-samples', type=int, default=1000,
                        help='Minimum samples per pair (default: 1000)')
    parser.add_argument('--save-dir', type=str, default='models/saved_multi',
                        help='Directory to save model')
    parser.add_argument('--config', type=str, default='./config/trading_params.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load pairs
    try:
        symbols = load_pairs_list(args.pairs_file)
        logger.info(f"Loaded {len(symbols)} pairs from {args.pairs_file}")
    except FileNotFoundError:
        logger.error(f"Pairs file not found: {args.pairs_file}")
        return 1
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    try:
        features, labels, pair_stats = prepare_multi_pair_data(
            symbols=symbols,
            data_dir=args.data_dir,
            timeframe=args.timeframe,
            min_samples=args.min_samples
        )
    except ValueError as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Train model
    logger.info("Starting training...")
    metrics = train_multi_pair_model(
        features=features,
        labels=labels,
        pair_stats=pair_stats,
        save_dir=args.save_dir,
        config_path=args.config
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("MULTI-PAIR TRAINING RESULTS")
    print("=" * 60)
    print(f"Pairs trained: {len(pair_stats)}")
    print(f"Total samples: {len(features):,}")
    print(f"\nTest Direction Accuracy: {metrics['test_direction_accuracy']:.2%}")
    
    if metrics['is_overfitting']:
        print("\n⚠️  WARNING: Overfitting detected!")
        print(f"Overfit metrics: {metrics['overfit_metrics']}")
    else:
        print("\n✓ No severe overfitting detected")
    
    # Top/bottom pairs by accuracy
    pair_metrics = metrics.get('pair_metrics', {})
    if pair_metrics:
        sorted_pairs = sorted(
            pair_metrics.items(),
            key=lambda x: x[1].get('direction_accuracy', 0),
            reverse=True
        )
        
        print("\nTop 5 pairs by accuracy:")
        for symbol, m in sorted_pairs[:5]:
            print(f"  {symbol}: {m['direction_accuracy']:.2%}")
        
        print("\nBottom 5 pairs by accuracy:")
        for symbol, m in sorted_pairs[-5:]:
            print(f"  {symbol}: {m['direction_accuracy']:.2%}")
    
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

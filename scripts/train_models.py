#!/usr/bin/env python
"""
AI Trading Bot - Model Training Script
Trains all ML models on historical data.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.utils import load_yaml_config
from src.data import DataCollector, DataPreprocessor
from src.features import FeatureEngine
from src.models import EnsembleModel, ModelTrainer


async def fetch_training_data(
    symbol: str,
    timeframe: str,
    days: int,
    exchange_id: str = "binance"
):
    """Fetch historical data for training."""
    import ccxt.async_support as ccxt
    
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")
    
    # Create exchange client
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
    })
    
    try:
        # Format symbol for CCXT
        formatted_symbol = symbol.replace('USDT', '/USDT')
        
        # Calculate time range
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        
        # Fetch OHLCV data
        all_ohlcv = []
        while True:
            ohlcv = await exchange.fetch_ohlcv(formatted_symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > int(end_time.timestamp() * 1000):
                break
            await asyncio.sleep(0.1)  # Rate limiting
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    finally:
        await exchange.close()


def train_models(
    df,
    config: dict,
    output_dir: str,
    use_walk_forward: bool = True
):
    """Train all models on the data with overfitting prevention."""
    logger.info("Starting model training with overfitting prevention...")
    
    # Load validation settings
    validation_config = config.get('validation', {})
    regularization_config = config.get('regularization', {})
    
    # Initialize components
    feature_engine = FeatureEngine(config.get('features', {}))
    
    # Merge configs for trainer
    trainer_config = {
        **config.get('training', {}),
        'train_ratio': validation_config.get('train_ratio', 0.70),
        'val_ratio': validation_config.get('val_ratio', 0.15),
        'regularization': regularization_config,
        'overfitting_thresholds': config.get('overfitting_thresholds', {}),
    }
    trainer = ModelTrainer(trainer_config)
    
    # Clean data - drop NaN values
    df = df.dropna()
    logger.info(f"Data cleaned, {len(df)} samples remaining")
    
    # Generate features
    logger.info("Generating features...")
    features = feature_engine.generate_all_features(df, normalize=True)
    
    # Convert categorical columns to numeric
    for col in features.columns:
        if features[col].dtype == 'object':
            logger.info(f"Encoding categorical column: {col}")
            features[col] = pd.Categorical(features[col]).codes
    
    # Prepare labels
    logger.info("Preparing labels...")
    labels = trainer.prepare_labels(
        df,
        forward_periods=5,
        direction_threshold=0.003
    )
    
    # Walk-forward validation if enabled
    if use_walk_forward and len(df) > 5000:
        logger.info("Performing walk-forward validation...")
        wf_results = trainer.walk_forward_validation(
            features, labels,
            train_months=validation_config.get('walk_forward_train_months', 6),
            test_months=validation_config.get('walk_forward_test_months', 1),
            model_config=config.get('model_params', {})
        )
        
        if wf_results:
            # Analyze walk-forward results
            from src.data import DataSplitter
            splitter = DataSplitter()
            fold_metrics = [r['metrics'] for r in wf_results]
            fold_stats = splitter.get_fold_stats(fold_metrics)
            
            logger.info("Walk-Forward Results:")
            for metric, stats in fold_stats.items():
                logger.info(
                    f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(CI: {stats['ci_lower']:.4f} - {stats['ci_upper']:.4f})"
                )
    
    # Train final ensemble
    logger.info("Training final ensemble model...")
    ensemble, metrics = trainer.train_ensemble(
        features,
        labels,
        model_config=config.get('model_params', {})
    )
    
    # Check for overfitting
    is_overfitting = metrics.get('is_overfitting', False)
    if is_overfitting:
        logger.warning("⚠️ WARNING: Model shows signs of overfitting!")
        logger.warning("Consider: reducing model complexity, adding more data, or increasing regularization")
    
    # Log results
    logger.info(f"Training complete. Metrics: {metrics}")
    
    # Save model only if not overfitting
    os.makedirs(output_dir, exist_ok=True)
    if not is_overfitting:
        ensemble.save(output_dir)
        logger.info(f"Model saved to {output_dir}")
    else:
        logger.warning("Model NOT saved due to overfitting concerns")
    
    # Perform cross-validation
    logger.info("Performing time-series cross-validation...")
    cv_results = trainer.cross_validate(features, labels)
    logger.info(f"CV Results: {cv_results}")
    
    return ensemble, metrics, cv_results


def main():
    parser = argparse.ArgumentParser(description="Train trading models")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        help="Data timeframe"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days of historical data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/saved",
        help="Output directory for saved models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/trading_params.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward validation"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_config(args.config)
    
    # Also load settings.yaml for validation config
    try:
        settings = load_yaml_config('./config/settings.yaml')
        config['validation'] = settings.get('validation', {})
        config['regularization'] = settings.get('regularization', {})
        config['overfitting_thresholds'] = settings.get('overfitting_thresholds', {})
    except:
        logger.warning("Could not load settings.yaml, using defaults")
    
    # Fetch data
    logger.info(f"Training models for {args.symbol}")
    df = asyncio.run(fetch_training_data(
        args.symbol,
        args.timeframe,
        args.days
    ))
    
    if df is None or df.empty:
        logger.error("Failed to fetch training data")
        return 1
        
    logger.info(f"Fetched {len(df)} candles")
    
    # Train models
    ensemble, metrics, cv_results = train_models(
        df, config, args.output,
        use_walk_forward=not args.no_walk_forward
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY (with overfitting checks)")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Training samples: {len(df)}")
    
    # Print metrics by split
    for split in ['train', 'val', 'test']:
        if split in metrics and isinstance(metrics[split], dict):
            print(f"\n{split.upper()} Metrics:")
            for k, v in metrics[split].items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
    
    # Overfitting warning
    if metrics.get('is_overfitting'):
        print("\n⚠️  OVERFITTING DETECTED!")
        print("The model may not perform well on new data.")
    else:
        print("\n✓ No severe overfitting detected")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

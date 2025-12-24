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
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")
    
    collector = DataCollector(
        exchange_id=exchange_id,
        symbol=symbol.replace('USDT', '/USDT'),
        testnet=False  # Use mainnet for historical data
    )
    
    await collector.start()
    
    try:
        df = await collector.fetch_historical_ohlcv(
            timeframe=timeframe,
            days=days
        )
        return df
    finally:
        await collector.stop()


def train_models(
    df,
    config: dict,
    output_dir: str
):
    """Train all models on the data."""
    logger.info("Starting model training...")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engine = FeatureEngine(config.get('features', {}))
    trainer = ModelTrainer(config.get('training', {}))
    
    # Validate and clean data
    is_valid, issues = preprocessor.validate_ohlcv(df)
    if not is_valid:
        logger.warning(f"Data validation issues: {issues}")
        
    df = preprocessor.clean_ohlcv(df)
    
    # Generate features
    logger.info("Generating features...")
    features = feature_engine.generate_all_features(df, normalize=True)
    
    # Prepare labels
    logger.info("Preparing labels...")
    labels = trainer.prepare_labels(
        df,
        forward_periods=5,
        direction_threshold=0.003
    )
    
    # Train ensemble
    logger.info("Training ensemble model...")
    ensemble, metrics = trainer.train_ensemble(
        features,
        labels,
        model_config=config.get('model_params', {})
    )
    
    # Log results
    logger.info(f"Training complete. Metrics: {metrics}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    ensemble.save(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    # Perform cross-validation
    logger.info("Performing cross-validation...")
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_config(args.config)
    
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
    ensemble, metrics, cv_results = train_models(df, config, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Training samples: {len(df)}")
    print(f"\nMetrics:")
    for model, model_metrics in metrics.items():
        if isinstance(model_metrics, dict):
            print(f"  {model}:")
            for k, v in model_metrics.items():
                print(f"    {k}: {v:.4f}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

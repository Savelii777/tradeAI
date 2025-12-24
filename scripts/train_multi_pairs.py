#!/usr/bin/env python
"""
AI Trading Bot - Multi-Pair Model Training Script
Trains ML models on multiple trading pairs with historical data.
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
from src.features import FeatureEngine
from src.models import EnsembleModel, ModelTrainer


# List of trading pairs
TRADING_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LINKUSDT",
    "ATOMUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "APTUSDT",
    "NEARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "FILUSDT",
    "INJUSDT",
]


async def fetch_pair_data(
    symbol: str,
    timeframe: str,
    days: int,
):
    """Fetch historical data for a single pair."""
    import ccxt.async_support as ccxt
    
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        formatted_symbol = symbol.replace('USDT', '/USDT')
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        
        all_ohlcv = []
        while True:
            try:
                ohlcv = await exchange.fetch_ohlcv(formatted_symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if since > int(end_time.timestamp() * 1000):
                    break
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                break
        
        if not all_ohlcv:
            return None
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None
    finally:
        await exchange.close()


async def fetch_all_pairs(pairs: list, timeframe: str, days: int):
    """Fetch data for all pairs sequentially to avoid rate limits."""
    all_data = []
    
    for symbol in pairs:
        df = await fetch_pair_data(symbol, timeframe, days)
        if df is not None and not df.empty:
            all_data.append(df)
        await asyncio.sleep(1)  # Pause between pairs
    
    if not all_data:
        return None
    
    # Combine all data
    combined = pd.concat(all_data, axis=0)
    logger.info(f"Total combined data: {len(combined)} candles from {len(all_data)} pairs")
    
    return combined


def train_models(
    df: pd.DataFrame,
    config: dict,
    output_dir: str
):
    """Train all models on the combined data."""
    logger.info("Starting model training on multi-pair data...")
    
    # Initialize components
    feature_engine = FeatureEngine(config.get('features', {}))
    trainer = ModelTrainer(config.get('training', {}))
    
    # Process each symbol's data separately to generate features
    all_features = []
    all_labels = []
    
    symbols = df['symbol'].unique()
    logger.info(f"Processing {len(symbols)} symbols...")
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.drop(columns=['symbol'])
        symbol_df = symbol_df.dropna()
        
        if len(symbol_df) < 500:
            logger.warning(f"Skipping {symbol}: insufficient data ({len(symbol_df)} candles)")
            continue
        
        try:
            # Generate features
            features = feature_engine.generate_all_features(symbol_df, normalize=True)
            
            # Convert categorical columns to numeric
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = pd.Categorical(features[col]).codes
            
            # Prepare labels
            labels = trainer.prepare_labels(
                symbol_df,
                forward_periods=5,
                direction_threshold=0.003
            )
            
            all_features.append(features)
            all_labels.append(labels)
            
            logger.info(f"Processed {symbol}: {len(features)} samples")
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
            continue
    
    if not all_features:
        logger.error("No valid data to train on!")
        return None, None, None
    
    # Combine all features and labels
    combined_features = pd.concat(all_features, axis=0, ignore_index=True)
    
    # Combine labels
    combined_labels = {}
    for key in all_labels[0].keys():
        combined_labels[key] = pd.concat([l[key] for l in all_labels], axis=0, ignore_index=True)
    
    logger.info(f"Combined training data: {len(combined_features)} samples")
    
    # Train ensemble
    logger.info("Training ensemble model...")
    ensemble, metrics = trainer.train_ensemble(
        combined_features,
        combined_labels,
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
    cv_results = trainer.cross_validate(combined_features, combined_labels)
    logger.info(f"CV Results: {cv_results}")
    
    return ensemble, metrics, cv_results


def main():
    parser = argparse.ArgumentParser(description="Train trading models on multiple pairs")
    parser.add_argument(
        "--pairs",
        type=str,
        nargs='+',
        default=TRADING_PAIRS,
        help="Trading pairs to train on"
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
        default=365,
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
    
    # Fetch data for all pairs
    logger.info(f"Training models on {len(args.pairs)} pairs for {args.days} days")
    print(f"\nPairs: {', '.join(args.pairs)}\n")
    
    df = asyncio.run(fetch_all_pairs(args.pairs, args.timeframe, args.days))
    
    if df is None or df.empty:
        logger.error("Failed to fetch training data")
        return 1
    
    # Train models
    ensemble, metrics, cv_results = train_models(df, config, args.output)
    
    if ensemble is None:
        logger.error("Training failed")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-PAIR TRAINING SUMMARY")
    print("="*60)
    print(f"Pairs: {len(args.pairs)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {args.days} days")
    print(f"Total training samples: {len(df)}")
    print(f"\nMetrics:")
    for model, model_metrics in metrics.items():
        if isinstance(model_metrics, dict):
            print(f"  {model}:")
            for k, v in model_metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

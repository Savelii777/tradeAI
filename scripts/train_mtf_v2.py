#!/usr/bin/env python
"""
MTF Trading Bot - Model Training Script V2
==========================================

Trains all ML models with V2 improvements:
- Class weights and calibration for direction model
- Regression-based timing model
- Feature selection integration
- Walk-forward validation
- RR 1:3 optimized targets
- Blacklist filtering

Usage:
    python scripts/train_mtf_v2.py --pairs "BTC,ETH,SOL" --days 90
    python scripts/train_mtf_v2.py --all-pairs --days 180 --use-feature-selection
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.utils.helpers import load_yaml_config
from src.features.feature_engine import FeatureEngine
from src.models.direction_v2 import (
    DirectionModelV2,
    create_direction_target_v2,
    create_direction_target_rr3,
)
from src.models.timing_v2 import (
    TimingModelV2,
    create_timing_target_v2,
    create_timing_target_rr3,
)
from src.models.strength import StrengthModel
from src.models.volatility import VolatilityModel
from src.models.ensemble_v2 import EnsembleModelV2
from src.utils.blacklist import PairBlacklist


@dataclass
class TrainingConfig:
    """Configuration for training V2 models."""
    
    # Data settings
    pairs: List[str] = field(default_factory=list)
    days: int = 90
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    base_timeframe: str = "5m"
    
    # Split settings
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Direction model settings
    direction_threshold_pct: float = 0.002
    use_class_weights: bool = True
    use_calibration: bool = True
    sideways_penalty: float = 1.5
    
    # Timing model settings
    timing_method: str = "ratio"
    timing_lookahead: int = 15
    
    # RR settings
    use_rr3_targets: bool = True
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 4.5
    
    # Feature selection
    use_feature_selection: bool = False
    selected_features_file: Optional[str] = None
    target_n_features: int = 50
    
    # Walk-forward settings
    use_walk_forward: bool = True
    wf_train_days: int = 60
    wf_test_days: int = 7
    wf_step_days: int = 7
    
    # Output
    output_dir: str = "models/mtf_v2"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        config = load_yaml_config(config_path)
        return cls(
            pairs=config.get("training", {}).get("pairs", []),
            days=config.get("training", {}).get("days", 90),
            timeframes=config.get("training", {}).get("features", {}).get("timeframes", ["1m", "5m", "15m"]),
            base_timeframe=config.get("training", {}).get("features", {}).get("base_timeframe", "5m"),
            train_ratio=config.get("training", {}).get("train_ratio", 0.70),
            val_ratio=config.get("training", {}).get("val_ratio", 0.15),
            test_ratio=config.get("training", {}).get("test_ratio", 0.15),
            direction_threshold_pct=config.get("direction", {}).get("target", {}).get("threshold_pct", 0.002),
            use_class_weights=config.get("direction", {}).get("use_class_weights", True),
            use_calibration=config.get("direction", {}).get("use_calibration", True),
            sideways_penalty=config.get("direction", {}).get("sideways_penalty", 1.5),
            timing_method=config.get("timing", {}).get("target", {}).get("method", "ratio"),
            timing_lookahead=config.get("timing", {}).get("target", {}).get("lookahead_bars", 15),
            use_rr3_targets=config.get("ensemble", {}).get("rr_ratio", 3.0) >= 3.0,
            sl_atr_mult=config.get("direction", {}).get("target", {}).get("sl_atr_mult", 1.5),
            tp_atr_mult=config.get("direction", {}).get("target", {}).get("tp_atr_mult", 4.5),
            use_walk_forward=config.get("walk_forward", {}).get("enabled", True),
            wf_train_days=config.get("walk_forward", {}).get("train_days", 60),
            wf_test_days=config.get("walk_forward", {}).get("test_days", 7),
            wf_step_days=config.get("walk_forward", {}).get("step_days", 7),
            use_feature_selection=config.get("feature_selection", {}).get("enabled", False),
            target_n_features=config.get("feature_selection", {}).get("target_features", 50),
            output_dir=config.get("training", {}).get("output_dir", "models/mtf_v2"),
        )


@dataclass
class TrainingResults:
    """Results from model training."""
    
    direction_metrics: Dict[str, Any] = field(default_factory=dict)
    timing_metrics: Dict[str, Any] = field(default_factory=dict)
    strength_metrics: Dict[str, Any] = field(default_factory=dict)
    volatility_metrics: Dict[str, Any] = field(default_factory=dict)
    walk_forward_results: Optional[Dict] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time_seconds: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    selected_features: Optional[List[str]] = None


class MTFTrainerV2:
    """
    Multi-Timeframe Model Trainer V2.
    
    Integrates all V2 improvements:
    - DirectionModelV2 with class weights and calibration
    - TimingModelV2 with regression output
    - Feature selection
    - Walk-forward validation
    - RR 1:3 optimized targets
    """
    
    DEFAULT_PAIRS = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
        "DOGE/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT",
        "LINK/USDT:USDT", "ATOM/USDT:USDT", "LTC/USDT:USDT", "UNI/USDT:USDT",
        "APT/USDT:USDT", "NEAR/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
        "FIL/USDT:USDT", "INJ/USDT:USDT", "BNB/USDT:USDT"
    ]
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.blacklist = PairBlacklist()
        self.feature_engine = FeatureEngine({})
        
        # Models
        self.direction_model: Optional[DirectionModelV2] = None
        self.timing_model: Optional[TimingModelV2] = None
        self.strength_model: Optional[StrengthModel] = None
        self.volatility_model: Optional[VolatilityModel] = None
        self.ensemble_model: Optional[EnsembleModelV2] = None
        
        # Selected features
        self.selected_features: Optional[List[str]] = None
        if config.selected_features_file and os.path.exists(config.selected_features_file):
            with open(config.selected_features_file, 'r') as f:
                data = json.load(f)
                self.selected_features = data.get('features', [])
                logger.info(f"Loaded {len(self.selected_features)} selected features")
    
    async def fetch_pair_data(
        self,
        pair: str,
        timeframe: str,
        days: int,
        exchange_id: str = "binance"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a pair.
        
        Args:
            pair: Trading pair symbol
            timeframe: Data timeframe
            days: Number of days to fetch
            exchange_id: Exchange to use
            
        Returns:
            DataFrame with OHLCV data or None
        """
        import ccxt.async_support as ccxt
        
        logger.info(f"Fetching {days} days of {timeframe} data for {pair}")
        
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        try:
            # Format symbol
            formatted_symbol = pair.replace(':USDT', '').replace('/', '')
            if 'USDT' not in formatted_symbol:
                formatted_symbol += 'USDT'
            formatted_symbol = formatted_symbol.replace('USDT', '/USDT')
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            since = int(start_time.timestamp() * 1000)
            
            # Fetch OHLCV data in chunks
            all_ohlcv = []
            while True:
                try:
                    ohlcv = await exchange.fetch_ohlcv(
                        formatted_symbol, timeframe, since=since, limit=1000
                    )
                except Exception as e:
                    logger.warning(f"Error fetching {formatted_symbol}: {e}")
                    break
                    
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if since > int(end_time.timestamp() * 1000):
                    break
                    
                await asyncio.sleep(0.1)  # Rate limiting
            
            if not all_ohlcv:
                logger.warning(f"No data fetched for {pair}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            logger.info(f"Fetched {len(df)} candles for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {pair}: {e}")
            return None
        finally:
            await exchange.close()
    
    async def fetch_all_pairs_data(
        self,
        pairs: List[str],
        timeframe: str,
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple pairs.
        
        Args:
            pairs: List of trading pairs
            timeframe: Data timeframe
            days: Number of days to fetch
            
        Returns:
            Dictionary mapping pair to DataFrame
        """
        data = {}
        for pair in pairs:
            # Skip blacklisted pairs
            if self.blacklist.is_blacklisted(pair):
                logger.info(f"Skipping blacklisted pair: {pair}")
                continue
            
            df = await self.fetch_pair_data(pair, timeframe, days)
            if df is not None and len(df) >= 500:
                data[pair] = df
            else:
                logger.warning(f"Insufficient data for {pair}, skipping")
        
        return data
    
    def prepare_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Prepare features for all pairs.
        
        Args:
            data: Dictionary of pair DataFrames
            
        Returns:
            Combined feature DataFrame and individual pair features
        """
        logger.info("Preparing features for all pairs...")
        
        all_features = []
        pair_features = {}
        
        for pair, df in data.items():
            try:
                # Generate features
                features = self.feature_engine.generate_all_features(df, normalize=True)
                
                # Add pair identifier
                features['pair'] = pair
                
                # Filter selected features if available
                if self.selected_features:
                    available = [f for f in self.selected_features if f in features.columns]
                    features = features[available + ['pair']]
                
                pair_features[pair] = features
                all_features.append(features)
                
                logger.info(f"Generated {len(features.columns)-1} features for {pair}")
                
            except Exception as e:
                logger.warning(f"Failed to generate features for {pair}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No features generated for any pair")
        
        # Combine all features
        combined = pd.concat(all_features, axis=0)
        combined = combined.dropna()
        
        logger.info(f"Combined features: {len(combined)} samples, {len(combined.columns)-1} features")
        
        return combined, pair_features
    
    def prepare_targets(
        self,
        data: Dict[str, pd.DataFrame],
        features: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Prepare targets for all models.
        
        Args:
            data: Dictionary of pair DataFrames
            features: Combined feature DataFrame
            
        Returns:
            Dictionary of target Series
        """
        logger.info("Preparing targets...")
        
        targets = {}
        direction_targets = []
        timing_targets = []
        
        for pair, df in data.items():
            if pair not in features['pair'].unique():
                continue
            
            try:
                # Direction target
                if self.config.use_rr3_targets:
                    dir_target = create_direction_target_rr3(
                        df,
                        sl_atr_mult=self.config.sl_atr_mult,
                        tp_atr_mult=self.config.tp_atr_mult,
                        atr_period=14
                    )
                else:
                    dir_target = create_direction_target_v2(
                        df,
                        lookahead=5,
                        threshold_pct=self.config.direction_threshold_pct
                    )
                
                dir_target = dir_target.loc[features[features['pair'] == pair].index]
                direction_targets.append(dir_target)
                
                # Timing target
                if self.config.use_rr3_targets:
                    time_target = create_timing_target_rr3(
                        df,
                        sl_atr_mult=self.config.sl_atr_mult,
                        tp_atr_mult=self.config.tp_atr_mult,
                        atr_period=14
                    )
                else:
                    time_target = create_timing_target_v2(
                        df,
                        method=self.config.timing_method,
                        lookahead=self.config.timing_lookahead,
                        atr_period=14
                    )
                
                time_target = time_target.loc[features[features['pair'] == pair].index]
                timing_targets.append(time_target)
                
            except Exception as e:
                logger.warning(f"Failed to create targets for {pair}: {e}")
                continue
        
        if direction_targets:
            targets['direction'] = pd.concat(direction_targets)
        if timing_targets:
            targets['timing'] = pd.concat(timing_targets)
        
        logger.info(f"Created targets: direction={len(targets.get('direction', []))}, "
                   f"timing={len(targets.get('timing', []))}")
        
        return targets
    
    def split_data(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data chronologically into train/val/test.
        
        Args:
            features: Feature DataFrame
            targets: Dictionary of target Series
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        # Sort by index
        features = features.sort_index()
        
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_idx = features.index[:train_end]
        val_idx = features.index[train_end:val_end]
        test_idx = features.index[val_end:]
        
        train_data = {
            'features': features.loc[train_idx].drop('pair', axis=1),
            'targets': {k: v.loc[v.index.intersection(train_idx)] for k, v in targets.items()},
            'pairs': features.loc[train_idx, 'pair']
        }
        
        val_data = {
            'features': features.loc[val_idx].drop('pair', axis=1),
            'targets': {k: v.loc[v.index.intersection(val_idx)] for k, v in targets.items()},
            'pairs': features.loc[val_idx, 'pair']
        }
        
        test_data = {
            'features': features.loc[test_idx].drop('pair', axis=1),
            'targets': {k: v.loc[v.index.intersection(test_idx)] for k, v in targets.items()},
            'pairs': features.loc[test_idx, 'pair']
        }
        
        logger.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return train_data, val_data, test_data
    
    def train_direction_model(
        self,
        train_data: Dict,
        val_data: Dict
    ) -> Dict[str, Any]:
        """
        Train the direction model with V2 improvements.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            
        Returns:
            Metrics dictionary
        """
        logger.info("Training DirectionModelV2...")
        
        X_train = train_data['features'].reset_index(drop=True)
        y_train = train_data['targets']['direction'].reset_index(drop=True)
        X_val = val_data['features'].reset_index(drop=True)
        y_val = val_data['targets']['direction'].reset_index(drop=True)
        
        # Ensure same length
        min_train = min(len(X_train), len(y_train))
        min_val = min(len(X_val), len(y_val))
        
        X_train = X_train.iloc[:min_train]
        y_train = y_train.iloc[:min_train]
        X_val = X_val.iloc[:min_val]
        y_val = y_val.iloc[:min_val]
        
        logger.info(f"Direction training data: X_train={len(X_train)}, y_train={len(y_train)}")
        
        # Initialize model
        self.direction_model = DirectionModelV2(
            use_class_weights=self.config.use_class_weights,
            use_calibration=self.config.use_calibration,
            sideways_penalty=self.config.sideways_penalty
        )
        
        # Train model
        train_result = self.direction_model.train(
            X_train, y_train,
            X_val, y_val
        )
        
        # Get predictions and calculate metrics
        train_preds = self.direction_model.predict(X_train)
        val_preds = self.direction_model.predict(X_val)
        
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        # Count class distribution (predictions are -1, 0, 1)
        train_dist = {-1: 0, 0: 0, 1: 0}
        val_dist = {-1: 0, 0: 0, 1: 0}
        for p in train_preds:
            train_dist[p] = train_dist.get(p, 0) + 1
        for p in val_preds:
            val_dist[p] = val_dist.get(p, 0) + 1
            
        train_dist_pct = {k: v / len(train_preds) * 100 for k, v in train_dist.items()}
        val_dist_pct = {k: v / len(val_preds) * 100 for k, v in val_dist.items()}
        
        logger.info(f"Train accuracy: {train_acc:.4f}")
        logger.info(f"Val accuracy: {val_acc:.4f}")
        logger.info(f"Train distribution: Down={train_dist_pct.get(-1, 0):.1f}%, Sideways={train_dist_pct.get(0, 0):.1f}%, Up={train_dist_pct.get(1, 0):.1f}%")
        logger.info(f"Val distribution: Down={val_dist_pct.get(-1, 0):.1f}%, Sideways={val_dist_pct.get(0, 0):.1f}%, Up={val_dist_pct.get(1, 0):.1f}%")
        
        return {
            'train': {'accuracy': train_acc},
            'val': {'accuracy': val_acc},
            'train_distribution': train_dist,
            'val_distribution': val_dist
        }
    
    def train_timing_model(
        self,
        train_data: Dict,
        val_data: Dict
    ) -> Dict[str, Any]:
        """
        Train the timing model with V2 regression output.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            
        Returns:
            Metrics dictionary
        """
        logger.info("Training TimingModelV2...")
        
        X_train = train_data['features'].reset_index(drop=True)
        y_train = train_data['targets']['timing'].reset_index(drop=True)
        X_val = val_data['features'].reset_index(drop=True)
        y_val = val_data['targets']['timing'].reset_index(drop=True)
        
        # Ensure same length
        min_train = min(len(X_train), len(y_train))
        min_val = min(len(X_val), len(y_val))
        
        X_train = X_train.iloc[:min_train]
        y_train = y_train.iloc[:min_train]
        X_val = X_val.iloc[:min_val]
        y_val = y_val.iloc[:min_val]
        
        logger.info(f"Timing training data: X_train={len(X_train)}, y_train={len(y_train)}")
        
        # Initialize model
        self.timing_model = TimingModelV2()
        
        # Train model
        train_result = self.timing_model.train(
            X_train, y_train,
            X_val, y_val
        )
        
        # Get predictions for metrics
        train_preds = self.timing_model.predict(X_train)
        val_preds = self.timing_model.predict(X_val)
        
        # Calculate correlation as metric
        train_corr = np.corrcoef(train_preds, y_train)[0, 1] if len(train_preds) > 1 else 0
        val_corr = np.corrcoef(val_preds, y_val)[0, 1] if len(val_preds) > 1 else 0
        
        logger.info(f"Timing predictions - Train: mean={train_preds.mean():.4f}, std={train_preds.std():.4f}, corr={train_corr:.4f}")
        logger.info(f"Timing predictions - Val: mean={val_preds.mean():.4f}, std={val_preds.std():.4f}, corr={val_corr:.4f}")
        
        return {
            'train': {'correlation': train_corr, 'mean': float(train_preds.mean())},
            'val': {'correlation': val_corr, 'mean': float(val_preds.mean())},
            'train_pred_mean': float(train_preds.mean()),
            'train_pred_std': float(train_preds.std()),
            'val_pred_mean': float(val_preds.mean()),
            'val_pred_std': float(val_preds.std())
        }
    
    def train_all_models(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict
    ) -> TrainingResults:
        """
        Train all models.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            
        Returns:
            TrainingResults object
        """
        import time
        from sklearn.metrics import accuracy_score
        start_time = time.time()
        
        results = TrainingResults()
        results.n_samples = len(train_data['features']) + len(val_data['features']) + len(test_data['features'])
        results.n_features = len(train_data['features'].columns)
        
        # Train direction model
        results.direction_metrics = self.train_direction_model(train_data, val_data)
        
        # Test direction model
        X_test = test_data['features'].reset_index(drop=True)
        y_test_dir = test_data['targets']['direction'].reset_index(drop=True)
        min_test = min(len(X_test), len(y_test_dir))
        X_test = X_test.iloc[:min_test]
        y_test_dir = y_test_dir.iloc[:min_test]
        
        test_preds = self.direction_model.predict(X_test)
        test_acc = accuracy_score(y_test_dir, test_preds)
        results.direction_metrics['test'] = {'accuracy': test_acc}
        
        # Log test distribution
        test_dist = {-1: 0, 0: 0, 1: 0}
        for p in test_preds:
            test_dist[p] = test_dist.get(p, 0) + 1
        test_dist_pct = {k: v / len(test_preds) * 100 for k, v in test_dist.items()}
        logger.info(f"Test direction: accuracy={test_acc:.4f}, down={test_dist_pct.get(-1, 0):.1f}%, sideways={test_dist_pct.get(0, 0):.1f}%, up={test_dist_pct.get(1, 0):.1f}%")
        results.direction_metrics['test_distribution'] = test_dist
        
        # Train timing model
        results.timing_metrics = self.train_timing_model(train_data, val_data)
        
        # Test timing model
        y_test_time = test_data['targets']['timing'].reset_index(drop=True)
        min_test_time = min(len(X_test), len(y_test_time))
        X_test_time = X_test.iloc[:min_test_time]
        y_test_time = y_test_time.iloc[:min_test_time]
        
        test_time_preds = self.timing_model.predict(X_test_time)
        test_time_corr = np.corrcoef(test_time_preds, y_test_time)[0, 1] if len(test_time_preds) > 1 else 0
        results.timing_metrics['test'] = {'correlation': test_time_corr}
        
        logger.info(f"Test timing predictions: mean={test_time_preds.mean():.4f}, std={test_time_preds.std():.4f}, corr={test_time_corr:.4f}")
        results.timing_metrics['test_pred_mean'] = float(test_time_preds.mean())
        results.timing_metrics['test_pred_std'] = float(test_time_preds.std())
        
        # Create ensemble with trained models
        self.ensemble_model = EnsembleModelV2(
            model_config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            use_v2_models=True,
            rr_ratio=self.config.tp_atr_mult / self.config.sl_atr_mult
        )
        # Replace with already trained models
        self.ensemble_model.direction_model = self.direction_model
        self.ensemble_model.timing_model = self.timing_model
        if self.strength_model:
            self.ensemble_model.strength_model = self.strength_model
        if self.volatility_model:
            self.ensemble_model.volatility_model = self.volatility_model
        # Mark ensemble as trained
        self.ensemble_model._is_trained = True
        
        # Feature importance
        if hasattr(self.direction_model.model, 'feature_importances_'):
            importances = self.direction_model.model.feature_importances_
            feature_names = train_data['features'].columns.tolist()
            results.feature_importance = dict(zip(feature_names, importances.tolist()))
        
        results.training_time_seconds = time.time() - start_time
        results.selected_features = self.selected_features
        
        return results
    
    def save_models(self, output_dir: str):
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save direction model
        if self.direction_model:
            direction_path = os.path.join(output_dir, "direction_v2.pkl")
            with open(direction_path, 'wb') as f:
                pickle.dump(self.direction_model, f)
            logger.info(f"Saved direction model to {direction_path}")
        
        # Save timing model
        if self.timing_model:
            timing_path = os.path.join(output_dir, "timing_v2.pkl")
            with open(timing_path, 'wb') as f:
                pickle.dump(self.timing_model, f)
            logger.info(f"Saved timing model to {timing_path}")
        
        # Save ensemble model
        if self.ensemble_model:
            ensemble_path = os.path.join(output_dir, "ensemble_v2.pkl")
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            logger.info(f"Saved ensemble model to {ensemble_path}")
        
        # Save config
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        logger.info(f"Saved config to {config_path}")
    
    async def train(self) -> TrainingResults:
        """
        Run the full training pipeline.
        
        Returns:
            TrainingResults object
        """
        logger.info("Starting MTF V2 training pipeline...")
        
        # Get pairs
        pairs = self.config.pairs or self.DEFAULT_PAIRS
        logger.info(f"Training on {len(pairs)} pairs")
        
        # Fetch data
        data = await self.fetch_all_pairs_data(
            pairs,
            self.config.base_timeframe,
            self.config.days
        )
        
        if not data:
            raise ValueError("No data fetched for any pair")
        
        logger.info(f"Fetched data for {len(data)} pairs")
        
        # Prepare features
        features, pair_features = self.prepare_features(data)
        
        # Prepare targets
        targets = self.prepare_targets(data, features)
        
        # Split data
        train_data, val_data, test_data = self.split_data(features, targets)
        
        # Train models
        results = self.train_all_models(train_data, val_data, test_data)
        
        # Save models
        self.save_models(self.config.output_dir)
        
        return results


def print_results(results: TrainingResults):
    """Print training results summary."""
    print("\n" + "=" * 70)
    print("MTF V2 TRAINING RESULTS")
    print("=" * 70)
    
    print(f"\nTotal samples: {results.n_samples}")
    print(f"Total features: {results.n_features}")
    print(f"Training time: {results.training_time_seconds:.1f}s")
    
    print("\n" + "-" * 40)
    print("DIRECTION MODEL (V2)")
    print("-" * 40)
    
    for split in ['train', 'val', 'test']:
        if split in results.direction_metrics:
            metrics = results.direction_metrics[split]
            print(f"\n{split.upper()}:")
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
    
    # Distribution check
    if 'test_distribution' in results.direction_metrics:
        dist = results.direction_metrics['test_distribution']
        # dist keys are -1, 0, 1 (down, sideways, up)
        total = sum(dist.values()) if sum(dist.values()) > 0 else 1
        down_pct = dist.get(-1, 0) / total * 100
        sideways_pct = dist.get(0, 0) / total * 100
        up_pct = dist.get(1, 0) / total * 100
        print(f"\nTest Distribution: down={down_pct:.1f}%, sideways={sideways_pct:.1f}%, up={up_pct:.1f}%")
        if sideways_pct < 50:
            print("✓ Sideways bias improved (target: <50%)")
        else:
            print("⚠ Sideways still dominant (target: <50%)")
    
    print("\n" + "-" * 40)
    print("TIMING MODEL (V2)")
    print("-" * 40)
    
    for split in ['train', 'val', 'test']:
        if split in results.timing_metrics:
            metrics = results.timing_metrics[split]
            print(f"\n{split.upper()}:")
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
    
    # Timing check
    if 'test_pred_mean' in results.timing_metrics:
        mean = results.timing_metrics['test_pred_mean']
        print(f"\nTest prediction mean: {mean:.4f}")
        if 0.35 <= mean <= 0.65:
            print("✓ Timing distribution good (target: 0.35-0.65)")
        else:
            print(f"⚠ Timing needs adjustment (got {mean:.3f}, target: 0.35-0.65)")
    
    print("\n" + "=" * 70)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train MTF V2 models")
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated list of pairs (e.g., 'BTC,ETH,SOL')"
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Use all default pairs"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of historical data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mtf_v2",
        help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_params.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--use-feature-selection",
        action="store_true",
        help="Use feature selection"
    )
    parser.add_argument(
        "--selected-features-file",
        type=str,
        default=None,
        help="Path to selected features JSON file"
    )
    parser.add_argument(
        "--use-rr3",
        action="store_true",
        default=True,
        help="Use RR 1:3 targets"
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward validation"
    )
    
    args = parser.parse_args()
    
    # Build config
    if os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()
    
    # Override with CLI args
    if args.pairs:
        config.pairs = [f"{p.strip()}/USDT:USDT" for p in args.pairs.split(",")]
    elif args.all_pairs:
        config.pairs = []  # Use defaults
    
    config.days = args.days
    config.output_dir = args.output
    config.use_feature_selection = args.use_feature_selection
    config.selected_features_file = args.selected_features_file
    config.use_rr3_targets = args.use_rr3
    config.use_walk_forward = not args.no_walk_forward
    
    # Initialize trainer
    trainer = MTFTrainerV2(config)
    
    # Run training
    try:
        results = await trainer.train()
        print_results(results)
        
        # Save results
        results_path = os.path.join(config.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'direction_metrics': results.direction_metrics,
                'timing_metrics': results.timing_metrics,
                'n_samples': results.n_samples,
                'n_features': results.n_features,
                'training_time_seconds': results.training_time_seconds,
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

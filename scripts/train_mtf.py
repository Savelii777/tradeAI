#!/usr/bin/env python3
"""
Multi-Timeframe (MTF) Training Script

Strategy:
- M15: Trend context (direction filter)
- M5:  Signal generation (main predictions)
- M1:  Entry timing optimization

Uses top 20 pairs by historical accuracy.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from src.models.ensemble import EnsembleModel
from src.models.validator import ModelValidator


# ============================================================
# TOP PAIRS SELECTION
# ============================================================

# Based on single-TF training results + liquidity
TOP_20_PAIRS = [
    # Top performers from training
    'XAUT/USDT:USDT',   # 93.98%
    'BTC/USDT:USDT',    # 82.79%
    'BNB/USDT:USDT',    # 79.86%
    'TONCOIN/USDT:USDT', # 76.00%
    'ETH/USDT:USDT',    # Major
    'SOL/USDT:USDT',    # Major
    'XRP/USDT:USDT',    # Major
    'DOGE/USDT:USDT',   # Major
    'ADA/USDT:USDT',    # Major
    'AVAX/USDT:USDT',   # Major
    'LINK/USDT:USDT',   # Major
    'DOT/USDT:USDT',    # Major
    'LTC/USDT:USDT',    # Major
    'BCH/USDT:USDT',    # Major
    'UNI/USDT:USDT',    # DeFi
    'AAVE/USDT:USDT',   # DeFi
    'SUI/USDT:USDT',    # L1
    'APT/USDT:USDT',    # L1
    'NEAR/USDT:USDT',   # L1
    'OP/USDT:USDT',     # L2
]

# Pairs to EXCLUDE (< 50% accuracy)
EXCLUDED_PAIRS = [
    'PIPPIN/USDT:USDT',
    'BEAT/USDT:USDT',
    'H/USDT:USDT',
    'XMR/USDT:USDT',
    'ICNT/USDT:USDT',
    'NIGHT/USDT:USDT',
    'RAVE/USDT:USDT',
]


# ============================================================
# MTF DATA LOADING (from local CSV files)
# ============================================================

def load_pair_data(symbol: str, data_dir: str = 'data/candles', timeframe: str = '5m') -> Optional[pd.DataFrame]:
    """Load OHLCV data for a single pair from CSV."""
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    filepath = Path(data_dir) / f"{safe_symbol}_{timeframe}.csv"
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def load_mtf_data(
    pair: str,
    data_dir: str = 'data/candles'
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load M1, M5, M15 data for a single pair from local CSV files.
    
    Returns:
        Dict with keys 'm1', 'm5', 'm15' containing OHLCV DataFrames
    """
    logger.debug(f"Loading MTF data for {pair}")
    
    mtf_data = {}
    
    for tf in ['1m', '5m', '15m']:
        df = load_pair_data(pair, data_dir, tf)
        
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {pair} {tf}")
            return None
        
        # Store with simplified key
        key = tf.replace('m', '')  # '1m' -> '1', '5m' -> '5'
        mtf_data[f'm{key}'] = df
    
    logger.debug(f"Loaded MTF data for {pair}: M1={len(mtf_data.get('m1', []))}, M5={len(mtf_data.get('m5', []))}, M15={len(mtf_data.get('m15', []))}")
    
    return mtf_data


# ============================================================
# MTF FEATURE ENGINEERING
# ============================================================

class MTFFeatureEngine:
    """
    Multi-Timeframe Feature Engineering.
    
    Generates features from M1, M5, M15 and combines them.
    """
    
    def __init__(self):
        self.feature_engine = FeatureEngine()
    
    def generate_m15_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        M15 trend context features.
        
        Focus on: trend direction, trend strength, key levels
        """
        features = pd.DataFrame(index=df.index)
        
        # Trend direction (EMA crossover)
        ema_fast = df['close'].ewm(span=8, adjust=False).mean()
        ema_slow = df['close'].ewm(span=21, adjust=False).mean()
        
        features['m15_trend'] = np.where(ema_fast > ema_slow, 1, -1)
        features['m15_trend_strength'] = (ema_fast - ema_slow) / df['close'] * 100
        
        # Trend momentum
        features['m15_rsi'] = self._calculate_rsi(df['close'], 14)
        features['m15_momentum'] = df['close'].pct_change(5) * 100
        
        # Volatility context
        features['m15_atr'] = self._calculate_atr(df, 14)
        features['m15_atr_pct'] = features['m15_atr'] / df['close'] * 100
        
        # Support/Resistance proximity
        features['m15_high_dist'] = (df['high'].rolling(20).max() - df['close']) / df['close'] * 100
        features['m15_low_dist'] = (df['close'] - df['low'].rolling(20).min()) / df['close'] * 100
        
        # Volume trend
        features['m15_volume_ma'] = df['volume'].rolling(10).mean()
        features['m15_volume_ratio'] = df['volume'] / features['m15_volume_ma']
        
        return features
    
    def generate_m5_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        M5 signal generation features.
        
        Focus on: entry signals, momentum, patterns
        """
        # Use full feature engine for M5
        features = self.feature_engine.generate_all_features(df)
        
        # Add prefix to all columns
        features.columns = [f'm5_{col}' for col in features.columns]
        
        return features
    
    def generate_m1_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        M1 timing optimization features.
        
        Focus on: micro-structure, momentum bursts, entry precision
        """
        features = pd.DataFrame(index=df.index)
        
        # Micro momentum
        features['m1_momentum_1'] = df['close'].pct_change(1) * 100
        features['m1_momentum_3'] = df['close'].pct_change(3) * 100
        features['m1_momentum_5'] = df['close'].pct_change(5) * 100
        
        # Micro RSI (fast)
        features['m1_rsi_5'] = self._calculate_rsi(df['close'], 5)
        features['m1_rsi_9'] = self._calculate_rsi(df['close'], 9)
        
        # Volume burst detection
        vol_ma = df['volume'].rolling(20).mean()
        features['m1_volume_spike'] = df['volume'] / vol_ma
        
        # Candle patterns
        features['m1_body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        features['m1_upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open'] * 100
        features['m1_lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open'] * 100
        
        # Micro trend
        features['m1_ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
        features['m1_ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        features['m1_micro_trend'] = np.where(features['m1_ema_3'] > features['m1_ema_8'], 1, -1)
        
        # Price vs VWAP (if volume available)
        if 'volume' in df.columns:
            vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            features['m1_vwap_dist'] = (df['close'] - vwap) / vwap * 100
        
        return features
    
    def align_timeframes(
        self,
        m1_df: pd.DataFrame,
        m5_df: pd.DataFrame,
        m15_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align all timeframes to M5 timestamps.
        
        M15 features are forward-filled to M5.
        M1 features are aggregated to M5.
        """
        # Generate features for each TF
        logger.debug("Generating M15 features...")
        m15_features = self.generate_m15_trend_features(m15_df)
        
        logger.debug("Generating M5 features...")
        m5_features = self.generate_m5_signal_features(m5_df)
        
        logger.debug("Generating M1 features...")
        m1_features = self.generate_m1_timing_features(m1_df)
        
        # Align to M5 index
        aligned = m5_features.copy()
        
        # Forward-fill M15 to M5 (each M15 covers 3 M5 candles)
        m15_resampled = m15_features.resample('5min').ffill()
        
        # Join M15 features
        for col in m15_features.columns:
            if col in m15_resampled.columns:
                aligned[col] = m15_resampled[col].reindex(aligned.index, method='ffill')
        
        # Aggregate M1 to M5 (last 5 M1 candles per M5)
        m1_agg = m1_features.resample('5min').agg({
            col: ['last', 'mean', 'std'] if 'momentum' in col or 'rsi' in col 
            else 'last'
            for col in m1_features.columns
        })
        
        # Flatten column names
        m1_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                         for col in m1_agg.columns]
        
        # Join M1 features
        for col in m1_agg.columns:
            aligned[col] = m1_agg[col].reindex(aligned.index, method='ffill')
        
        return aligned.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()


# ============================================================
# MTF TRAINING PIPELINE
# ============================================================

def prepare_mtf_data(
    pairs: List[str],
    data_dir: str = 'data/candles',
    min_samples: int = 1000
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare MTF training data for all pairs.
    """
    mtf_engine = MTFFeatureEngine()
    
    all_features = []
    all_targets = []
    pair_info = []
    
    for pair in pairs:
        if pair in EXCLUDED_PAIRS:
            logger.warning(f"Skipping excluded pair: {pair}")
            continue
        
        # Load MTF data from local files
        mtf_data = load_mtf_data(pair, data_dir)
        
        if mtf_data is None:
            logger.warning(f"No MTF data for {pair}")
            continue
        
        try:
            # Align timeframes and generate features
            features = mtf_engine.align_timeframes(
                mtf_data['m1'],
                mtf_data['m5'],
                mtf_data['m15']
            )
            
            if len(features) < min_samples:
                logger.warning(f"{pair}: only {len(features)} samples, skipping")
                continue
            
            # Create target Dict (like in training.py prepare_labels)
            close = mtf_data['m5']['close']
            forward_periods = 5
            
            # Direction (simple: next close > current close)
            forward_return = close.pct_change(periods=forward_periods).shift(-forward_periods)
            rolling_vol = close.pct_change().rolling(window=100).std()
            threshold = np.maximum(rolling_vol, 0.003)
            
            direction = pd.Series(0, index=mtf_data['m5'].index)  # Sideways
            direction[forward_return > threshold] = 1  # Up
            direction[forward_return < -threshold] = -1  # Down
            direction = direction.reindex(features.index).dropna()
            
            # Strength (movement magnitude)
            atr = (mtf_data['m5']['high'] - mtf_data['m5']['low']).ewm(span=14).mean()
            strength = (forward_return.abs() * close / atr).fillna(0)
            strength = strength.reindex(features.index).dropna()
            
            # Volatility (future volatility)
            future_vol = close.pct_change().rolling(window=forward_periods).std().shift(-forward_periods)
            volatility = future_vol.fillna(method='bfill')
            volatility = volatility.reindex(features.index).dropna()
            
            # Timing (good entry timing)
            future_high = mtf_data['m5']['high'].rolling(window=forward_periods).max().shift(-forward_periods)
            future_low = mtf_data['m5']['low'].rolling(window=forward_periods).min().shift(-forward_periods)
            max_gain = (future_high - close) / close
            max_loss = (close - future_low) / close
            timing = ((max_gain > max_loss) & (max_gain > 0.005)).astype(int)
            timing = timing.reindex(features.index).dropna()
            
            # Find common index
            common_idx = features.index.intersection(direction.index).intersection(
                strength.index).intersection(volatility.index).intersection(timing.index)
            
            features = features.loc[common_idx]
            target = {
                'direction': direction.loc[common_idx],
                'strength': strength.loc[common_idx],
                'volatility': volatility.loc[common_idx],
                'timing': timing.loc[common_idx]
            }
            
            # Add pair identifier
            features['pair'] = pair
            
            all_features.append(features)
            all_targets.append(target)
            
            logger.info(f"✓ {pair}: {len(features)} MTF samples")
            
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid MTF data collected")
    
    # Combine all pairs - reset index to avoid duplicates
    X = pd.concat(all_features, ignore_index=True)
    
    # Combine targets (list of dicts -> dict of concatenated series) - also reset index
    y = {
        'direction': pd.concat([t['direction'].reset_index(drop=True) for t in all_targets], ignore_index=True),
        'strength': pd.concat([t['strength'].reset_index(drop=True) for t in all_targets], ignore_index=True),
        'volatility': pd.concat([t['volatility'].reset_index(drop=True) for t in all_targets], ignore_index=True),
        'timing': pd.concat([t['timing'].reset_index(drop=True) for t in all_targets], ignore_index=True)
    }
    
    # Store pair info before dropping
    pairs_col = X['pair'].copy()
    X = X.drop(columns=['pair'])
    
    # Convert object columns to categorical codes (LightGBM requirement)
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.debug(f"Converting object column to numeric: {col}")
            X[col] = pd.Categorical(X[col]).codes
    
    logger.info(f"\nTotal MTF samples: {len(X):,}")
    logger.info(f"MTF features: {len(X.columns)}")
    
    return X, y, pairs_col


def mtf_chronological_split(
    X: pd.DataFrame,
    y: Dict[str, pd.Series],
    pairs: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple:
    """
    Chronological split respecting time order within each pair.
    Since we used reset_index, X and y have aligned integer indices.
    """
    train_dfs, val_dfs, test_dfs = [], [], []
    train_ys = {'direction': [], 'strength': [], 'volatility': [], 'timing': []}
    val_ys = {'direction': [], 'strength': [], 'volatility': [], 'timing': []}
    test_ys = {'direction': [], 'strength': [], 'volatility': [], 'timing': []}
    
    for pair in pairs.unique():
        mask = pairs == pair
        indices = np.where(mask)[0]  # Get positional indices
        
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        train_dfs.append(X.iloc[train_idx])
        val_dfs.append(X.iloc[val_idx])
        test_dfs.append(X.iloc[test_idx])
        
        # Split each target component using same indices
        for key in ['direction', 'strength', 'volatility', 'timing']:
            train_ys[key].append(y[key].iloc[train_idx])
            val_ys[key].append(y[key].iloc[val_idx])
            test_ys[key].append(y[key].iloc[test_idx])
    
    X_train = pd.concat(train_dfs, ignore_index=True)
    X_val = pd.concat(val_dfs, ignore_index=True)
    X_test = pd.concat(test_dfs, ignore_index=True)
    
    # Concatenate target dicts
    y_train = {k: pd.concat(v, ignore_index=True) for k, v in train_ys.items()}
    y_val = {k: pd.concat(v, ignore_index=True) for k, v in val_ys.items()}
    y_test = {k: pd.concat(v, ignore_index=True) for k, v in test_ys.items()}
    
    # Verify lengths match
    logger.debug(f"X_test: {len(X_test)}, y_test['direction']: {len(y_test['direction'])}")
    
    logger.info(f"MTF Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_mtf_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Dict[str, pd.Series],
    y_val: Dict[str, pd.Series],
    y_test: Dict[str, pd.Series],
    save_path: str = 'models/saved_mtf'
) -> Dict:
    """
    Train MTF ensemble model.
    """
    logger.info("Training MTF ensemble...")
    
    # Initialize ensemble
    ensemble = EnsembleModel()
    
    # Train
    ensemble.train_all(X_train, y_train, X_val, y_val)
    
    # Evaluate on test
    logger.info("Evaluating on test set...")
    predictions = ensemble.predict(X_test)
    
    # Calculate metrics - compare direction
    if isinstance(predictions, dict):
        pred_direction = predictions.get('direction', predictions.get('prediction'))
    else:
        pred_direction = predictions
    
    if hasattr(pred_direction, 'values'):
        pred_direction = pred_direction.values
    
    y_test_dir = y_test['direction'].values if isinstance(y_test, dict) else y_test.values
    
    # Debug: Check lengths
    logger.debug(f"pred_direction shape: {len(pred_direction)}, y_test_dir shape: {len(y_test_dir)}")
    
    # y_test should already be correctly sized from the split function
    # If not, truncate to match X_test length
    if len(y_test_dir) != len(pred_direction):
        logger.warning(f"Shape mismatch! Truncating y_test to match prediction length")
        y_test_dir = y_test_dir[:len(pred_direction)]
    
    test_accuracy = (pred_direction == y_test_dir).mean()
    
    # Save model
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(str(save_dir))
    
    logger.info(f"MTF Model saved to {save_path}")
    
    return {
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'y_test': y_test
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MTF Model Training')
    parser.add_argument('--data-dir', type=str, default='data/candles', help='Data directory')
    parser.add_argument('--min-samples', type=int, default=500, help='Min samples per pair')
    parser.add_argument('--pairs', type=int, default=20, help='Number of top pairs')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MULTI-TIMEFRAME (MTF) TRAINING")
    logger.info("=" * 60)
    logger.info(f"Timeframes: M15 (trend) + M5 (signal) + M1 (timing)")
    logger.info(f"Top pairs: {args.pairs}")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Select pairs
    pairs = TOP_20_PAIRS[:args.pairs]
    logger.info(f"\nTraining on pairs: {pairs}")
    
    # Load and prepare MTF data
    logger.info("\nLoading MTF data...")
    X, y, pairs_col = prepare_mtf_data(
        pairs, 
        data_dir=args.data_dir,
        min_samples=args.min_samples
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = mtf_chronological_split(
        X, y, pairs_col
    )
    
    # Train model
    results = train_mtf_model(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("MTF TRAINING RESULTS")
    print("=" * 60)
    print(f"Pairs: {len(pairs)}")
    print(f"Total samples: {len(X):,}")
    print(f"MTF Features: {len(X.columns)}")
    print(f"\nTest Direction Accuracy: {results['test_accuracy']:.2%}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()

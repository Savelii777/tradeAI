"""
AI Trading Bot - Data Splitter
Provides proper chronological data splitting to prevent data leakage.
"""

from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd


class DataSplitter:
    """
    Splits time series data chronologically to prevent data leakage.
    
    Key principles:
    - No random shuffling - strict chronological order
    - Test data is NEVER seen during training
    - Walk-forward validation simulates real trading
    
    Attributes:
        logger: Logger instance.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize DataSplitter.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def split_chronological(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/validation/test sets.
        
        Data is split in time order:
        [---- Train (70%) ----][-- Val (15%) --][-- Test (15%) --]
        
        Args:
            data: DataFrame with datetime index or 'timestamp' column.
            train_ratio: Fraction for training (default 0.70).
            val_ratio: Fraction for validation (default 0.15).
            test_ratio: Fraction for testing (default 0.15).
            
        Returns:
            Tuple of (train_df, val_df, test_df).
            
        Raises:
            ValueError: If ratios don't sum to 1.0 or data is too small.
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        if len(data) < 100:
            raise ValueError(f"Data too small for splitting: {len(data)} rows")
        
        # Ensure data is sorted by time
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        elif 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()
        
        self.logger.info(
            f"Chronological split: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )
        
        # Log time ranges
        if isinstance(data.index, pd.DatetimeIndex):
            self.logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]}")
            self.logger.info(f"Val: {val_df.index[0]} to {val_df.index[-1]}")
            self.logger.info(f"Test: {test_df.index[0]} to {test_df.index[-1]}")
        
        return train_df, val_df, test_df
    
    def walk_forward_split(
        self,
        data: pd.DataFrame,
        train_months: int = 6,
        test_months: int = 1,
        min_folds: int = 6,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        Generate walk-forward validation splits.
        
        Walk-forward process:
        1. Train on first N months
        2. Test on next M months  
        3. Slide window forward by M months
        4. Repeat until end of data
        
        This simulates real trading where model is periodically retrained.
        
        Args:
            data: DataFrame with datetime index.
            train_months: Number of months for training window.
            test_months: Number of months for test window.
            min_folds: Minimum number of folds required.
            
        Yields:
            Tuple of (train_df, test_df, fold_number).
            
        Raises:
            ValueError: If data doesn't have datetime index or too small.
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index or 'timestamp' column")
        
        data = data.sort_index()
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Calculate window sizes
        train_window = timedelta(days=train_months * 30)
        test_window = timedelta(days=test_months * 30)
        
        # Check if we have enough data
        total_needed = train_window + test_window * min_folds
        data_duration = end_date - start_date
        
        if data_duration < total_needed:
            self.logger.warning(
                f"Data duration ({data_duration.days} days) may not be enough "
                f"for {min_folds} folds with {train_months}m train + {test_months}m test"
            )
        
        fold = 0
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + train_window
            test_end = train_end + test_window
            
            # Check if we've reached the end
            if test_end > end_date:
                break
            
            # Extract train and test data
            train_df = data[
                (data.index >= current_train_start) & 
                (data.index < train_end)
            ].copy()
            
            test_df = data[
                (data.index >= train_end) & 
                (data.index < test_end)
            ].copy()
            
            if len(train_df) < 100 or len(test_df) < 20:
                self.logger.warning(
                    f"Fold {fold}: insufficient data (train={len(train_df)}, "
                    f"test={len(test_df)}), skipping"
                )
                current_train_start += test_window
                continue
            
            fold += 1
            self.logger.info(
                f"Fold {fold}: train {train_df.index[0].date()} to "
                f"{train_df.index[-1].date()}, test {test_df.index[0].date()} "
                f"to {test_df.index[-1].date()}"
            )
            
            yield train_df, test_df, fold
            
            # Slide window forward
            current_train_start += test_window
        
        if fold < min_folds:
            self.logger.warning(
                f"Only generated {fold} folds, less than minimum {min_folds}"
            )
    
    def expanding_window_split(
        self,
        data: pd.DataFrame,
        initial_train_months: int = 3,
        test_months: int = 1,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        Generate expanding window splits.
        
        Unlike walk-forward, training window expands to include all past data:
        Fold 1: [Train: 0-3m] [Test: 3-4m]
        Fold 2: [Train: 0-4m] [Test: 4-5m]
        Fold 3: [Train: 0-5m] [Test: 5-6m]
        
        Args:
            data: DataFrame with datetime index.
            initial_train_months: Initial training period in months.
            test_months: Test period in months.
            
        Yields:
            Tuple of (train_df, test_df, fold_number).
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index")
        
        data = data.sort_index()
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        initial_train = timedelta(days=initial_train_months * 30)
        test_window = timedelta(days=test_months * 30)
        
        fold = 0
        train_end = start_date + initial_train
        
        while True:
            test_end = train_end + test_window
            
            if test_end > end_date:
                break
            
            # Train on all data from start to train_end
            train_df = data[data.index < train_end].copy()
            test_df = data[
                (data.index >= train_end) & 
                (data.index < test_end)
            ].copy()
            
            if len(train_df) < 100 or len(test_df) < 20:
                train_end += test_window
                continue
            
            fold += 1
            self.logger.info(
                f"Expanding fold {fold}: train from start to "
                f"{train_df.index[-1].date()}, test to {test_df.index[-1].date()}"
            )
            
            yield train_df, test_df, fold
            
            train_end += test_window
    
    def get_fold_stats(
        self,
        fold_results: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics across all folds.
        
        Args:
            fold_results: List of metric dictionaries from each fold.
            
        Returns:
            Dictionary with mean, std, and confidence intervals for each metric.
        """
        if not fold_results:
            return {}
        
        # Get all metric names
        metrics = set()
        for result in fold_results:
            metrics.update(result.keys())
        
        stats = {}
        for metric in metrics:
            values = [r.get(metric, np.nan) for r in fold_results]
            values = [v for v in values if not np.isnan(v)]
            
            if not values:
                continue
            
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            
            # 95% confidence interval
            ci_margin = 1.96 * std / np.sqrt(n) if n > 1 else 0
            
            stats[metric] = {
                'mean': mean,
                'std': std,
                'min': np.min(values),
                'max': np.max(values),
                'ci_lower': mean - ci_margin,
                'ci_upper': mean + ci_margin,
                'n_folds': n,
            }
        
        return stats
    
    def validate_no_overlap(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> bool:
        """
        Validate that splits don't overlap in time.
        
        Args:
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            
        Returns:
            True if no overlap, False otherwise.
        """
        def get_time_range(df):
            if isinstance(df.index, pd.DatetimeIndex):
                return df.index[0], df.index[-1]
            elif 'timestamp' in df.columns:
                return df['timestamp'].min(), df['timestamp'].max()
            return None, None
        
        train_start, train_end = get_time_range(train_df)
        val_start, val_end = get_time_range(val_df)
        test_start, test_end = get_time_range(test_df)
        
        if train_end is None or val_start is None or test_start is None:
            self.logger.warning("Cannot validate overlap: missing time info")
            return True
        
        # Check chronological order
        if train_end >= val_start:
            self.logger.error(
                f"Train/Val overlap: train ends {train_end}, val starts {val_start}"
            )
            return False
        
        if val_end >= test_start:
            self.logger.error(
                f"Val/Test overlap: val ends {val_end}, test starts {test_start}"
            )
            return False
        
        self.logger.info("✓ No data leakage: all splits are chronologically ordered")
        return True

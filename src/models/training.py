"""
AI Trading Bot - Model Training Pipeline
Orchestrates training of all ML models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from .ensemble import EnsembleModel


class ModelTrainer:
    """
    Manages training, validation, and evaluation of trading models.
    
    Features:
    - Time-series cross-validation
    - Walk-forward analysis
    - Hyperparameter optimization
    - Model versioning
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config or {}
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.val_ratio = self.config.get('val_ratio', 0.15)
        self.n_splits = self.config.get('n_splits', 5)
        
        self.training_history: List[Dict] = []
        self.best_model: Optional[EnsembleModel] = None
        self.best_score = -np.inf
        
    def prepare_labels(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        direction_threshold: float = 0.003
    ) -> Dict[str, pd.Series]:
        """
        Prepare labels for all models.
        
        Args:
            df: DataFrame with OHLCV data.
            forward_periods: Number of periods to look ahead.
            direction_threshold: Threshold for direction classification.
            
        Returns:
            Dictionary of label series for each model.
        """
        labels = {}
        close = df['close']
        
        # Direction labels
        forward_return = close.pct_change(periods=forward_periods).shift(-forward_periods)
        
        # Adaptive threshold based on volatility
        rolling_vol = close.pct_change().rolling(window=100).std()
        threshold = np.maximum(rolling_vol, direction_threshold)
        
        direction = pd.Series(0, index=df.index)  # Sideways
        direction[forward_return > threshold] = 1  # Up
        direction[forward_return < -threshold] = -1  # Down
        labels['direction'] = direction
        
        # Strength labels (absolute movement in ATR)
        atr = self._calculate_atr(df, 14)
        labels['strength'] = (forward_return.abs() * close / atr).fillna(0)
        
        # Volatility labels (future volatility)
        future_volatility = close.pct_change().rolling(window=forward_periods).std().shift(-forward_periods)
        labels['volatility'] = future_volatility.fillna(method='bfill')
        
        # Timing labels
        future_high = df['high'].rolling(window=forward_periods).max().shift(-forward_periods)
        future_low = df['low'].rolling(window=forward_periods).min().shift(-forward_periods)
        
        max_gain = (future_high - close) / close
        max_loss = (close - future_low) / close
        
        # Good entry: max gain > max loss and significant gain
        labels['timing'] = ((max_gain > max_loss) & (max_gain > 0.005)).astype(int)
        
        return labels
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean()
        
    def split_data(
        self,
        features: pd.DataFrame,
        labels: Dict[str, pd.Series]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train/val/test sets.
        
        Args:
            features: Feature DataFrame.
            labels: Dictionary of label series.
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries.
        """
        n = len(features)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_data = {
            'X': features.iloc[:train_end],
            'y': {k: v.iloc[:train_end] for k, v in labels.items()}
        }
        
        val_data = {
            'X': features.iloc[train_end:val_end],
            'y': {k: v.iloc[train_end:val_end] for k, v in labels.items()}
        }
        
        test_data = {
            'X': features.iloc[val_end:],
            'y': {k: v.iloc[val_end:] for k, v in labels.items()}
        }
        
        logger.info(f"Data split: train={len(train_data['X'])}, "
                   f"val={len(val_data['X'])}, test={len(test_data['X'])}")
        
        return train_data, val_data, test_data
        
    def train_ensemble(
        self,
        features: pd.DataFrame,
        labels: Dict[str, pd.Series],
        model_config: Optional[Dict] = None
    ) -> Tuple[EnsembleModel, Dict[str, Any]]:
        """
        Train the full ensemble model.
        
        Args:
            features: Feature DataFrame.
            labels: Dictionary of label series.
            model_config: Model configuration.
            
        Returns:
            Tuple of (trained model, training metrics).
        """
        # Clean data
        features, labels = self._clean_data(features, labels)
        
        # Split data
        train_data, val_data, test_data = self.split_data(features, labels)
        
        # Create and train ensemble
        ensemble = EnsembleModel(model_config=model_config)
        
        train_metrics = ensemble.train_all(
            train_data['X'], train_data['y'],
            val_data['X'], val_data['y']
        )
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(
            ensemble,
            test_data['X'],
            test_data['y']
        )
        
        # Store training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'data_size': len(features),
            'train_size': len(train_data['X']),
            'val_size': len(val_data['X']),
            'test_size': len(test_data['X'])
        }
        self.training_history.append(training_record)
        
        # Update best model if improved
        composite_score = test_metrics.get('direction_accuracy', 0) * 0.5 + \
                         test_metrics.get('timing_auc', 0) * 0.3 + \
                         (1 - test_metrics.get('strength_mae', 1)) * 0.2
                         
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_model = ensemble
            logger.info(f"New best model with score: {composite_score:.4f}")
            
        return ensemble, {**train_metrics, 'test': test_metrics}
        
    def _clean_data(
        self,
        features: pd.DataFrame,
        labels: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Remove rows with NaN values."""
        # Get common valid indices
        valid_mask = ~features.isnull().any(axis=1)
        
        for label_series in labels.values():
            valid_mask &= ~label_series.isnull()
            
        features = features[valid_mask]
        labels = {k: v[valid_mask] for k, v in labels.items()}
        
        logger.info(f"Clean data: {valid_mask.sum()} valid samples")
        return features, labels
        
    def evaluate_model(
        self,
        model: EnsembleModel,
        X: pd.DataFrame,
        y: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Trained ensemble model.
            X: Feature DataFrame.
            y: Dictionary of label series.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        predictions = model.predict(X)
        
        # Direction metrics
        if 'direction' in y and 'direction' in predictions:
            direction_pred = predictions['direction']
            direction_true = y['direction'].values
            
            metrics['direction_accuracy'] = accuracy_score(direction_true, direction_pred)
            metrics['direction_f1'] = f1_score(
                direction_true, direction_pred, average='weighted'
            )
            
        # Strength metrics
        if 'strength' in y and 'strength' in predictions:
            strength_pred = predictions['strength']
            strength_true = y['strength'].values
            
            metrics['strength_mae'] = mean_absolute_error(strength_true, strength_pred)
            
        # Timing metrics
        if 'timing' in y and 'timing' in predictions:
            from sklearn.metrics import roc_auc_score
            
            timing_pred = predictions['timing']
            timing_true = y['timing'].values
            
            try:
                metrics['timing_auc'] = roc_auc_score(timing_true, timing_pred)
            except ValueError:
                metrics['timing_auc'] = 0.5
                
        return metrics
        
    def walk_forward_validation(
        self,
        features: pd.DataFrame,
        labels: Dict[str, pd.Series],
        train_months: int = 6,
        test_months: int = 1,
        model_config: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform walk-forward validation.
        
        Args:
            features: Feature DataFrame.
            labels: Dictionary of label series.
            train_months: Months of training data.
            test_months: Months of test data.
            model_config: Model configuration.
            
        Returns:
            List of validation results for each fold.
        """
        features, labels = self._clean_data(features, labels)
        
        # Determine split points based on dates
        dates = features.index
        if not isinstance(dates, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, using integer splits")
            return self._walk_forward_integer(features, labels, model_config)
            
        results = []
        start_date = dates.min()
        end_date = dates.max()
        
        current_date = start_date + pd.DateOffset(months=train_months)
        
        while current_date + pd.DateOffset(months=test_months) <= end_date:
            train_end = current_date
            test_end = current_date + pd.DateOffset(months=test_months)
            
            # Split data
            train_mask = dates < train_end
            test_mask = (dates >= train_end) & (dates < test_end)
            
            X_train = features[train_mask]
            y_train = {k: v[train_mask] for k, v in labels.items()}
            X_test = features[test_mask]
            y_test = {k: v[test_mask] for k, v in labels.items()}
            
            if len(X_train) < 1000 or len(X_test) < 100:
                current_date += pd.DateOffset(months=test_months)
                continue
                
            # Train model
            ensemble = EnsembleModel(model_config=model_config)
            ensemble.train_all(X_train, y_train)
            
            # Evaluate
            test_metrics = self.evaluate_model(ensemble, X_test, y_test)
            
            results.append({
                'train_start': dates[train_mask].min(),
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': test_metrics
            })
            
            logger.info(f"Fold {len(results)}: {test_metrics}")
            current_date += pd.DateOffset(months=test_months)
            
        return results
        
    def _walk_forward_integer(
        self,
        features: pd.DataFrame,
        labels: Dict[str, pd.Series],
        model_config: Optional[Dict]
    ) -> List[Dict]:
        """Walk-forward with integer indices."""
        n = len(features)
        train_size = int(n * 0.6)
        test_size = int(n * 0.1)
        step_size = test_size
        
        results = []
        start = 0
        
        while start + train_size + test_size <= n:
            train_end = start + train_size
            test_end = train_end + test_size
            
            X_train = features.iloc[start:train_end]
            y_train = {k: v.iloc[start:train_end] for k, v in labels.items()}
            X_test = features.iloc[train_end:test_end]
            y_test = {k: v.iloc[train_end:test_end] for k, v in labels.items()}
            
            ensemble = EnsembleModel(model_config=model_config)
            ensemble.train_all(X_train, y_train)
            
            test_metrics = self.evaluate_model(ensemble, X_test, y_test)
            
            results.append({
                'fold': len(results) + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': test_metrics
            })
            
            start += step_size
            
        return results
        
    def cross_validate(
        self,
        features: pd.DataFrame,
        labels: Dict[str, pd.Series],
        model_config: Optional[Dict] = None
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            features: Feature DataFrame.
            labels: Dictionary of label series.
            model_config: Model configuration.
            
        Returns:
            Dictionary of metric lists across folds.
        """
        features, labels = self._clean_data(features, labels)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_results = {
            'direction_accuracy': [],
            'direction_f1': [],
            'strength_mae': [],
            'timing_auc': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            X_train = features.iloc[train_idx]
            y_train = {k: v.iloc[train_idx] for k, v in labels.items()}
            X_test = features.iloc[test_idx]
            y_test = {k: v.iloc[test_idx] for k, v in labels.items()}
            
            ensemble = EnsembleModel(model_config=model_config)
            ensemble.train_all(X_train, y_train)
            
            metrics = self.evaluate_model(ensemble, X_test, y_test)
            
            for key in cv_results:
                if key in metrics:
                    cv_results[key].append(metrics[key])
                    
            logger.info(f"Fold {fold + 1}/{self.n_splits}: {metrics}")
            
        # Calculate summary statistics
        summary = {}
        for key, values in cv_results.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                
        logger.info(f"CV Summary: {summary}")
        return cv_results
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training runs."""
        if not self.training_history:
            return {'message': 'No training history'}
            
        return {
            'total_runs': len(self.training_history),
            'best_score': self.best_score,
            'latest_run': self.training_history[-1],
            'history': self.training_history
        }

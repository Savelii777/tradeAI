"""
AI Trading Bot - Timing Model
Predicts optimal entry timing.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import joblib


class TimingModel:
    """
    Predicts optimal entry timing.
    
    Answers: Should we enter now or wait for a better price?
    Output: Probability of now being a good entry point.
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict] = None
    ):
        """
        Initialize the timing model.
        
        Args:
            model_type: Type of model ('lightgbm', 'catboost').
            params: Model hyperparameters.
        """
        self.model_type = model_type
        self.params = params or self._default_params()
        self.model = None
        self.feature_names: List[str] = []
        self._is_trained = False
        
    def _default_params(self) -> Dict:
        """Get default model parameters with strong regularization."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'n_estimators': 500,
                'max_depth': 4,              # Ограничено
                'num_leaves': 15,            # Меньше
                'min_child_samples': 200,    # Увеличено
                'learning_rate': 0.02,       # Медленнее
                'subsample': 0.7,
                'subsample_freq': 3,
                'colsample_bytree': 0.5,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42,
                'verbosity': -1,
                'force_row_wise': True
            }
        else:
            return {}
            
    def _create_model(self) -> Any:
        """Create the underlying model."""
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**self.params)
        elif self.model_type == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def create_timing_labels(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        threshold_atr: float = 0.5
    ) -> pd.Series:
        """
        Create labels for timing model.
        
        A good entry is when price moves favorably within N periods
        without first moving against us significantly.
        
        Args:
            df: DataFrame with OHLCV data.
            forward_periods: Number of periods to look forward.
            threshold_atr: ATR threshold for adverse movement.
            
        Returns:
            Binary series (1 = good entry, 0 = bad entry).
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14).mean()
        
        # For bullish signals: check if low doesn't go too far below entry
        # before price moves up
        future_lows = low.rolling(window=forward_periods).min().shift(-forward_periods)
        future_highs = high.rolling(window=forward_periods).max().shift(-forward_periods)
        
        # Max adverse excursion before favorable move
        adverse_long = (close - future_lows) / atr
        favorable_long = (future_highs - close) / atr
        
        # Good long entry: favorable > adverse and favorable is significant
        good_long_entry = (favorable_long > adverse_long) & (favorable_long > 1.0)
        
        return good_long_entry.astype(int)
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the timing model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (0 or 1).
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            Training metrics.
        """
        self.feature_names = X_train.columns.tolist()
        self.model = self._create_model()
        
        if X_val is not None and y_val is not None:
            if self.model_type == "lightgbm":
                import lightgbm as lgb
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=100)
                    ]
                )
            else:
                self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            self.model.fit(X_train, y_train)
            
        self._is_trained = True
        
        # Calculate training metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        
        metrics = {
            'train_auc': train_auc,
            'train_accuracy': train_acc
        }
        
        if X_val is not None:
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            metrics['val_auc'] = val_auc
            metrics['val_accuracy'] = val_acc
            
        logger.info(f"Timing model trained: {metrics}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict if now is a good entry time.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of binary predictions.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of good entry.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        return self.model.predict_proba(X)[:, 1]
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))
            
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
    def save(self, path: str) -> None:
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_trained': self._is_trained
        }, path)
        logger.info(f"Timing model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self._is_trained = data['is_trained']
        logger.info(f"Timing model loaded from {path}")

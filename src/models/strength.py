"""
AI Trading Bot - Strength Prediction Model
Predicts the magnitude of price movement.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
import joblib


class StrengthModel:
    """
    Predicts the strength/magnitude of price movement.
    
    Uses gradient boosting for regression.
    Output: Expected movement in ATR units.
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict] = None
    ):
        """
        Initialize the strength model.
        
        Args:
            model_type: Type of model ('lightgbm', 'catboost', 'xgboost').
            params: Model hyperparameters.
        """
        self.model_type = model_type
        self.params = params or self._default_params()
        self.model = None
        self.feature_names: List[str] = []
        self._is_trained = False
        
    def _default_params(self) -> Dict:
        """Get default model parameters."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'regression',
                'metric': 'mae',
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbosity': -1
            }
        elif self.model_type == "catboost":
            return {
                'loss_function': 'MAE',
                'iterations': 300,
                'depth': 6,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbose': False
            }
        else:
            return {}
            
    def _create_model(self) -> Any:
        """Create the underlying model."""
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(**self.params)
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**self.params)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train the strength model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (movement magnitude).
            X_val: Validation features.
            y_val: Validation labels.
            sample_weight: Sample weights for training.
            
        Returns:
            Training metrics.
        """
        self.feature_names = X_train.columns.tolist()
        self.model = self._create_model()
        
        if X_val is not None and y_val is not None:
            if self.model_type == "lightgbm":
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    sample_weight=sample_weight
                )
            elif self.model_type == "catboost":
                self.model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    sample_weight=sample_weight
                )
            else:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    sample_weight=sample_weight
                )
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            
        self._is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_mae = np.mean(np.abs(train_pred - y_train))
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = np.mean(np.abs(val_pred - y_val))
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            metrics['val_mae'] = val_mae
            metrics['val_rmse'] = val_rmse
            
        logger.info(f"Strength model trained: {metrics}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict movement strength.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of predicted movement magnitudes.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        return self.model.predict(X)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
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
        logger.info(f"Strength model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self._is_trained = data['is_trained']
        logger.info(f"Strength model loaded from {path}")

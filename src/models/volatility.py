"""
AI Trading Bot - Volatility Prediction Model
Predicts expected volatility for position sizing and stop-loss calculation.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
import joblib


class VolatilityModel:
    """
    Predicts expected volatility for the next N periods.
    
    Uses gradient boosting for regression.
    Output: Expected volatility (ATR or standard deviation).
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict] = None
    ):
        """
        Initialize the volatility model.
        
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
        """Get default model parameters with strong regularization."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': 500,
                'max_depth': 4,
                'num_leaves': 15,
                'min_child_samples': 200,
                'learning_rate': 0.02,
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
            return lgb.LGBMRegressor(**self.params)
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the volatility model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (future volatility).
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
        train_pred = self.model.predict(X_train)
        train_mae = np.mean(np.abs(train_pred - y_train))
        
        metrics = {'train_mae': train_mae}
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = np.mean(np.abs(val_pred - y_val))
            metrics['val_mae'] = val_mae
            
        logger.info(f"Volatility model trained: {metrics}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future volatility.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of predicted volatility values.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        # Ensure predictions are positive
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)
        
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
        logger.info(f"Volatility model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self._is_trained = data['is_trained']
        logger.info(f"Volatility model loaded from {path}")

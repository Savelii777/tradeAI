"""
AI Trading Bot - Direction Prediction Model
Predicts the direction of price movement.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import joblib


class DirectionModel:
    """
    Predicts price direction (up, down, sideways).
    
    Uses gradient boosting (LightGBM) as the primary model.
    Output: Probabilities for each direction class.
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict] = None
    ):
        """
        Initialize the direction model.
        
        Args:
            model_type: Type of model ('lightgbm', 'catboost', 'xgboost').
            params: Model hyperparameters.
        """
        self.model_type = model_type
        self.params = params or self._default_params()
        self.model = None
        self.feature_names: List[str] = []
        self.classes = [-1, 0, 1]  # Down, Sideways, Up
        self._is_trained = False
        
    def _default_params(self) -> Dict:
        """Get default model parameters."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 63,
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
                'loss_function': 'MultiClass',
                'iterations': 500,
                'depth': 8,
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
            return lgb.LGBMClassifier(**self.params)
        elif self.model_type == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**self.params)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**self.params)
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
        Train the direction model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (-1, 0, 1).
            X_val: Validation features.
            y_val: Validation labels.
            sample_weight: Sample weights for training.
            
        Returns:
            Training metrics.
        """
        self.feature_names = X_train.columns.tolist()
        self.model = self._create_model()
        
        # Map labels to 0, 1, 2 for model training
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        
        if X_val is not None and y_val is not None:
            y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})
            
            if self.model_type == "lightgbm":
                self.model.fit(
                    X_train, y_train_mapped,
                    eval_set=[(X_val, y_val_mapped)],
                    sample_weight=sample_weight
                )
            elif self.model_type == "catboost":
                self.model.fit(
                    X_train, y_train_mapped,
                    eval_set=(X_val, y_val_mapped),
                    sample_weight=sample_weight
                )
            else:
                self.model.fit(
                    X_train, y_train_mapped,
                    eval_set=[(X_val, y_val_mapped)],
                    sample_weight=sample_weight
                )
        else:
            self.model.fit(X_train, y_train_mapped, sample_weight=sample_weight)
            
        self._is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train_mapped).mean()
        
        metrics = {'train_accuracy': train_accuracy}
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = (val_pred == y_val_mapped).mean()
            metrics['val_accuracy'] = val_accuracy
            
        logger.info(f"Direction model trained: {metrics}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict direction classes.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of predictions (-1, 0, 1).
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        pred = self.model.predict(X)
        # Map back to original classes
        return np.array([-1, 0, 1])[pred]
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of shape (n_samples, 3) with probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
            
        return self.model.predict_proba(X)
        
    def get_prediction_with_confidence(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with confidence scores.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Tuple of (predictions, confidence scores).
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        predictions = np.array([-1, 0, 1])[predictions]
        confidence = np.max(proba, axis=1)
        
        return predictions, confidence
        
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
        logger.info(f"Direction model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self._is_trained = data['is_trained']
        logger.info(f"Direction model loaded from {path}")

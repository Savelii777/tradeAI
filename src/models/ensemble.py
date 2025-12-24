"""
AI Trading Bot - Ensemble Model
Combines predictions from multiple models.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import joblib

from .direction import DirectionModel
from .strength import StrengthModel
from .volatility import VolatilityModel
from .timing import TimingModel


class EnsembleModel:
    """
    Ensemble that combines multiple specialized models.
    
    Components:
    - Direction model: Predicts up/down/sideways
    - Strength model: Predicts movement magnitude
    - Volatility model: Predicts expected volatility
    - Timing model: Predicts entry quality
    
    Output: Combined trading signal with confidence.
    """
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ensemble model.
        
        Args:
            model_config: Configuration for individual models.
            weights: Weights for combining model outputs.
        """
        self.config = model_config or {}
        self.weights = weights or {
            'direction': 0.4,
            'strength': 0.2,
            'timing': 0.2,
            'volatility': 0.2
        }
        
        # Initialize component models
        self.direction_model = DirectionModel(
            model_type=self.config.get('direction', {}).get('type', 'lightgbm'),
            params=self.config.get('direction', {}).get('params')
        )
        
        self.strength_model = StrengthModel(
            model_type=self.config.get('strength', {}).get('type', 'lightgbm'),
            params=self.config.get('strength', {}).get('params')
        )
        
        self.volatility_model = VolatilityModel(
            model_type=self.config.get('volatility', {}).get('type', 'lightgbm'),
            params=self.config.get('volatility', {}).get('params')
        )
        
        self.timing_model = TimingModel(
            model_type=self.config.get('timing', {}).get('type', 'lightgbm'),
            params=self.config.get('timing', {}).get('params')
        )
        
        # Meta-model for combining predictions
        self.meta_model = None
        self.use_meta_model = self.config.get('use_meta_model', False)
        
        self._is_trained = False
        
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: Dict[str, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all component models.
        
        Args:
            X_train: Training features.
            y_train: Dictionary of training labels for each model.
            X_val: Validation features.
            y_val: Dictionary of validation labels.
            
        Returns:
            Training metrics for each model.
        """
        metrics = {}
        
        # Train direction model
        if 'direction' in y_train:
            logger.info("Training direction model...")
            metrics['direction'] = self.direction_model.train(
                X_train, y_train['direction'],
                X_val, y_val.get('direction') if y_val else None
            )
            
        # Train strength model
        if 'strength' in y_train:
            logger.info("Training strength model...")
            metrics['strength'] = self.strength_model.train(
                X_train, y_train['strength'],
                X_val, y_val.get('strength') if y_val else None
            )
            
        # Train volatility model
        if 'volatility' in y_train:
            logger.info("Training volatility model...")
            metrics['volatility'] = self.volatility_model.train(
                X_train, y_train['volatility'],
                X_val, y_val.get('volatility') if y_val else None
            )
            
        # Train timing model
        if 'timing' in y_train:
            logger.info("Training timing model...")
            metrics['timing'] = self.timing_model.train(
                X_train, y_train['timing'],
                X_val, y_val.get('timing') if y_val else None
            )
            
        # Train meta-model if requested
        if self.use_meta_model:
            self._train_meta_model(X_train, y_train, X_val, y_val)
            
        self._is_trained = True
        logger.info("Ensemble training complete")
        
        return metrics
        
    def _train_meta_model(
        self,
        X_train: pd.DataFrame,
        y_train: Dict[str, pd.Series],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[Dict[str, pd.Series]]
    ) -> None:
        """Train the meta-model on base model predictions."""
        # Get base model predictions
        base_preds = self._get_base_predictions(X_train)
        
        # Create meta-features
        meta_features = pd.DataFrame(base_preds, index=X_train.index)
        
        # Train meta-model (simple gradient boosting)
        import lightgbm as lgb
        
        self.meta_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=-1
        )
        
        # Use direction as target for meta-model
        if 'direction' in y_train:
            y_meta = (y_train['direction'] > 0).astype(int)
            self.meta_model.fit(meta_features, y_meta)
            logger.info("Meta-model trained")
            
    def _get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all base models."""
        predictions = {}
        
        if self.direction_model._is_trained:
            dir_proba = self.direction_model.predict_proba(X)
            predictions['dir_prob_down'] = dir_proba[:, 0]
            predictions['dir_prob_sideways'] = dir_proba[:, 1]
            predictions['dir_prob_up'] = dir_proba[:, 2]
            
        if self.strength_model._is_trained:
            predictions['strength'] = self.strength_model.predict(X)
            
        if self.volatility_model._is_trained:
            predictions['volatility'] = self.volatility_model.predict(X)
            
        if self.timing_model._is_trained:
            predictions['timing'] = self.timing_model.predict_proba(X)
            
        return predictions
        
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all models.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Dictionary with all model predictions.
        """
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained")
            
        predictions = {}
        
        # Direction prediction
        if self.direction_model._is_trained:
            dir_pred, dir_conf = self.direction_model.get_prediction_with_confidence(X)
            predictions['direction'] = dir_pred
            predictions['direction_confidence'] = dir_conf
            predictions['direction_proba'] = self.direction_model.predict_proba(X)
            
        # Strength prediction
        if self.strength_model._is_trained:
            predictions['strength'] = self.strength_model.predict(X)
            
        # Volatility prediction
        if self.volatility_model._is_trained:
            predictions['volatility'] = self.volatility_model.predict(X)
            
        # Timing prediction
        if self.timing_model._is_trained:
            predictions['timing'] = self.timing_model.predict_proba(X)
            
        return predictions
        
    def get_trading_signal(
        self,
        X: pd.DataFrame,
        min_direction_prob: float = 0.6,
        min_strength: float = 1.5,
        min_timing: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate trading signals from ensemble predictions.
        
        Args:
            X: Feature DataFrame (single row or multiple rows).
            min_direction_prob: Minimum direction probability threshold.
            min_strength: Minimum expected movement threshold.
            min_timing: Minimum timing score threshold.
            
        Returns:
            Dictionary with trading signals.
        """
        predictions = self.predict(X)
        
        # Calculate composite score
        direction_score = np.zeros(len(X))
        
        if 'direction_proba' in predictions:
            proba = predictions['direction_proba']
            # Score: positive for up probability, negative for down
            direction_score = proba[:, 2] - proba[:, 0]
            
        strength_score = predictions.get('strength', np.ones(len(X)))
        timing_score = predictions.get('timing', np.ones(len(X)))
        volatility = predictions.get('volatility', np.ones(len(X)))
        
        # Combined score
        composite_score = (
            self.weights['direction'] * direction_score +
            self.weights['strength'] * np.clip(strength_score / 3, 0, 1) +
            self.weights['timing'] * timing_score
        )
        
        # Generate signals
        signals = []
        for i in range(len(X)):
            direction_prob = predictions.get('direction_proba', np.array([[0.33, 0.34, 0.33]]))[i]
            
            signal = {
                'index': X.index[i] if hasattr(X, 'index') else i,
                'composite_score': composite_score[i],
                'direction_score': direction_score[i],
                'direction_proba': direction_prob.tolist(),
                'strength': strength_score[i] if isinstance(strength_score, np.ndarray) else strength_score,
                'timing': timing_score[i] if isinstance(timing_score, np.ndarray) else timing_score,
                'volatility': volatility[i] if isinstance(volatility, np.ndarray) else volatility,
                'signal': 'hold',
                'confidence': 0.0
            }
            
            # Determine signal
            max_prob_idx = np.argmax(direction_prob)
            max_prob = direction_prob[max_prob_idx]
            
            if max_prob >= min_direction_prob:
                str_val = signal['strength'] if isinstance(signal['strength'], (int, float)) else 1.0
                tim_val = signal['timing'] if isinstance(signal['timing'], (int, float)) else 1.0
                
                if str_val >= min_strength and tim_val >= min_timing:
                    if max_prob_idx == 2:  # Up
                        signal['signal'] = 'buy'
                        signal['confidence'] = max_prob
                    elif max_prob_idx == 0:  # Down
                        signal['signal'] = 'sell'
                        signal['confidence'] = max_prob
                        
            signals.append(signal)
            
        return signals if len(signals) > 1 else signals[0]
        
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update model combination weights."""
        self.weights.update(new_weights)
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"Updated ensemble weights: {self.weights}")
        
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from all models."""
        importance = {}
        
        if self.direction_model._is_trained:
            importance['direction'] = self.direction_model.get_feature_importance()
            
        if self.strength_model._is_trained:
            importance['strength'] = self.strength_model.get_feature_importance()
            
        if self.volatility_model._is_trained:
            importance['volatility'] = self.volatility_model.get_feature_importance()
            
        if self.timing_model._is_trained:
            importance['timing'] = self.timing_model.get_feature_importance()
            
        return importance
        
    def save(self, directory: str) -> None:
        """Save all models to directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.direction_model.save(f"{directory}/direction_model.joblib")
        self.strength_model.save(f"{directory}/strength_model.joblib")
        self.volatility_model.save(f"{directory}/volatility_model.joblib")
        self.timing_model.save(f"{directory}/timing_model.joblib")
        
        # Save ensemble metadata
        joblib.dump({
            'weights': self.weights,
            'config': self.config,
            'use_meta_model': self.use_meta_model,
            'is_trained': self._is_trained
        }, f"{directory}/ensemble_meta.joblib")
        
        if self.meta_model is not None:
            joblib.dump(self.meta_model, f"{directory}/meta_model.joblib")
            
        logger.info(f"Ensemble saved to {directory}")
        
    def load(self, directory: str) -> None:
        """Load all models from directory."""
        self.direction_model.load(f"{directory}/direction_model.joblib")
        self.strength_model.load(f"{directory}/strength_model.joblib")
        self.volatility_model.load(f"{directory}/volatility_model.joblib")
        self.timing_model.load(f"{directory}/timing_model.joblib")
        
        # Load ensemble metadata
        meta = joblib.load(f"{directory}/ensemble_meta.joblib")
        self.weights = meta['weights']
        self.config = meta['config']
        self.use_meta_model = meta['use_meta_model']
        self._is_trained = meta['is_trained']
        
        import os
        if os.path.exists(f"{directory}/meta_model.joblib"):
            self.meta_model = joblib.load(f"{directory}/meta_model.joblib")
            
        logger.info(f"Ensemble loaded from {directory}")

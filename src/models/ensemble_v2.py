"""
AI Trading Bot - Ensemble Model V2
Combines predictions from multiple models with improved signal generation.

IMPROVEMENTS:
- Uses DirectionModelV2 with class weights
- Uses TimingModelV2 with regression
- Dynamic thresholds based on volatility
- Integrated blacklist support
- Better position sizing
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
import joblib
import lightgbm as lgb

from .direction_v2 import DirectionModelV2, create_direction_target_v2, create_direction_target_rr3
from .timing_v2 import TimingModelV2, create_timing_target_v2, create_timing_target_rr3
from .strength import StrengthModel
from .volatility import VolatilityModel


@dataclass
class TradingSignal:
    """Trading signal with all relevant information."""
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    direction_proba: List[float]
    strength: float
    timing: float
    volatility: float
    
    # Position sizing
    suggested_sl_atr: float = 1.5
    suggested_tp_atr: float = 4.5  # RR 1:3
    position_size_mult: float = 1.0
    
    # Metadata
    reason: str = ""
    pair: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'direction_proba': self.direction_proba,
            'strength': self.strength,
            'timing': self.timing,
            'volatility': self.volatility,
            'suggested_sl_atr': self.suggested_sl_atr,
            'suggested_tp_atr': self.suggested_tp_atr,
            'position_size_mult': self.position_size_mult,
            'reason': self.reason,
            'pair': self.pair
        }


class EnsembleModelV2:
    """
    Ensemble that combines multiple specialized models (V2).
    
    Improvements over V1:
    - DirectionModelV2 with class weights and calibration
    - TimingModelV2 with regression output
    - Dynamic thresholds based on volatility
    - Blacklist integration
    - RR 1:3 optimized signal generation
    """
    
    # Default signal thresholds
    DEFAULT_MIN_DIRECTION_PROB = 0.45  # Lower than V1 (was 0.60)
    DEFAULT_MIN_STRENGTH = 0.3
    DEFAULT_MIN_TIMING = 0.4  # Higher threshold now that timing is calibrated
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None,
        use_v2_models: bool = True,
        rr_ratio: float = 3.0,  # Risk:Reward ratio
    ):
        """
        Initialize the ensemble model.
        
        Args:
            model_config: Configuration for individual models.
            weights: Weights for combining model outputs.
            use_v2_models: Whether to use V2 models (recommended).
            rr_ratio: Target risk:reward ratio for signals.
        """
        self.config = model_config or {}
        self.weights = weights or {
            'direction': 0.4,
            'strength': 0.2,
            'timing': 0.25,  # Increased weight for better timing
            'volatility': 0.15
        }
        self.use_v2_models = use_v2_models
        self.rr_ratio = rr_ratio
        
        # Initialize component models
        if use_v2_models:
            self.direction_model = DirectionModelV2(
                model_type=self.config.get('direction', {}).get('type', 'lightgbm'),
                params=self.config.get('direction', {}).get('params'),
                use_class_weights=True,
                use_calibration=True,
                sideways_penalty=1.5
            )
            
            self.timing_model = TimingModelV2(
                model_type=self.config.get('timing', {}).get('type', 'lightgbm'),
                params=self.config.get('timing', {}).get('params'),
                output_type='regression',
                normalize_output=True
            )
        else:
            # Fallback to V1 models
            from .direction import DirectionModel
            from .timing import TimingModel
            self.direction_model = DirectionModel(
                model_type=self.config.get('direction', {}).get('type', 'lightgbm'),
                params=self.config.get('direction', {}).get('params')
            )
            self.timing_model = TimingModel(
                model_type=self.config.get('timing', {}).get('type', 'lightgbm'),
                params=self.config.get('timing', {}).get('params')
            )
        
        self.strength_model = StrengthModel(
            model_type=self.config.get('strength', {}).get('type', 'lightgbm'),
            params=self.config.get('strength', {}).get('params')
        )
        
        self.volatility_model = VolatilityModel(
            model_type=self.config.get('volatility', {}).get('type', 'lightgbm'),
            params=self.config.get('volatility', {}).get('params')
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
        
        # Check if y_val is valid
        has_y_val = y_val is not None and (not hasattr(y_val, 'empty') or not y_val.empty)
        
        # Train direction model
        if 'direction' in y_train:
            logger.info("Training direction model V2...")
            metrics['direction'] = self.direction_model.train(
                X_train, y_train['direction'],
                X_val, y_val['direction'] if has_y_val and 'direction' in y_val else None
            )
            
        # Train strength model
        if 'strength' in y_train:
            logger.info("Training strength model...")
            metrics['strength'] = self.strength_model.train(
                X_train, y_train['strength'],
                X_val, y_val['strength'] if has_y_val and 'strength' in y_val else None
            )
            
        # Train volatility model
        if 'volatility' in y_train:
            logger.info("Training volatility model...")
            metrics['volatility'] = self.volatility_model.train(
                X_train, y_train['volatility'],
                X_val, y_val['volatility'] if has_y_val and 'volatility' in y_val else None
            )
            
        # Train timing model
        if 'timing' in y_train:
            logger.info("Training timing model V2...")
            metrics['timing'] = self.timing_model.train(
                X_train, y_train['timing'],
                X_val, y_val['timing'] if has_y_val and 'timing' in y_val else None
            )
            
        # Train meta-model if requested
        if self.use_meta_model:
            self._train_meta_model(X_train, y_train, X_val, y_val)
            
        self._is_trained = True
        logger.info("Ensemble V2 training complete")
        
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
        
        # Train meta-model
        self.meta_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=-1
        )
        
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
            predictions['timing'] = self.timing_model.predict(X)
            
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
            predictions['timing'] = self.timing_model.predict(X)
            
        return predictions
    
    def get_trading_signal(
        self,
        X: pd.DataFrame,
        min_direction_prob: float = None,
        min_strength: float = None,
        min_timing: float = None,
        pair: Optional[str] = None,
        use_dynamic_thresholds: bool = True
    ) -> List[TradingSignal]:
        """
        Generate trading signals from ensemble predictions.
        
        Args:
            X: Feature DataFrame.
            min_direction_prob: Minimum direction probability threshold.
            min_strength: Minimum expected movement threshold.
            min_timing: Minimum timing score threshold.
            pair: Trading pair for logging.
            use_dynamic_thresholds: Adjust thresholds based on volatility.
            
        Returns:
            List of TradingSignal objects.
        """
        # Use defaults if not specified
        min_direction_prob = min_direction_prob or self.DEFAULT_MIN_DIRECTION_PROB
        min_strength = min_strength or self.DEFAULT_MIN_STRENGTH
        min_timing = min_timing or self.DEFAULT_MIN_TIMING
        
        predictions = self.predict(X)
        
        # Get volatility for dynamic thresholds
        volatility = predictions.get('volatility', np.ones(len(X)) * 0.02)
        
        signals = []
        
        for i in range(len(X)):
            direction_proba = predictions.get('direction_proba', np.array([[0.33, 0.34, 0.33]]))[i]
            strength = predictions.get('strength', np.ones(len(X)))[i]
            timing = predictions.get('timing', np.ones(len(X)) * 0.5)[i]
            vol = volatility[i] if isinstance(volatility, np.ndarray) else volatility
            
            # Dynamic threshold adjustment based on volatility
            if use_dynamic_thresholds:
                # Higher volatility = require more confidence
                vol_adjustment = 1 + max(0, vol - 0.02) * 10  # Scale by excess volatility
                adj_min_direction = min(0.65, min_direction_prob * vol_adjustment)
                adj_min_timing = min(0.6, min_timing * vol_adjustment)
            else:
                adj_min_direction = min_direction_prob
                adj_min_timing = min_timing
            
            # Create base signal
            signal = TradingSignal(
                signal='hold',
                confidence=0.0,
                direction_proba=direction_proba.tolist(),
                strength=float(strength),
                timing=float(timing),
                volatility=float(vol),
                pair=pair,
                reason='no_signal'
            )
            
            # Determine direction
            p_down, p_sideways, p_up = direction_proba
            
            # Check for long signal
            if p_up > adj_min_direction and p_up > p_down:
                if strength >= min_strength and timing >= adj_min_timing:
                    signal.signal = 'buy'
                    signal.confidence = p_up
                    signal.reason = 'long_signal'
                    
                    # Calculate position sizing multiplier
                    signal.position_size_mult = self._calculate_position_mult(
                        confidence=p_up,
                        timing=timing,
                        volatility=vol
                    )
                    
                    # Set SL/TP levels
                    signal.suggested_sl_atr = 1.5
                    signal.suggested_tp_atr = 1.5 * self.rr_ratio  # RR 1:3 = 4.5 ATR
                    
                else:
                    signal.reason = f'filters_not_met: strength={strength:.2f}, timing={timing:.2f}'
            
            # Check for short signal
            elif p_down > adj_min_direction and p_down > p_up:
                if strength >= min_strength and timing >= adj_min_timing:
                    signal.signal = 'sell'
                    signal.confidence = p_down
                    signal.reason = 'short_signal'
                    
                    signal.position_size_mult = self._calculate_position_mult(
                        confidence=p_down,
                        timing=timing,
                        volatility=vol
                    )
                    
                    signal.suggested_sl_atr = 1.5
                    signal.suggested_tp_atr = 1.5 * self.rr_ratio
                    
                else:
                    signal.reason = f'filters_not_met: strength={strength:.2f}, timing={timing:.2f}'
            
            else:
                signal.reason = f'direction_unclear: up={p_up:.2f}, down={p_down:.2f}, thresh={adj_min_direction:.2f}'
            
            signals.append(signal)
        
        return signals
    
    def _calculate_position_mult(
        self,
        confidence: float,
        timing: float,
        volatility: float,
        base_mult: float = 1.0
    ) -> float:
        """
        Calculate position size multiplier based on signal quality.
        
        Higher confidence and timing = larger position
        Higher volatility = smaller position
        
        Returns multiplier in range [0.5, 1.5]
        """
        # Confidence factor: 0.8-1.2 based on confidence
        conf_factor = 0.8 + (confidence - 0.5) * 0.8
        
        # Timing factor: 0.8-1.2 based on timing score
        timing_factor = 0.8 + (timing - 0.3) * 0.6
        
        # Volatility factor: 0.7-1.0 (inverse relationship)
        vol_factor = max(0.7, 1.0 - (volatility - 0.02) * 5)
        
        multiplier = base_mult * conf_factor * timing_factor * vol_factor
        
        return max(0.5, min(1.5, multiplier))
    
    def get_signal_summary(
        self,
        X: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Get summary statistics of signals."""
        signals = self.get_trading_signal(X, **kwargs)
        
        buy_count = sum(1 for s in signals if s.signal == 'buy')
        sell_count = sum(1 for s in signals if s.signal == 'sell')
        hold_count = sum(1 for s in signals if s.signal == 'hold')
        
        avg_confidence = np.mean([s.confidence for s in signals if s.signal != 'hold']) if buy_count + sell_count > 0 else 0
        
        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'signal_rate': (buy_count + sell_count) / len(signals) if signals else 0,
            'avg_confidence': avg_confidence
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update model combination weights."""
        self.weights.update(new_weights)
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
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model predictions."""
        stats = {
            'is_trained': self._is_trained,
            'use_v2_models': self.use_v2_models,
            'rr_ratio': self.rr_ratio,
            'weights': self.weights
        }
        
        if self.use_v2_models and hasattr(self.direction_model, 'get_class_distribution_stats'):
            stats['direction_stats'] = self.direction_model.get_class_distribution_stats()
        
        return stats
    
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
            'use_v2_models': self.use_v2_models,
            'rr_ratio': self.rr_ratio,
            'is_trained': self._is_trained
        }, f"{directory}/ensemble_meta.joblib")
        
        if self.meta_model is not None:
            joblib.dump(self.meta_model, f"{directory}/meta_model.joblib")
            
        logger.info(f"Ensemble V2 saved to {directory}")
    
    def load(self, directory: str) -> None:
        """Load all models from directory."""
        # Check if V2 models exist
        v2_direction_path = f"{directory}/direction_model.joblib"
        
        # Try to detect if V2 models
        try:
            meta = joblib.load(f"{directory}/ensemble_meta.joblib")
            self.use_v2_models = meta.get('use_v2_models', False)
        except:
            self.use_v2_models = False
        
        # Load appropriate direction model
        if self.use_v2_models:
            self.direction_model = DirectionModelV2()
            self.timing_model = TimingModelV2()
        else:
            from .direction import DirectionModel
            from .timing import TimingModel
            self.direction_model = DirectionModel()
            self.timing_model = TimingModel()
        
        self.direction_model.load(f"{directory}/direction_model.joblib")
        self.strength_model.load(f"{directory}/strength_model.joblib")
        self.volatility_model.load(f"{directory}/volatility_model.joblib")
        self.timing_model.load(f"{directory}/timing_model.joblib")
        
        # Load ensemble metadata
        meta = joblib.load(f"{directory}/ensemble_meta.joblib")
        self.weights = meta['weights']
        self.config = meta['config']
        self.use_meta_model = meta['use_meta_model']
        self.rr_ratio = meta.get('rr_ratio', 2.0)
        self._is_trained = meta['is_trained']
        
        import os
        if os.path.exists(f"{directory}/meta_model.joblib"):
            self.meta_model = joblib.load(f"{directory}/meta_model.joblib")
            
        logger.info(f"Ensemble {'V2' if self.use_v2_models else 'V1'} loaded from {directory}")


# Backward compatibility
EnsembleModel = EnsembleModelV2

"""
Multi-Period Ensemble Model

Combines predictions from models trained on different time periods:
- 30 days (fresh patterns, highest weight)
- 90 days (balanced)
- 365 days (conservative, regime awareness)

The ensemble provides:
1. Weighted average predictions
2. Consensus checking (protection against regime changes)
3. Position sizing based on agreement level
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from loguru import logger


@dataclass
class EnsembleSignal:
    """Signal from the ensemble."""
    signal: int  # 1 = long, -1 = short, 0 = hold
    confidence: float  # 0-1, weighted average
    direction_proba: List[float]  # [down, sideways, up] weighted avg
    timing_proba: float
    strength: float
    volatility: float
    
    # Consensus info
    agreement_level: int  # 3 = all agree, 2 = majority, 1 = split
    position_size_multiplier: float  # 1.0 for full, 0.7 for reduced
    individual_signals: Dict[str, int]  # {'30d': 1, '90d': 1, '365d': 0}
    individual_confidences: Dict[str, float]
    
    # Protection
    is_protected: bool  # True if models disagree (skip trade)
    protection_reason: Optional[str]


class SinglePeriodModel:
    """Wrapper for a single period model (e.g., 30d, 90d, 365d)."""
    
    def __init__(self, model_dir: str, period_name: str):
        self.model_dir = Path(model_dir)
        self.period_name = period_name
        self.direction_model = None
        self.timing_model = None
        self.strength_model = None
        self.volatility_model = None
        self._is_loaded = False
        
    def load(self) -> bool:
        """Load models from directory."""
        try:
            self.direction_model = joblib.load(self.model_dir / 'direction_model.joblib')
            self.timing_model = joblib.load(self.model_dir / 'timing_model.joblib')
            self.strength_model = joblib.load(self.model_dir / 'strength_model.joblib')
            self.volatility_model = joblib.load(self.model_dir / 'volatility_model.joblib')
            self._is_loaded = True
            logger.info(f"Loaded {self.period_name} model from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.period_name} model: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Get predictions from this model."""
        if not self._is_loaded:
            raise RuntimeError(f"{self.period_name} model not loaded")
        
        # Direction prediction
        direction_proba = self.direction_model.predict_proba(X)[0]
        direction_pred = np.argmax(direction_proba)
        
        # Timing prediction
        timing_proba = self.timing_model.predict_proba(X)[0]
        
        # Strength and volatility
        strength = self.strength_model.predict(X)[0]
        volatility = self.volatility_model.predict(X)[0]
        
        # Map direction: 0=down, 1=sideways, 2=up
        direction_map = {0: -1, 1: 0, 2: 1}
        direction = direction_map.get(direction_pred, 0)
        
        # Signal: only if good timing and confident direction
        signal = 0
        if timing_proba[1] > 0.3 and max(direction_proba) > 0.50:
            if direction == 1:
                signal = 1
            elif direction == -1:
                signal = -1
        
        confidence = max(direction_proba) * timing_proba[1]
        
        return {
            'signal': signal,
            'direction': direction,
            'direction_proba': direction_proba.tolist(),
            'timing_proba': timing_proba[1],
            'strength': strength,
            'volatility': volatility,
            'confidence': confidence
        }


class MultiPeriodEnsemble:
    """
    Ensemble of models trained on different time periods.
    
    Weights:
    - 30d: 0.50 (fresh patterns, most relevant)
    - 90d: 0.35 (balanced view)
    - 365d: 0.15 (regime awareness, conservative)
    
    Consensus rules:
    - 3/3 agree: Strong signal, full position
    - 2/3 agree: Normal signal, 70% position
    - No consensus: Skip trade (protection)
    """
    
    # Default weights
    DEFAULT_WEIGHTS = {
        '30d': 0.50,
        '90d': 0.35,
        '365d': 0.15
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        require_consensus: bool = True,
        min_agreement: int = 2  # Minimum models that must agree
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.require_consensus = require_consensus
        self.min_agreement = min_agreement
        
        self.models: Dict[str, SinglePeriodModel] = {}
        self._is_loaded = False
        
    def load(self, model_paths: Dict[str, str]) -> bool:
        """
        Load all models.
        
        Args:
            model_paths: {'30d': './models/v1_fresh', '90d': './models/v1_90d', ...}
        """
        logger.info("Loading Multi-Period Ensemble...")
        
        success_count = 0
        for period, path in model_paths.items():
            if period not in self.weights:
                logger.warning(f"Period {period} not in weights, skipping")
                continue
                
            model = SinglePeriodModel(path, period)
            if model.load():
                self.models[period] = model
                success_count += 1
        
        if success_count == 0:
            logger.error("No models loaded!")
            return False
        
        # Normalize weights for loaded models
        total_weight = sum(self.weights[p] for p in self.models.keys())
        self.weights = {p: w/total_weight for p, w in self.weights.items() if p in self.models}
        
        self._is_loaded = True
        logger.info(f"Ensemble loaded: {list(self.models.keys())} with weights {self.weights}")
        return True
    
    @property
    def feature_names(self) -> List[str]:
        """Get expected feature names from first loaded model."""
        if not self.models:
            return []
        first_model = list(self.models.values())[0]
        return first_model.direction_model.feature_name_
    
    @property
    def n_features(self) -> int:
        """Get expected number of features."""
        return len(self.feature_names)
    
    def get_trading_signal(
        self,
        X: pd.DataFrame,
        min_direction_prob: float = 0.50,
        min_strength: float = 0.30,
        min_timing: float = 0.01
    ) -> EnsembleSignal:
        """
        Get ensemble trading signal.
        
        Returns weighted average of all models with consensus check.
        """
        if not self._is_loaded:
            raise RuntimeError("Ensemble not loaded")
        
        # Get predictions from each model
        predictions = {}
        for period, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[period] = pred
            except Exception as e:
                logger.warning(f"Model {period} prediction failed: {e}")
        
        if not predictions:
            return self._create_hold_signal("All models failed")
        
        # Extract individual signals and confidences
        individual_signals = {p: pred['signal'] for p, pred in predictions.items()}
        individual_confidences = {p: pred['confidence'] for p, pred in predictions.items()}
        
        # Calculate weighted averages
        weighted_direction_proba = np.zeros(3)
        weighted_timing = 0.0
        weighted_strength = 0.0
        weighted_volatility = 0.0
        total_weight = 0.0
        
        for period, pred in predictions.items():
            w = self.weights.get(period, 0)
            weighted_direction_proba += np.array(pred['direction_proba']) * w
            weighted_timing += pred['timing_proba'] * w
            weighted_strength += pred['strength'] * w
            weighted_volatility += pred['volatility'] * w
            total_weight += w
        
        if total_weight > 0:
            weighted_direction_proba /= total_weight
            weighted_timing /= total_weight
            weighted_strength /= total_weight
            weighted_volatility /= total_weight
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            predictions[p]['confidence'] * self.weights.get(p, 0) 
            for p in predictions
        ) / total_weight if total_weight > 0 else 0
        
        # Check consensus
        signals = list(individual_signals.values())
        long_count = sum(1 for s in signals if s == 1)
        short_count = sum(1 for s in signals if s == -1)
        hold_count = sum(1 for s in signals if s == 0)
        
        # Determine agreement level and final signal
        agreement_level = 0
        final_signal = 0
        position_multiplier = 1.0
        is_protected = False
        protection_reason = None
        
        total_models = len(signals)
        
        if long_count == total_models:
            # All agree LONG
            agreement_level = 3
            final_signal = 1
            position_multiplier = 1.0
            
        elif short_count == total_models:
            # All agree SHORT
            agreement_level = 3
            final_signal = -1
            position_multiplier = 1.0
            
        elif long_count >= self.min_agreement:
            # Majority LONG
            agreement_level = 2
            final_signal = 1
            position_multiplier = 0.7
            
        elif short_count >= self.min_agreement:
            # Majority SHORT
            agreement_level = 2
            final_signal = -1
            position_multiplier = 0.7
            
        else:
            # No consensus - PROTECTION
            agreement_level = 1
            final_signal = 0
            position_multiplier = 0.0
            is_protected = True
            protection_reason = f"No consensus: LONG={long_count}, SHORT={short_count}, HOLD={hold_count}"
        
        # Additional protection: if 365d strongly disagrees
        if '365d' in predictions:
            long_term_signal = predictions['365d']['signal']
            long_term_conf = predictions['365d']['confidence']
            
            if final_signal != 0 and long_term_signal == -final_signal and long_term_conf > 0.6:
                # Long-term model strongly disagrees - reduce position
                position_multiplier *= 0.5
                protection_reason = f"365d strongly disagrees (conf={long_term_conf:.2f})"
        
        return EnsembleSignal(
            signal=final_signal,
            confidence=weighted_confidence,
            direction_proba=weighted_direction_proba.tolist(),
            timing_proba=weighted_timing,
            strength=weighted_strength,
            volatility=weighted_volatility,
            agreement_level=agreement_level,
            position_size_multiplier=position_multiplier,
            individual_signals=individual_signals,
            individual_confidences=individual_confidences,
            is_protected=is_protected,
            protection_reason=protection_reason
        )
    
    def _create_hold_signal(self, reason: str) -> EnsembleSignal:
        """Create a HOLD signal with protection."""
        return EnsembleSignal(
            signal=0,
            confidence=0.0,
            direction_proba=[0.33, 0.34, 0.33],
            timing_proba=0.0,
            strength=0.0,
            volatility=0.0,
            agreement_level=0,
            position_size_multiplier=0.0,
            individual_signals={},
            individual_confidences={},
            is_protected=True,
            protection_reason=reason
        )


def create_default_ensemble(models_dir: str = './models') -> MultiPeriodEnsemble:
    """
    Create ensemble with default model paths.
    
    Expected structure:
    - models/v1_fresh (30d)
    - models/v1_90d (90d)
    - models/v1_365d (365d)
    """
    ensemble = MultiPeriodEnsemble()
    
    model_paths = {
        '30d': f'{models_dir}/v1_fresh',
        '90d': f'{models_dir}/v1_90d',
        '365d': f'{models_dir}/v1_365d'
    }
    
    ensemble.load(model_paths)
    return ensemble

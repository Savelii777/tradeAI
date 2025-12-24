"""
AI Trading Bot - Direction Prediction Model V2
Predicts the direction of price movement.

FIXES:
- Class weights to handle sideways bias
- Probability calibration (Platt scaling)
- Lower threshold for direction classification
- Better target definition
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight


class DirectionModelV2:
    """
    Predicts price direction (up, down, sideways) with improved class balancing.
    
    Improvements over V1:
    - Automatic class weight computation
    - Probability calibration for better thresholds
    - Focal loss option for hard examples
    - Post-processing for sideways bias correction
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict] = None,
        use_class_weights: bool = True,
        use_calibration: bool = True,
        calibration_method: str = "sigmoid",  # 'sigmoid' (Platt) or 'isotonic'
        sideways_penalty: float = 1.0,  # No extra penalty - balanced weights are enough
    ):
        """
        Initialize the direction model.
        
        Args:
            model_type: Type of model ('lightgbm', 'catboost', 'xgboost').
            params: Model hyperparameters.
            use_class_weights: Whether to use class weights for imbalanced data.
            use_calibration: Whether to calibrate probabilities.
            calibration_method: 'sigmoid' (Platt scaling) or 'isotonic'.
            sideways_penalty: Extra penalty multiplier for sideways class weights.
        """
        self.model_type = model_type
        self.params = params or self._default_params()
        self.use_class_weights = use_class_weights
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.sideways_penalty = sideways_penalty
        
        self.model = None
        self.calibrated_model = None
        self.class_weights_dict: Dict[int, float] = {}
        self.feature_names: List[str] = []
        self.classes = [-1, 0, 1]  # Down, Sideways, Up
        self._is_trained = False
        self._is_calibrated = False
        
        # Statistics for monitoring
        self.train_class_distribution: Dict[int, float] = {}
        self.pred_class_distribution: Dict[int, float] = {}
        
    def _default_params(self) -> Dict:
        """Get default model parameters optimized for balanced predictions."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': 300,
                'max_depth': 4,              # Middle ground (was 3, too weak)
                'num_leaves': 12,            # Middle ground (was 8, too weak)
                'min_child_samples': 200,    # Middle ground (was 500, too strong)
                'learning_rate': 0.02,       # Middle ground
                'subsample': 0.6,            # Middle ground
                'subsample_freq': 1,
                'colsample_bytree': 0.4,     # Middle ground
                'reg_alpha': 0.5,            # Middle ground
                'reg_lambda': 0.5,           # Middle ground
                'random_state': 42,
                'verbosity': -1,
                'force_row_wise': True,
                'is_unbalance': False,
            }
        elif self.model_type == "catboost":
            return {
                'loss_function': 'MultiClass',
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            }
        else:
            return {}
    
    def _compute_class_weights(
        self, 
        y: pd.Series,
        sideways_penalty: float = 1.0
    ) -> Dict[int, float]:
        """
        Compute class weights to balance predictions toward natural distribution.
        
        Uses power=0.7 scaling (between sqrt and linear) to moderately boost minority classes.
        
        Args:
            y: Target labels (-1, 0, 1)
            sideways_penalty: Extra multiplier for sideways class weight (not used)
            
        Returns:
            Dictionary mapping class labels to weights
        """
        # Count class frequencies
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})
        counts = y_mapped.value_counts().sort_index()
        total = len(y_mapped)
        
        # Calculate frequencies
        freq_down = counts.get(0, 0) / total
        freq_side = counts.get(1, 0) / total  
        freq_up = counts.get(2, 0) / total
        
        # Power-based inverse frequency weights
        # power=0.85 gives best balance of sideways ~50% and accuracy
        # (1/0.2)^0.85 = 3.74, (1/0.6)^0.85 = 1.54 - ratio 2.4
        
        power = 0.85
        w_down = (1.0 / max(freq_down, 0.1)) ** power
        w_side = (1.0 / max(freq_side, 0.1)) ** power
        w_up = (1.0 / max(freq_up, 0.1)) ** power
        
        # Normalize so average weight = 1
        avg_w = (w_down + w_side + w_up) / 3
        w_down = w_down / avg_w
        w_side = w_side / avg_w
        w_up = w_up / avg_w
        
        # After normalization for freq=0.2/0.6/0.2:
        # w_down ~ 1.25, w_side ~ 0.59, w_up ~ 1.25
        # This should give ~40% sideways predictions
        
        weight_dict = {0: w_down, 1: w_side, 2: w_up}
        
        logger.info(f"Computed POWER class weights (p={power}): Down={w_down:.3f}, "
                   f"Sideways={w_side:.3f}, Up={w_up:.3f}")
        
        return weight_dict
    
    def _compute_sample_weights(
        self, 
        y: pd.Series, 
        class_weights: Dict[int, float]
    ) -> np.ndarray:
        """Convert class weights to sample weights."""
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})
        sample_weights = np.array([class_weights[c] for c in y_mapped])
        return sample_weights
            
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
        Train the direction model with class balancing.
        
        Args:
            X_train: Training features.
            y_train: Training labels (-1, 0, 1).
            X_val: Validation features.
            y_val: Validation labels.
            sample_weight: External sample weights (overrides class weights).
            
        Returns:
            Training metrics.
        """
        self.feature_names = X_train.columns.tolist()
        self.model = self._create_model()
        
        # Store class distribution for monitoring
        self.train_class_distribution = y_train.value_counts(normalize=True).to_dict()
        logger.info(f"Training class distribution: {self.train_class_distribution}")
        
        # Compute class weights if enabled
        if self.use_class_weights and sample_weight is None:
            self.class_weights_dict = self._compute_class_weights(
                y_train, 
                self.sideways_penalty
            )
            sample_weight = self._compute_sample_weights(y_train, self.class_weights_dict)
        
        # Map labels to 0, 1, 2 for model training
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        
        if X_val is not None and y_val is not None:
            y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})
            
            if self.model_type == "lightgbm":
                import lightgbm as lgb
                self.model.fit(
                    X_train, y_train_mapped,
                    eval_set=[(X_val, y_val_mapped)],
                    sample_weight=sample_weight,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=100)
                    ]
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
        
        # Calibrate if enabled
        if self.use_calibration and X_val is not None and y_val is not None:
            self._calibrate(X_val, y_val)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train_mapped).mean()
        
        # Store prediction distribution
        unique, counts = np.unique(train_pred, return_counts=True)
        self.pred_class_distribution = dict(zip(unique, counts / len(train_pred)))
        logger.info(f"Training prediction distribution: {self.pred_class_distribution}")
        
        metrics = {
            'train_accuracy': train_accuracy,
            'class_balance': self._compute_class_balance_score(train_pred, y_train_mapped)
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = (val_pred == y_val_mapped).mean()
            metrics['val_accuracy'] = val_accuracy
            
            # Val prediction distribution
            unique, counts = np.unique(val_pred, return_counts=True)
            val_dist = dict(zip(unique, counts / len(val_pred)))
            logger.info(f"Validation prediction distribution: {val_dist}")
            
        logger.info(f"Direction model V2 trained: {metrics}")
        return metrics
    
    def _calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Calibrate model probabilities using validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info(f"Calibrating probabilities with {self.calibration_method}...")
        
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})
        
        try:
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method=self.calibration_method,
                cv='prefit'
            )
            self.calibrated_model.fit(X_val, y_val_mapped)
            self._is_calibrated = True
            logger.info("Probability calibration complete")
        except Exception as e:
            logger.warning(f"Calibration failed: {e}. Using uncalibrated probabilities.")
            self._is_calibrated = False
    
    def _compute_class_balance_score(
        self, 
        predictions: np.ndarray, 
        true_labels: np.ndarray
    ) -> float:
        """
        Compute how balanced the predictions are.
        
        Score close to 1 = balanced, close to 0 = very imbalanced
        """
        pred_dist = np.bincount(predictions, minlength=3) / len(predictions)
        # Ideal distribution
        ideal = np.array([0.33, 0.34, 0.33])
        # KL-divergence-like score
        balance = 1 - np.sum(np.abs(pred_dist - ideal)) / 2
        return float(balance)
        
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
        Predict class probabilities (calibrated if available).
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of shape (n_samples, 3) with probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        
        if self._is_calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        
        return self.model.predict_proba(X)
    
    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        up_threshold: float = 0.40,
        down_threshold: float = 0.40,
        sideways_threshold: float = 0.50
    ) -> np.ndarray:
        """
        Predict with custom thresholds per class.
        
        This helps reduce sideways bias by requiring higher confidence for sideways.
        
        Args:
            X: Feature DataFrame
            up_threshold: Min probability to predict "up"
            down_threshold: Min probability to predict "down"
            sideways_threshold: Min probability to predict "sideways"
            
        Returns:
            Array of predictions (-1, 0, 1)
        """
        proba = self.predict_proba(X)
        
        predictions = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            p_down, p_sideways, p_up = proba[i]
            
            # Check if we have strong directional signal
            if p_up > up_threshold and p_up > p_down:
                predictions[i] = 1  # Up
            elif p_down > down_threshold and p_down > p_up:
                predictions[i] = -1  # Down
            elif p_sideways > sideways_threshold:
                predictions[i] = 0  # Sideways
            else:
                # Default to direction with higher probability
                if p_up > p_down:
                    predictions[i] = 1
                elif p_down > p_up:
                    predictions[i] = -1
                else:
                    predictions[i] = 0
                    
        return predictions
        
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
    
    def get_directional_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get directional confidence: positive = bullish, negative = bearish.
        
        Returns value in range [-1, 1] where:
        - 1 = 100% confident bullish
        - 0 = neutral/sideways
        - -1 = 100% confident bearish
        """
        proba = self.predict_proba(X)
        # P(up) - P(down), normalized by (1 - P(sideways))
        p_up = proba[:, 2]
        p_down = proba[:, 0]
        p_sideways = proba[:, 1]
        
        # Directional confidence
        direction_conf = (p_up - p_down) / (p_up + p_down + 1e-8)
        # Weight by non-sideways probability
        direction_conf *= (1 - p_sideways)
        
        return direction_conf
        
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
    
    def get_class_distribution_stats(self) -> Dict:
        """Get training and prediction class distribution statistics."""
        return {
            'train_distribution': self.train_class_distribution,
            'pred_distribution': self.pred_class_distribution,
            'class_weights': self.class_weights_dict,
            'is_calibrated': self._is_calibrated
        }
        
    def save(self, path: str) -> None:
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'class_weights_dict': self.class_weights_dict,
            'is_trained': self._is_trained,
            'is_calibrated': self._is_calibrated,
            'use_class_weights': self.use_class_weights,
            'use_calibration': self.use_calibration,
            'calibration_method': self.calibration_method,
            'sideways_penalty': self.sideways_penalty,
            'train_class_distribution': self.train_class_distribution,
            'pred_class_distribution': self.pred_class_distribution,
        }, path)
        logger.info(f"Direction model V2 saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.calibrated_model = data.get('calibrated_model')
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self.class_weights_dict = data.get('class_weights_dict', {})
        self._is_trained = data['is_trained']
        self._is_calibrated = data.get('is_calibrated', False)
        self.use_class_weights = data.get('use_class_weights', True)
        self.use_calibration = data.get('use_calibration', True)
        self.calibration_method = data.get('calibration_method', 'sigmoid')
        self.sideways_penalty = data.get('sideways_penalty', 1.5)
        self.train_class_distribution = data.get('train_class_distribution', {})
        self.pred_class_distribution = data.get('pred_class_distribution', {})
        logger.info(f"Direction model V2 loaded from {path}")


# =============================================================================
# Target Creation Functions
# =============================================================================

def create_direction_target_v2(
    df: pd.DataFrame,
    lookahead: int = 5,
    threshold_pct: float = 0.002,  # Fixed 0.2% threshold (lower than adaptive)
    use_adaptive_threshold: bool = False,
    adaptive_multiplier: float = 0.5  # Multiplier for rolling volatility
) -> pd.Series:
    """
    Create direction target with lower thresholds to reduce sideways bias.
    
    Args:
        df: DataFrame with OHLCV data
        lookahead: Number of bars to look ahead
        threshold_pct: Fixed threshold percentage
        use_adaptive_threshold: Whether to use volatility-based threshold
        adaptive_multiplier: Multiplier for adaptive threshold
        
    Returns:
        Series with direction labels (-1, 0, 1)
    """
    close = df['close']
    
    # Calculate future return
    future_return = close.pct_change(periods=lookahead).shift(-lookahead)
    
    if use_adaptive_threshold:
        # Adaptive threshold based on rolling volatility
        rolling_vol = close.pct_change().rolling(window=50).std()
        threshold = rolling_vol * adaptive_multiplier
        threshold = threshold.clip(lower=threshold_pct)  # Min threshold
    else:
        # Fixed threshold (reduces sideways predictions)
        threshold = threshold_pct
    
    # Create direction labels
    direction = pd.Series(0, index=df.index)  # Default sideways
    direction[future_return > threshold] = 1   # Up
    direction[future_return < -threshold] = -1  # Down
    
    # Log distribution
    dist = direction.value_counts(normalize=True)
    logger.info(f"Target distribution: Down={dist.get(-1, 0):.1%}, "
               f"Sideways={dist.get(0, 0):.1%}, Up={dist.get(1, 0):.1%}")
    
    return direction


def create_direction_target_rr3(
    df: pd.DataFrame,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 4.5,  # RR 1:3
    max_lookahead: int = 50,
    atr_period: int = 14
) -> pd.Series:
    """
    Create direction target based on whether TP or SL is hit first.
    
    This trains the model to predict moves that reach RR 1:3.
    
    Args:
        df: DataFrame with OHLCV data
        sl_atr_mult: Stop loss in ATR multiples
        tp_atr_mult: Take profit in ATR multiples
        max_lookahead: Maximum bars to look ahead
        atr_period: ATR calculation period
        
    Returns:
        Series with direction labels (-1, 0, 1)
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
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    
    direction = pd.Series(0, index=df.index)  # Default sideways
    
    for i in range(len(df) - max_lookahead):
        entry_price = close.iloc[i]
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # Long trade levels
        sl_long = entry_price - sl_atr_mult * current_atr
        tp_long = entry_price + tp_atr_mult * current_atr
        
        # Short trade levels
        sl_short = entry_price + sl_atr_mult * current_atr
        tp_short = entry_price - tp_atr_mult * current_atr
        
        # Look forward
        future_highs = high.iloc[i+1:i+1+max_lookahead]
        future_lows = low.iloc[i+1:i+1+max_lookahead]
        
        # Check long trade outcome
        long_tp_hit = (future_highs >= tp_long).any()
        long_sl_hit = (future_lows <= sl_long).any()
        
        if long_tp_hit and not long_sl_hit:
            direction.iloc[i] = 1  # Profitable long
        elif long_sl_hit and long_tp_hit:
            # Both hit - check which first
            tp_idx = future_highs[future_highs >= tp_long].index[0] if long_tp_hit else None
            sl_idx = future_lows[future_lows <= sl_long].index[0] if long_sl_hit else None
            if tp_idx is not None and (sl_idx is None or tp_idx < sl_idx):
                direction.iloc[i] = 1  # TP hit first
        
        # Check short trade outcome
        short_tp_hit = (future_lows <= tp_short).any()
        short_sl_hit = (future_highs >= sl_short).any()
        
        if direction.iloc[i] == 0:  # Only if not already long
            if short_tp_hit and not short_sl_hit:
                direction.iloc[i] = -1  # Profitable short
            elif short_sl_hit and short_tp_hit:
                tp_idx = future_lows[future_lows <= tp_short].index[0] if short_tp_hit else None
                sl_idx = future_highs[future_highs >= sl_short].index[0] if short_sl_hit else None
                if tp_idx is not None and (sl_idx is None or tp_idx < sl_idx):
                    direction.iloc[i] = -1  # TP hit first
    
    # Log distribution
    dist = direction.value_counts(normalize=True)
    logger.info(f"RR3 Target distribution: Down={dist.get(-1, 0):.1%}, "
               f"Sideways={dist.get(0, 0):.1%}, Up={dist.get(1, 0):.1%}")
    
    return direction


# Backward compatibility alias
DirectionModel = DirectionModelV2

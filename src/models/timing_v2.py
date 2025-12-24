"""
AI Trading Bot - Timing Model V2
Predicts optimal entry timing using regression approach.

FIXES:
- Regression instead of classification (better calibration)
- Ratio-based target (favorable / adverse)
- Longer lookahead period
- Better definition of "good entry"
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import joblib


class TimingModelV2:
    """
    Predicts optimal entry timing using regression.
    
    Instead of binary classification (good/bad), predicts a continuous
    "timing quality" score based on:
    - Favorable move / Adverse move ratio
    - How quickly favorable move occurs
    - Risk-adjusted entry quality
    
    Output: Score from 0 to 1+ where higher = better entry timing
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict] = None,
        output_type: str = "regression",  # 'regression' or 'classification'
        normalize_output: bool = True,
    ):
        """
        Initialize the timing model.
        
        Args:
            model_type: Type of model ('lightgbm', 'catboost').
            params: Model hyperparameters.
            output_type: 'regression' for continuous scores or 'classification' for binary.
            normalize_output: Whether to normalize regression output to [0, 1].
        """
        self.model_type = model_type
        self.output_type = output_type
        self.normalize_output = normalize_output
        self.params = params or self._default_params()
        self.model = None
        self.feature_names: List[str] = []
        self._is_trained = False
        
        # Statistics for normalization
        self.score_mean: float = 0.5
        self.score_std: float = 0.2
        self.score_min: float = 0.0
        self.score_max: float = 1.0
        
    def _default_params(self) -> Dict:
        """Get default model parameters with balanced regularization."""
        base_params = {
            'boosting_type': 'gbdt',
            'n_estimators': 200,
            'max_depth': 4,               # Middle ground
            'num_leaves': 12,             # Middle ground
            'min_child_samples': 200,     # Middle ground
            'learning_rate': 0.02,        # Middle ground
            'subsample': 0.6,             # Middle ground
            'subsample_freq': 1,
            'colsample_bytree': 0.4,      # Middle ground
            'reg_alpha': 0.5,             # Middle ground
            'reg_lambda': 0.5,            # Middle ground
            'random_state': 42,
            'verbosity': -1,
            'force_row_wise': True
        }
        
        if self.output_type == "regression":
            base_params['objective'] = 'regression'
            base_params['metric'] = 'mae'
        else:
            base_params['objective'] = 'binary'
            base_params['metric'] = 'auc'
            
        return base_params
            
    def _create_model(self) -> Any:
        """Create the underlying model."""
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            if self.output_type == "regression":
                return lgb.LGBMRegressor(**self.params)
            else:
                return lgb.LGBMClassifier(**self.params)
        elif self.model_type == "catboost":
            if self.output_type == "regression":
                from catboost import CatBoostRegressor
                return CatBoostRegressor(**self.params)
            else:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(**self.params)
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
        Train the timing model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (0-1 scores or binary).
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            Training metrics.
        """
        self.feature_names = X_train.columns.tolist()
        self.model = self._create_model()
        
        # Store normalization stats
        if self.output_type == "regression":
            self.score_mean = y_train.mean()
            self.score_std = y_train.std()
            self.score_min = y_train.min()
            self.score_max = y_train.max()
            logger.info(f"Timing target stats: mean={self.score_mean:.3f}, "
                       f"std={self.score_std:.3f}, range=[{self.score_min:.3f}, {self.score_max:.3f}]")
        
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
        
        if self.output_type == "regression":
            train_mae = np.mean(np.abs(train_pred - y_train))
            train_corr = np.corrcoef(train_pred, y_train)[0, 1]
            metrics = {
                'train_mae': train_mae,
                'train_correlation': train_corr,
                'train_mean_pred': np.mean(train_pred),
                'train_std_pred': np.std(train_pred)
            }
        else:
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
            metrics = {'train_auc': train_auc}
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            if self.output_type == "regression":
                metrics['val_mae'] = np.mean(np.abs(val_pred - y_val))
                metrics['val_correlation'] = np.corrcoef(val_pred, y_val)[0, 1]
                metrics['val_mean_pred'] = np.mean(val_pred)
            else:
                from sklearn.metrics import roc_auc_score
                metrics['val_auc'] = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
            
        logger.info(f"Timing model V2 trained: {metrics}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict timing quality scores.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of timing scores (0-1 range if normalized).
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        
        if self.output_type == "regression":
            scores = self.model.predict(X)
            
            if self.normalize_output:
                # Normalize to [0, 1] using min-max scaling
                scores = (scores - self.score_min) / (self.score_max - self.score_min + 1e-8)
                scores = np.clip(scores, 0, 1)
            
            return scores
        else:
            # Classification: return probability of good timing
            return self.model.predict_proba(X)[:, 1]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict timing probabilities (for backward compatibility).
        
        For regression: normalizes output to [0, 1]
        For classification: returns actual probabilities
        """
        return self.predict(X)
    
    def get_timing_signal(
        self,
        X: pd.DataFrame,
        good_threshold: float = 0.6,
        bad_threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get timing signals based on thresholds.
        
        Args:
            X: Feature DataFrame
            good_threshold: Threshold for "good" timing
            bad_threshold: Threshold below which timing is "bad"
            
        Returns:
            Tuple of (signals, scores) where signals are 1 (good), 0 (neutral), -1 (bad)
        """
        scores = self.predict(X)
        
        signals = np.zeros(len(X), dtype=int)
        signals[scores >= good_threshold] = 1
        signals[scores < bad_threshold] = -1
        
        return signals, scores
    
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
            'is_trained': self._is_trained,
            'output_type': self.output_type,
            'normalize_output': self.normalize_output,
            'score_mean': self.score_mean,
            'score_std': self.score_std,
            'score_min': self.score_min,
            'score_max': self.score_max,
        }, path)
        logger.info(f"Timing model V2 saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self._is_trained = data['is_trained']
        self.output_type = data.get('output_type', 'classification')
        self.normalize_output = data.get('normalize_output', True)
        self.score_mean = data.get('score_mean', 0.5)
        self.score_std = data.get('score_std', 0.2)
        self.score_min = data.get('score_min', 0.0)
        self.score_max = data.get('score_max', 1.0)
        logger.info(f"Timing model V2 loaded from {path}")


# =============================================================================
# Target Creation Functions
# =============================================================================

def create_timing_target_v2(
    df: pd.DataFrame,
    lookahead: int = 15,  # Increased from 5 to 15 (75 min on M5)
    atr_period: int = 14,
    method: str = "ratio"  # 'ratio', 'percentile', 'relative'
) -> pd.Series:
    """
    Create timing target using improved methodology.
    
    Methods:
    - 'ratio': favorable_move / adverse_move (capped at 3)
    - 'percentile': 1 if favorable move > 75th percentile
    - 'relative': 1 if better than median entry
    
    Args:
        df: DataFrame with OHLCV data
        lookahead: Number of bars to look ahead
        atr_period: ATR calculation period
        method: Target calculation method
        
    Returns:
        Series with timing scores (0-1 for regression, binary for classification)
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
    
    # Calculate future highs and lows
    future_highs = high.rolling(window=lookahead).max().shift(-lookahead)
    future_lows = low.rolling(window=lookahead).min().shift(-lookahead)
    
    # For long entries:
    # Favorable = how much price goes up
    # Adverse = how much price goes down before going up
    favorable_long = (future_highs - close) / atr
    adverse_long = (close - future_lows) / atr
    
    # For short entries (symmetric)
    favorable_short = (close - future_lows) / atr
    adverse_short = (future_highs - close) / atr
    
    if method == "ratio":
        # Ratio method: favorable / adverse (higher = better timing)
        # Take max of long/short opportunity
        ratio_long = favorable_long / (adverse_long + 0.01)  # Avoid div by zero
        ratio_short = favorable_short / (adverse_short + 0.01)
        
        # Use best opportunity
        ratio = np.maximum(ratio_long, ratio_short)
        
        # Normalize to [0, 1] range
        # Ratio > 1 means favorable > adverse
        # Cap at 3 and normalize
        timing_score = np.clip(ratio / 3, 0, 1)
        
    elif method == "percentile":
        # Percentile method: 1 if favorable move > 75th percentile
        favorable = np.maximum(favorable_long, favorable_short)
        threshold = favorable.rolling(window=500, min_periods=100).quantile(0.75)
        timing_score = (favorable > threshold).astype(float)
        
    elif method == "relative":
        # Relative method: compare to median
        favorable = np.maximum(favorable_long, favorable_short)
        adverse = np.minimum(adverse_long, adverse_short)
        
        median_favorable = favorable.rolling(window=500, min_periods=100).median()
        median_adverse = adverse.rolling(window=500, min_periods=100).median()
        
        # Good timing if better than median on both dimensions
        good_timing = (favorable > median_favorable) & (adverse < median_adverse)
        timing_score = good_timing.astype(float)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    timing_score = pd.Series(timing_score, index=df.index)
    
    # Log statistics
    logger.info(f"Timing target stats ({method}): "
               f"mean={timing_score.mean():.3f}, "
               f"std={timing_score.std():.3f}, "
               f"min={timing_score.min():.3f}, "
               f"max={timing_score.max():.3f}")
    
    return timing_score.fillna(0.5)


def create_timing_target_rr3(
    df: pd.DataFrame,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 4.5,  # RR 1:3
    max_lookahead: int = 50,
    atr_period: int = 14
) -> pd.Series:
    """
    Create timing target for RR 1:3 trades.
    
    Score based on:
    - How quickly TP is reached (if reached)
    - Whether SL is hit first
    - Risk-adjusted quality
    
    Returns:
        Series with timing scores (0-1)
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
    
    timing_score = pd.Series(0.5, index=df.index)  # Default neutral
    
    for i in range(len(df) - max_lookahead):
        entry_price = close.iloc[i]
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # Long trade levels
        sl_long = entry_price - sl_atr_mult * current_atr
        tp_long = entry_price + tp_atr_mult * current_atr
        
        # Future prices
        future_highs = high.iloc[i+1:i+1+max_lookahead]
        future_lows = low.iloc[i+1:i+1+max_lookahead]
        
        # Check long outcome
        tp_hit_idx = None
        sl_hit_idx = None
        
        for j, (h, l) in enumerate(zip(future_highs, future_lows)):
            if h >= tp_long and tp_hit_idx is None:
                tp_hit_idx = j
            if l <= sl_long and sl_hit_idx is None:
                sl_hit_idx = j
            if tp_hit_idx is not None and sl_hit_idx is not None:
                break
        
        # Calculate timing score
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx):
            # TP hit first - good timing
            # Score higher if TP reached quickly
            speed_bonus = 1 - (tp_hit_idx / max_lookahead)
            timing_score.iloc[i] = 0.6 + 0.4 * speed_bonus
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx):
            # SL hit first - bad timing
            timing_score.iloc[i] = 0.2
        else:
            # Neither hit - neutral
            # Check how close it got to TP
            max_favorable = (future_highs.max() - entry_price) / current_atr
            tp_progress = min(max_favorable / tp_atr_mult, 1.0)
            timing_score.iloc[i] = 0.3 + 0.3 * tp_progress
    
    logger.info(f"RR3 Timing target stats: "
               f"mean={timing_score.mean():.3f}, "
               f"std={timing_score.std():.3f}")
    
    return timing_score


# Backward compatibility
TimingModel = TimingModelV2

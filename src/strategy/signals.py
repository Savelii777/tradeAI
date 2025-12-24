"""
AI Trading Bot - Signal Generation
Generates trading signals from model predictions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.constants import SignalType, OrderSide


@dataclass
class TradingSignal:
    """Trading signal with all relevant information."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    direction: int  # 1 for long, -1 for short
    confidence: float
    expected_move: float
    volatility: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    metadata: Dict[str, Any]


class SignalGenerator:
    """
    Generates trading signals from ML model predictions.
    
    Combines:
    - Direction probability
    - Expected movement strength
    - Entry timing quality
    - Volatility expectations
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize signal generator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Entry thresholds
        self.min_direction_prob = self.config.get('min_direction_probability', 0.60)
        self.min_expected_move = self.config.get('min_expected_move_atr', 1.5)
        self.min_timing_score = self.config.get('min_timing_score', 0.5)
        
        # Exit parameters
        self.stop_loss_atr = self.config.get('stop_loss_atr_multiplier', 1.5)
        self.take_profit_rr = self.config.get('take_profit_min_rr', 2.0)
        
        # Signal history
        self._signal_history: List[TradingSignal] = []
        
    def generate_signal(
        self,
        predictions: Dict[str, Any],
        current_price: float,
        atr: float,
        symbol: str = "BTCUSDT",
        timestamp: Optional[datetime] = None
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from predictions.
        
        Args:
            predictions: Dictionary with model predictions.
            current_price: Current market price.
            atr: Current ATR value.
            symbol: Trading symbol.
            timestamp: Signal timestamp.
            
        Returns:
            TradingSignal if conditions met, None otherwise.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        # Extract predictions
        direction_proba = predictions.get('direction_proba', [0.33, 0.34, 0.33])
        if isinstance(direction_proba, np.ndarray) and direction_proba.ndim == 2:
            direction_proba = direction_proba[0]
            
        strength = predictions.get('strength', 0)
        if isinstance(strength, np.ndarray):
            strength = strength[0] if len(strength) > 0 else 0
            
        timing = predictions.get('timing', 0)
        if isinstance(timing, np.ndarray):
            timing = timing[0] if len(timing) > 0 else 0
            
        volatility = predictions.get('volatility', atr)
        if isinstance(volatility, np.ndarray):
            volatility = volatility[0] if len(volatility) > 0 else atr
            
        # Determine direction
        max_prob_idx = np.argmax(direction_proba)
        max_prob = direction_proba[max_prob_idx]
        
        # Check entry conditions
        if max_prob < self.min_direction_prob:
            logger.debug(f"Signal rejected: direction prob {max_prob:.3f} < {self.min_direction_prob}")
            return None
            
        if strength < self.min_expected_move:
            logger.debug(f"Signal rejected: strength {strength:.3f} < {self.min_expected_move}")
            return None
            
        if timing < self.min_timing_score:
            logger.debug(f"Signal rejected: timing {timing:.3f} < {self.min_timing_score}")
            return None
            
        # Determine signal direction
        if max_prob_idx == 2:  # Up
            direction = 1
            signal_type = SignalType.BUY
        elif max_prob_idx == 0:  # Down
            direction = -1
            signal_type = SignalType.SELL
        else:  # Sideways
            logger.debug("Signal rejected: sideways prediction")
            return None
            
        # Calculate stop loss and take profit
        stop_distance = atr * self.stop_loss_atr
        take_profit_distance = stop_distance * self.take_profit_rr
        
        if direction == 1:  # Long
            stop_loss = current_price - stop_distance
            take_profit = current_price + take_profit_distance
        else:  # Short
            stop_loss = current_price + stop_distance
            take_profit = current_price - take_profit_distance
            
        # Create signal
        signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            confidence=max_prob,
            expected_move=strength * atr,
            volatility=volatility,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=0,  # To be calculated by position sizing
            metadata={
                'direction_proba': list(direction_proba),
                'timing_score': timing,
                'atr': atr
            }
        )
        
        self._signal_history.append(signal)
        logger.info(f"Signal generated: {signal_type.value} {symbol} @ {current_price:.2f}, "
                   f"confidence={max_prob:.3f}")
        
        return signal
        
    def filter_signal(
        self,
        signal: TradingSignal,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Apply additional filters to a signal.
        
        Args:
            signal: Trading signal to filter.
            filters: Dictionary of filter conditions.
            
        Returns:
            True if signal passes all filters.
        """
        # Volatility filter
        if 'max_volatility' in filters:
            if signal.volatility > filters['max_volatility']:
                logger.debug(f"Signal filtered: volatility {signal.volatility:.4f} > max")
                return False
                
        if 'min_volatility' in filters:
            if signal.volatility < filters['min_volatility']:
                logger.debug(f"Signal filtered: volatility {signal.volatility:.4f} < min")
                return False
                
        # Confidence filter
        if 'min_confidence' in filters:
            if signal.confidence < filters['min_confidence']:
                logger.debug(f"Signal filtered: confidence {signal.confidence:.3f} < min")
                return False
                
        # Time filter
        if 'allowed_hours' in filters:
            if signal.timestamp.hour not in filters['allowed_hours']:
                logger.debug(f"Signal filtered: hour {signal.timestamp.hour} not allowed")
                return False
                
        return True
        
    def get_recent_signals(
        self,
        n: int = 10,
        signal_type: Optional[SignalType] = None
    ) -> List[TradingSignal]:
        """
        Get recent signals.
        
        Args:
            n: Number of signals to return.
            signal_type: Filter by signal type.
            
        Returns:
            List of recent signals.
        """
        signals = self._signal_history[-n:]
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
            
        return signals
        
    def get_signal_stats(self) -> Dict[str, Any]:
        """Get statistics about generated signals."""
        if not self._signal_history:
            return {'total_signals': 0}
            
        buy_signals = sum(1 for s in self._signal_history if s.signal_type == SignalType.BUY)
        sell_signals = sum(1 for s in self._signal_history if s.signal_type == SignalType.SELL)
        avg_confidence = np.mean([s.confidence for s in self._signal_history])
        
        return {
            'total_signals': len(self._signal_history),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence
        }
        
    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history = []

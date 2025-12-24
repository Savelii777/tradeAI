"""
AI Trading Bot - Position Sizing
Calculates optimal position sizes based on risk parameters.
"""

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger


class PositionSizer:
    """
    Calculates optimal position sizes.
    
    Methods:
    - Fixed fractional
    - Kelly criterion
    - Volatility-adjusted
    - Dynamic based on performance
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize position sizer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Risk parameters
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)
        self.max_position_size = self.config.get('max_position_size', 0.20)
        
        # Sizing method
        self.method = self.config.get('method', 'volatility_adjusted')
        self.base_size = self.config.get('base_size', 0.01)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)
        
        # Dynamic adjustment
        self.reduce_after_losses = self.config.get('reduce_after_losses', True)
        self.reduction_factor = self.config.get('reduction_factor', 0.5)
        self.recovery_trades = self.config.get('recovery_trades', 3)
        
        # State
        self._consecutive_losses = 0
        self._size_reduction = 1.0
        
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float = 1.0,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size.
        
        Args:
            account_balance: Current account balance.
            entry_price: Planned entry price.
            stop_loss: Stop loss price.
            signal_confidence: Model confidence (0-1).
            volatility: Current volatility (ATR).
            win_rate: Historical win rate.
            avg_win_loss_ratio: Average win/loss ratio.
            
        Returns:
            Dictionary with position sizing details.
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            logger.warning("Zero risk per unit - cannot calculate position size")
            return {
                'position_size': 0,
                'position_value': 0,
                'risk_amount': 0,
                'reason': 'Zero risk per unit'
            }
            
        # Calculate base position size using selected method
        if self.method == 'fixed':
            base_risk = account_balance * self.base_size
            
        elif self.method == 'kelly':
            kelly_size = self._kelly_criterion(
                win_rate or 0.5,
                avg_win_loss_ratio or 1.0
            )
            base_risk = account_balance * kelly_size
            
        elif self.method == 'volatility_adjusted':
            # Adjust based on volatility
            if volatility and volatility > 0:
                vol_factor = 1 / (1 + volatility * 10)  # Reduce size with higher vol
            else:
                vol_factor = 1.0
                
            base_risk = account_balance * self.max_risk_per_trade * vol_factor
            
        else:
            base_risk = account_balance * self.max_risk_per_trade
            
        # Adjust for confidence
        adjusted_risk = base_risk * signal_confidence
        
        # Apply loss reduction
        adjusted_risk *= self._size_reduction
        
        # Calculate position size in units
        position_size = adjusted_risk / risk_per_unit
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Apply maximum position size limit
        max_value = account_balance * self.max_position_size
        if position_value > max_value:
            position_value = max_value
            position_size = position_value / entry_price
            adjusted_risk = position_size * risk_per_unit
            
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': adjusted_risk,
            'risk_percent': adjusted_risk / account_balance * 100,
            'method': self.method,
            'confidence_factor': signal_confidence,
            'reduction_factor': self._size_reduction
        }
        
    def _kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly criterion fraction.
        
        Kelly% = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        
        Args:
            win_rate: Historical win rate.
            win_loss_ratio: Average win / average loss.
            
        Returns:
            Kelly fraction (capped and reduced).
        """
        if win_loss_ratio == 0:
            return 0
            
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Cap at max risk
        kelly = min(kelly, self.max_risk_per_trade)
        
        # Apply fractional Kelly (half-Kelly is safer)
        kelly *= self.kelly_fraction
        
        # Ensure non-negative
        return max(kelly, 0)
        
    def update_after_trade(
        self,
        is_win: bool
    ) -> None:
        """
        Update state after a trade.
        
        Args:
            is_win: Whether the trade was profitable.
        """
        if is_win:
            self._consecutive_losses = 0
            # Gradually restore size
            if self._size_reduction < 1.0:
                self._size_reduction = min(1.0, self._size_reduction * 1.2)
        else:
            self._consecutive_losses += 1
            
            if self.reduce_after_losses and self._consecutive_losses >= 2:
                self._size_reduction = max(
                    self.reduction_factor,
                    self._size_reduction * 0.7
                )
                logger.warning(f"Position size reduced to {self._size_reduction:.1%} "
                              f"after {self._consecutive_losses} consecutive losses")
                
    def reset(self) -> None:
        """Reset state."""
        self._consecutive_losses = 0
        self._size_reduction = 1.0
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            'consecutive_losses': self._consecutive_losses,
            'size_reduction': self._size_reduction,
            'method': self.method
        }
        
    def calculate_lot_size(
        self,
        position_value: float,
        lot_size: float = 0.001,
        min_notional: float = 10.0
    ) -> float:
        """
        Round position to valid lot size.
        
        Args:
            position_value: Position value in quote currency.
            lot_size: Minimum lot size.
            min_notional: Minimum order value.
            
        Returns:
            Rounded lot size.
        """
        # Round down to lot size
        lots = int(position_value / lot_size) * lot_size
        
        # Ensure minimum notional
        if lots < min_notional:
            logger.warning(f"Position {lots} below minimum notional {min_notional}")
            return 0
            
        return lots

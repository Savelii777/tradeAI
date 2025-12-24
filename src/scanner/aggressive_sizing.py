"""
AI Trading Bot - Aggressive Position Sizing
100% deposit with leverage for maximum efficiency.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class LeverageCalculation:
    """Result of leverage and position size calculation."""
    deposit_used: float  # Always 100%
    position_value: float  # Total position value with leverage
    position_size: float  # Position size in base currency
    leverage: int  # Calculated leverage
    stop_loss: float
    stop_distance_pct: float
    risk_amount: float  # Amount at risk (should equal fixed_risk * deposit)
    risk_percent: float  # Risk as percentage
    take_profit: float
    take_profit_pct: float
    risk_reward_ratio: float
    margin_required: float
    liquidation_price: float
    entry_price: float


class AggressiveSizer:
    """
    Aggressive position sizer for 100% deposit usage.
    
    Key principles:
    - Always use 100% of available deposit
    - Calculate leverage based on stop distance and desired risk
    - Fixed 5% risk per trade
    - Single position at a time
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize aggressive sizer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Risk parameters
        self.fixed_risk_pct = self.config.get('fixed_risk_pct', 0.05)  # 5%
        
        # Leverage limits
        self.min_leverage = self.config.get('min_leverage', 5)
        self.max_leverage = self.config.get('max_leverage', 20)
        
        # Take profit parameters
        self.take_profit_rr = self.config.get('take_profit_rr', 3.0)  # Risk:Reward
        
        # Safety parameters
        self.liquidation_buffer = self.config.get('liquidation_buffer', 0.02)  # 2% buffer
        self.maintenance_margin = self.config.get('maintenance_margin', 0.005)  # 0.5%
        
    def calculate(
        self,
        deposit: float,
        entry_price: float,
        stop_loss: float,
        direction: int  # 1 for long, -1 for short
    ) -> LeverageCalculation:
        """
        Calculate position size and leverage for aggressive trading.
        
        Formula:
        leverage = fixed_risk_pct / stop_distance_pct
        position_value = deposit * leverage
        position_size = position_value / entry_price
        
        Args:
            deposit: Current account deposit.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            direction: 1 for long, -1 for short.
            
        Returns:
            LeverageCalculation with all position parameters.
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = stop_distance / entry_price
        
        if stop_distance_pct == 0:
            logger.error("Zero stop distance - cannot calculate position")
            return self._create_zero_calculation(deposit, entry_price)
            
        # Calculate required leverage for fixed risk
        required_leverage = self.fixed_risk_pct / stop_distance_pct
        
        # Apply leverage limits
        leverage = int(min(max(required_leverage, self.min_leverage), self.max_leverage))
        
        # Recalculate actual risk with capped leverage
        actual_risk_pct = stop_distance_pct * leverage
        
        # Calculate position
        position_value = deposit * leverage
        position_size = position_value / entry_price
        
        # Calculate risk amount
        risk_amount = deposit * actual_risk_pct
        
        # Calculate take profit
        tp_distance = stop_distance * self.take_profit_rr
        if direction == 1:  # Long
            take_profit = entry_price + tp_distance
        else:  # Short
            take_profit = entry_price - tp_distance
            
        take_profit_pct = tp_distance / entry_price
        
        # Calculate liquidation price
        # For long: liquidation = entry - (deposit / position_size) * (1 - maintenance_margin)
        # For short: liquidation = entry + (deposit / position_size) * (1 - maintenance_margin)
        margin_per_unit = deposit / position_size
        liq_distance = margin_per_unit * (1 - self.maintenance_margin)
        
        if direction == 1:
            liquidation_price = entry_price - liq_distance
        else:
            liquidation_price = entry_price + liq_distance
            
        # Validate stop is not past liquidation
        if direction == 1 and stop_loss <= liquidation_price:
            logger.warning(f"Stop loss {stop_loss:.4f} is below liquidation {liquidation_price:.4f}")
        elif direction == -1 and stop_loss >= liquidation_price:
            logger.warning(f"Stop loss {stop_loss:.4f} is above liquidation {liquidation_price:.4f}")
            
        return LeverageCalculation(
            deposit_used=1.0,  # 100%
            position_value=position_value,
            position_size=position_size,
            leverage=leverage,
            stop_loss=stop_loss,
            stop_distance_pct=stop_distance_pct,
            risk_amount=risk_amount,
            risk_percent=actual_risk_pct,
            take_profit=take_profit,
            take_profit_pct=take_profit_pct,
            risk_reward_ratio=self.take_profit_rr,
            margin_required=deposit,
            liquidation_price=liquidation_price,
            entry_price=entry_price
        )
        
    def calculate_from_stop_pct(
        self,
        deposit: float,
        entry_price: float,
        stop_distance_pct: float,
        direction: int
    ) -> LeverageCalculation:
        """
        Calculate position from stop distance percentage.
        
        Args:
            deposit: Current account deposit.
            entry_price: Entry price.
            stop_distance_pct: Stop distance as percentage (e.g., 0.003 for 0.3%).
            direction: 1 for long, -1 for short.
            
        Returns:
            LeverageCalculation with all position parameters.
        """
        if direction == 1:
            stop_loss = entry_price * (1 - stop_distance_pct)
        else:
            stop_loss = entry_price * (1 + stop_distance_pct)
            
        return self.calculate(deposit, entry_price, stop_loss, direction)
        
    def calculate_optimal_stop(
        self,
        deposit: float,
        entry_price: float,
        direction: int,
        target_leverage: int = 10
    ) -> float:
        """
        Calculate optimal stop distance for target leverage.
        
        Args:
            deposit: Current account deposit.
            entry_price: Entry price.
            direction: 1 for long, -1 for short.
            target_leverage: Desired leverage.
            
        Returns:
            Stop loss price.
        """
        # stop_distance_pct = fixed_risk_pct / leverage
        stop_distance_pct = self.fixed_risk_pct / target_leverage
        
        if direction == 1:
            stop_loss = entry_price * (1 - stop_distance_pct)
        else:
            stop_loss = entry_price * (1 + stop_distance_pct)
            
        return stop_loss
        
    def _create_zero_calculation(
        self,
        deposit: float,
        entry_price: float
    ) -> LeverageCalculation:
        """Create zero calculation for error cases."""
        return LeverageCalculation(
            deposit_used=0,
            position_value=0,
            position_size=0,
            leverage=0,
            stop_loss=0,
            stop_distance_pct=0,
            risk_amount=0,
            risk_percent=0,
            take_profit=0,
            take_profit_pct=0,
            risk_reward_ratio=0,
            margin_required=0,
            liquidation_price=0,
            entry_price=entry_price
        )
        
    def validate_position(
        self,
        calc: LeverageCalculation,
        direction: int,
        min_rr: float = 2.0
    ) -> Dict[str, Any]:
        """
        Validate position parameters before execution.
        
        Args:
            calc: Leverage calculation to validate.
            direction: Position direction.
            min_rr: Minimum risk/reward ratio.
            
        Returns:
            Validation result with 'valid' boolean and 'issues' list.
        """
        issues = []
        
        # Check leverage limits
        if calc.leverage < self.min_leverage:
            issues.append(f"Leverage {calc.leverage}x below minimum {self.min_leverage}x")
        if calc.leverage > self.max_leverage:
            issues.append(f"Leverage {calc.leverage}x above maximum {self.max_leverage}x")
            
        # Check risk/reward
        if calc.risk_reward_ratio < min_rr:
            issues.append(f"Risk/reward {calc.risk_reward_ratio:.1f} below minimum {min_rr}")
            
        # Check stop vs liquidation
        if direction == 1:
            if calc.stop_loss <= calc.liquidation_price:
                issues.append("Stop loss would be hit after liquidation")
        else:
            if calc.stop_loss >= calc.liquidation_price:
                issues.append("Stop loss would be hit after liquidation")
                
        # Check position size
        if calc.position_size <= 0:
            issues.append("Invalid position size")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
        
    def get_leverage_table(
        self,
        stop_distances: list = None
    ) -> Dict[float, int]:
        """
        Get leverage for different stop distances.
        
        Args:
            stop_distances: List of stop distances to calculate.
            
        Returns:
            Dict mapping stop distance to leverage.
        """
        if stop_distances is None:
            stop_distances = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
            
        table = {}
        for stop_pct in stop_distances:
            leverage = self.fixed_risk_pct / stop_pct
            leverage = int(min(max(leverage, self.min_leverage), self.max_leverage))
            table[stop_pct] = leverage
            
        return table
        
    def log_position_summary(self, calc: LeverageCalculation, direction: int) -> None:
        """Log detailed position summary."""
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info("=== AGGRESSIVE POSITION ===")
        logger.info(f"Direction: {dir_str}")
        logger.info(f"Entry: {calc.entry_price:.4f}")
        logger.info(f"Stop Loss: {calc.stop_loss:.4f} ({calc.stop_distance_pct:.2%})")
        logger.info(f"Take Profit: {calc.take_profit:.4f} ({calc.take_profit_pct:.2%})")
        logger.info(f"Leverage: {calc.leverage}x")
        logger.info(f"Position Value: ${calc.position_value:.2f}")
        logger.info(f"Position Size: {calc.position_size:.6f}")
        logger.info(f"Risk Amount: ${calc.risk_amount:.2f} ({calc.risk_percent:.1%})")
        logger.info(f"Risk/Reward: 1:{calc.risk_reward_ratio:.1f}")
        logger.info(f"Liquidation: {calc.liquidation_price:.4f}")
        logger.info("===========================")

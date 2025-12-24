"""
AI Trading Bot - Risk Manager

Comprehensive risk management for V1 model.
Controls position sizing, daily limits, drawdown limits, and cooldowns.

This class ONLY manages risk - it does NOT change model logic!
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Tuple

from loguru import logger


class RiskManager:
    """
    Risk manager for V1 model.
    
    Does NOT change model logic, only limits risk exposure.
    
    Features:
    - Per-trade risk limits
    - Daily loss limits
    - Maximum drawdown protection
    - Consecutive loss cooldowns
    - Pair blacklisting
    - Dynamic position sizing based on drawdown
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration dictionary
        """
        config = config or {}
        
        # Per-trade limits
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_position_size_pct = config.get('max_position_size_pct', 0.10)  # 10%
        
        # Daily limits
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5%
        self.max_daily_trades = config.get('max_daily_trades', 20)
        
        # Drawdown limits
        self.max_drawdown = config.get('max_drawdown', 0.20)  # 20%
        self.drawdown_reduce_threshold = config.get('drawdown_reduce_size', 0.10)  # 10%
        
        # Consecutive losses
        self.max_consecutive_losses = config.get('max_consecutive_losses', 4)
        self.cooldown_minutes = config.get('cooldown_after_losses', 60)
        
        # Pair blacklist
        self.blacklist: Set[str] = set(config.get('blacklist', []))
        
        # State tracking
        self.daily_pnl = 0.0
        self.peak_capital = 0.0
        self.current_capital = 0.0
        self.consecutive_losses = 0
        self.last_loss_time: Optional[datetime] = None
        self.trades_today = 0
        self.is_trading_allowed = True
        self._last_reset_date: Optional[datetime] = None
        
        logger.info(f"RiskManager initialized: max_risk={self.max_risk_per_trade:.1%}, "
                   f"max_dd={self.max_drawdown:.1%}, blacklist={self.blacklist}")
    
    def can_trade(self, pair: str, capital: float) -> Tuple[bool, str]:
        """
        Check if a trade is allowed.
        
        Args:
            pair: Trading pair symbol
            capital: Current capital
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Update capital tracking
        self.current_capital = capital
        if self.peak_capital == 0:
            self.peak_capital = capital
        else:
            self.peak_capital = max(self.peak_capital, capital)
        
        # Check if new day - reset daily counters
        self._maybe_reset_daily()
        
        # Check blacklist
        if pair in self.blacklist:
            return False, f"Pair {pair} is blacklisted"
        
        # Normalize pair format for blacklist check
        pair_normalized = pair.replace('/', '').replace(':', '').replace('USDT', '/USDT:USDT')
        if pair_normalized in self.blacklist:
            return False, f"Pair {pair} is blacklisted"
        
        # Check if trading is paused
        if not self.is_trading_allowed:
            return False, "Trading paused due to risk limits"
        
        # Check daily loss limit
        if self.peak_capital > 0:
            daily_loss_pct = -self.daily_pnl / self.peak_capital
            if daily_loss_pct >= self.max_daily_loss:
                self.is_trading_allowed = False
                return False, f"Daily loss limit reached: {daily_loss_pct:.2%}"
        
        # Check max drawdown
        current_dd = self._calculate_drawdown()
        if current_dd >= self.max_drawdown:
            self.is_trading_allowed = False
            return False, f"Max drawdown reached: {current_dd:.2%}"
        
        # Check consecutive losses cooldown
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.last_loss_time:
                time_since_loss = (datetime.now() - self.last_loss_time).total_seconds() / 60
                if time_since_loss < self.cooldown_minutes:
                    remaining = self.cooldown_minutes - time_since_loss
                    return False, f"Cooldown after {self.consecutive_losses} losses: {remaining:.0f}min left"
                else:
                    # Cooldown finished, reset consecutive losses
                    self.consecutive_losses = 0
                    logger.info("Cooldown finished, resetting consecutive losses counter")
        
        # Check daily trade limit
        if self.trades_today >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.trades_today}/{self.max_daily_trades}"
        
        return True, "OK"
    
    def calculate_position_size(
        self, 
        capital: float, 
        stop_loss_pct: float,
        signal_confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on risk and current drawdown.
        
        Args:
            capital: Current capital
            stop_loss_pct: Stop loss as percentage of entry price
            signal_confidence: Optional confidence from model (0-1)
            
        Returns:
            Position size in currency units
        """
        # Base risk amount
        risk_amount = capital * self.max_risk_per_trade
        
        # Reduce size based on drawdown
        current_dd = self._calculate_drawdown()
        
        if current_dd >= 0.15:
            # In deep drawdown (>15%): reduce to 25% of normal
            risk_amount *= 0.25
            logger.warning(f"Deep drawdown {current_dd:.1%}: reducing position to 25%")
        elif current_dd >= self.drawdown_reduce_threshold:
            # In moderate drawdown (>10%): reduce to 50% of normal
            risk_amount *= 0.50
            logger.info(f"Moderate drawdown {current_dd:.1%}: reducing position to 50%")
        
        # Optionally scale by confidence
        if signal_confidence < 1.0:
            # Scale position by confidence (but keep at least 50%)
            confidence_scale = max(0.5, signal_confidence)
            risk_amount *= confidence_scale
        
        # Calculate position size from risk and stop loss
        if stop_loss_pct > 0:
            position_size = risk_amount / stop_loss_pct
        else:
            # Default to 1.5% stop loss
            position_size = risk_amount / 0.015
        
        # Cap at max position size
        max_position = capital * self.max_position_size_pct
        position_size = min(position_size, max_position)
        
        return position_size
    
    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """
        Record trade result for tracking.
        
        Args:
            pnl: Profit/loss amount
            is_win: Whether trade was profitable
        """
        self.daily_pnl += pnl
        self.trades_today += 1
        
        if is_win:
            self.consecutive_losses = 0
            logger.debug(f"Win recorded: PnL=${pnl:.2f}, consecutive losses reset")
        else:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
            logger.debug(f"Loss recorded: PnL=${pnl:.2f}, consecutive losses={self.consecutive_losses}")
        
        # Update capital
        self.current_capital += pnl
        
        # Log status after significant events
        if self.consecutive_losses >= 3:
            logger.warning(f"High consecutive losses: {self.consecutive_losses}")
        
        if self.daily_pnl < 0 and abs(self.daily_pnl) > self.peak_capital * 0.03:
            logger.warning(f"Daily loss >3%: ${self.daily_pnl:.2f}")
    
    def reset_daily(self) -> None:
        """
        Reset daily counters. Call at midnight UTC.
        """
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.is_trading_allowed = True
        self._last_reset_date = datetime.now().date()
        logger.info("Daily counters reset")
    
    def _maybe_reset_daily(self) -> None:
        """Check if we need to reset daily counters."""
        current_date = datetime.now().date()
        
        if self._last_reset_date is None:
            self._last_reset_date = current_date
        elif current_date > self._last_reset_date:
            self.reset_daily()
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current risk manager status.
        
        Returns:
            Dictionary with status information
        """
        current_dd = self._calculate_drawdown()
        
        return {
            'is_trading_allowed': self.is_trading_allowed,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.peak_capital if self.peak_capital > 0 else 0,
            'current_drawdown': current_dd,
            'current_drawdown_pct': f"{current_dd:.2%}",
            'consecutive_losses': self.consecutive_losses,
            'trades_today': self.trades_today,
            'peak_capital': self.peak_capital,
            'current_capital': self.current_capital,
            'position_size_multiplier': self._get_position_multiplier(current_dd)
        }
    
    def _get_position_multiplier(self, drawdown: float) -> float:
        """Get position size multiplier based on drawdown."""
        if drawdown >= 0.15:
            return 0.25
        elif drawdown >= self.drawdown_reduce_threshold:
            return 0.50
        return 1.0
    
    def set_initial_capital(self, capital: float) -> None:
        """
        Set initial capital and peak.
        
        Args:
            capital: Initial capital amount
        """
        self.current_capital = capital
        self.peak_capital = capital
        logger.info(f"Initial capital set: ${capital:.2f}")
    
    def add_to_blacklist(self, pair: str) -> None:
        """Add pair to blacklist."""
        self.blacklist.add(pair)
        logger.info(f"Added {pair} to blacklist")
    
    def remove_from_blacklist(self, pair: str) -> None:
        """Remove pair from blacklist."""
        self.blacklist.discard(pair)
        logger.info(f"Removed {pair} from blacklist")
    
    def resume_trading(self) -> None:
        """Manually resume trading after pause."""
        self.is_trading_allowed = True
        self.consecutive_losses = 0
        logger.info("Trading manually resumed")


def load_risk_config(config_path: str = "config/risk_management.yaml") -> Dict:
    """
    Load risk configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    from pathlib import Path
    
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Risk config not found: {config_path}, using defaults")
        return {}
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('risk', config)

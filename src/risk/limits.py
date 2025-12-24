"""
AI Trading Bot - Risk Limits
Enforces trading limits and restrictions.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


class RiskLimits:
    """
    Enforces risk limits at various levels.
    
    Limits:
    - Per-trade limits
    - Daily limits
    - Weekly limits
    - Maximum drawdown
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize risk limits.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Per-trade limits
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)
        self.max_position_size = self.config.get('max_position_size', 0.20)
        
        # Daily limits
        self.max_daily_loss = self.config.get('max_daily_loss', 0.03)
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 5)
        self.max_trades_per_day = self.config.get('max_trades_per_day', 20)
        self.cooldown_period = self.config.get('cooldown_after_losses', 7200)
        
        # Weekly limits
        self.max_weekly_loss = self.config.get('max_weekly_loss', 0.07)
        
        # Global limits
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        self.min_balance = self.config.get('min_balance', 100)
        
        # State tracking
        self._daily_pnl = 0
        self._weekly_pnl = 0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._peak_balance = 0
        self._cooldown_until: Optional[datetime] = None
        self._last_reset_date: Optional[datetime] = None
        self._trading_paused = False
        self._pause_reason = ""
        
    def check_trade_allowed(
        self,
        account_balance: float,
        trade_risk: float,
        position_value: float
    ) -> Dict[str, Any]:
        """
        Check if a trade is allowed.
        
        Args:
            account_balance: Current account balance.
            trade_risk: Risk amount for the trade.
            position_value: Position value.
            
        Returns:
            Dictionary with allowed status and reasons.
        """
        self._maybe_reset_daily()
        
        result = {
            'allowed': True,
            'reasons': [],
            'checks': {}
        }
        
        # Check if trading is paused
        if self._trading_paused:
            result['allowed'] = False
            result['reasons'].append(f"Trading paused: {self._pause_reason}")
            return result
            
        # Check cooldown
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            result['allowed'] = False
            remaining = (self._cooldown_until - datetime.utcnow()).seconds
            result['reasons'].append(f"In cooldown period ({remaining}s remaining)")
            return result
            
        # Check minimum balance
        if account_balance < self.min_balance:
            result['allowed'] = False
            result['reasons'].append(f"Balance {account_balance} below minimum {self.min_balance}")
            result['checks']['min_balance'] = False
        else:
            result['checks']['min_balance'] = True
            
        # Check per-trade risk limit
        risk_ratio = trade_risk / account_balance
        if risk_ratio > self.max_risk_per_trade:
            result['allowed'] = False
            result['reasons'].append(
                f"Trade risk {risk_ratio:.2%} exceeds max {self.max_risk_per_trade:.2%}"
            )
            result['checks']['trade_risk'] = False
        else:
            result['checks']['trade_risk'] = True
            
        # Check position size limit
        position_ratio = position_value / account_balance
        if position_ratio > self.max_position_size:
            result['allowed'] = False
            result['reasons'].append(
                f"Position size {position_ratio:.2%} exceeds max {self.max_position_size:.2%}"
            )
            result['checks']['position_size'] = False
        else:
            result['checks']['position_size'] = True
            
        # Check daily trade limit
        if self._daily_trades >= self.max_trades_per_day:
            result['allowed'] = False
            result['reasons'].append(f"Daily trade limit ({self.max_trades_per_day}) reached")
            result['checks']['daily_trades'] = False
        else:
            result['checks']['daily_trades'] = True
            
        # Check daily loss limit
        projected_loss = abs(self._daily_pnl) + trade_risk
        daily_loss_ratio = projected_loss / account_balance
        if self._daily_pnl < 0 and daily_loss_ratio > self.max_daily_loss:
            result['allowed'] = False
            result['reasons'].append(f"Daily loss limit ({self.max_daily_loss:.2%}) reached")
            result['checks']['daily_loss'] = False
        else:
            result['checks']['daily_loss'] = True
            
        # Check weekly loss limit
        weekly_loss_ratio = abs(self._weekly_pnl) / account_balance if self._weekly_pnl < 0 else 0
        if weekly_loss_ratio > self.max_weekly_loss:
            result['allowed'] = False
            result['reasons'].append(f"Weekly loss limit ({self.max_weekly_loss:.2%}) reached")
            result['checks']['weekly_loss'] = False
        else:
            result['checks']['weekly_loss'] = True
            
        # Check drawdown
        if self._peak_balance > 0:
            current_drawdown = (self._peak_balance - account_balance) / self._peak_balance
            if current_drawdown > self.max_drawdown:
                result['allowed'] = False
                result['reasons'].append(f"Max drawdown ({self.max_drawdown:.2%}) exceeded")
                result['checks']['drawdown'] = False
                self._pause_trading(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
            else:
                result['checks']['drawdown'] = True
                
        return result
        
    def record_trade_result(
        self,
        pnl: float,
        is_win: bool
    ) -> None:
        """
        Record a trade result.
        
        Args:
            pnl: Trade profit/loss.
            is_win: Whether trade was profitable.
        """
        self._maybe_reset_daily()
        
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._daily_trades += 1
        
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._start_cooldown()
                
        logger.debug(f"Trade recorded: PnL={pnl:.2f}, daily={self._daily_pnl:.2f}, "
                    f"consecutive_losses={self._consecutive_losses}")
                    
    def update_balance(self, balance: float) -> None:
        """Update peak balance for drawdown calculation."""
        if balance > self._peak_balance:
            self._peak_balance = balance
            
    def _start_cooldown(self) -> None:
        """Start cooldown period after consecutive losses."""
        self._cooldown_until = datetime.utcnow() + timedelta(seconds=self.cooldown_period)
        logger.warning(f"Cooldown started for {self.cooldown_period}s "
                      f"after {self._consecutive_losses} consecutive losses")
                      
    def _pause_trading(self, reason: str) -> None:
        """Pause trading."""
        self._trading_paused = True
        self._pause_reason = reason
        logger.error(f"Trading paused: {reason}")
        
    def resume_trading(self) -> None:
        """Resume trading."""
        self._trading_paused = False
        self._pause_reason = ""
        logger.info("Trading resumed")
        
    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if new day."""
        now = datetime.utcnow()
        
        if self._last_reset_date is None:
            self._last_reset_date = now
            return
            
        if now.date() > self._last_reset_date.date():
            self._daily_pnl = 0
            self._daily_trades = 0
            self._last_reset_date = now
            logger.info("Daily limits reset")
            
            # Reset weekly on Monday
            if now.weekday() == 0:
                self._weekly_pnl = 0
                logger.info("Weekly limits reset")
                
    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'trading_allowed': not self._trading_paused,
            'pause_reason': self._pause_reason,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'daily_trades': self._daily_trades,
            'consecutive_losses': self._consecutive_losses,
            'peak_balance': self._peak_balance,
            'cooldown_until': self._cooldown_until.isoformat() if self._cooldown_until else None,
            'in_cooldown': self._cooldown_until and datetime.utcnow() < self._cooldown_until
        }
        
    def reset(self) -> None:
        """Reset all limits."""
        self._daily_pnl = 0
        self._weekly_pnl = 0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._peak_balance = 0
        self._cooldown_until = None
        self._trading_paused = False
        self._pause_reason = ""
        logger.info("Risk limits reset")

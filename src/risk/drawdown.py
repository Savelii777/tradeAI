"""
AI Trading Bot - Drawdown Control
Monitors and controls account drawdown.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class DrawdownController:
    """
    Monitors and manages account drawdown.
    
    Features:
    - Real-time drawdown tracking
    - Drawdown-based position sizing
    - Recovery tracking
    - Emergency stop trigger
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize drawdown controller.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Thresholds
        self.warning_threshold = self.config.get('warning_threshold', 0.05)
        self.critical_threshold = self.config.get('critical_threshold', 0.10)
        self.max_threshold = self.config.get('max_threshold', 0.15)
        
        # Position reduction settings
        self.reduce_at_warning = self.config.get('reduce_at_warning', 0.75)
        self.reduce_at_critical = self.config.get('reduce_at_critical', 0.50)
        
        # State
        self._equity_history: List[Dict] = []
        self._peak_equity = 0
        self._current_drawdown = 0
        self._max_drawdown_reached = 0
        self._drawdown_start_time: Optional[datetime] = None
        self._emergency_stop_triggered = False
        
    def update(
        self,
        current_equity: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update with current equity.
        
        Args:
            current_equity: Current account equity.
            timestamp: Update timestamp.
            
        Returns:
            Drawdown status.
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._drawdown_start_time = None
            
        # Calculate current drawdown
        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        else:
            self._current_drawdown = 0
            
        # Track max drawdown
        if self._current_drawdown > self._max_drawdown_reached:
            self._max_drawdown_reached = self._current_drawdown
            
        # Record equity history
        self._equity_history.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'peak': self._peak_equity,
            'drawdown': self._current_drawdown
        })
        
        # Keep last 1000 records
        if len(self._equity_history) > 1000:
            self._equity_history = self._equity_history[-1000:]
            
        # Track drawdown start
        if self._current_drawdown > 0 and self._drawdown_start_time is None:
            self._drawdown_start_time = timestamp
            
        # Check for emergency stop (before getting status)
        if self._current_drawdown >= self.max_threshold:
            self._emergency_stop_triggered = True
            logger.error(f"EMERGENCY STOP: Drawdown {self._current_drawdown:.2%} "
                        f"exceeded max threshold {self.max_threshold:.2%}")
                        
        # Determine status level
        status = self._get_status_level()
                        
        return status
        
    def _get_status_level(self) -> Dict[str, Any]:
        """Get current drawdown status."""
        if self._current_drawdown >= self.max_threshold:
            level = "emergency"
            position_multiplier = 0
            message = "Maximum drawdown exceeded - trading stopped"
        elif self._current_drawdown >= self.critical_threshold:
            level = "critical"
            position_multiplier = self.reduce_at_critical
            message = f"Critical drawdown level: {self._current_drawdown:.2%}"
        elif self._current_drawdown >= self.warning_threshold:
            level = "warning"
            position_multiplier = self.reduce_at_warning
            message = f"Warning drawdown level: {self._current_drawdown:.2%}"
        else:
            level = "normal"
            position_multiplier = 1.0
            message = "Drawdown within normal limits"
            
        return {
            'level': level,
            'current_drawdown': self._current_drawdown,
            'max_drawdown': self._max_drawdown_reached,
            'peak_equity': self._peak_equity,
            'position_multiplier': position_multiplier,
            'message': message,
            'emergency_stop': self._emergency_stop_triggered,
            'drawdown_duration': self._get_drawdown_duration()
        }
        
    def _get_drawdown_duration(self) -> Optional[float]:
        """Get duration of current drawdown in hours."""
        if self._drawdown_start_time is None:
            return None
        duration = datetime.utcnow() - self._drawdown_start_time
        return duration.total_seconds() / 3600
        
    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on drawdown."""
        if self._current_drawdown >= self.max_threshold:
            return 0
        elif self._current_drawdown >= self.critical_threshold:
            return self.reduce_at_critical
        elif self._current_drawdown >= self.warning_threshold:
            return self.reduce_at_warning
        return 1.0
        
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self._equity_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(self._equity_history)
        df.set_index('timestamp', inplace=True)
        return df
        
    def get_drawdown_statistics(self) -> Dict[str, Any]:
        """Get drawdown statistics."""
        if not self._equity_history:
            return {'message': 'No history available'}
            
        df = self.get_equity_curve()
        
        # Calculate running drawdown series
        drawdown_series = df['drawdown']
        
        # Find drawdown periods
        is_in_drawdown = drawdown_series > 0
        drawdown_starts = is_in_drawdown & ~is_in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~is_in_drawdown & is_in_drawdown.shift(1, fill_value=False)
        
        return {
            'current_drawdown': self._current_drawdown,
            'max_drawdown': self._max_drawdown_reached,
            'avg_drawdown': drawdown_series.mean(),
            'drawdown_std': drawdown_series.std(),
            'time_in_drawdown': (drawdown_series > 0).mean(),
            'peak_equity': self._peak_equity,
            'drawdown_duration_hours': self._get_drawdown_duration()
        }
        
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (manual override)."""
        self._emergency_stop_triggered = False
        logger.warning("Emergency stop reset manually")
        
    def reset(self) -> None:
        """Reset all state."""
        self._equity_history = []
        self._peak_equity = 0
        self._current_drawdown = 0
        self._max_drawdown_reached = 0
        self._drawdown_start_time = None
        self._emergency_stop_triggered = False
        logger.info("Drawdown controller reset")

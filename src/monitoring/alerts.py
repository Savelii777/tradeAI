"""
AI Trading Bot - Alert System
Manages alerts and notifications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import asyncio

from loguru import logger

from src.utils.constants import AlertLevel


@dataclass
class Alert:
    """Alert representation."""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    sent_to: List[str] = field(default_factory=list)


class AlertManager:
    """
    Manages alert generation and delivery.
    
    Features:
    - Alert categorization
    - Multiple notification channels
    - Alert deduplication
    - Alert history
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Notification handlers
        self._handlers: Dict[str, Callable] = {}
        
        # Alert storage
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        
        # Deduplication
        self._recent_alerts: Dict[str, datetime] = {}
        self._dedup_window = self.config.get('dedup_window_seconds', 300)
        
    def register_handler(
        self,
        channel: str,
        handler: Callable
    ) -> None:
        """
        Register a notification handler.
        
        Args:
            channel: Channel name (e.g., 'telegram', 'email').
            handler: Handler function.
        """
        self._handlers[channel] = handler
        logger.info(f"Registered alert handler: {channel}")
        
    async def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        data: Optional[Dict] = None,
        channels: Optional[List[str]] = None
    ) -> Alert:
        """
        Create and send an alert.
        
        Args:
            level: Alert level.
            title: Alert title.
            message: Alert message.
            data: Additional data.
            channels: Channels to send to (default: all).
            
        Returns:
            Created alert.
        """
        # Check deduplication
        dedup_key = f"{level.value}:{title}"
        if dedup_key in self._recent_alerts:
            time_since = (datetime.utcnow() - self._recent_alerts[dedup_key]).seconds
            if time_since < self._dedup_window:
                logger.debug(f"Alert deduplicated: {title}")
                return None
                
        # Create alert
        self._alert_counter += 1
        alert = Alert(
            id=f"ALT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}",
            level=level,
            title=title,
            message=message,
            data=data or {}
        )
        
        self._alerts.append(alert)
        self._recent_alerts[dedup_key] = alert.timestamp
        
        # Log alert
        log_func = {
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.INFO: logger.info
        }.get(level, logger.info)
        
        log_func(f"[{level.value.upper()}] {title}: {message}")
        
        # Send to channels
        channels = channels or list(self._handlers.keys())
        for channel in channels:
            if channel in self._handlers:
                try:
                    await self._send_alert(alert, channel)
                    alert.sent_to.append(channel)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel}: {e}")
                    
        return alert
        
    async def _send_alert(self, alert: Alert, channel: str) -> None:
        """Send alert to a channel."""
        handler = self._handlers.get(channel)
        if not handler:
            return
            
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)
            
    # Convenience methods for different alert levels
    
    async def critical(
        self,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ) -> Alert:
        """Send critical alert."""
        return await self.create_alert(AlertLevel.CRITICAL, title, message, data)
        
    async def warning(
        self,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ) -> Alert:
        """Send warning alert."""
        return await self.create_alert(AlertLevel.WARNING, title, message, data)
        
    async def info(
        self,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ) -> Alert:
        """Send info alert."""
        return await self.create_alert(AlertLevel.INFO, title, message, data)
        
    # Pre-defined alerts
    
    async def alert_position_opened(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float
    ) -> Alert:
        """Alert for position opened."""
        return await self.info(
            "Position Opened",
            f"Opened {side.upper()} {size} {symbol} @ {entry_price}",
            {'symbol': symbol, 'side': side, 'size': size, 'price': entry_price}
        )
        
    async def alert_position_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_percent: float
    ) -> Alert:
        """Alert for position closed."""
        emoji = "✅" if pnl > 0 else "❌"
        return await self.info(
            "Position Closed",
            f"{emoji} {symbol} PnL: {pnl:.2f} ({pnl_percent:.2f}%)",
            {'symbol': symbol, 'pnl': pnl, 'pnl_percent': pnl_percent}
        )
        
    async def alert_daily_limit_reached(self, loss: float) -> Alert:
        """Alert for daily loss limit."""
        return await self.warning(
            "Daily Loss Limit",
            f"Daily loss limit reached: {loss:.2f}. Trading paused.",
            {'loss': loss}
        )
        
    async def alert_drawdown_warning(self, drawdown: float) -> Alert:
        """Alert for drawdown warning."""
        return await self.warning(
            "Drawdown Warning",
            f"Account drawdown: {drawdown:.2%}. Position sizes reduced.",
            {'drawdown': drawdown}
        )
        
    async def alert_drawdown_critical(self, drawdown: float) -> Alert:
        """Alert for critical drawdown."""
        return await self.critical(
            "Critical Drawdown",
            f"CRITICAL: Drawdown {drawdown:.2%}. Trading stopped.",
            {'drawdown': drawdown}
        )
        
    async def alert_connection_lost(self, exchange: str) -> Alert:
        """Alert for lost exchange connection."""
        return await self.critical(
            "Connection Lost",
            f"Lost connection to {exchange}. Attempting reconnect.",
            {'exchange': exchange}
        )
        
    async def alert_model_retrained(
        self,
        model_name: str,
        metrics: Dict
    ) -> Alert:
        """Alert for model retrain completion."""
        return await self.info(
            "Model Retrained",
            f"Model {model_name} retrained. Accuracy: {metrics.get('accuracy', 'N/A')}",
            metrics
        )
        
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get recent alerts."""
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
            
        return alerts[-limit:]
        
    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get unacknowledged alerts."""
        return [a for a in self._alerts if not a.acknowledged]
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

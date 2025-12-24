"""
AI Trading Bot - Logging Configuration
Enhanced logging with structured output.
"""

import sys
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger


class TradingLogger:
    """
    Enhanced logging for trading operations.
    
    Features:
    - Structured logging
    - Trade-specific logging
    - Performance logging
    - Error tracking
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "7 days"
    ):
        """
        Initialize the trading logger.
        
        Args:
            log_file: Path to log file.
            log_level: Logging level.
            rotation: Log rotation size.
            retention: Log retention period.
        """
        self.log_file = log_file
        self.log_level = log_level
        
        self._configure_logger(rotation, retention)
        
    def _configure_logger(
        self,
        rotation: str,
        retention: str
    ) -> None:
        """Configure loguru logger."""
        # Remove default handler
        logger.remove()
        
        # Console handler with color
        logger.add(
            sys.stderr,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # File handler
        if self.log_file:
            # Main log file
            logger.add(
                self.log_file,
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                       "{name}:{function}:{line} - {message}",
                rotation=rotation,
                retention=retention,
                compression="gz"
            )
            
            # Trade-specific log
            trade_log = self.log_file.replace('.log', '_trades.log')
            logger.add(
                trade_log,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
                filter=lambda record: record["extra"].get("trade_log", False),
                rotation=rotation,
                retention=retention
            )
            
            # Error log
            error_log = self.log_file.replace('.log', '_errors.log')
            logger.add(
                error_log,
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                       "{name}:{function}:{line} - {message}\n{exception}",
                rotation=rotation,
                retention=retention,
                backtrace=True,
                diagnose=True
            )
            
    def log_trade(
        self,
        action: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        **kwargs
    ) -> None:
        """
        Log a trade event.
        
        Args:
            action: Trade action (open, close, modify).
            symbol: Trading symbol.
            side: Trade side.
            price: Trade price.
            quantity: Trade quantity.
            **kwargs: Additional trade details.
        """
        trade_info = {
            'action': action,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            **kwargs
        }
        
        trade_str = " | ".join([f"{k}={v}" for k, v in trade_info.items()])
        
        with logger.contextualize(trade_log=True):
            logger.info(f"TRADE | {trade_str}")
            
    def log_signal(
        self,
        signal_type: str,
        symbol: str,
        confidence: float,
        **kwargs
    ) -> None:
        """Log a trading signal."""
        logger.info(
            f"SIGNAL | type={signal_type} | symbol={symbol} | "
            f"confidence={confidence:.3f} | {kwargs}"
        )
        
    def log_order(
        self,
        order_id: str,
        action: str,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        status: str = ""
    ) -> None:
        """Log an order event."""
        logger.info(
            f"ORDER | id={order_id} | action={action} | symbol={symbol} | "
            f"type={order_type} | side={side} | qty={quantity} | "
            f"price={price} | status={status}"
        )
        
    def log_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        current_price: float,
        pnl: float,
        pnl_percent: float
    ) -> None:
        """Log position status."""
        logger.info(
            f"POSITION | id={position_id} | symbol={symbol} | side={side} | "
            f"size={size} | entry={entry_price} | current={current_price} | "
            f"pnl={pnl:.2f} ({pnl_percent:.2f}%)"
        )
        
    def log_risk(
        self,
        event: str,
        **kwargs
    ) -> None:
        """Log risk-related event."""
        logger.warning(f"RISK | event={event} | {kwargs}")
        
    def log_performance(
        self,
        period: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Log performance metrics."""
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"PERFORMANCE | period={period} | {metrics_str}")
        
    def log_model(
        self,
        event: str,
        model_name: str,
        **kwargs
    ) -> None:
        """Log model-related event."""
        logger.info(f"MODEL | event={event} | model={model_name} | {kwargs}")
        
    def log_error(
        self,
        error: Exception,
        context: str = ""
    ) -> None:
        """Log an error with context."""
        logger.exception(f"ERROR | context={context} | error={str(error)}")
        
    def log_heartbeat(
        self,
        component: str,
        status: str = "ok",
        **kwargs
    ) -> None:
        """Log component heartbeat."""
        logger.debug(f"HEARTBEAT | component={component} | status={status} | {kwargs}")


# Global logger instance
trading_logger = TradingLogger()

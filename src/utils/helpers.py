"""
AI Trading Bot - Helper Functions
Utility functions used throughout the application.
"""

import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from cryptography.fernet import Fernet
from loguru import logger


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Dictionary containing configuration.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        file_path: Path to save the YAML file.
    """
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def encrypt_value(value: str, key: bytes) -> str:
    """
    Encrypt a string value using Fernet encryption.
    
    Args:
        value: Value to encrypt.
        key: Encryption key (32 bytes, base64-encoded).
        
    Returns:
        Encrypted value as string.
    """
    fernet = Fernet(key)
    return fernet.encrypt(value.encode()).decode()


def decrypt_value(encrypted_value: str, key: bytes) -> str:
    """
    Decrypt an encrypted string value.
    
    Args:
        encrypted_value: Encrypted value.
        key: Encryption key.
        
    Returns:
        Decrypted value.
    """
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_value.encode()).decode()


def generate_encryption_key() -> bytes:
    """
    Generate a new Fernet encryption key.
    
    Returns:
        New encryption key.
    """
    return Fernet.generate_key()


def get_current_timestamp() -> datetime:
    """
    Get current UTC timestamp.
    
    Returns:
        Current datetime in UTC.
    """
    return datetime.now(timezone.utc)


def timestamp_to_ms(dt: datetime) -> int:
    """
    Convert datetime to milliseconds timestamp.
    
    Args:
        dt: Datetime object.
        
    Returns:
        Timestamp in milliseconds.
    """
    return int(dt.timestamp() * 1000)


def ms_to_timestamp(ms: int) -> datetime:
    """
    Convert milliseconds timestamp to datetime.
    
    Args:
        ms: Timestamp in milliseconds.
        
    Returns:
        Datetime object in UTC.
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to the nearest tick size.
    
    Args:
        price: Price to round.
        tick_size: Minimum tick size.
        
    Returns:
        Rounded price.
    """
    return round(price / tick_size) * tick_size


def round_to_lot(quantity: float, lot_size: float) -> float:
    """
    Round quantity to the nearest lot size.
    
    Args:
        quantity: Quantity to round.
        lot_size: Minimum lot size.
        
    Returns:
        Rounded quantity.
    """
    return round(quantity / lot_size) * lot_size


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    fee_rate: float = 0.001
) -> Dict[str, float]:
    """
    Calculate profit/loss for a trade.
    
    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Position quantity.
        side: Position side ('long' or 'short').
        fee_rate: Trading fee rate.
        
    Returns:
        Dictionary with PnL details.
    """
    if side.lower() == 'long':
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity
    
    entry_fee = entry_price * quantity * fee_rate
    exit_fee = exit_price * quantity * fee_rate
    total_fees = entry_fee + exit_fee
    
    net_pnl = gross_pnl - total_fees
    pnl_percent = (net_pnl / (entry_price * quantity)) * 100
    
    return {
        'gross_pnl': gross_pnl,
        'fees': total_fees,
        'net_pnl': net_pnl,
        'pnl_percent': pnl_percent
    }


def normalize_series(
    series: pd.Series,
    window: int = 500,
    method: str = 'zscore'
) -> pd.Series:
    """
    Normalize a pandas Series using rolling statistics.
    
    Args:
        series: Series to normalize.
        window: Rolling window size.
        method: Normalization method ('zscore', 'minmax').
        
    Returns:
        Normalized series.
    """
    if method == 'zscore':
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
        return (series - rolling_mean) / rolling_std
    elif method == 'minmax':
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()
        range_val = rolling_max - rolling_min
        range_val = range_val.replace(0, 1)  # Avoid division by zero
        return (series - rolling_min) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def clip_outliers(
    series: pd.Series,
    std_threshold: float = 3.0
) -> pd.Series:
    """
    Clip outliers beyond N standard deviations.
    
    Args:
        series: Series to process.
        std_threshold: Number of standard deviations for clipping.
        
    Returns:
        Series with clipped outliers.
    """
    mean = series.mean()
    std = series.std()
    lower_bound = mean - std_threshold * std
    upper_bound = mean + std_threshold * std
    return series.clip(lower=lower_bound, upper=upper_bound)


def encode_cyclical(value: float, max_value: float) -> tuple:
    """
    Encode a cyclical feature using sin/cos transformation.
    
    Args:
        value: Value to encode (e.g., hour of day).
        max_value: Maximum value in the cycle (e.g., 24 for hours).
        
    Returns:
        Tuple of (sin_value, cos_value).
    """
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


def get_trading_session(hour: int) -> str:
    """
    Determine the trading session based on hour (UTC).
    
    Args:
        hour: Hour of day (0-23).
        
    Returns:
        Trading session name.
    """
    if 0 <= hour < 8:
        return "asian"
    elif 7 <= hour < 16:
        return "european"
    else:
        return "american"


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Series of returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of trading periods per year.
        
    Returns:
        Sharpe ratio.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor from a list of trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key.
        
    Returns:
        Profit factor.
    """
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Series of equity values.
        
    Returns:
        Dictionary with drawdown metrics.
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_idx = drawdown.idxmin()
    
    # Find peak before max drawdown
    peak_idx = equity_curve[:max_drawdown_idx].idxmax()
    
    return {
        'max_drawdown': abs(max_drawdown),
        'max_drawdown_date': max_drawdown_idx,
        'peak_date': peak_idx,
        'peak_value': equity_curve[peak_idx],
        'trough_value': equity_curve[max_drawdown_idx]
    }


def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB"
) -> None:
    """
    Configure the application logger.
    
    Args:
        log_file: Path to log file (optional).
        log_level: Logging level.
        rotation: Log rotation size.
    """
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File handler
    if log_file:
        logger.add(
            sink=log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            rotation=rotation,
            retention="7 days",
            compression="gz"
        )

"""
AI Trading Bot - Data Module
Provides data collection, storage, and preprocessing functionality.

This module handles all market data operations:
- Data collection from exchanges (REST API)
- Multi-pair parallel data collection
- Real-time data streaming (WebSocket)
- Data storage (SQLite/PostgreSQL with optional Redis caching)
- Data preprocessing for ML models
"""

from .models import (
    Candle,
    OrderBook,
    OrderBookLevel,
    Trade,
    FundingRate,
    OpenInterest,
    SymbolInfo,
    MarketData,
)
from .collector import DataCollector
from .multi_collector import MultiPairCollector
from .websocket_manager import WebSocketManager
from .storage import DataStorage
from .preprocessor import DataPreprocessor, Scaler
from .exceptions import (
    DataCollectionError,
    ExchangeConnectionError,
    RateLimitError,
    InvalidSymbolError,
    DataParsingError,
    StorageError,
    DatabaseConnectionError,
    WebSocketError,
    WebSocketConnectionError,
    WebSocketSubscriptionError,
)


__all__ = [
    # Models
    'Candle',
    'OrderBook',
    'OrderBookLevel',
    'Trade',
    'FundingRate',
    'OpenInterest',
    'SymbolInfo',
    'MarketData',
    # Collectors
    'DataCollector',
    'MultiPairCollector',
    # WebSocket
    'WebSocketManager',
    # Storage
    'DataStorage',
    # Preprocessor
    'DataPreprocessor',
    'Scaler',
    # Exceptions
    'DataCollectionError',
    'ExchangeConnectionError',
    'RateLimitError',
    'InvalidSymbolError',
    'DataParsingError',
    'StorageError',
    'DatabaseConnectionError',
    'WebSocketError',
    'WebSocketConnectionError',
    'WebSocketSubscriptionError',
]

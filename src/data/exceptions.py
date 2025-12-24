"""
AI Trading Bot - Data Module Exceptions
Custom exceptions for data collection, storage, and processing.
"""


class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    pass


class ExchangeConnectionError(DataCollectionError):
    """Exception raised when connection to exchange fails."""
    pass


class RateLimitError(DataCollectionError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float = 1.0):
        """
        Initialize RateLimitError.
        
        Args:
            message: Error message.
            retry_after: Suggested wait time in seconds before retry.
        """
        super().__init__(message)
        self.retry_after = retry_after


class InvalidSymbolError(DataCollectionError):
    """Exception raised when an invalid trading symbol is used."""
    pass


class DataParsingError(DataCollectionError):
    """Exception raised when data from exchange cannot be parsed."""
    pass


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


class DatabaseConnectionError(StorageError):
    """Exception raised when database connection fails."""
    pass


class WebSocketError(Exception):
    """Base exception for WebSocket errors."""
    pass


class WebSocketConnectionError(WebSocketError):
    """Exception raised when WebSocket connection fails."""
    pass


class WebSocketSubscriptionError(WebSocketError):
    """Exception raised when subscription to a channel fails."""
    pass

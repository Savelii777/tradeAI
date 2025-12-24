"""
AI Trading Bot - Data Models
Dataclass models for all market data types used in the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Candle:
    """
    Represents one OHLCV candle.
    
    Attributes:
        timestamp: Candle open time.
        open: Opening price.
        high: Highest price.
        low: Lowest price.
        close: Closing price.
        volume: Trading volume.
        symbol: Trading pair symbol.
        timeframe: Candle timeframe (e.g., "1m", "1h").
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


@dataclass
class OrderBookLevel:
    """
    Represents one level of the order book.
    
    Attributes:
        price: Price level.
        quantity: Total quantity at this price level.
    """
    price: float
    quantity: float


@dataclass
class OrderBook:
    """
    Represents a full order book snapshot.
    
    Attributes:
        symbol: Trading pair symbol.
        timestamp: Time of the snapshot.
        bids: List of bid levels (sorted by price descending).
        asks: List of ask levels (sorted by price ascending).
    """
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """
        Get the best (highest) bid level.
        
        Returns:
            Best bid level or None if no bids.
        """
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """
        Get the best (lowest) ask level.
        
        Returns:
            Best ask level or None if no asks.
        """
        return self.asks[0] if self.asks else None
    
    def get_spread_percent(self) -> Optional[float]:
        """
        Calculate the bid-ask spread as a percentage.
        
        Returns:
            Spread percentage or None if order book is empty.
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
            
        if best_bid.price == 0:
            return None
            
        return ((best_ask.price - best_bid.price) / best_bid.price) * 100
    
    def get_total_volume(self, levels: int = 10) -> Dict[str, float]:
        """
        Get the total volume on N levels from each side.
        
        Args:
            levels: Number of levels to sum from each side.
            
        Returns:
            Dictionary with 'bid_volume' and 'ask_volume'.
        """
        bid_volume = sum(level.quantity for level in self.bids[:levels])
        ask_volume = sum(level.quantity for level in self.asks[:levels])
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume
        }


@dataclass
class Trade:
    """
    Represents one trade from the trade stream.
    
    Attributes:
        timestamp: Time of the trade.
        symbol: Trading pair symbol.
        price: Trade price.
        quantity: Trade quantity.
        side: Trade side ("buy" or "sell").
    """
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: str  # "buy" or "sell"


@dataclass
class FundingRate:
    """
    Represents funding rate information for perpetual futures.
    
    Attributes:
        symbol: Trading pair symbol.
        rate: Current funding rate.
        next_funding_time: Time of next funding.
    """
    symbol: str
    rate: float
    next_funding_time: datetime


@dataclass
class OpenInterest:
    """
    Represents open interest data.
    
    Attributes:
        symbol: Trading pair symbol.
        value: Open interest value.
        timestamp: Time of the data.
    """
    symbol: str
    value: float
    timestamp: datetime


@dataclass
class SymbolInfo:
    """
    Represents trading pair information.
    
    Attributes:
        symbol: Trading pair symbol.
        base_asset: Base asset (e.g., "BTC").
        quote_asset: Quote asset (e.g., "USDT").
        price_precision: Number of decimal places for price.
        quantity_precision: Number of decimal places for quantity.
        min_quantity: Minimum order quantity.
        max_leverage: Maximum allowed leverage.
        tick_size: Minimum price increment.
        status: Trading status (e.g., "active").
    """
    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    min_quantity: float
    max_leverage: int
    tick_size: float
    status: str


@dataclass
class MarketData:
    """
    Aggregated market data for one trading pair.
    
    Attributes:
        symbol: Trading pair symbol.
        candles: Dictionary of candles by timeframe.
        orderbook: Current order book snapshot.
        recent_trades: List of recent trades.
        funding: Current funding rate info.
        open_interest: Current open interest.
        last_update: Time of last update.
    """
    symbol: str
    candles: Dict[str, List[Candle]] = field(default_factory=dict)
    orderbook: Optional[OrderBook] = None
    recent_trades: List[Trade] = field(default_factory=list)
    funding: Optional[FundingRate] = None
    open_interest: Optional[OpenInterest] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

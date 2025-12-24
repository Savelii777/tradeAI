"""
AI Trading Bot - Data Collector
Collects market data from exchange for a single trading pair via REST API.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional, Any
import logging

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
from .exceptions import (
    DataCollectionError,
    ExchangeConnectionError,
    RateLimitError,
    InvalidSymbolError,
    DataParsingError,
)


class DataCollector:
    """
    Collects market data from exchange for a single trading pair.
    
    Uses REST API to fetch historical and current market data including
    candles, order book, trades, funding rate, and open interest.
    
    Attributes:
        exchange_client: CCXT exchange client instance.
        symbol: Trading pair symbol.
        logger: Logger instance.
    """
    
    def __init__(
        self,
        exchange_client: Any,
        symbol: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DataCollector.
        
        Args:
            exchange_client: CCXT exchange client or compatible.
            symbol: Trading pair symbol (e.g., "BTC/USDT:USDT").
            logger: Optional logger instance.
        """
        self.exchange_client = exchange_client
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)
        
        # Internal caches
        self._candle_cache: dict = {}
        self._symbol_info_cache: Optional[SymbolInfo] = None
        self._last_request_time: float = 0
        
        # Retry settings
        self._max_retries = 3
        self._retry_delay = 1.0
        
    async def _retry_async(self, func, *args, **kwargs):
        """
        Execute async function with retry logic.
        
        Args:
            func: Async function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.
            
        Returns:
            Function result.
            
        Raises:
            DataCollectionError: If all retries fail.
        """
        last_error = None
        
        for attempt in range(self._max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                
                # Check for rate limit errors
                if 'rate' in str(e).lower() or 'limit' in str(e).lower():
                    wait_time = min(self._retry_delay * (2 ** attempt), 30)
                    self.logger.warning(
                        f"Rate limit hit, waiting {wait_time}s before retry"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                    
                # Check for invalid symbol
                if 'symbol' in str(e).lower() and 'invalid' in str(e).lower():
                    raise InvalidSymbolError(f"Invalid symbol: {self.symbol}")
                    
                # Check for connection errors
                if 'connection' in str(e).lower() or 'network' in str(e).lower():
                    wait_time = self._retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Connection error ({error_name}), retry {attempt + 1}/{self._max_retries}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                    
                # For other errors, retry with exponential backoff
                if attempt < self._max_retries - 1:
                    wait_time = self._retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Error ({error_name}): {e}, retry {attempt + 1}/{self._max_retries}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
        raise DataCollectionError(f"Failed after {self._max_retries} attempts: {last_error}")
    
    def _run_sync(self, func, *args, **kwargs):
        """
        Execute synchronous function with retry logic.
        
        Args:
            func: Synchronous function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.
            
        Returns:
            Function result.
        """
        last_error = None
        
        for attempt in range(self._max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                
                # Check for rate limit
                if 'rate' in str(e).lower() or 'limit' in str(e).lower():
                    import time
                    wait_time = min(self._retry_delay * (2 ** attempt), 30)
                    self.logger.warning(
                        f"Rate limit hit, waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue
                    
                # Check for invalid symbol
                if 'symbol' in str(e).lower() and 'invalid' in str(e).lower():
                    raise InvalidSymbolError(f"Invalid symbol: {self.symbol}")
                    
                # Retry with backoff
                if attempt < self._max_retries - 1:
                    import time
                    wait_time = self._retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Error ({error_name}): {e}, retry {attempt + 1}/{self._max_retries}"
                    )
                    time.sleep(wait_time)
                else:
                    raise
                    
        raise DataCollectionError(f"Failed after {self._max_retries} attempts: {last_error}")

    def fetch_candles(
        self,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Candle]:
        """
        Fetch historical OHLCV candles.
        
        Args:
            timeframe: Candle timeframe (e.g., "1m", "1h", "4h", "1d").
            limit: Number of candles to fetch (default: 100).
            since: Start time for fetching candles (optional).
            
        Returns:
            List of Candle objects.
            
        Raises:
            DataCollectionError: If fetching fails.
        """
        try:
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)
                
            ohlcv = self._run_sync(
                self.exchange_client.fetch_ohlcv,
                self.symbol,
                timeframe,
                since_ms,
                limit
            )
            
            candles = []
            for data in ohlcv:
                candle = Candle(
                    timestamp=datetime.fromtimestamp(data[0] / 1000, tz=timezone.utc),
                    open=float(data[1]),
                    high=float(data[2]),
                    low=float(data[3]),
                    close=float(data[4]),
                    volume=float(data[5]) if data[5] else 0.0,
                    symbol=self.symbol,
                    timeframe=timeframe
                )
                candles.append(candle)
                
            self.logger.debug(
                f"Fetched {len(candles)} candles for {self.symbol} {timeframe}"
            )
            return candles
            
        except InvalidSymbolError:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            raise DataCollectionError(f"Failed to fetch candles: {e}")

    def fetch_orderbook(self, depth: int = 20) -> OrderBook:
        """
        Fetch current order book.
        
        Args:
            depth: Number of levels to fetch from each side (default: 20).
            
        Returns:
            OrderBook object.
            
        Raises:
            DataCollectionError: If fetching fails.
        """
        try:
            raw_book = self._run_sync(
                self.exchange_client.fetch_order_book,
                self.symbol,
                depth
            )
            
            bids = [
                OrderBookLevel(price=float(bid[0]), quantity=float(bid[1]))
                for bid in raw_book.get('bids', [])[:depth]
            ]
            
            asks = [
                OrderBookLevel(price=float(ask[0]), quantity=float(ask[1]))
                for ask in raw_book.get('asks', [])[:depth]
            ]
            
            timestamp = datetime.now(timezone.utc)
            if 'timestamp' in raw_book and raw_book['timestamp']:
                timestamp = datetime.fromtimestamp(
                    raw_book['timestamp'] / 1000, tz=timezone.utc
                )
            
            return OrderBook(
                symbol=self.symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            raise DataCollectionError(f"Failed to fetch orderbook: {e}")

    def fetch_recent_trades(self, limit: int = 100) -> List[Trade]:
        """
        Fetch recent trades.
        
        Args:
            limit: Number of trades to fetch (default: 100).
            
        Returns:
            List of Trade objects.
            
        Raises:
            DataCollectionError: If fetching fails.
        """
        try:
            raw_trades = self._run_sync(
                self.exchange_client.fetch_trades,
                self.symbol,
                None,
                limit
            )
            
            trades = []
            for trade_data in raw_trades:
                trade = Trade(
                    timestamp=datetime.fromtimestamp(
                        trade_data['timestamp'] / 1000, tz=timezone.utc
                    ),
                    symbol=self.symbol,
                    price=float(trade_data['price']),
                    quantity=float(trade_data['amount']),
                    side=trade_data.get('side', 'buy')
                )
                trades.append(trade)
                
            return trades
            
        except Exception as e:
            self.logger.error(f"Error fetching trades: {e}")
            raise DataCollectionError(f"Failed to fetch trades: {e}")

    def fetch_funding_rate(self) -> FundingRate:
        """
        Fetch current funding rate for perpetual futures.
        
        Returns:
            FundingRate object.
            
        Raises:
            DataCollectionError: If fetching fails.
        """
        try:
            # Try to get funding rate - method varies by exchange
            if hasattr(self.exchange_client, 'fetch_funding_rate'):
                raw_funding = self._run_sync(
                    self.exchange_client.fetch_funding_rate,
                    self.symbol
                )
            elif hasattr(self.exchange_client, 'fapiPublicGetPremiumIndex'):
                # Binance futures specific
                raw_funding = self._run_sync(
                    self.exchange_client.fapiPublicGetPremiumIndex,
                    {'symbol': self.symbol.replace('/', '').replace(':USDT', '')}
                )
            else:
                # Try generic funding rates method
                raw_funding = self._run_sync(
                    self.exchange_client.fetch_funding_rates,
                    [self.symbol]
                )
                if isinstance(raw_funding, dict) and self.symbol in raw_funding:
                    raw_funding = raw_funding[self.symbol]
            
            # Parse the response based on structure
            rate = 0.0
            next_time = datetime.now(timezone.utc)
            
            if isinstance(raw_funding, dict):
                rate = float(raw_funding.get('fundingRate', 0) or 
                           raw_funding.get('rate', 0) or 0)
                
                next_ts = raw_funding.get('nextFundingTime') or \
                         raw_funding.get('fundingTimestamp')
                if next_ts:
                    if isinstance(next_ts, (int, float)):
                        next_time = datetime.fromtimestamp(
                            next_ts / 1000 if next_ts > 1e10 else next_ts,
                            tz=timezone.utc
                        )
                    elif isinstance(next_ts, datetime):
                        next_time = next_ts
            
            return FundingRate(
                symbol=self.symbol,
                rate=rate,
                next_funding_time=next_time
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rate: {e}")
            raise DataCollectionError(f"Failed to fetch funding rate: {e}")

    def fetch_open_interest(self) -> OpenInterest:
        """
        Fetch current open interest.
        
        Returns:
            OpenInterest object.
            
        Raises:
            DataCollectionError: If fetching fails.
        """
        try:
            if hasattr(self.exchange_client, 'fetch_open_interest'):
                raw_oi = self._run_sync(
                    self.exchange_client.fetch_open_interest,
                    self.symbol
                )
            elif hasattr(self.exchange_client, 'fapiPublicGetOpenInterest'):
                # Binance futures specific
                symbol_clean = self.symbol.replace('/', '').replace(':USDT', '')
                raw_oi = self._run_sync(
                    self.exchange_client.fapiPublicGetOpenInterest,
                    {'symbol': symbol_clean}
                )
            else:
                raise DataCollectionError(
                    "Exchange does not support open interest"
                )
            
            value = 0.0
            timestamp = datetime.now(timezone.utc)
            
            if isinstance(raw_oi, dict):
                value = float(raw_oi.get('openInterest', 0) or 
                            raw_oi.get('openInterestAmount', 0) or 0)
                
                ts = raw_oi.get('timestamp') or raw_oi.get('time')
                if ts:
                    timestamp = datetime.fromtimestamp(
                        ts / 1000 if ts > 1e10 else ts,
                        tz=timezone.utc
                    )
            
            return OpenInterest(
                symbol=self.symbol,
                value=value,
                timestamp=timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching open interest: {e}")
            raise DataCollectionError(f"Failed to fetch open interest: {e}")

    def fetch_symbol_info(self) -> SymbolInfo:
        """
        Fetch information about the trading pair.
        
        Returns:
            SymbolInfo object.
            
        Raises:
            DataCollectionError: If fetching fails.
        """
        try:
            # Load markets if not already loaded
            if not self.exchange_client.markets:
                self._run_sync(self.exchange_client.load_markets)
            
            market = self.exchange_client.markets.get(self.symbol)
            
            if not market:
                raise InvalidSymbolError(f"Symbol not found: {self.symbol}")
            
            # Extract precision info
            price_precision = market.get('precision', {}).get('price', 8)
            if isinstance(price_precision, float):
                price_precision = int(-1 * (price_precision // 1)) if price_precision < 1 else int(price_precision)
                
            quantity_precision = market.get('precision', {}).get('amount', 8)
            if isinstance(quantity_precision, float):
                quantity_precision = int(-1 * (quantity_precision // 1)) if quantity_precision < 1 else int(quantity_precision)
            
            # Extract limits
            limits = market.get('limits', {})
            min_quantity = limits.get('amount', {}).get('min', 0) or 0
            
            # Get leverage info
            max_leverage = 1
            if 'leverage' in limits:
                max_leverage = limits['leverage'].get('max', 1) or 1
            elif market.get('type') in ('future', 'swap'):
                max_leverage = 125  # Default for most futures exchanges
            
            # Get tick size
            tick_size = 0.01
            if 'precision' in market and 'price' in market['precision']:
                prec = market['precision']['price']
                if isinstance(prec, int):
                    tick_size = 10 ** (-prec)
                else:
                    tick_size = prec
            
            symbol_info = SymbolInfo(
                symbol=self.symbol,
                base_asset=market.get('base', ''),
                quote_asset=market.get('quote', ''),
                price_precision=price_precision,
                quantity_precision=quantity_precision,
                min_quantity=float(min_quantity),
                max_leverage=int(max_leverage),
                tick_size=float(tick_size),
                status='active' if market.get('active', True) else 'inactive'
            )
            
            self._symbol_info_cache = symbol_info
            return symbol_info
            
        except InvalidSymbolError:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching symbol info: {e}")
            raise DataCollectionError(f"Failed to fetch symbol info: {e}")

    def fetch_all_data(
        self,
        timeframes: Optional[List[str]] = None,
        candles_limit: int = 100
    ) -> MarketData:
        """
        Fetch all available data for the trading pair.
        
        Args:
            timeframes: List of timeframes to fetch candles for.
                       Default: ["1m", "5m", "1h", "4h"].
            candles_limit: Number of candles to fetch per timeframe.
            
        Returns:
            MarketData object with all collected data.
            
        Raises:
            DataCollectionError: If any critical data fetch fails.
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "1h", "4h"]
            
        market_data = MarketData(
            symbol=self.symbol,
            candles={},
            last_update=datetime.now(timezone.utc)
        )
        
        errors = []
        
        # Fetch candles for each timeframe
        for tf in timeframes:
            try:
                candles = self.fetch_candles(timeframe=tf, limit=candles_limit)
                market_data.candles[tf] = candles
            except Exception as e:
                self.logger.warning(f"Failed to fetch {tf} candles: {e}")
                errors.append(f"candles_{tf}: {e}")
        
        # Fetch order book
        try:
            market_data.orderbook = self.fetch_orderbook()
        except Exception as e:
            self.logger.warning(f"Failed to fetch orderbook: {e}")
            errors.append(f"orderbook: {e}")
        
        # Fetch recent trades
        try:
            market_data.recent_trades = self.fetch_recent_trades()
        except Exception as e:
            self.logger.warning(f"Failed to fetch trades: {e}")
            errors.append(f"trades: {e}")
        
        # Fetch funding rate (futures only)
        try:
            market_data.funding = self.fetch_funding_rate()
        except Exception as e:
            self.logger.debug(f"Funding rate not available: {e}")
        
        # Fetch open interest (futures only)
        try:
            market_data.open_interest = self.fetch_open_interest()
        except Exception as e:
            self.logger.debug(f"Open interest not available: {e}")
        
        market_data.last_update = datetime.now(timezone.utc)
        
        # Log summary
        self.logger.info(
            f"Fetched data for {self.symbol}: "
            f"{len(market_data.candles)} timeframes, "
            f"orderbook: {market_data.orderbook is not None}, "
            f"trades: {len(market_data.recent_trades)}, "
            f"funding: {market_data.funding is not None}"
        )
        
        if errors:
            self.logger.warning(f"Some data fetch errors: {errors}")
        
        return market_data

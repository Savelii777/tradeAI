"""
AI Trading Bot - Multi-Pair Data Collector
Parallel data collection for multiple trading pairs.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
import logging
import time

from .models import Candle, MarketData
from .collector import DataCollector
from .exceptions import DataCollectionError


class MultiPairCollector:
    """
    Collects market data for multiple trading pairs in parallel.
    
    Manages a pool of DataCollector instances and coordinates
    parallel data fetching with rate limiting.
    
    Attributes:
        exchange_client: CCXT exchange client instance.
        symbols: List of trading pair symbols.
        timeframes: List of timeframes to collect.
        max_concurrent: Maximum concurrent requests.
        logger: Logger instance.
    """
    
    def __init__(
        self,
        exchange_client: Any,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        max_concurrent: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MultiPairCollector.
        
        Args:
            exchange_client: CCXT exchange client or compatible.
            symbols: List of trading pair symbols.
            timeframes: List of timeframes to collect.
            max_concurrent: Maximum number of concurrent requests.
            logger: Optional logger instance.
        """
        self.exchange_client = exchange_client
        self.symbols = symbols
        self.timeframes = timeframes or ["5m", "1h", "4h"]
        self.max_concurrent = max_concurrent
        self.logger = logger or logging.getLogger(__name__)
        
        # Create collectors for each symbol
        self._collectors: Dict[str, DataCollector] = {
            symbol: DataCollector(exchange_client, symbol, logger)
            for symbol in symbols
        }
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Rate limiting
        self._request_count = 0
        self._request_window_start = time.time()
        self._requests_per_minute = 1200  # Default rate limit
        
        # Get rate limit from exchange if available
        if hasattr(exchange_client, 'rateLimit'):
            # rateLimit in ccxt is in milliseconds per request
            rate_limit_ms = exchange_client.rateLimit
            self._requests_per_minute = int(60000 / rate_limit_ms) if rate_limit_ms > 0 else 1200

    async def _rate_limit_wait(self) -> None:
        """Wait if approaching rate limit."""
        current_time = time.time()
        window_duration = current_time - self._request_window_start
        
        # Reset window if more than 60 seconds
        if window_duration > 60:
            self._request_count = 0
            self._request_window_start = current_time
            return
        
        # Check if approaching limit
        if self._request_count >= self._requests_per_minute * 0.8:
            # Wait until window resets
            wait_time = 60 - window_duration + 1
            self.logger.warning(
                f"Approaching rate limit, waiting {wait_time:.1f}s"
            )
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._request_window_start = time.time()

    async def _collect_symbol_data(
        self,
        symbol: str,
        data_types: List[str]
    ) -> Optional[MarketData]:
        """
        Collect data for a single symbol with concurrency control.
        
        Args:
            symbol: Trading pair symbol.
            data_types: Types of data to collect.
            
        Returns:
            MarketData object or None if failed.
        """
        async with self._semaphore:
            await self._rate_limit_wait()
            
            try:
                collector = self._collectors.get(symbol)
                if not collector:
                    collector = DataCollector(
                        self.exchange_client, symbol, self.logger
                    )
                    self._collectors[symbol] = collector
                
                market_data = MarketData(
                    symbol=symbol,
                    last_update=datetime.now(timezone.utc)
                )
                
                # Fetch candles
                if "candles" in data_types:
                    for tf in self.timeframes:
                        try:
                            self._request_count += 1
                            candles = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: collector.fetch_candles(tf, 100)
                            )
                            market_data.candles[tf] = candles
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to fetch {tf} candles for {symbol}: {e}"
                            )
                
                # Fetch order book
                if "orderbook" in data_types:
                    try:
                        self._request_count += 1
                        market_data.orderbook = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: collector.fetch_orderbook(20)
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to fetch orderbook for {symbol}: {e}"
                        )
                
                # Fetch trades
                if "trades" in data_types:
                    try:
                        self._request_count += 1
                        market_data.recent_trades = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: collector.fetch_recent_trades(100)
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to fetch trades for {symbol}: {e}"
                        )
                
                # Fetch funding rate
                if "funding" in data_types:
                    try:
                        self._request_count += 1
                        market_data.funding = await asyncio.get_event_loop().run_in_executor(
                            None,
                            collector.fetch_funding_rate
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Funding rate not available for {symbol}: {e}"
                        )
                
                # Fetch open interest
                if "oi" in data_types:
                    try:
                        self._request_count += 1
                        market_data.open_interest = await asyncio.get_event_loop().run_in_executor(
                            None,
                            collector.fetch_open_interest
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Open interest not available for {symbol}: {e}"
                        )
                
                market_data.last_update = datetime.now(timezone.utc)
                return market_data
                
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")
                return None

    async def fetch_all_symbols(
        self,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, MarketData]:
        """
        Fetch data for all symbols in parallel.
        
        Args:
            data_types: Types of data to fetch.
                       Options: "candles", "orderbook", "trades", "funding", "oi".
                       Default: ["candles"].
                       
        Returns:
            Dictionary mapping symbol to MarketData.
        """
        if data_types is None:
            data_types = ["candles"]
            
        self.logger.info(
            f"Fetching {data_types} for {len(self.symbols)} symbols"
        )
        
        # Create tasks for all symbols
        tasks = [
            self._collect_symbol_data(symbol, data_types)
            for symbol in self.symbols
        ]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        data_dict: Dict[str, MarketData] = {}
        errors = []
        
        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error for {symbol}: {result}")
                errors.append(symbol)
            elif result is not None:
                data_dict[symbol] = result
        
        self.logger.info(
            f"Fetched data for {len(data_dict)}/{len(self.symbols)} symbols"
        )
        
        if errors:
            self.logger.warning(f"Failed symbols: {errors}")
        
        return data_dict

    async def fetch_symbols_candles(
        self,
        timeframe: str,
        limit: int = 100
    ) -> Dict[str, List[Candle]]:
        """
        Fetch only candles for all symbols (optimized method).
        
        Args:
            timeframe: Candle timeframe.
            limit: Number of candles per symbol.
            
        Returns:
            Dictionary mapping symbol to list of candles.
        """
        self.logger.info(
            f"Fetching {timeframe} candles for {len(self.symbols)} symbols"
        )
        
        async def fetch_single(symbol: str) -> tuple:
            async with self._semaphore:
                await self._rate_limit_wait()
                self._request_count += 1
                
                try:
                    collector = self._collectors.get(symbol)
                    if not collector:
                        collector = DataCollector(
                            self.exchange_client, symbol, self.logger
                        )
                        self._collectors[symbol] = collector
                    
                    candles = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: collector.fetch_candles(timeframe, limit)
                    )
                    return symbol, candles
                except Exception as e:
                    self.logger.warning(f"Failed to fetch candles for {symbol}: {e}")
                    return symbol, []
        
        # Execute all tasks
        tasks = [fetch_single(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        candles_dict: Dict[str, List[Candle]] = {}
        
        for result in results:
            if isinstance(result, Exception):
                continue
            symbol, candles = result
            if candles:
                candles_dict[symbol] = candles
        
        return candles_dict

    async def get_tradeable_symbols(
        self,
        min_volume_usd: float = 10_000_000,
        quote_asset: str = "USDT"
    ) -> List[str]:
        """
        Get list of all tradeable futures pairs that meet criteria.
        
        Args:
            min_volume_usd: Minimum 24h volume in USD.
            quote_asset: Quote asset to filter by.
            
        Returns:
            List of symbol strings sorted by volume.
        """
        try:
            # Load markets
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange_client.load_markets
            )
            
            markets = self.exchange_client.markets
            tradeable: List[tuple] = []
            
            for symbol, market in markets.items():
                # Filter by type (futures/swap)
                market_type = market.get('type', '')
                if market_type not in ('future', 'swap'):
                    continue
                    
                # Filter by quote asset
                if market.get('quote', '') != quote_asset:
                    continue
                    
                # Filter by active status
                if not market.get('active', True):
                    continue
                    
                # Get 24h volume if available
                volume = 0.0
                if hasattr(self.exchange_client, 'fetch_ticker'):
                    try:
                        ticker = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda s=symbol: self.exchange_client.fetch_ticker(s)
                        )
                        volume = ticker.get('quoteVolume', 0) or 0
                    except Exception:
                        pass
                
                # Filter by minimum volume
                if volume >= min_volume_usd:
                    tradeable.append((symbol, volume))
            
            # Sort by volume descending
            tradeable.sort(key=lambda x: x[1], reverse=True)
            
            result = [symbol for symbol, _ in tradeable]
            
            self.logger.info(
                f"Found {len(result)} tradeable symbols with >{min_volume_usd:,.0f}$ volume"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting tradeable symbols: {e}")
            raise DataCollectionError(f"Failed to get tradeable symbols: {e}")

    def add_symbol(self, symbol: str) -> None:
        """
        Add a new symbol to the collector.
        
        Args:
            symbol: Trading pair symbol to add.
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self._collectors[symbol] = DataCollector(
                self.exchange_client, symbol, self.logger
            )

    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the collector.
        
        Args:
            symbol: Trading pair symbol to remove.
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            self._collectors.pop(symbol, None)

    def get_collector(self, symbol: str) -> Optional[DataCollector]:
        """
        Get the DataCollector for a specific symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            DataCollector instance or None.
        """
        return self._collectors.get(symbol)

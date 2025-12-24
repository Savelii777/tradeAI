"""
AI Trading Bot - Unit Tests for Data Module - Collector
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.collector import DataCollector
from src.data.models import Candle, OrderBook, Trade, FundingRate, OpenInterest, SymbolInfo
from src.data.exceptions import DataCollectionError, InvalidSymbolError


@pytest.fixture
def mock_exchange():
    """Create a mock exchange client."""
    exchange = Mock()
    exchange.markets = {}
    exchange.rateLimit = 100
    return exchange


@pytest.fixture
def collector(mock_exchange):
    """Create a DataCollector with mock exchange."""
    return DataCollector(
        exchange_client=mock_exchange,
        symbol="BTC/USDT:USDT"
    )


class TestDataCollectorInit:
    """Tests for DataCollector initialization."""
    
    def test_initialization(self, mock_exchange):
        """Test basic initialization."""
        collector = DataCollector(mock_exchange, "BTC/USDT:USDT")
        
        assert collector.symbol == "BTC/USDT:USDT"
        assert collector.exchange_client == mock_exchange


class TestFetchCandles:
    """Tests for fetch_candles method."""
    
    def test_fetch_candles_success(self, collector, mock_exchange):
        """Test successful candle fetch."""
        # Mock response
        mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 42000.0, 42500.0, 41500.0, 42200.0, 1000.0],
            [1704070800000, 42200.0, 42800.0, 42000.0, 42600.0, 1200.0],
        ]
        
        result = collector.fetch_candles("1h", limit=2)
        
        assert len(result) == 2
        assert isinstance(result[0], Candle)
        assert result[0].open == 42000.0
        assert result[0].close == 42200.0
        assert result[0].symbol == "BTC/USDT:USDT"
        assert result[0].timeframe == "1h"
    
    def test_fetch_candles_with_since(self, collector, mock_exchange):
        """Test fetching candles with since parameter."""
        mock_exchange.fetch_ohlcv.return_value = []
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        collector.fetch_candles("1h", since=since)
        
        # Verify since was converted to milliseconds
        call_args = mock_exchange.fetch_ohlcv.call_args
        assert call_args[0][2] is not None  # since parameter
    
    def test_fetch_candles_error(self, collector, mock_exchange):
        """Test error handling in candle fetch."""
        mock_exchange.fetch_ohlcv.side_effect = Exception("API error")
        
        with pytest.raises(DataCollectionError):
            collector.fetch_candles("1h")


class TestFetchOrderbook:
    """Tests for fetch_orderbook method."""
    
    def test_fetch_orderbook_success(self, collector, mock_exchange):
        """Test successful orderbook fetch."""
        mock_exchange.fetch_order_book.return_value = {
            'bids': [[42000.0, 1.0], [41999.0, 2.0]],
            'asks': [[42001.0, 1.5], [42002.0, 2.5]],
            'timestamp': 1704067200000
        }
        
        result = collector.fetch_orderbook(depth=2)
        
        assert isinstance(result, OrderBook)
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0].price == 42000.0
        assert result.asks[0].price == 42001.0
    
    def test_fetch_orderbook_error(self, collector, mock_exchange):
        """Test error handling in orderbook fetch."""
        mock_exchange.fetch_order_book.side_effect = Exception("API error")
        
        with pytest.raises(DataCollectionError):
            collector.fetch_orderbook()


class TestFetchRecentTrades:
    """Tests for fetch_recent_trades method."""
    
    def test_fetch_trades_success(self, collector, mock_exchange):
        """Test successful trades fetch."""
        mock_exchange.fetch_trades.return_value = [
            {
                'timestamp': 1704067200000,
                'price': 42000.0,
                'amount': 0.5,
                'side': 'buy'
            },
            {
                'timestamp': 1704067201000,
                'price': 42001.0,
                'amount': 0.3,
                'side': 'sell'
            }
        ]
        
        result = collector.fetch_recent_trades(limit=2)
        
        assert len(result) == 2
        assert isinstance(result[0], Trade)
        assert result[0].price == 42000.0
        assert result[0].side == 'buy'
    
    def test_fetch_trades_error(self, collector, mock_exchange):
        """Test error handling in trades fetch."""
        mock_exchange.fetch_trades.side_effect = Exception("API error")
        
        with pytest.raises(DataCollectionError):
            collector.fetch_recent_trades()


class TestFetchFundingRate:
    """Tests for fetch_funding_rate method."""
    
    def test_fetch_funding_with_fetch_funding_rate(self, collector, mock_exchange):
        """Test fetching funding rate using fetch_funding_rate method."""
        mock_exchange.fetch_funding_rate.return_value = {
            'fundingRate': 0.0001,
            'nextFundingTime': 1704067200000
        }
        
        result = collector.fetch_funding_rate()
        
        assert isinstance(result, FundingRate)
        assert result.rate == 0.0001
    
    def test_fetch_funding_error(self, collector, mock_exchange):
        """Test error handling in funding fetch."""
        mock_exchange.fetch_funding_rate.side_effect = Exception("API error")
        delattr(mock_exchange, 'fapiPublicGetPremiumIndex')
        delattr(mock_exchange, 'fetch_funding_rates')
        
        with pytest.raises(DataCollectionError):
            collector.fetch_funding_rate()


class TestFetchOpenInterest:
    """Tests for fetch_open_interest method."""
    
    def test_fetch_oi_success(self, collector, mock_exchange):
        """Test successful open interest fetch."""
        mock_exchange.fetch_open_interest.return_value = {
            'openInterest': 50000000.0,
            'timestamp': 1704067200000
        }
        
        result = collector.fetch_open_interest()
        
        assert isinstance(result, OpenInterest)
        assert result.value == 50000000.0
    
    def test_fetch_oi_error(self, collector, mock_exchange):
        """Test error handling in OI fetch."""
        mock_exchange.fetch_open_interest.side_effect = Exception("API error")
        delattr(mock_exchange, 'fapiPublicGetOpenInterest')
        
        with pytest.raises(DataCollectionError):
            collector.fetch_open_interest()


class TestFetchSymbolInfo:
    """Tests for fetch_symbol_info method."""
    
    def test_fetch_symbol_info_success(self, collector, mock_exchange):
        """Test successful symbol info fetch."""
        mock_exchange.markets = {
            'BTC/USDT:USDT': {
                'base': 'BTC',
                'quote': 'USDT',
                'precision': {'price': 2, 'amount': 3},
                'limits': {
                    'amount': {'min': 0.001},
                    'leverage': {'max': 125}
                },
                'type': 'swap',
                'active': True
            }
        }
        
        result = collector.fetch_symbol_info()
        
        assert isinstance(result, SymbolInfo)
        assert result.symbol == "BTC/USDT:USDT"
        assert result.base_asset == "BTC"
        assert result.quote_asset == "USDT"
    
    def test_fetch_symbol_info_invalid_symbol(self, collector, mock_exchange):
        """Test fetching info for invalid symbol."""
        mock_exchange.markets = {}
        mock_exchange.load_markets.return_value = {}
        
        with pytest.raises(InvalidSymbolError):
            collector.fetch_symbol_info()


class TestFetchAllData:
    """Tests for fetch_all_data method."""
    
    def test_fetch_all_data_success(self, collector, mock_exchange):
        """Test fetching all data."""
        # Mock all methods
        mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 42000.0, 42500.0, 41500.0, 42200.0, 1000.0]
        ]
        mock_exchange.fetch_order_book.return_value = {
            'bids': [[42000.0, 1.0]],
            'asks': [[42001.0, 1.5]],
            'timestamp': None
        }
        mock_exchange.fetch_trades.return_value = []
        mock_exchange.fetch_funding_rate.return_value = {'rate': 0.0001}
        mock_exchange.fetch_open_interest.return_value = {'openInterest': 50000000}
        
        result = collector.fetch_all_data(timeframes=["1h"], candles_limit=1)
        
        assert result.symbol == "BTC/USDT:USDT"
        assert "1h" in result.candles
        assert result.orderbook is not None
    
    def test_fetch_all_data_partial_failure(self, collector, mock_exchange):
        """Test that partial failures don't break the whole request."""
        mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 42000.0, 42500.0, 41500.0, 42200.0, 1000.0]
        ]
        mock_exchange.fetch_order_book.side_effect = Exception("API error")
        mock_exchange.fetch_trades.side_effect = Exception("API error")
        mock_exchange.fetch_funding_rate.side_effect = Exception("Not available")
        mock_exchange.fetch_open_interest.side_effect = Exception("Not available")
        
        # Should still return data, not raise
        result = collector.fetch_all_data(timeframes=["1h"])
        
        assert result.symbol == "BTC/USDT:USDT"
        assert "1h" in result.candles


class TestRetryLogic:
    """Tests for retry logic."""
    
    def test_retry_on_network_error(self, collector, mock_exchange):
        """Test that network errors trigger retry."""
        # Fail twice, then succeed
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("connection refused"),
            Exception("timeout"),
            [[1704067200000, 42000.0, 42500.0, 41500.0, 42200.0, 1000.0]]
        ]
        
        result = collector.fetch_candles("1h", limit=1)
        
        assert len(result) == 1
        assert mock_exchange.fetch_ohlcv.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

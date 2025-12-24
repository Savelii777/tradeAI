"""
AI Trading Bot - Unit Tests for Data Module - Models
"""

import pytest
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.models import (
    Candle,
    OrderBook,
    OrderBookLevel,
    Trade,
    FundingRate,
    OpenInterest,
    SymbolInfo,
    MarketData,
)


class TestCandle:
    """Tests for Candle dataclass."""
    
    def test_candle_creation(self):
        """Test creating a candle."""
        now = datetime.now(timezone.utc)
        candle = Candle(
            timestamp=now,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0,
            symbol="BTC/USDT:USDT",
            timeframe="1h"
        )
        
        assert candle.timestamp == now
        assert candle.open == 50000.0
        assert candle.high == 51000.0
        assert candle.low == 49000.0
        assert candle.close == 50500.0
        assert candle.volume == 100.0
        assert candle.symbol == "BTC/USDT:USDT"
        assert candle.timeframe == "1h"


class TestOrderBookLevel:
    """Tests for OrderBookLevel dataclass."""
    
    def test_level_creation(self):
        """Test creating an order book level."""
        level = OrderBookLevel(price=50000.0, quantity=1.5)
        
        assert level.price == 50000.0
        assert level.quantity == 1.5


class TestOrderBook:
    """Tests for OrderBook dataclass."""
    
    @pytest.fixture
    def sample_orderbook(self):
        """Create a sample order book."""
        now = datetime.now(timezone.utc)
        bids = [
            OrderBookLevel(price=49990.0, quantity=1.0),
            OrderBookLevel(price=49980.0, quantity=2.0),
            OrderBookLevel(price=49970.0, quantity=3.0),
        ]
        asks = [
            OrderBookLevel(price=50010.0, quantity=1.5),
            OrderBookLevel(price=50020.0, quantity=2.5),
            OrderBookLevel(price=50030.0, quantity=3.5),
        ]
        return OrderBook(
            symbol="BTC/USDT:USDT",
            timestamp=now,
            bids=bids,
            asks=asks
        )
    
    def test_orderbook_creation(self, sample_orderbook):
        """Test creating an order book."""
        assert sample_orderbook.symbol == "BTC/USDT:USDT"
        assert len(sample_orderbook.bids) == 3
        assert len(sample_orderbook.asks) == 3
    
    def test_get_best_bid(self, sample_orderbook):
        """Test getting best bid."""
        best_bid = sample_orderbook.get_best_bid()
        
        assert best_bid is not None
        assert best_bid.price == 49990.0
        assert best_bid.quantity == 1.0
    
    def test_get_best_ask(self, sample_orderbook):
        """Test getting best ask."""
        best_ask = sample_orderbook.get_best_ask()
        
        assert best_ask is not None
        assert best_ask.price == 50010.0
        assert best_ask.quantity == 1.5
    
    def test_get_spread_percent(self, sample_orderbook):
        """Test calculating spread."""
        spread = sample_orderbook.get_spread_percent()
        
        assert spread is not None
        # Spread = (50010 - 49990) / 49990 * 100 = 0.04%
        assert abs(spread - 0.04) < 0.01
    
    def test_get_total_volume(self, sample_orderbook):
        """Test getting total volume."""
        volumes = sample_orderbook.get_total_volume(levels=2)
        
        assert volumes['bid_volume'] == 3.0  # 1.0 + 2.0
        assert volumes['ask_volume'] == 4.0  # 1.5 + 2.5
    
    def test_empty_orderbook(self):
        """Test empty order book methods."""
        empty_book = OrderBook(
            symbol="BTC/USDT:USDT",
            timestamp=datetime.now(timezone.utc),
            bids=[],
            asks=[]
        )
        
        assert empty_book.get_best_bid() is None
        assert empty_book.get_best_ask() is None
        assert empty_book.get_spread_percent() is None


class TestTrade:
    """Tests for Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a trade."""
        now = datetime.now(timezone.utc)
        trade = Trade(
            timestamp=now,
            symbol="BTC/USDT:USDT",
            price=50000.0,
            quantity=0.5,
            side="buy"
        )
        
        assert trade.timestamp == now
        assert trade.symbol == "BTC/USDT:USDT"
        assert trade.price == 50000.0
        assert trade.quantity == 0.5
        assert trade.side == "buy"


class TestFundingRate:
    """Tests for FundingRate dataclass."""
    
    def test_funding_rate_creation(self):
        """Test creating a funding rate."""
        next_time = datetime.now(timezone.utc)
        funding = FundingRate(
            symbol="BTC/USDT:USDT",
            rate=0.0001,
            next_funding_time=next_time
        )
        
        assert funding.symbol == "BTC/USDT:USDT"
        assert funding.rate == 0.0001
        assert funding.next_funding_time == next_time


class TestOpenInterest:
    """Tests for OpenInterest dataclass."""
    
    def test_open_interest_creation(self):
        """Test creating open interest."""
        now = datetime.now(timezone.utc)
        oi = OpenInterest(
            symbol="BTC/USDT:USDT",
            value=50000000.0,
            timestamp=now
        )
        
        assert oi.symbol == "BTC/USDT:USDT"
        assert oi.value == 50000000.0
        assert oi.timestamp == now


class TestSymbolInfo:
    """Tests for SymbolInfo dataclass."""
    
    def test_symbol_info_creation(self):
        """Test creating symbol info."""
        info = SymbolInfo(
            symbol="BTC/USDT:USDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=3,
            min_quantity=0.001,
            max_leverage=125,
            tick_size=0.01,
            status="active"
        )
        
        assert info.symbol == "BTC/USDT:USDT"
        assert info.base_asset == "BTC"
        assert info.quote_asset == "USDT"
        assert info.price_precision == 2
        assert info.quantity_precision == 3
        assert info.min_quantity == 0.001
        assert info.max_leverage == 125
        assert info.tick_size == 0.01
        assert info.status == "active"


class TestMarketData:
    """Tests for MarketData dataclass."""
    
    def test_market_data_creation(self):
        """Test creating market data."""
        now = datetime.now(timezone.utc)
        market_data = MarketData(
            symbol="BTC/USDT:USDT",
            last_update=now
        )
        
        assert market_data.symbol == "BTC/USDT:USDT"
        assert market_data.candles == {}
        assert market_data.orderbook is None
        assert market_data.recent_trades == []
        assert market_data.funding is None
        assert market_data.open_interest is None
    
    def test_market_data_with_candles(self):
        """Test market data with candles."""
        now = datetime.now(timezone.utc)
        candle = Candle(
            timestamp=now,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0,
            symbol="BTC/USDT:USDT",
            timeframe="1h"
        )
        
        market_data = MarketData(
            symbol="BTC/USDT:USDT",
            candles={"1h": [candle]}
        )
        
        assert "1h" in market_data.candles
        assert len(market_data.candles["1h"]) == 1
        assert market_data.candles["1h"][0].close == 50500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

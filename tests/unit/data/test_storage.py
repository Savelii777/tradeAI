"""
AI Trading Bot - Unit Tests for Data Module - Storage
"""

import pytest
import os
import tempfile
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.models import Candle, Trade, FundingRate, OpenInterest
from src.data.storage import DataStorage


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def storage(temp_db_path):
    """Create a DataStorage instance with temp database."""
    return DataStorage(db_path=temp_db_path)


@pytest.fixture
def sample_candles():
    """Create sample candles."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    candles = []
    
    for i in range(10):
        candle = Candle(
            timestamp=base_time + timedelta(hours=i),
            open=50000 + i * 10,
            high=50100 + i * 10,
            low=49900 + i * 10,
            close=50050 + i * 10,
            volume=100 + i,
            symbol="BTC/USDT:USDT",
            timeframe="1h"
        )
        candles.append(candle)
    
    return candles


@pytest.fixture
def sample_trades():
    """Create sample trades."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    trades = []
    
    for i in range(10):
        trade = Trade(
            timestamp=base_time + timedelta(minutes=i),
            symbol="BTC/USDT:USDT",
            price=50000 + i,
            quantity=0.1 * (i + 1),
            side="buy" if i % 2 == 0 else "sell"
        )
        trades.append(trade)
    
    return trades


class TestStorageInitialization:
    """Tests for DataStorage initialization."""
    
    def test_initialization(self, temp_db_path):
        """Test basic initialization."""
        storage = DataStorage(db_path=temp_db_path)
        
        assert storage.db_path == temp_db_path
        assert storage.use_redis == False
    
    def test_creates_tables(self, temp_db_path):
        """Test that tables are created."""
        storage = DataStorage(db_path=temp_db_path)
        
        # Check tables exist by trying to query them
        import sqlite3
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'candles' in tables
        assert 'trades' in tables
        assert 'funding_history' in tables
        assert 'oi_history' in tables
        
        conn.close()


class TestSaveCandles:
    """Tests for save_candles method."""
    
    def test_save_candles(self, storage, sample_candles):
        """Test saving candles."""
        count = storage.save_candles(sample_candles)
        
        assert count == 10
    
    def test_upsert_candles(self, storage, sample_candles):
        """Test that upsert works correctly."""
        # Save first
        storage.save_candles(sample_candles)
        
        # Modify and save again
        sample_candles[0] = Candle(
            timestamp=sample_candles[0].timestamp,
            open=99999,
            high=sample_candles[0].high,
            low=sample_candles[0].low,
            close=sample_candles[0].close,
            volume=sample_candles[0].volume,
            symbol=sample_candles[0].symbol,
            timeframe=sample_candles[0].timeframe
        )
        storage.save_candles(sample_candles)
        
        # Retrieve and check
        result = storage.get_candles("BTC/USDT:USDT", "1h")
        assert len(result) == 10  # No duplicates
        assert result[0].open == 99999  # Updated value
    
    def test_save_empty_list(self, storage):
        """Test saving empty list."""
        count = storage.save_candles([])
        assert count == 0


class TestGetCandles:
    """Tests for get_candles method."""
    
    def test_get_all_candles(self, storage, sample_candles):
        """Test getting all candles."""
        storage.save_candles(sample_candles)
        
        result = storage.get_candles("BTC/USDT:USDT", "1h")
        
        assert len(result) == 10
        assert result[0].symbol == "BTC/USDT:USDT"
        assert result[0].timeframe == "1h"
    
    def test_get_candles_with_limit(self, storage, sample_candles):
        """Test getting candles with limit."""
        storage.save_candles(sample_candles)
        
        result = storage.get_candles("BTC/USDT:USDT", "1h", limit=5)
        
        assert len(result) == 5
    
    def test_get_candles_with_time_filter(self, storage, sample_candles):
        """Test getting candles with time filter."""
        storage.save_candles(sample_candles)
        
        start = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        
        result = storage.get_candles(
            "BTC/USDT:USDT", "1h",
            start_time=start,
            end_time=end
        )
        
        assert len(result) == 4  # Hours 3, 4, 5, 6
    
    def test_get_candles_wrong_symbol(self, storage, sample_candles):
        """Test getting candles for wrong symbol."""
        storage.save_candles(sample_candles)
        
        result = storage.get_candles("ETH/USDT:USDT", "1h")
        
        assert len(result) == 0


class TestSaveTrades:
    """Tests for save_trades method."""
    
    def test_save_trades(self, storage, sample_trades):
        """Test saving trades."""
        count = storage.save_trades(sample_trades)
        
        assert count == 10
    
    def test_get_trades(self, storage, sample_trades):
        """Test getting trades."""
        storage.save_trades(sample_trades)
        
        result = storage.get_trades("BTC/USDT:USDT")
        
        assert len(result) == 10
        assert result[0].side in ["buy", "sell"]


class TestSaveFunding:
    """Tests for save_funding method."""
    
    def test_save_and_get_funding(self, storage):
        """Test saving and getting funding rates."""
        now = datetime.now(timezone.utc)
        funding = FundingRate(
            symbol="BTC/USDT:USDT",
            rate=0.0001,
            next_funding_time=now
        )
        
        storage.save_funding(funding)
        
        result = storage.get_funding_history("BTC/USDT:USDT", days=1)
        
        assert len(result) == 1
        assert result[0].rate == 0.0001


class TestCaching:
    """Tests for Redis caching methods."""
    
    def test_cache_without_redis(self, storage):
        """Test cache operations without Redis."""
        # Should not raise, just return False/None
        assert storage.cache_set("test_key", "test_value") == False
        assert storage.cache_get("test_key") is None
        assert storage.cache_delete("test_key") == False


class TestCleanup:
    """Tests for cleanup_old_data method."""
    
    def test_cleanup(self, storage):
        """Test cleaning up old data."""
        # Create old candles
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        old_candles = [
            Candle(
                timestamp=old_time,
                open=100, high=110, low=90, close=105, volume=10,
                symbol="BTC/USDT:USDT", timeframe="1h"
            )
        ]
        
        # Create recent candles
        recent_time = datetime.now(timezone.utc) - timedelta(days=1)
        recent_candles = [
            Candle(
                timestamp=recent_time,
                open=200, high=210, low=190, close=205, volume=20,
                symbol="BTC/USDT:USDT", timeframe="1h"
            )
        ]
        
        storage.save_candles(old_candles)
        storage.save_candles(recent_candles)
        
        # Cleanup with 30 day retention
        deleted = storage.cleanup_old_data(days_to_keep=30)
        
        # Old candle should be deleted
        assert deleted >= 1
        
        # Recent candle should remain
        result = storage.get_candles("BTC/USDT:USDT", "1h")
        assert len(result) == 1


class TestClose:
    """Tests for close method."""
    
    def test_close(self, storage):
        """Test closing connections."""
        # Should not raise
        storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

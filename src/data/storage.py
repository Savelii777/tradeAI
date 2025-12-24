"""
AI Trading Bot - Data Storage
Handles storage and retrieval of historical and operational data.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
import logging

from .models import Candle, Trade, FundingRate, OpenInterest
from .exceptions import StorageError, DatabaseConnectionError


class DataStorage:
    """
    Manages data storage in SQLite/PostgreSQL with optional Redis caching.
    
    Provides methods for storing and retrieving historical candles,
    trades, funding rates, and open interest data.
    
    Attributes:
        db_path: Path to SQLite database or PostgreSQL connection string.
        use_redis: Whether to use Redis for caching.
        logger: Logger instance.
    """
    
    def __init__(
        self,
        db_path: str = "data/trading.db",
        use_redis: bool = False,
        redis_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DataStorage.
        
        Args:
            db_path: Path to SQLite database file or PostgreSQL connection string.
            use_redis: Whether to enable Redis caching.
            redis_url: Redis connection URL (e.g., "redis://localhost:6379").
            logger: Optional logger instance.
        """
        self.db_path = db_path
        self.use_redis = use_redis
        self.redis_url = redis_url
        self.logger = logger or logging.getLogger(__name__)
        
        # Database connection
        self._connection = None
        self._is_postgres = db_path.startswith('postgresql://') or db_path.startswith('postgres://')
        
        # Redis client
        self._redis = None
        if use_redis and redis_url:
            self._init_redis(redis_url)
        
        # Initialize database
        self._init_database()

    def _init_redis(self, redis_url: str) -> None:
        """Initialize Redis connection."""
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self.logger.info("Redis connection established")
        except ImportError:
            self.logger.warning("redis package not installed, caching disabled")
            self.use_redis = False
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}, caching disabled")
            self.use_redis = False

    def _init_database(self) -> None:
        """Initialize database and create tables if needed."""
        try:
            if self._is_postgres:
                self._init_postgres()
            else:
                self._init_sqlite()
            self.logger.info("Database initialized")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise DatabaseConnectionError(f"Failed to initialize database: {e}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        import os
        
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create candles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Create index on candles
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_timestamp
                ON candles(symbol, timeframe, timestamp)
            ''')
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    side TEXT NOT NULL
                )
            ''')
            
            # Create index on trades
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
                ON trades(symbol, timestamp)
            ''')
            
            # Create funding_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS funding_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rate REAL NOT NULL
                )
            ''')
            
            # Create index on funding_history
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_funding_symbol_timestamp
                ON funding_history(symbol, timestamp)
            ''')
            
            # Create oi_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oi_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    value REAL NOT NULL
                )
            ''')
            
            # Create index on oi_history
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_oi_symbol_timestamp
                ON oi_history(symbol, timestamp)
            ''')
            
            conn.commit()

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL database."""
        try:
            import psycopg2
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create candles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS candles (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(50) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                ''')
                
                # Create index
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_timestamp
                    ON candles(symbol, timeframe, timestamp)
                ''')
                
                # Create trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        quantity DOUBLE PRECISION NOT NULL,
                        side VARCHAR(10) NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
                    ON trades(symbol, timestamp)
                ''')
                
                # Create funding_history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS funding_history (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        rate DOUBLE PRECISION NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_funding_symbol_timestamp
                    ON funding_history(symbol, timestamp)
                ''')
                
                # Create oi_history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS oi_history (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        value DOUBLE PRECISION NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_oi_symbol_timestamp
                    ON oi_history(symbol, timestamp)
                ''')
                
                conn.commit()
                
        except ImportError:
            raise DatabaseConnectionError(
                "psycopg2 package required for PostgreSQL: pip install psycopg2-binary"
            )

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = None
        try:
            if self._is_postgres:
                import psycopg2
                conn = psycopg2.connect(self.db_path)
            else:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()

    def save_candles(self, candles: List[Candle]) -> int:
        """
        Save candles to database using upsert.
        
        Args:
            candles: List of Candle objects to save.
            
        Returns:
            Number of saved records.
        """
        if not candles:
            return 0
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                saved = 0
                
                for candle in candles:
                    if self._is_postgres:
                        cursor.execute('''
                            INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                        ''', (
                            candle.symbol, candle.timeframe, candle.timestamp,
                            candle.open, candle.high, candle.low, candle.close, candle.volume
                        ))
                    else:
                        cursor.execute('''
                            INSERT OR REPLACE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            candle.symbol, candle.timeframe, candle.timestamp.isoformat(),
                            candle.open, candle.high, candle.low, candle.close, candle.volume
                        ))
                    saved += 1
                
                conn.commit()
                self.logger.debug(f"Saved {saved} candles to database")
                return saved
                
        except Exception as e:
            self.logger.error(f"Error saving candles: {e}")
            raise StorageError(f"Failed to save candles: {e}")

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Get candles from database.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start_time: Start time filter.
            end_time: End time filter.
            limit: Maximum number of candles to return.
            
        Returns:
            List of Candle objects.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                params = [symbol, timeframe]
                query = '''
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = ? AND timeframe = ?
                '''
                
                if self._is_postgres:
                    query = query.replace('?', '%s')
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time.isoformat() if not self._is_postgres else start_time)
                    
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time.isoformat() if not self._is_postgres else end_time)
                
                query += ' ORDER BY timestamp ASC'
                
                if limit:
                    query += f' LIMIT {limit}'
                
                if self._is_postgres:
                    query = query.replace('?', '%s')
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                candles = []
                for row in rows:
                    if self._is_postgres:
                        ts = row[2]
                    else:
                        ts_str = row['timestamp'] if isinstance(row, sqlite3.Row) else row[2]
                        ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                    
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    
                    candle = Candle(
                        timestamp=ts,
                        open=float(row[3] if not isinstance(row, sqlite3.Row) else row['open']),
                        high=float(row[4] if not isinstance(row, sqlite3.Row) else row['high']),
                        low=float(row[5] if not isinstance(row, sqlite3.Row) else row['low']),
                        close=float(row[6] if not isinstance(row, sqlite3.Row) else row['close']),
                        volume=float(row[7] if not isinstance(row, sqlite3.Row) else row['volume']),
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    candles.append(candle)
                
                return candles
                
        except Exception as e:
            self.logger.error(f"Error getting candles: {e}")
            raise StorageError(f"Failed to get candles: {e}")

    def save_trades(self, trades: List[Trade]) -> int:
        """
        Save trades to database.
        
        Args:
            trades: List of Trade objects to save.
            
        Returns:
            Number of saved records.
        """
        if not trades:
            return 0
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for trade in trades:
                    if self._is_postgres:
                        cursor.execute('''
                            INSERT INTO trades (symbol, timestamp, price, quantity, side)
                            VALUES (%s, %s, %s, %s, %s)
                        ''', (trade.symbol, trade.timestamp, trade.price, trade.quantity, trade.side))
                    else:
                        cursor.execute('''
                            INSERT INTO trades (symbol, timestamp, price, quantity, side)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (trade.symbol, trade.timestamp.isoformat(), trade.price, trade.quantity, trade.side))
                
                conn.commit()
                self.logger.debug(f"Saved {len(trades)} trades to database")
                return len(trades)
                
        except Exception as e:
            self.logger.error(f"Error saving trades: {e}")
            raise StorageError(f"Failed to save trades: {e}")

    def get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """
        Get trades from database.
        
        Args:
            symbol: Trading pair symbol.
            start_time: Start time filter.
            end_time: End time filter.
            limit: Maximum number of trades to return.
            
        Returns:
            List of Trade objects.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                params = [symbol]
                query = '''
                    SELECT symbol, timestamp, price, quantity, side
                    FROM trades
                    WHERE symbol = ?
                '''
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time.isoformat() if not self._is_postgres else start_time)
                    
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time.isoformat() if not self._is_postgres else end_time)
                
                query += ' ORDER BY timestamp ASC'
                
                if limit:
                    query += f' LIMIT {limit}'
                
                if self._is_postgres:
                    query = query.replace('?', '%s')
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    ts_val = row[1] if not isinstance(row, sqlite3.Row) else row['timestamp']
                    if isinstance(ts_val, str):
                        ts = datetime.fromisoformat(ts_val)
                    else:
                        ts = ts_val
                    
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    
                    trade = Trade(
                        timestamp=ts,
                        symbol=symbol,
                        price=float(row[2] if not isinstance(row, sqlite3.Row) else row['price']),
                        quantity=float(row[3] if not isinstance(row, sqlite3.Row) else row['quantity']),
                        side=row[4] if not isinstance(row, sqlite3.Row) else row['side']
                    )
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            raise StorageError(f"Failed to get trades: {e}")

    def save_funding(self, funding: FundingRate) -> None:
        """
        Save funding rate to history.
        
        Args:
            funding: FundingRate object to save.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if self._is_postgres:
                    cursor.execute('''
                        INSERT INTO funding_history (symbol, timestamp, rate)
                        VALUES (%s, %s, %s)
                    ''', (funding.symbol, funding.next_funding_time, funding.rate))
                else:
                    cursor.execute('''
                        INSERT INTO funding_history (symbol, timestamp, rate)
                        VALUES (?, ?, ?)
                    ''', (funding.symbol, funding.next_funding_time.isoformat(), funding.rate))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving funding: {e}")
            raise StorageError(f"Failed to save funding: {e}")

    def get_funding_history(
        self,
        symbol: str,
        days: int = 30
    ) -> List[FundingRate]:
        """
        Get funding rate history.
        
        Args:
            symbol: Trading pair symbol.
            days: Number of days of history.
            
        Returns:
            List of FundingRate objects.
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if self._is_postgres:
                    cursor.execute('''
                        SELECT symbol, timestamp, rate
                        FROM funding_history
                        WHERE symbol = %s AND timestamp >= %s
                        ORDER BY timestamp ASC
                    ''', (symbol, start_time))
                else:
                    cursor.execute('''
                        SELECT symbol, timestamp, rate
                        FROM funding_history
                        WHERE symbol = ? AND timestamp >= ?
                        ORDER BY timestamp ASC
                    ''', (symbol, start_time.isoformat()))
                
                rows = cursor.fetchall()
                
                funding_rates = []
                for row in rows:
                    ts_val = row[1] if not isinstance(row, sqlite3.Row) else row['timestamp']
                    if isinstance(ts_val, str):
                        ts = datetime.fromisoformat(ts_val)
                    else:
                        ts = ts_val
                    
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    
                    funding = FundingRate(
                        symbol=symbol,
                        rate=float(row[2] if not isinstance(row, sqlite3.Row) else row['rate']),
                        next_funding_time=ts
                    )
                    funding_rates.append(funding)
                
                return funding_rates
                
        except Exception as e:
            self.logger.error(f"Error getting funding history: {e}")
            raise StorageError(f"Failed to get funding history: {e}")

    def save_open_interest(self, oi: OpenInterest) -> None:
        """
        Save open interest to history.
        
        Args:
            oi: OpenInterest object to save.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if self._is_postgres:
                    cursor.execute('''
                        INSERT INTO oi_history (symbol, timestamp, value)
                        VALUES (%s, %s, %s)
                    ''', (oi.symbol, oi.timestamp, oi.value))
                else:
                    cursor.execute('''
                        INSERT INTO oi_history (symbol, timestamp, value)
                        VALUES (?, ?, ?)
                    ''', (oi.symbol, oi.timestamp.isoformat(), oi.value))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving open interest: {e}")
            raise StorageError(f"Failed to save open interest: {e}")

    def cache_set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set a value in Redis cache.
        
        Args:
            key: Cache key.
            value: Value to cache (will be JSON serialized).
            ttl: Time to live in seconds (default: 5 minutes).
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.use_redis or not self._redis:
            return False
            
        try:
            serialized = json.dumps(value, default=str)
            self._redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None if not found.
        """
        if not self.use_redis or not self._redis:
            return None
            
        try:
            data = self._redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            return None

    def cache_delete(self, key: str) -> bool:
        """
        Delete a value from Redis cache.
        
        Args:
            key: Cache key.
            
        Returns:
            True if deleted, False otherwise.
        """
        if not self.use_redis or not self._redis:
            return False
            
        try:
            self._redis.delete(key)
            return True
        except Exception as e:
            self.logger.warning(f"Cache delete error: {e}")
            return False

    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Delete data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to keep.
            
        Returns:
            Total number of deleted records.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        total_deleted = 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete old candles
                if self._is_postgres:
                    cursor.execute(
                        'DELETE FROM candles WHERE timestamp < %s',
                        (cutoff,)
                    )
                else:
                    cursor.execute(
                        'DELETE FROM candles WHERE timestamp < ?',
                        (cutoff.isoformat(),)
                    )
                total_deleted += cursor.rowcount
                
                # Delete old trades
                if self._is_postgres:
                    cursor.execute(
                        'DELETE FROM trades WHERE timestamp < %s',
                        (cutoff,)
                    )
                else:
                    cursor.execute(
                        'DELETE FROM trades WHERE timestamp < ?',
                        (cutoff.isoformat(),)
                    )
                total_deleted += cursor.rowcount
                
                # Delete old funding history
                if self._is_postgres:
                    cursor.execute(
                        'DELETE FROM funding_history WHERE timestamp < %s',
                        (cutoff,)
                    )
                else:
                    cursor.execute(
                        'DELETE FROM funding_history WHERE timestamp < ?',
                        (cutoff.isoformat(),)
                    )
                total_deleted += cursor.rowcount
                
                # Delete old OI history
                if self._is_postgres:
                    cursor.execute(
                        'DELETE FROM oi_history WHERE timestamp < %s',
                        (cutoff,)
                    )
                else:
                    cursor.execute(
                        'DELETE FROM oi_history WHERE timestamp < ?',
                        (cutoff.isoformat(),)
                    )
                total_deleted += cursor.rowcount
                
                conn.commit()
                
                # VACUUM for SQLite
                if not self._is_postgres:
                    conn.execute('VACUUM')
                
            self.logger.info(f"Cleaned up {total_deleted} old records")
            return total_deleted
            
        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")
            raise StorageError(f"Failed to cleanup data: {e}")

    def close(self) -> None:
        """Close all connections."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None

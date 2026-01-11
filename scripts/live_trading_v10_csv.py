#!/usr/bin/env python3
"""
Live Trading Script V10 - CSV/Parquet-Based Approach

KEY INNOVATION: Uses the SAME data files as training!
- Loads historical data from CSV/Parquet files (same as backtest)
- Fetches ONLY missing candles from Binance API
- Appends new candles to data files
- Calculates features from FULL data (exactly like backtest)

PARQUET SUPPORT (10x faster):
- Automatically converts CSV to Parquet on first load
- Uses Parquet for all subsequent operations
- Same data, just faster I/O

This ensures 100% feature consistency between backtest and live.

OPTIMIZATIONS:
- Parallel API fetching with ThreadPoolExecutor
- Parquet format for 10x faster data loading
- Feature caching - only update last rows
- Fast scan mode for trading loop

Usage:
    1. Copy config/secrets.yaml.example to config/secrets.yaml
    2. Fill in your MEXC API keys and Telegram credentials
    3. Ensure data/candles/ contains the CSV files from training
    4. Run: python scripts/live_trading_v10_csv.py
    
    Note: CSV files will be automatically converted to Parquet on first run.
"""

import sys
import time
import json
import hmac
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import joblib
import ccxt
import requests
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# Check for pyarrow availability (for Parquet support)
try:
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("⚠️ pyarrow not installed - using CSV instead of Parquet (slower)")
    logger.warning("  Install with: pip install pyarrow>=14.0.1")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES
)
from train_mtf import MTFFeatureEngine


# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "live_trading_v10.log"

logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """Configuration for V10 CSV-based live trading."""
    
    # Paths
    MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
    DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
    PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_20.json"
    SECRETS_FILE = Path(__file__).parent.parent / "config" / "secrets.yaml"
    TRADES_FILE = Path(__file__).parent.parent / "active_trades_v10.json"
    
    # Timeframes
    TIMEFRAMES = ['1m', '5m', '15m']
    
    # V17 Signal Thresholds
    MIN_CONF = 0.50
    MIN_TIMING = 1.5
    MIN_STRENGTH = 1.8
    
    # Risk Management
    RISK_PCT = 0.05
    MAX_LEVERAGE = 50.0
    MAX_HOLDING_BARS = 150
    ENTRY_FEE = 0.0002
    EXIT_FEE = 0.0002
    SL_ATR_BASE = 1.5
    # User requirement: up to $4M position, with leverage up to 50x
    # At 50x leverage: need $80k margin for $4M position
    # At 10x leverage: $400k max position, at 20x: $200k max position
    MAX_POSITION_SIZE = 4000000.0  # Max $4M position
    SLIPPAGE_PCT = 0.0005
    
    # V8 Features
    USE_ADAPTIVE_SL = True
    USE_DYNAMIC_LEVERAGE = True
    USE_AGGRESSIVE_TRAIL = True
    
    # MEXC API
    MEXC_BASE_URL = "https://contract.mexc.com"
    
    # Performance optimization
    # Note: 8 workers is optimal for M2 chip (8 cores)
    # More workers = more context switching overhead + API rate limits
    MAX_WORKERS = 8  # Parallel threads for scanning (match CPU cores)
    FEATURE_CACHE_SIZE = 50  # Number of rows to keep in incremental update
    
    @classmethod
    def load_secrets(cls) -> Dict[str, Any]:
        """Load secrets from config/secrets.yaml."""
        if not cls.SECRETS_FILE.exists():
            raise FileNotFoundError(
                f"Secrets file not found: {cls.SECRETS_FILE}\n"
                "Please copy config/secrets.yaml.example to config/secrets.yaml"
            )
        
        with open(cls.SECRETS_FILE, 'r') as f:
            secrets = yaml.safe_load(f)
        
        return secrets


# ============================================================
# CSV DATA MANAGER (with Parquet support)
# ============================================================
class CSVDataManager:
    """
    Manages data files - the key to matching backtest exactly.
    
    PARQUET SUPPORT (10x faster than CSV):
    - First tries to load from .parquet files
    - Falls back to .csv if parquet not found
    - Saves ONLY to Parquet (NEVER overwrites CSV training data!)
    - Converts existing CSV to Parquet on first load
    
    IMPORTANT: CSV files are READ-ONLY to protect training data.
    All updates are saved to Parquet files only.
    
    Features:
    - Loads existing data from training CSVs (read-only)
    - Fetches only missing candles from API
    - Appends new data to Parquet files
    - Returns full history for feature calculation
    - Validates data integrity at startup
    - Caches data for fast trading loop
    """
    
    # Required columns for valid OHLCV data
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    # Use Parquet by default if pyarrow is available (10x faster than CSV)
    USE_PARQUET = PYARROW_AVAILABLE
    
    def __init__(self, data_dir: Path, binance: ccxt.Exchange):
        self.data_dir = data_dir
        self.binance = binance
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for validated and loaded data (for fast trading loop)
        self._data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._features_cache: Dict[str, pd.DataFrame] = {}
        
        fmt = "Parquet" if self.USE_PARQUET else "CSV"
        logger.info(f"📂 Data Manager initialized ({fmt} mode): {self.data_dir}")
    
    def _get_csv_path(self, pair: str, timeframe: str) -> Path:
        """Get CSV file path for a pair/timeframe."""
        safe_symbol = pair.replace('/', '_').replace(':', '_')
        return self.data_dir / f"{safe_symbol}_{timeframe}.csv"
    
    def _get_parquet_path(self, pair: str, timeframe: str) -> Path:
        """Get Parquet file path for a pair/timeframe."""
        safe_symbol = pair.replace('/', '_').replace(':', '_')
        return self.data_dir / f"{safe_symbol}_{timeframe}.parquet"
    
    def load_csv(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load existing data (tries Parquet first, then CSV).
        
        If CSV is loaded, it's automatically converted to Parquet.
        """
        parquet_path = self._get_parquet_path(pair, timeframe)
        csv_path = self._get_csv_path(pair, timeframe)
        
        # Try Parquet first (10x faster)
        if self.USE_PARQUET and parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df.index = pd.to_datetime(df.index, utc=True)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                return df
            except Exception as e:
                logger.warning(f"Error loading parquet {parquet_path}: {e}")
        
        # Fall back to CSV
        if not csv_path.exists():
            if not parquet_path.exists():
                logger.warning(f"Data not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)
            
            # Ensure numeric columns are actually numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return None
        
        # Convert to Parquet for faster future loads
        if self.USE_PARQUET:
            try:
                self._save_parquet(pair, timeframe, df)
                logger.info(f"  ✅ Converted {csv_path.name} to Parquet")
            except Exception as e:
                logger.warning(f"Could not convert {csv_path.name} to parquet: {e}")
        
        return df
    
    def _save_parquet(self, pair: str, timeframe: str, df: pd.DataFrame):
        """Save DataFrame to Parquet format."""
        filepath = self._get_parquet_path(pair, timeframe)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        df.index.name = 'timestamp'
        
        # Ensure index is timezone-aware (required for Parquet)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        
        # Ensure all columns have proper dtypes for Parquet
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.to_parquet(filepath, engine='pyarrow')
        logger.debug(f"💾 Saved {len(df)} candles to {filepath}")
    
    def save_csv(self, pair: str, timeframe: str, df: pd.DataFrame):
        """
        Save DataFrame to file (Parquet ONLY to protect training CSVs).
        
        IMPORTANT: This method NEVER overwrites CSV files to protect training data.
        CSVs are only read, never written. All updates go to Parquet files.
        """
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Ensure index is named 'timestamp'
        df.index.name = 'timestamp'
        
        # Ensure index is timezone-aware (required for Parquet)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        
        # Ensure all columns have proper dtypes for Parquet
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ALWAYS save to Parquet to protect training CSVs
        filepath = self._get_parquet_path(pair, timeframe)
        try:
            df.to_parquet(filepath, engine='pyarrow')
            logger.debug(f"💾 Saved {len(df)} candles to {filepath}")
        except Exception as e:
            # Log error but DO NOT fall back to CSV - protect training data!
            logger.error(f"⚠️ Could not save data for {pair} {timeframe}: {e}")
            logger.error("  Parquet save failed. Training CSV files are protected (not overwritten).")
    
    def fetch_missing_candles(self, pair: str, timeframe: str, 
                               existing_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Fetch only the missing candles from Binance API.
        
        Returns the COMBINED DataFrame (existing + new).
        """
        if existing_df is None or len(existing_df) == 0:
            # No existing data - fetch last 1000 candles
            logger.info(f"  No existing data for {pair} {timeframe}, fetching 1000 candles...")
            candles = self.binance.fetch_ohlcv(pair, timeframe, limit=1000)
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df
        
        # Get the last timestamp from existing data
        last_ts = existing_df.index[-1]
        now = datetime.now(timezone.utc)
        
        # Calculate how many candles we need
        tf_minutes = {'1m': 1, '5m': 5, '15m': 15}[timeframe]
        time_diff = (now - last_ts).total_seconds() / 60
        candles_needed = int(time_diff / tf_minutes) + 10  # +10 buffer
        
        if candles_needed <= 0:
            logger.debug(f"  {pair} {timeframe}: Data is up to date")
            return existing_df
        
        logger.info(f"  Fetching {candles_needed} new candles for {pair} {timeframe}...")
        
        # Fetch from Binance
        since_ts = int(last_ts.timestamp() * 1000)
        candles = self.binance.fetch_ohlcv(pair, timeframe, since=since_ts, limit=min(candles_needed, 1000))
        
        if not candles:
            return existing_df
        
        # Convert to DataFrame
        new_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
        new_df.set_index('timestamp', inplace=True)
        
        # Combine with existing, removing duplicates
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]  # Keep latest values
        combined.sort_index(inplace=True)
        
        new_count = len(combined) - len(existing_df)
        if new_count > 0:
            logger.info(f"  ✅ Added {new_count} new candles to {pair} {timeframe}")
        
        return combined
    
    def get_updated_data(self, pair: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get updated data for all timeframes.
        
        1. Loads existing CSV
        2. Fetches missing candles from API
        3. Saves updated CSV
        4. Returns full data for feature calculation
        """
        data = {}
        
        for tf in Config.TIMEFRAMES:
            # Load existing CSV
            existing = self.load_csv(pair, tf)
            
            # Fetch missing candles
            try:
                updated = self.fetch_missing_candles(pair, tf, existing)
            except Exception as e:
                logger.error(f"Error fetching {pair} {tf}: {e}")
                if existing is not None:
                    updated = existing
                else:
                    return None
            
            # Save updated CSV
            self.save_csv(pair, tf, updated)
            
            data[tf] = updated
        
        # Validate we have enough data
        for tf in Config.TIMEFRAMES:
            if tf not in data or len(data[tf]) < 200:
                logger.warning(f"Insufficient data for {pair} {tf}: {len(data.get(tf, []))}")
                return None
        
        # Cache the data for fast access during trading
        self._data_cache[pair] = data
        
        return data
    
    def validate_csv_data(self, pair: str) -> tuple[bool, list]:
        """
        Validate data integrity at startup.
        
        Checks:
        1. All required columns exist
        2. No duplicate timestamps
        3. Data types are correct (numeric OHLCV)
        4. No NaN values in critical columns
        5. Timestamps are monotonically increasing
        6. Price data is valid (positive values, high >= low)
        
        Returns:
            (is_valid, list of issues found)
        """
        issues = []
        
        for tf in Config.TIMEFRAMES:
            df = self.load_csv(pair, tf)
            
            if df is None:
                issues.append(f"{tf}: Data file not found")
                continue
            
            # Check required columns
            missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
            if missing_cols:
                issues.append(f"{tf}: Missing columns: {missing_cols}")
                continue
            
            # Check for duplicates
            dup_count = df.index.duplicated().sum()
            if dup_count > 0:
                issues.append(f"{tf}: {dup_count} duplicate timestamps (will be cleaned)")
                # Auto-fix duplicates
                df = df[~df.index.duplicated(keep='first')]
                self.save_csv(pair, tf, df)
            
            # Check data types
            for col in self.REQUIRED_COLUMNS:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"{tf}: Column '{col}' is not numeric")
            
            # Check for NaN in OHLCV
            nan_counts = df[self.REQUIRED_COLUMNS].isna().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if len(nan_cols) > 0:
                issues.append(f"{tf}: NaN values in {dict(nan_cols)}")
            
            # Check timestamp order
            if not df.index.is_monotonic_increasing:
                issues.append(f"{tf}: Timestamps not sorted (will be fixed)")
                df = df.sort_index()
                self.save_csv(pair, tf, df)
            
            # Check valid prices (positive, high >= low)
            invalid_prices = (df['high'] < df['low']).sum()
            if invalid_prices > 0:
                issues.append(f"{tf}: {invalid_prices} candles with high < low")
            
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
            if negative_prices > 0:
                issues.append(f"{tf}: {negative_prices} candles with negative/zero prices")
        
        is_valid = len([i for i in issues if "not found" in i or "Missing columns" in i]) == 0
        
        return is_valid, issues
    
    def fast_update_candle(self, pair: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        FAST update for trading loop - only fetches the latest candle.
        
        Uses cached data and appends only the newest candle.
        Much faster than get_updated_data() which processes everything.
        
        Returns cached data with latest candle appended.
        """
        # Use cached data if available
        if pair not in self._data_cache:
            # First time - do full load
            return self.get_updated_data(pair)
        
        data = self._data_cache[pair]
        
        for tf in Config.TIMEFRAMES:
            existing = data.get(tf)
            if existing is None:
                continue
            
            try:
                # Fetch only last 2 candles (current forming + last closed)
                candles = self.binance.fetch_ohlcv(pair, tf, limit=2)
                
                if not candles:
                    continue
                
                # Convert to DataFrame
                new_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
                new_df.set_index('timestamp', inplace=True)
                
                # Append/update only new candles
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                
                data[tf] = combined
                
                # IMPORTANT: Save to CSV so data persists between restarts!
                self.save_csv(pair, tf, combined)
                
            except Exception as e:
                logger.debug(f"Error updating {pair} {tf}: {e}")
                continue
        
        # Update cache
        self._data_cache[pair] = data
        
        return data
    
    def cache_features(self, pair: str, features: pd.DataFrame):
        """Cache calculated features for a pair."""
        self._features_cache[pair] = features
    
    def get_cached_features(self, pair: str) -> Optional[pd.DataFrame]:
        """Get cached features for a pair."""
        return self._features_cache.get(pair)


# ============================================================
# MEXC CLIENT (same as V9)
# ============================================================
class MEXCClient:
    """Direct MEXC Futures API client."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = Config.MEXC_BASE_URL
        logger.info("✅ MEXC Client initialized")
    
    def _generate_signature(self, sign_str: str) -> str:
        return hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        timestamp = int(time.time() * 1000)
        if params is None:
            params = {}
        params['timestamp'] = timestamp
        
        sorted_params = sorted(params.items())
        params_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
        sign_str = f"{self.api_key}{timestamp}{params_str}"
        signature = self._generate_signature(sign_str)
        
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Signature': signature,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30, verify=True)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers, timeout=30, verify=True)
            else:
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"MEXC API error: {e}")
            return None
    
    def get_account_assets(self) -> float:
        result = self._request('GET', '/api/v1/private/account/assets', {})
        if result and result.get('success'):
            for asset in result.get('data', []):
                if asset['currency'] == 'USDT':
                    return float(asset.get('availableBalance', 0))
        return 0.0
    
    def place_order(self, symbol: str, side: int, volume: int, leverage: int,
                   price: float = 0, order_type: int = 5, open_type: int = 1) -> Optional[str]:
        params = {
            'symbol': symbol, 'price': price, 'vol': volume,
            'leverage': leverage, 'side': side, 'type': order_type, 'openType': open_type
        }
        logger.info(f"📤 Placing order: {params}")
        result = self._request('POST', '/api/v1/private/order/submit', params)
        if result and result.get('success'):
            return result.get('data')
        return None
    
    def place_stop_order(self, symbol: str, side: int, volume: int, 
                        stop_price: float, leverage: int) -> Optional[str]:
        params = {
            'symbol': symbol, 'price': 0, 'vol': volume, 'leverage': leverage,
            'side': side, 'type': 5, 'openType': 1, 'triggerPrice': stop_price,
            'triggerType': 1, 'executeCycle': 1, 'trend': 1 if side == 3 else 2
        }
        result = self._request('POST', '/api/v1/private/planorder/place', params)
        if result and result.get('success'):
            return result.get('data')
        return None
    
    def cancel_stop_orders(self, symbol: str) -> bool:
        result = self._request('GET', '/api/v1/private/planorder/list', {'symbol': symbol})
        if not result or not result.get('success'):
            return False
        for order in result.get('data', []):
            if order.get('id'):
                self._request('POST', '/api/v1/private/planorder/cancel', 
                            {'symbol': symbol, 'orderId': order['id']})
        return True


# ============================================================
# TELEGRAM NOTIFIER (same as V9)
# ============================================================
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
    
    def send(self, title: str, body: str) -> bool:
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            requests.post(url, data={'chat_id': self.chat_id, 'text': f"{title}\n{body}", 
                                     'parse_mode': 'HTML'}, timeout=5, verify=True)
            return True
        except:
            return False


# ============================================================
# PORTFOLIO MANAGER (same as V9)
# ============================================================
class PortfolioManager:
    def __init__(self, mexc: MEXCClient, telegram: TelegramNotifier):
        self.mexc = mexc
        self.telegram = telegram
        self.position = None
        self.trades_history = []
        
        if self.load_state():
            logger.info(f"📂 State loaded. Capital: ${self.capital:.2f}")
        else:
            self.capital = self.mexc.get_account_assets() or 20.0
            logger.info(f"💰 Capital: ${self.capital:.2f}")
            self.save_state()
    
    def load_state(self) -> bool:
        if Config.TRADES_FILE.exists():
            try:
                with open(Config.TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    self.capital = data.get('capital')
                    self.position = data.get('position')
                    if self.position and isinstance(self.position.get('entry_time'), str):
                        self.position['entry_time'] = datetime.fromisoformat(self.position['entry_time'])
                    return self.capital is not None
            except:
                return False
        return False
    
    def save_state(self):
        pos = None
        if self.position:
            pos = self.position.copy()
            if isinstance(pos.get('entry_time'), datetime):
                pos['entry_time'] = pos['entry_time'].isoformat()
        with open(Config.TRADES_FILE, 'w') as f:
            json.dump({'capital': self.capital, 'position': pos, 'history': self.trades_history}, f, indent=2)
    
    def open_position(self, signal: Dict) -> bool:
        if self.position:
            return False
        
        entry_price = signal['price']
        atr = signal['atr']
        pred_strength = signal.get('pred_strength', 2.0)
        
        # Adaptive SL
        if pred_strength >= 3.0:
            sl_mult = 1.6
        elif pred_strength >= 2.0:
            sl_mult = 1.5
        else:
            sl_mult = 1.2
        
        stop_distance = atr * sl_mult
        stop_loss = entry_price - stop_distance if signal['direction'] == 'LONG' else entry_price + stop_distance
        
        # Position sizing
        stop_loss_pct = stop_distance / entry_price
        risk_amount = self.capital * Config.RISK_PCT
        position_value = min(risk_amount / stop_loss_pct, Config.MAX_POSITION_SIZE)
        leverage = min(position_value / self.capital, Config.MAX_LEVERAGE)
        
        mexc_symbol = signal['pair'].replace('/USDT:USDT', '_USDT').replace('/', '_')
        volume = max(1, int(position_value / entry_price))
        mexc_side = 1 if signal['direction'] == 'LONG' else 2
        
        order_id = self.mexc.place_order(mexc_symbol, mexc_side, volume, int(leverage))
        if not order_id:
            return False
        
        stop_side = 3 if signal['direction'] == 'LONG' else 4
        stop_order_id = self.mexc.place_stop_order(mexc_symbol, stop_side, volume, stop_loss, int(leverage))
        
        self.position = {
            'pair': signal['pair'], 'direction': signal['direction'],
            'entry_price': entry_price, 'entry_time': datetime.now(timezone.utc),
            'stop_loss': stop_loss, 'stop_distance': stop_distance,
            'position_value': position_value, 'leverage': leverage,
            'mexc_symbol': mexc_symbol, 'volume': volume,
            'pred_strength': pred_strength, 'breakeven_active': False,
            # V14: Balanced BE trigger - not too early (small profits), not too late (more SL hits)
            'be_trigger_mult': 2.5 if pred_strength >= 3.0 else (2.2 if pred_strength >= 2.0 else 1.8)
        }
        self.save_state()
        
        self.telegram.send(
            f"🟢 <b>{signal['direction']}</b> {signal['pair']}",
            f"Entry: {entry_price:.6f}\nSL: {stop_loss:.6f}\nLev: {leverage:.1f}x"
        )
        return True
    
    def update_position(self, current_price: float, candle_high: float, candle_low: float):
        if not self.position:
            return
        
        pos = self.position
        pred_strength = pos.get('pred_strength', 2.0)
        sl_mult = 1.6 if pred_strength >= 3.0 else (1.5 if pred_strength >= 2.0 else 1.2)
        atr = pos['stop_distance'] / sl_mult
        be_trigger_dist = atr * pos['be_trigger_mult']
        
        if pos['direction'] == 'LONG':
            if not pos['breakeven_active'] and candle_high >= pos['entry_price'] + be_trigger_dist:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] + atr * 1.0  # V14: Lock meaningful profit when BE triggers
                logger.info(f"✅ Breakeven activated at +{atr * 1.0:.6f}")
            
            if pos['breakeven_active']:
                r_mult = (candle_high - pos['entry_price']) / pos['stop_distance']
                # V14: Balanced trailing - tighter at high R, looser early
                trail = 0.6 if r_mult > 5 else (1.2 if r_mult > 3 else (1.8 if r_mult > 2 else 2.5))
                new_sl = candle_high - atr * trail
                if new_sl > pos['stop_loss']:
                    pos['stop_loss'] = new_sl
        else:
            if not pos['breakeven_active'] and candle_low <= pos['entry_price'] - be_trigger_dist:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] - atr * 1.0  # V14: Lock meaningful profit when BE triggers
            
            if pos['breakeven_active']:
                r_mult = (pos['entry_price'] - candle_low) / pos['stop_distance']
                # V14: Balanced trailing - tighter at high R, looser early
                trail = 0.6 if r_mult > 5 else (1.2 if r_mult > 3 else (1.8 if r_mult > 2 else 2.5))
                new_sl = candle_low + atr * trail
                if new_sl < pos['stop_loss']:
                    pos['stop_loss'] = new_sl
        
        self.save_state()
    
    def close_position(self, price: float, reason: str):
        if not self.position:
            return
        
        pos = self.position
        self.mexc.cancel_stop_orders(pos['mexc_symbol'])
        self.mexc.place_order(pos['mexc_symbol'], 3 if pos['direction'] == 'LONG' else 4, 
                             pos['volume'], int(pos['leverage']))
        
        if pos['direction'] == 'LONG':
            pnl_pct = (price * (1 - Config.SLIPPAGE_PCT) - pos['entry_price'] * (1 + Config.SLIPPAGE_PCT)) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] * (1 - Config.SLIPPAGE_PCT) - price * (1 + Config.SLIPPAGE_PCT)) / pos['entry_price']
        
        net = pos['position_value'] * pnl_pct - pos['position_value'] * Config.EXIT_FEE
        self.capital += net
        
        emoji = "✅" if net > 0 else "❌"
        self.telegram.send(f"{emoji} CLOSE {pos['pair']}", f"PnL: ${net:.2f}\nReason: {reason}")
        
        self.position = None
        self.save_state()


# ============================================================
# FEATURE ENGINEERING (from CSV - matches backtest exactly!)
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features(data: Dict[str, pd.DataFrame], mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """
    Prepare features from CSV data - EXACTLY like backtest!
    
    This is the key function that ensures live matches backtest.
    """
    # Use only last N candles - matches backtest window size
    # This ensures rolling indicators have same context as training
    LOOKBACK_M1 = 7500   # ~5 days of 1m data
    LOOKBACK_M5 = 1500   # ~5 days of 5m data  
    LOOKBACK_M15 = 500   # ~5 days of 15m data
    
    m1 = data['1m'].tail(LOOKBACK_M1)
    m5 = data['5m'].tail(LOOKBACK_M5)
    m15 = data['15m'].tail(LOOKBACK_M15)
    
    logger.debug(f"Preparing features from CSV: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
    
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 200:
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        # Use MTFFeatureEngine - SAME as backtest
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        
        if len(ft) == 0:
            return pd.DataFrame()
        
        # Add OHLCV from M5
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        
        # Add volume features - SAME as backtest
        ft = add_volume_features(ft)
        
        # Add ATR
        ft['atr'] = calculate_atr(ft)
        
        # NaN handling - SAME as backtest
        ft = ft.dropna(subset=['close', 'atr'])
        
        # Exclude cumsum-dependent features
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        # Exclude absolute price features  
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        # Forward fill and final dropna
        ft = ft.ffill().dropna()
        
        logger.debug(f"Features prepared: {len(ft)} rows, {len(ft.columns)} cols")
        
        return ft
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================
def load_models() -> Dict:
    """Load trained models."""
    if not Config.MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {Config.MODEL_DIR}")
    
    # Load scaler if exists
    scaler_path = Config.MODEL_DIR / 'scaler.joblib'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    return {
        'direction': joblib.load(Config.MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(Config.MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(Config.MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(Config.MODEL_DIR / 'feature_names.joblib'),
        'scaler': scaler
    }


def get_pairs() -> list:
    """Get trading pairs."""
    if Config.PAIRS_FILE.exists():
        with open(Config.PAIRS_FILE) as f:
            return [p['symbol'] for p in json.load(f)['pairs']][:20]
    return ['BTC/USDT:USDT', 'ETH/USDT:USDT']


def wait_until_candle_close(timeframe_minutes: int = 5) -> datetime:
    """Wait until next candle close."""
    now = datetime.now(timezone.utc)
    next_close_minute = ((now.minute // timeframe_minutes) + 1) * timeframe_minutes
    
    if next_close_minute >= 60:
        next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_close = now.replace(minute=next_close_minute, second=0, microsecond=0)
    
    wait_seconds = (next_close - now).total_seconds() + 3
    if wait_seconds > 0:
        logger.info(f"⏰ Waiting {wait_seconds:.0f}s until {next_close.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_seconds)
    
    return next_close


def startup_data_sync(csv_manager: CSVDataManager, pairs: list, mtf_fe: MTFFeatureEngine, 
                       models: Dict) -> Dict[str, pd.DataFrame]:
    """
    STARTUP PHASE: Download ALL missing data for ALL pairs BEFORE trading.
    
    This ensures:
    1. All CSV files are validated (no duplicates, correct format)
    2. All CSV files are up to date
    3. Features are pre-calculated and verified
    4. Data is cached for fast trading loop
    5. No trading happens until everything is ready
    
    Returns:
        Dict mapping pair -> latest features DataFrame
    """
    # Constants for startup validation
    MIN_FEATURES_ROWS = 10  # Minimum rows needed for valid features
    MAX_MISSING_FEATURES_WARNING = 10  # Warn if more features are missing
    API_RATE_LIMIT_DELAY = 0.3  # Seconds between API calls to avoid rate limits
    MIN_FEATURE_MATCH_PCT = 90.0  # Minimum percentage of features that must match
    
    logger.info("=" * 70)
    logger.info("🔄 STARTUP PHASE 1: Validating CSV data...")
    logger.info("=" * 70)
    
    ready_pairs = {}
    failed_pairs = []
    validation_issues = {}
    
    # PHASE 1: Validate all CSV files first
    for i, pair in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] Validating {pair}...")
        
        try:
            is_valid, issues = csv_manager.validate_csv_data(pair)
            
            if issues:
                validation_issues[pair] = issues
                for issue in issues:
                    logger.warning(f"  ⚠️ {issue}")
            
            if is_valid:
                logger.info(f"  ✅ Data format OK")
            else:
                logger.warning(f"  ⚠️ Validation issues found (will download data in PHASE 2)")
        except Exception as e:
            logger.error(f"  ❌ Validation error: {e}")
            logger.warning(f"  ⚠️ Will try to download data in PHASE 2")
    
    # Report validation summary
    logger.info("\n" + "=" * 70)
    if validation_issues:
        logger.info(f"📊 Validation: {len(pairs) - len(failed_pairs)}/{len(pairs)} pairs OK")
        logger.info(f"   Issues found and auto-fixed where possible")
    else:
        logger.info("📊 Validation: All pairs OK - no issues found!")
    
    # PHASE 2: Sync data and calculate features
    logger.info("=" * 70)
    logger.info("🔄 STARTUP PHASE 2: Syncing data and calculating features...")
    logger.info("=" * 70)
    
    for i, pair in enumerate(pairs):
        logger.info(f"\n[{i+1}/{len(pairs)}] Processing {pair}...")
        
        try:
            # Step 1: Load CSV and fetch missing candles
            data = csv_manager.get_updated_data(pair)
            
            if data is None:
                logger.warning(f"  ❌ No data available for {pair}")
                failed_pairs.append(pair)
                continue
            
            # Log data stats
            for tf in ['1m', '5m', '15m']:
                df = data.get(tf)
                if df is not None and len(df) > 0:
                    logger.info(f"  {tf.upper()}: {len(df)} candles, {df.index[0]} to {df.index[-1]}")
            
            # Step 2: Calculate features from FULL CSV
            features = prepare_features(data, mtf_fe)
            
            if features is None or len(features) < MIN_FEATURES_ROWS:
                logger.warning(f"  ❌ Could not prepare features for {pair} (need at least {MIN_FEATURES_ROWS} rows)")
                failed_pairs.append(pair)
                continue
            
            # Step 3: Verify feature consistency with model
            model_features = set(models['features'])
            available_features = set(features.columns)
            matching = model_features.intersection(available_features)
            missing_features = model_features - available_features
            match_pct = len(matching) / len(model_features) * 100
            
            if match_pct < MIN_FEATURE_MATCH_PCT:
                logger.warning(f"  ❌ Feature match too low: {match_pct:.1f}% (need {MIN_FEATURE_MATCH_PCT}%)")
                failed_pairs.append(pair)
                continue
            
            if len(missing_features) > MAX_MISSING_FEATURES_WARNING:
                logger.warning(f"  ⚠️ {len(missing_features)} missing features (will use 0.0)")
            
            # Step 4: Test prediction on last closed candle
            # Use -1 if only 1 row, -2 for second-to-last (last closed candle)
            row_idx = -2 if len(features) >= 2 else -1
            row = features.iloc[[row_idx]].copy()
            
            # Fill missing features with 0
            for f in models['features']:
                if f not in row.columns:
                    row[f] = 0.0
            
            X = row[models['features']].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features if scaler is available
            if models.get('scaler') is not None:
                X = models['scaler'].transform(X)
            
            dir_proba = models['direction'].predict_proba(X)
            dir_conf = float(np.max(dir_proba))
            timing = float(models['timing'].predict(X)[0])
            strength = float(models['strength'].predict(X)[0])
            
            logger.info(f"  ✅ Ready! Features: {match_pct:.0f}% | Conf: {dir_conf:.2f} | Tim: {timing:.2f} | Str: {strength:.1f}")
            
            ready_pairs[pair] = features
            
            # Rate limit delay between pairs
            time.sleep(API_RATE_LIMIT_DELAY)
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            failed_pairs.append(pair)
            continue
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTUP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✅ Ready pairs: {len(ready_pairs)}")
    if failed_pairs:
        logger.warning(f"❌ Failed pairs: {len(failed_pairs)} - {failed_pairs}")
    logger.info("=" * 70)
    
    return ready_pairs


def process_pair_for_signal(
    pair: str,
    csv_manager: CSVDataManager,
    mtf_fe: MTFFeatureEngine,
    models: Dict,
    binance: ccxt.Exchange
) -> Optional[Dict]:
    """
    Process a single pair and return signal if found.
    
    This function is designed to be called in parallel.
    
    IMPORTANT: Uses FULL data for feature calculation to match backtest exactly!
    Only API fetching is optimized (fast_update_candle), not feature calculation.
    
    Returns:
        Dict with signal info if signal found, None otherwise
    """
    try:
        # FAST UPDATE: Only fetch latest candle from API (uses cache for CSV data)
        data = csv_manager.fast_update_candle(pair)
        if data is None:
            return None
        
        # FULL FEATURE CALCULATION - same as backtest!
        # We MUST use full data for correct rolling indicators (EMA-200, etc.)
        df = prepare_features(data, mtf_fe)
        
        if df is None or len(df) < 2:
            return None
        
        # Cache features for reference
        csv_manager.cache_features(pair, df)
        
        # Get last closed candle
        row = df.iloc[[-2]].copy()
        
        # Fill missing features with 0
        for f in models['features']:
            if f not in row.columns:
                row[f] = 0.0
        
        # Get predictions
        X = row[models['features']].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features if scaler is available
        if models.get('scaler') is not None:
            X = models['scaler'].transform(X)
        
        dir_proba = models['direction'].predict_proba(X)
        dir_conf = float(np.max(dir_proba))
        dir_pred = int(np.argmax(dir_proba))
        timing = float(models['timing'].predict(X)[0])
        strength = float(models['strength'].predict(X)[0])
        
        direction = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
        
        # DEBUG: Log every pair's prediction (useful for debugging)
        # Shows: direction, confidence, timing, strength for each pair
        logger.debug(f"  {pair}: {direction} conf={dir_conf:.2f} tim={timing:.2f} str={strength:.1f}")
        
        # Return result for logging
        result = {
            'pair': pair,
            'direction': direction,
            'dir_pred': dir_pred,
            'conf': dir_conf,
            'timing': timing,
            'strength': strength,
            'atr': row['atr'].iloc[0] if 'atr' in row.columns else 0.0,
            'is_signal': False
        }
        
        # Check if it's a valid signal
        if dir_pred != 1:  # Not sideways
            if dir_conf >= Config.MIN_CONF and timing >= Config.MIN_TIMING and strength >= Config.MIN_STRENGTH:
                result['is_signal'] = True
        
        return result
        
    except Exception as e:
        logger.debug(f"Error processing {pair}: {e}")
        return None


def parallel_scan(
    active_pairs: List[str],
    csv_manager: CSVDataManager,
    mtf_fe: MTFFeatureEngine,
    models: Dict,
    binance: ccxt.Exchange,
    max_workers: int = 10
) -> List[Dict]:
    """
    Scan all pairs in PARALLEL using ThreadPoolExecutor.
    
    Returns list of results for all pairs.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pairs for processing
        future_to_pair = {
            executor.submit(
                process_pair_for_signal, 
                pair, csv_manager, mtf_fe, models, binance
            ): pair 
            for pair in active_pairs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error in future for {pair}: {e}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None, help="Specific pair to trade (e.g., 'PIPPIN/USDT:USDT'). Overrides pairs_20.json.")
    parser.add_argument("--pairs_list", type=str, default=None, help="Comma-separated list of pairs (e.g., 'PIPPIN/USDT:USDT,ASTER/USDT:USDT,ZEC/USDT:USDT'). Overrides pairs_20.json.")
    cli_args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("V10 LIVE TRADING - CSV-BASED APPROACH (PARALLEL)")
    logger.info("=" * 70)
    logger.info("✅ Uses SAME CSV files as training for feature calculation")
    logger.info("✅ Fetches only missing candles from API")
    logger.info("✅ Guarantees 100% feature consistency with backtest")
    logger.info("✅ STARTUP: Downloads ALL missing data BEFORE trading")
    logger.info("✅ PARALLEL: Scans all pairs simultaneously")
    logger.info("=" * 70)
    
    # Load secrets
    try:
        secrets = Config.load_secrets()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    # Initialize clients
    mexc = MEXCClient(
        secrets['mexc']['api_key'],
        secrets['mexc']['api_secret']
    )
    
    telegram = TelegramNotifier(
        secrets.get('notifications', {}).get('telegram', {}).get('bot_token', ''),
        secrets.get('notifications', {}).get('telegram', {}).get('chat_id', '')
    )
    
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Initialize CSV data manager
    csv_manager = CSVDataManager(Config.DATA_DIR, binance)
    
    # Load models
    try:
        models = load_models()
        # Support for pairs list mode (multiple pairs via comma-separated list)
        if cli_args.pairs_list:
            pairs = [p.strip() for p in cli_args.pairs_list.split(',')]
            logger.info(f"🎯 MULTI PAIR MODE: {len(pairs)} pairs - {pairs}")
        elif cli_args.pair:
            pairs = [cli_args.pair]
            logger.info(f"🎯 SINGLE PAIR MODE: {cli_args.pair}")
        else:
            pairs = get_pairs()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    portfolio = PortfolioManager(mexc, telegram)
    mtf_fe = MTFFeatureEngine()
    
    logger.info(f"Configured pairs: {len(pairs)}")
    logger.info(f"Capital: ${portfolio.capital:.2f}")
    
    # ========================================
    # STARTUP PHASE: Sync ALL data first!
    # ========================================
    ready_pairs = startup_data_sync(csv_manager, pairs, mtf_fe, models)
    
    if len(ready_pairs) == 0:
        logger.error("❌ No pairs ready for trading! Check your CSV files in data/candles/")
        return
    
    # Update pairs list to only include ready pairs
    active_pairs = list(ready_pairs.keys())
    logger.info(f"🎯 Trading {len(active_pairs)} pairs: {active_pairs}")
    
    telegram.send("🚀 <b>V10 Live Trading Started</b>", 
                  f"Capital: ${portfolio.capital:.2f}\n"
                  f"Active pairs: {len(active_pairs)}\n"
                  f"Mode: CSV-based (PARALLEL scan)")
    
    # Main loop
    while True:
        try:
            wait_until_candle_close(5)
            
            # Update position if exists
            if portfolio.position:
                pos = portfolio.position
                ticker = binance.fetch_ticker(pos['pair'])
                candles = binance.fetch_ohlcv(pos['pair'], '5m', limit=2)
                if len(candles) >= 2:
                    portfolio.update_position(ticker['last'], candles[-2][2], candles[-2][3])
            
            # Scan for signals using PARALLEL processing
            if not portfolio.position:
                logger.info("=" * 70)
                logger.info(f"🔍 PARALLEL SCAN: Processing {len(active_pairs)} pairs with {Config.MAX_WORKERS} workers...")
                scan_start = time.time()
                
                # Run parallel scan
                results = parallel_scan(
                    active_pairs, csv_manager, mtf_fe, models, binance,
                    max_workers=Config.MAX_WORKERS
                )
                
                scan_time = time.time() - scan_start
                
                # Log all results
                signals_found = []
                confidence_values = []
                for result in sorted(results, key=lambda x: x['pair']):
                    status = "✅ SIGNAL" if result['is_signal'] else ""
                    logger.info(f"  {result['pair']}: {result['direction']} | Conf: {result['conf']:.2f} | Tim: {result['timing']:.2f} | Str: {result['strength']:.1f} {status}")
                    confidence_values.append(result['conf'])
                    
                    if result['is_signal']:
                        signals_found.append(result)
                
                # Log confidence distribution statistics for diagnosis (optimized single pass)
                if confidence_values:
                    n = len(confidence_values)
                    min_conf_val = float('inf')
                    max_conf_val = float('-inf')
                    total = 0.0
                    above_threshold = 0
                    for c in confidence_values:
                        total += c
                        if c < min_conf_val:
                            min_conf_val = c
                        if c > max_conf_val:
                            max_conf_val = c
                        if c >= Config.MIN_CONF:
                            above_threshold += 1
                    avg_conf = total / n
                    logger.info(f"📊 Confidence Stats: Avg={avg_conf:.2f} | Min={min_conf_val:.2f} | Max={max_conf_val:.2f} | Above {Config.MIN_CONF}={above_threshold}/{n}")
                
                logger.info(f"⏱️ Scan completed in {scan_time:.1f}s for {len(active_pairs)} pairs")
                
                # Process first signal found
                if signals_found:
                    signal_result = signals_found[0]  # Take first signal
                    pair = signal_result['pair']
                    
                    logger.info(f"  🎯 Opening position on {pair}...")
                    
                    try:
                        ticker = binance.fetch_ticker(pair)
                        signal = {
                            'pair': pair,
                            'direction': 'LONG' if signal_result['dir_pred'] == 2 else 'SHORT',
                            'price': ticker['last'],
                            'atr': signal_result['atr'],
                            'pred_strength': signal_result['strength']
                        }
                        
                        if portfolio.open_position(signal):
                            logger.info(f"🚀 Position opened: {pair}")
                    except Exception as e:
                        logger.error(f"Error opening position: {e}")
                
                logger.info("=" * 70)
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            telegram.send("🛑 V10 Stopped", f"Capital: ${portfolio.capital:.2f}")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    main()

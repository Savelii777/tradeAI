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
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
from catboost import CatBoostClassifier

# MEXC WEB API requires curl_cffi for TLS fingerprinting bypass
try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False
    logger.warning("⚠️ curl_cffi not installed - MEXC WEB API won't work")
    logger.warning("  Install with: pip install curl-cffi")

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
# ENSEMBLE MODEL CLASS (needed for loading trained model)
# ============================================================
class EnsembleDirectionModel(ClassifierMixin, BaseEstimator):
    """
    Ensemble of LightGBM + CatBoost for more stable predictions.
    Required for unpickling the trained model.
    """
    _estimator_type = "classifier"
    
    def __init__(self, lgb_params=None, use_catboost=True):
        self.lgb_params = lgb_params
        self.use_catboost = use_catboost
        self.lgb_model = None
        self.catboost_model = None
        self.weights = {'lgb': 0.5, 'catboost': 0.5}
        self._classes = np.array([0, 1, 2])
    
    @property
    def classes_(self):
        return self._classes
        
    def fit(self, X, y, sample_weight=None):
        lgb_default = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 50,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1
        }
        lgb_default.update(self.lgb_params or {})
        
        self.lgb_model = lgb.LGBMClassifier(**lgb_default)
        self.lgb_model.fit(X, y, sample_weight=sample_weight)
        
        if self.use_catboost:
            self.catboost_model = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.03,
                l2_leaf_reg=3, loss_function='MultiClass',
                classes_count=3, random_seed=42, verbose=False
            )
            self.catboost_model.fit(X, y, sample_weight=sample_weight)
        
        self._classes = np.unique(y)
        return self
    
    def predict_proba(self, X):
        lgb_proba = self.lgb_model.predict_proba(X)
        if self.use_catboost and self.catboost_model is not None:
            cat_proba = self.catboost_model.predict_proba(X)
            return self.weights['lgb'] * lgb_proba + self.weights['catboost'] * cat_proba
        return lgb_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    @property
    def feature_importances_(self):
        if self.lgb_model is not None:
            return self.lgb_model.feature_importances_
        return None


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
    
    # Signal Thresholds (V13 - higher quality)
    MIN_CONF = 0.65  # V13: stricter confidence threshold
    MIN_TIMING = 1.8
    MIN_STRENGTH = 2.5
    CONFIDENCE_BOOST_THRESHOLD = 0.75  # V13: boost risk above this confidence
    
    # Risk Management (V13 - dynamic sizing)
    BASE_RISK_PCT = 0.05  # V13: base risk 5%
    MAX_RISK_PCT = 0.07   # V13: max risk 7% for high confidence
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
    
    # V13 Filters
    USE_REGIME_FILTER = True  # V13: skip low volatility regimes
    USE_MTF_CONFIRMATION = True  # V13: require multi-timeframe confirmation
    
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
# V13 HELPER FUNCTIONS
# ============================================================
def calculate_dynamic_risk(confidence: float, timing: float) -> float:
    """
    V13: Dynamic risk sizing based on signal quality.
    
    Base: 5% risk
    High confidence (>0.75) + strong timing (>2.5): 7% risk
    """
    if confidence >= Config.CONFIDENCE_BOOST_THRESHOLD and timing >= 2.5:
        return Config.MAX_RISK_PCT  # 7%
    elif confidence >= Config.CONFIDENCE_BOOST_THRESHOLD:
        return 0.06  # 6%
    else:
        return Config.BASE_RISK_PCT  # 5%


def check_regime_filter(df: pd.DataFrame) -> bool:
    """
    V13: Filter out low volatility regimes.
    
    Returns True if regime is tradeable, False to skip.
    """
    if not Config.USE_REGIME_FILTER:
        return True
    
    if 'atr' not in df.columns or len(df) < 20:
        return True
    
    # Get recent ATR values
    recent_atr = df['atr'].tail(20)
    current_atr = recent_atr.iloc[-1]
    avg_atr = recent_atr.mean()
    
    # Skip if ATR too low (< 50% of average)
    if current_atr < avg_atr * 0.5:
        return False
    
    return True


def check_mtf_confirmation(data: Dict[str, pd.DataFrame], direction: str) -> bool:
    """
    V13: Multi-timeframe trend confirmation.
    
    Checks if M5 and M15 trends align with M1 signal direction.
    """
    if not Config.USE_MTF_CONFIRMATION:
        return True
    
    try:
        # Get recent closes
        m5_close = data['5m']['close'].tail(20)
        m15_close = data['15m']['close'].tail(10)
        
        # Calculate simple trends (current vs average)
        m5_trend = m5_close.iloc[-1] > m5_close.mean()
        m15_trend = m15_close.iloc[-1] > m15_close.mean()
        
        if direction == 'LONG':
            return m5_trend and m15_trend
        elif direction == 'SHORT':
            return (not m5_trend) and (not m15_trend)
        
        return True
    except:
        return True  # On error, don't filter


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
            
            # Retry logic for API errors
            candles = None
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Fetch only last 2 candles (current forming + last closed)
                    candles = self.binance.fetch_ohlcv(pair, tf, limit=2)
                    break  # Success
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.debug(f"Retry {retry+1}/{max_retries} for {pair} {tf}: {e}")
                        time.sleep(0.5)
                    else:
                        logger.warning(f"Failed to update {pair} {tf} after {max_retries} retries")
            
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
# MEXC WEB API CLIENT (Bypass maintenance mode)
# Uses internal WEB API instead of public OpenAPI
# Based on: https://github.com/biberhund/MEXC_Future_Order_API_Maintenance_Bypass
# ============================================================
class MEXCClient:
    """MEXC Futures WEB API client - bypasses maintenance mode."""
    
    def __init__(self, web_cookie: str):
        """
        Initialize MEXC WEB API client.
        
        Args:
            web_cookie: Browser cookie (u_id starting with 'WEB...')
                       Get from browser DevTools > Application > Cookies > futures.mexc.com
        """
        self.web_cookie = web_cookie
        self.base_url = "https://futures.mexc.com"
        
        if not CURL_CFFI_AVAILABLE:
            raise ImportError("curl_cffi is required for MEXC WEB API. Install with: pip install curl-cffi")
        
        logger.info("✅ MEXC WEB API Client initialized")
    
    def _md5(self, value: str) -> str:
        """Calculate MD5 hash."""
        return hashlib.md5(value.encode('utf-8')).hexdigest()
    
    def _generate_signature(self, obj: dict) -> dict:
        """Generate MEXC WEB API signature."""
        date_now = str(int(time.time() * 1000))
        g = self._md5(self.web_cookie + date_now)[7:]
        s = json.dumps(obj, separators=(',', ':'))
        sign = self._md5(date_now + s + g)
        return {'time': date_now, 'sign': sign}
    
    def _request(self, method: str, endpoint: str, obj: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to MEXC WEB API."""
        if obj is None:
            obj = {}
        
        signature = self._generate_signature(obj)
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'x-mxc-sign': signature['sign'],
            'x-mxc-nonce': signature['time'],
            'x-kl-ajax-request': 'Ajax_Request',
            'Authorization': self.web_cookie,
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Origin': 'https://futures.mexc.com',
            'Referer': 'https://futures.mexc.com/'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if method == 'GET':
                    if obj:
                        params = '&'.join([f"{k}={v}" for k, v in obj.items()])
                        url = f"{url}?{params}"
                    response = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=15)
                else:
                    response = curl_requests.post(url, headers=headers, json=obj, impersonate="chrome", timeout=15)
                
                return response.json()
            except Exception as e:
                logger.warning(f"MEXC WEB API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        return None
    
    def get_contract_size(self, symbol: str) -> float:
        """Get contract size for a symbol. Returns how many units 1 contract represents."""
        try:
            url = f"{self.base_url}/api/v1/contract/detail?symbol={symbol}"
            response = curl_requests.get(url, impersonate="chrome", timeout=10)
            data = response.json()
            if data.get('success'):
                contract_size = float(data['data'].get('contractSize', 1.0))
                return contract_size
        except Exception as e:
            logger.warning(f"Failed to get contract size for {symbol}: {e}")
        return 0.0001  # Default fallback for most coins
    
    def get_account_assets(self) -> float:
        """Get USDT balance."""
        result = self._request('GET', '/api/v1/private/account/assets', {})
        if result and result.get('success'):
            for asset in result.get('data', []):
                if asset.get('currency') == 'USDT':
                    return float(asset.get('availableBalance', 0))
        return 0.0
    
    def get_open_positions(self, symbol: str = None) -> tuple:
        """
        Get open positions.
        Returns (positions_list, success_flag).
        If API fails, returns ([], False) so caller knows not to trust result.
        """
        obj = {'symbol': symbol} if symbol else {}
        result = self._request('GET', '/api/v1/private/position/open_positions', obj)
        if result and result.get('success'):
            return result.get('data', []), True
        return [], False
    
    def change_leverage(self, symbol: str, leverage: int, position_type: int = 1) -> bool:
        """Change leverage (position_type: 1=long, 2=short)."""
        obj = {
            "symbol": symbol,
            "leverage": leverage,
            "openType": 1,  # Isolated
            "positionType": position_type
        }
        result = self._request('POST', '/api/v1/private/position/change_leverage', obj)
        return result and result.get('success', False)
    
    def place_order(self, symbol: str, side: int, volume: int, leverage: int,
                   price: float = 0, order_type: int = 5, open_type: int = 1) -> Optional[str]:
        """
        Place order on MEXC Futures.
        
        Args:
            symbol: e.g., 'BTC_USDT'
            side: 1=open long, 2=close short, 3=open short, 4=close long
            volume: Number of contracts
            leverage: Leverage multiplier
            order_type: 1=limit, 5=market
            open_type: 1=isolated, 2=cross
        """
        # Set leverage first (for both long and short)
        self.change_leverage(symbol, leverage, position_type=1)
        self.change_leverage(symbol, leverage, position_type=2)
        
        obj = {
            "symbol": symbol,
            "side": side,
            "openType": open_type,
            "type": str(order_type),
            "vol": volume,
            "leverage": leverage,
            "priceProtect": "0"
        }
        
        if price > 0 and order_type == 1:
            obj["price"] = price
        
        logger.info(f"📤 Placing order: {symbol} side={side} vol={volume} lev={leverage}x")
        result = self._request('POST', '/api/v1/private/order/create', obj)
        
        if result and result.get('success'):
            order_id = result.get('data', {}).get('orderId')
            logger.info(f"✅ Order placed! ID: {order_id}")
            return str(order_id)
        else:
            logger.error(f"❌ Order failed: {result}")
            return None
    
    def _get_price_precision(self, price: float) -> int:
        """Determine price precision based on price magnitude.
        
        MEXC Futures precision requirements (conservative):
        - High prices (>=1000): 2 decimals
        - Medium (>=10): 2-3 decimals  
        - Low (>=1): 3 decimals (NEAR, LINK, etc.)
        - Very low (>=0.01): 4 decimals
        - Tiny (<0.01): 5-6 decimals (PEPE, etc.)
        """
        if price >= 1000:
            return 1  # BTC at 95000 -> 95000.0
        elif price >= 100:
            return 2  # BCH at 600 -> 600.00
        elif price >= 10:
            return 2  # SOL at 180 -> 180.00
        elif price >= 1:
            return 3  # NEAR at 1.70 -> 1.700
        elif price >= 0.01:
            return 4  # Small coins
        else:
            return 5  # Meme coins like PEPE
    
    def place_stop_order(self, symbol: str, side: int, volume: int, 
                        stop_price: float, leverage: int) -> Optional[str]:
        """
        Place stop loss order using position-based TP/SL.
        
        Args:
            symbol: e.g., 'BTC_USDT'
            side: 4=close long (SL for LONG position), 2=close short (SL for SHORT position)
            volume: Number of contracts
            stop_price: Trigger price
            leverage: Leverage (not used for position-based SL, but kept for interface compatibility)
        """
        # Get position ID first
        positions, api_success = self.get_open_positions(symbol)
        if not api_success or not positions:
            logger.warning(f"No position found for {symbol}, cannot place stop order")
            return None
        
        # side=4 means close long, so we need LONG position (positionType=1)
        # side=2 means close short, so we need SHORT position (positionType=2)
        position_type = 1 if side == 4 else 2  # 4=close long -> need long pos, 2=close short -> need short pos
        position = None
        for pos in positions:
            if pos.get('symbol') == symbol and pos.get('positionType') == position_type:
                position = pos
                break
        
        if not position:
            logger.warning(f"No matching position for {symbol} side={side}")
            return None
        
        position_id = int(position.get('positionId'))
        
        # Round stop price based on price magnitude
        precision = self._get_price_precision(stop_price)
        stop_price = round(stop_price, precision)
        
        obj = {
            "positionId": position_id,
            "vol": volume,
            "stopLossPrice": stop_price,
            "lossTrend": 1  # 1=last price
        }
        
        logger.info(f"📤 Placing stop order @ {stop_price} for position {position_id}")
        result = self._request('POST', '/api/v1/private/stoporder/place', obj)
        
        if result and result.get('success'):
            logger.info(f"✅ Stop order placed!")
            return str(result.get('data', ''))
        else:
            logger.error(f"❌ Stop order failed: {result}")
            return None
    
    def cancel_stop_orders(self, symbol: str) -> bool:
        """Cancel all stop orders for a symbol."""
        obj = {'symbol': symbol}
        result = self._request('POST', '/api/v1/private/stoporder/cancel_all', obj)
        if result and result.get('success'):
            logger.info(f"✅ Cancelled stop orders for {symbol}")
            return True
        return False
    
    def update_stop_order(self, stop_order_id: int, new_stop_price: float) -> bool:
        """Update/move an existing stop order (trailing stop)."""
        precision = self._get_price_precision(new_stop_price)
        new_stop_price = round(new_stop_price, precision)
        obj = {
            "stopPlanOrderId": stop_order_id,
            "stopLossPrice": new_stop_price,
            "lossTrend": 1
        }
        result = self._request('POST', '/api/v1/private/stoporder/change_plan_price', obj)
        return result and result.get('success', False)
    
    def get_history_positions(self, symbol: str = None, page_num: int = 1, page_size: int = 20) -> list:
        """
        Get closed positions history from exchange.
        Returns list of closed positions with PnL data.
        """
        obj = {
            "page_num": page_num,
            "page_size": page_size
        }
        if symbol:
            obj["symbol"] = symbol
        
        result = self._request('GET', '/api/v1/private/position/list/history_positions', obj)
        if result and result.get('success'):
            return result.get('data', [])
        return []
    
    def get_order_deals(self, symbol: str, order_id: str) -> list:
        """Get filled trades for an order."""
        obj = {
            "symbol": symbol,
            "orderId": order_id
        }
        result = self._request('GET', '/api/v1/private/order/deal_details', obj)
        if result and result.get('success'):
            return result.get('data', [])
        return []


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
# PORTFOLIO MANAGER - Exchange-First (No local JSON state)
# ============================================================
class PortfolioManager:
    """
    Portfolio manager that works directly with exchange.
    NO LOCAL JSON STATE - exchange is the source of truth.
    
    On startup: reads open positions from exchange
    During trading: syncs with exchange every iteration
    """
    
    def __init__(self, mexc: MEXCClient, telegram: TelegramNotifier):
        self.mexc = mexc
        self.telegram = telegram
        self.position = None  # Local cache, synced from exchange
        self.trades_history = []
        
        # Get capital from exchange
        self.capital = self.mexc.get_account_assets()
        if self.capital <= 0:
            logger.error("❌ Could not get balance from MEXC!")
            self.capital = 0
        
        logger.info(f"💰 Capital from exchange: ${self.capital:.2f}")
        
        # Sync position from exchange on startup
        self._sync_position_from_exchange()
    
    def _sync_position_from_exchange(self) -> bool:
        """
        Sync local position state from exchange.
        Exchange is the source of truth - no JSON files.
        
        Returns True if we have an open position.
        """
        positions, api_success = self.mexc.get_open_positions()
        
        if not api_success:
            logger.warning("⚠️ Could not get positions from exchange")
            return self.position is not None
        
        if not positions:
            if self.position:
                logger.info(f"📊 Position {self.position.get('pair')} closed on exchange")
                self._handle_closed_position()
            self.position = None
            return False
        
        # We have an open position on exchange
        exchange_pos = positions[0]  # Take first position
        symbol = exchange_pos.get('symbol', '')
        pos_type = exchange_pos.get('positionType', 1)  # 1=long, 2=short
        
        # Convert MEXC symbol to Binance format
        pair = symbol.replace('_USDT', '/USDT:USDT')
        direction = 'LONG' if pos_type == 1 else 'SHORT'
        
        # If we don't have local position or it's different, create from exchange
        if not self.position or self.position.get('mexc_symbol') != symbol:
            entry_price = float(exchange_pos.get('openAvgPrice', 0))
            volume = int(float(exchange_pos.get('holdVol', 0)))
            leverage = int(float(exchange_pos.get('leverage', 10)))
            position_value = float(exchange_pos.get('positionValue', 0))
            
            # Get stop order info
            stop_order_id = None
            # Note: We don't have stop order ID from position API, will be set when we place one
            
            self.position = {
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': datetime.now(timezone.utc),  # Approximate
                'stop_loss': entry_price * (0.98 if direction == 'LONG' else 1.02),  # Default 2%
                'stop_distance': entry_price * 0.02,  # Will be recalculated
                'position_value': position_value,
                'leverage': leverage,
                'mexc_symbol': symbol,
                'volume': volume,
                'contract_size': self.mexc.get_contract_size(symbol),
                'stop_order_id': stop_order_id,
                'pred_strength': 2.0,  # Default
                'breakeven_active': False,
                'be_trigger_mult': 2.2  # Default
            }
            
            logger.info(f"📊 Synced position from exchange: {pair} {direction}")
            logger.info(f"   Entry: {entry_price:.6f} | Vol: {volume} | Lev: {leverage}x")
        
        return True
    
    def _handle_closed_position(self):
        """Handle position that was closed on exchange (by stop or manually)."""
        if not self.position:
            return
        
        pos = self.position
        mexc_symbol = pos.get('mexc_symbol')
        direction = pos.get('direction')
        position_type = 1 if direction == 'LONG' else 2
        
        # Get trade data from exchange history
        history = self.mexc.get_history_positions(mexc_symbol)
        
        # Find the most recent closed position
        exchange_trade = None
        for h in history:
            if h.get('symbol') == mexc_symbol and h.get('positionType') == position_type:
                exchange_trade = h
                break
        
        if exchange_trade:
            pnl_dollar = float(exchange_trade.get('realised', 0))
            entry_price = float(exchange_trade.get('openAvgPrice', pos.get('entry_price', 0)))
            exit_price = float(exchange_trade.get('closeAvgPrice', entry_price))
            position_value = float(exchange_trade.get('positionValue', pos.get('position_value', 1)))
            pnl_pct = (pnl_dollar / position_value) * 100 if position_value else 0
        else:
            # Fallback
            entry_price = pos.get('entry_price', 0)
            exit_price = entry_price
            pnl_dollar = 0
            pnl_pct = 0
            position_value = pos.get('position_value', 1)
        
        # Update capital from exchange
        real_balance = self.mexc.get_account_assets()
        if real_balance > 0:
            old_capital = self.capital
            self.capital = real_balance
            if abs(old_capital - real_balance) > 0.01:
                logger.info(f"💰 Capital synced: ${old_capital:.2f} → ${self.capital:.2f}")
        
        # Determine reason
        if pnl_dollar < 0:
            reason = 'stop_loss'
            emoji = "❌"
        elif pnl_pct < 0.5:
            reason = 'breakeven_stop'
            emoji = "🛡️"
        else:
            reason = 'take_profit'
            emoji = "✅"
        
        logger.info(f"{emoji} Trade closed: {pos['pair']} {direction}")
        logger.info(f"   Entry: {entry_price:.6f} | Exit: {exit_price:.6f}")
        logger.info(f"   PnL: ${pnl_dollar:+.2f} ({pnl_pct:+.1f}%) | Reason: {reason}")
        
        self.telegram.send(
            f"{emoji} Closed {pos['pair']}",
            f"Entry: {entry_price:.6f}\nExit: {exit_price:.6f}\nPnL: ${pnl_dollar:+.2f} ({pnl_pct:+.1f}%)\nReason: {reason}\nCapital: ${self.capital:.2f}"
        )
        
        # Add to history
        self.trades_history.append({
            'pair': pos['pair'],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_dollar': pnl_dollar,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'exit_time': datetime.now(timezone.utc).isoformat()
        })
    
    def has_position(self) -> bool:
        """Check if we have an open position (syncs with exchange first)."""
        return self._sync_position_from_exchange()
    
    def sync_capital(self):
        """Sync capital from exchange."""
        real_balance = self.mexc.get_account_assets()
        if real_balance > 0:
            self.capital = real_balance
    
    def open_position(self, signal: Dict) -> bool:
        """Open a new position."""
        # First sync with exchange
        if self.has_position():
            logger.warning("⚠️ Already have an open position on exchange")
            return False
        
        # Get fresh balance
        self.sync_capital()
        if self.capital <= 0:
            logger.error("❌ No balance available")
            return False
        
        # Apply slippage to entry price
        raw_price = signal['price']
        slippage_mult = 1 + Config.SLIPPAGE_PCT if signal['direction'] == 'LONG' else 1 - Config.SLIPPAGE_PCT
        entry_price = raw_price * slippage_mult
        
        atr = signal['atr']
        pred_strength = signal.get('pred_strength', 2.0)
        
        # === ATR-BASED STOP LOSS (дальше, стабильнее) ===
        if pred_strength >= 3.0:
            sl_mult = 1.6
        elif pred_strength >= 2.0:
            sl_mult = 1.5
        else:
            sl_mult = 1.2
        
        stop_distance = atr * sl_mult
        stop_loss = entry_price - stop_distance if signal['direction'] == 'LONG' else entry_price + stop_distance
        
        # === V13: DYNAMIC RISK-BASED POSITION SIZING ===
        # Risk scales 5-7% based on signal quality
        confidence = signal.get('confidence', Config.MIN_CONF)
        timing = signal.get('timing', Config.MIN_TIMING)
        RISK_PCT = calculate_dynamic_risk(confidence, timing)
        risk_amount = self.capital * RISK_PCT
        logger.info(f"📊 Dynamic Risk: {RISK_PCT*100:.1f}% (conf={confidence:.2f}, timing={timing:.1f})")
        sl_pct = stop_distance / entry_price
        
        # position_size = risk_amount / sl_pct
        position_value = risk_amount / sl_pct
        
        # Ограничения
        max_position_by_leverage = self.capital * Config.MAX_LEVERAGE
        original_position = position_value
        position_value = min(position_value, max_position_by_leverage, Config.MAX_POSITION_SIZE)
        
        # CRITICAL: Если position урезан, нужно уменьшить стоп чтобы сохранить 5% риск!
        if position_value < original_position:
            # Пересчитываем стоп для сохранения риска
            new_sl_pct = risk_amount / position_value
            new_stop_distance = new_sl_pct * entry_price
            
            # Обновляем стоп
            if signal['direction'] == 'LONG':
                stop_loss = entry_price - new_stop_distance
            else:
                stop_loss = entry_price + new_stop_distance
            stop_distance = new_stop_distance
            sl_pct = new_sl_pct
            logger.warning(f"⚠️ Position capped! Adjusted stop to {sl_pct*100:.2f}% for 5% risk")
        
        # Рассчитываем итоговое плечо
        leverage = position_value / self.capital
        leverage = min(leverage, int(Config.MAX_LEVERAGE))
        
        logger.info(f"📊 ATR: {atr:.6f} | SL dist: {stop_distance:.6f} ({sl_pct*100:.2f}%) | Leverage: {leverage:.1f}x")
        
        # Минимальная проверка
        min_position_value = 1.0
        if position_value < min_position_value:
            logger.error(f"❌ Position too small: ${position_value:.2f}")
            return False
        
        mexc_symbol = signal['pair'].replace('/USDT:USDT', '_USDT').replace('/', '_')
        
        # Get contract size
        contract_size = self.mexc.get_contract_size(mexc_symbol)
        contract_value = entry_price * contract_size
        volume = max(1, int(position_value / contract_value))
        
        actual_position_value = volume * contract_value
        
        # Place order
        mexc_side = 1 if signal['direction'] == 'LONG' else 3
        
        logger.info(f"📊 Opening: ${actual_position_value:.2f} | Lev: {leverage:.0f}x | Vol: {volume}")
        
        order_id = self.mexc.place_order(mexc_symbol, mexc_side, volume, int(leverage))
        if not order_id:
            return False
        
        # Place stop order
        stop_side = 4 if signal['direction'] == 'LONG' else 2
        stop_order_id = self.mexc.place_stop_order(mexc_symbol, stop_side, volume, stop_loss, int(leverage))
        
        if not stop_order_id:
            logger.error("❌ STOP ORDER FAILED! Closing position...")
            close_side = 4 if signal['direction'] == 'LONG' else 2
            self.mexc.place_order(mexc_symbol, close_side, volume, int(leverage))
            self.telegram.send("⚠️ EMERGENCY CLOSE", f"Stop order failed for {signal['pair']}")
            return False
        
        # Cache position locally (exchange is still source of truth)
        self.position = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': datetime.now(timezone.utc),
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'position_value': actual_position_value,
            'leverage': leverage,
            'mexc_symbol': mexc_symbol,
            'volume': volume,
            'contract_size': contract_size,
            'stop_order_id': int(stop_order_id),
            'pred_strength': pred_strength,
            'breakeven_active': False,
            'be_trigger_mult': 2.5 if pred_strength >= 3.0 else (2.2 if pred_strength >= 2.0 else 1.8)
        }
        
        self.telegram.send(
            f"🟢 <b>{signal['direction']}</b> {signal['pair']}",
            f"Entry: {entry_price:.6f}\nSL: {stop_loss:.6f}\nLev: {leverage:.1f}x"
        )
        return True
    
    def update_position(self, current_price: float, candle_high: float, candle_low: float):
        """Update position (breakeven, trailing stop) - ATR-BASED."""
        # First sync with exchange
        if not self.has_position():
            return
        
        pos = self.position
        pred_strength = pos.get('pred_strength', 2.0)
        
        # Get ATR from stop_distance
        sl_mult = 1.6 if pred_strength >= 3.0 else (1.5 if pred_strength >= 2.0 else 1.2)
        atr = pos['stop_distance'] / sl_mult
        
        # === ATR-BASED BREAKEVEN TRIGGER ===
        be_trigger_mult = 2.5 if pred_strength >= 3.0 else (2.2 if pred_strength >= 2.0 else 1.8)
        be_trigger_dist = atr * be_trigger_mult
        be_margin_dist = atr * 1.0  # Lock 1.0 ATR profit
        
        if pos['direction'] == 'LONG':
            # Breakeven trigger - ATR-based
            if not pos.get('breakeven_active') and candle_high >= pos['entry_price'] + be_trigger_dist:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] + be_margin_dist  # Lock 1.0 ATR
                
                if pos.get('stop_order_id'):
                    self.mexc.update_stop_order(pos['stop_order_id'], pos['stop_loss'])
                else:
                    new_stop_id = self.mexc.place_stop_order(
                        pos['mexc_symbol'], 4, pos['volume'],
                        pos['stop_loss'], int(pos.get('leverage', 10))
                    )
                    if new_stop_id:
                        pos['stop_order_id'] = int(new_stop_id)
                
                logger.info(f"✅ Breakeven SL set to {pos['stop_loss']:.4f} (1.0 ATR locked)")
            
            # Trailing - R-based multipliers
            if pos.get('breakeven_active'):
                r_mult = (candle_high - pos['entry_price']) / pos['stop_distance']
                if r_mult > 5:
                    trail_mult = 0.6
                elif r_mult > 3:
                    trail_mult = 1.2
                elif r_mult > 2:
                    trail_mult = 1.8
                else:
                    trail_mult = 2.5
                
                new_sl = candle_high - atr * trail_mult
                
                if new_sl > pos['stop_loss']:
                    old_sl = pos['stop_loss']
                    pos['stop_loss'] = new_sl
                    
                    if pos.get('stop_order_id'):
                        if self.mexc.update_stop_order(pos['stop_order_id'], new_sl):
                            logger.info(f"🔄 Trailing SL: {old_sl:.4f} → {new_sl:.4f}")
                    else:
                        new_stop_id = self.mexc.place_stop_order(
                            pos['mexc_symbol'], 4, pos['volume'], new_sl, int(pos.get('leverage', 10))
                        )
                        if new_stop_id:
                            pos['stop_order_id'] = int(new_stop_id)
        
        else:  # SHORT
            if not pos.get('breakeven_active') and candle_low <= pos['entry_price'] - be_trigger_dist:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] - be_margin_dist  # Lock 1.0 ATR
                
                if pos.get('stop_order_id'):
                    self.mexc.update_stop_order(pos['stop_order_id'], pos['stop_loss'])
                else:
                    new_stop_id = self.mexc.place_stop_order(
                        pos['mexc_symbol'], 2, pos['volume'],
                        pos['stop_loss'], int(pos.get('leverage', 10))
                    )
                    if new_stop_id:
                        pos['stop_order_id'] = int(new_stop_id)
                
                logger.info(f"✅ Breakeven SL set to {pos['stop_loss']:.4f} (1.0 ATR locked)")
            
            if pos.get('breakeven_active'):
                r_mult = (pos['entry_price'] - candle_low) / pos['stop_distance']
                if r_mult > 5:
                    trail_mult = 0.6
                elif r_mult > 3:
                    trail_mult = 1.2
                elif r_mult > 2:
                    trail_mult = 1.8
                else:
                    trail_mult = 2.5
                
                new_sl = candle_low + atr * trail_mult
                
                if new_sl < pos['stop_loss']:
                    old_sl = pos['stop_loss']
                    pos['stop_loss'] = new_sl
                    
                    if pos.get('stop_order_id'):
                        if self.mexc.update_stop_order(pos['stop_order_id'], new_sl):
                            logger.info(f"🔄 Trailing SL: {old_sl:.4f} → {new_sl:.4f}")
                    else:
                        new_stop_id = self.mexc.place_stop_order(
                            pos['mexc_symbol'], 2, pos['volume'], new_sl, int(pos.get('leverage', 10))
                        )
                        if new_stop_id:
                            pos['stop_order_id'] = int(new_stop_id)
    
    def close_position(self, price: float, reason: str):
        """Manually close position."""
        if not self.position:
            return
        
        pos = self.position
        self.mexc.cancel_stop_orders(pos['mexc_symbol'])
        
        close_side = 4 if pos['direction'] == 'LONG' else 2
        self.mexc.place_order(pos['mexc_symbol'], close_side, pos['volume'], int(pos['leverage']))
        
        self.position = None
        self.sync_capital()
        
        self.telegram.send(f"📊 CLOSE {pos['pair']}", f"Reason: {reason}\nCapital: ${self.capital:.2f}")


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
            
            if missing_features:
                logger.info(f"  ℹ️ Missing features ({len(missing_features)}): {sorted(missing_features)}")
            
            if len(missing_features) > MAX_MISSING_FEATURES_WARNING:
                logger.warning(f"  ⚠️ {len(missing_features)} missing features (will use 0.0): {missing_features}")
            
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


def validate_data_quality(ready_pairs: Dict[str, pd.DataFrame], models: Dict, 
                          csv_manager: CSVDataManager) -> bool:
    """
    Full data quality validation before trading starts.
    
    Checks:
    1. All required features present (100%)
    2. No NaN in critical features
    3. OHLCV data integrity (low <= open/close <= high)
    4. Sufficient data for each pair
    5. Feature value ranges are reasonable
    
    Returns:
        True if all checks pass, False if critical issues found
    """
    logger.info("\n" + "=" * 70)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 70)
    
    model_features = set(models['features'])
    issues = []
    warnings_list = []
    
    for pair, features in ready_pairs.items():
        pair_issues = []
        pair_warnings = []
        
        # 1. Check feature completeness
        available = set(features.columns)
        missing = model_features - available
        match_pct = (len(model_features) - len(missing)) / len(model_features) * 100
        
        if match_pct < 100:
            pair_issues.append(f"Features: {match_pct:.1f}% (missing: {missing})")
        
        # 2. Check for NaN in last row (what we'll use for prediction)
        last_row = features.iloc[-1]
        nan_features = []
        for f in models['features']:
            if f in features.columns and pd.isna(last_row[f]):
                nan_features.append(f)
        
        if nan_features:
            pair_warnings.append(f"NaN in last row: {nan_features}")
        
        # 3. Check OHLCV integrity in CSV data
        for tf in ['5m']:  # Check main timeframe
            df_tf = csv_manager.load_csv(pair, tf)
            
            if df_tf is not None and len(df_tf) > 0:
                df = df_tf.tail(100)  # Check last 100 candles
                
                # Price logic: low <= open <= high AND low <= close <= high
                invalid_prices = ((df['low'] > df['open']) | (df['low'] > df['close']) |
                                 (df['high'] < df['open']) | (df['high'] < df['close']))
                if invalid_prices.any():
                    pair_issues.append(f"Invalid OHLCV logic: {invalid_prices.sum()} candles")
                
                # Check for zero/negative prices
                zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
                if zero_prices.any():
                    pair_issues.append(f"Zero/negative prices: {zero_prices.sum()} candles")
                
                # Check for zero volume
                zero_volume = df['volume'] <= 0
                if zero_volume.any():
                    pair_warnings.append(f"Zero volume: {zero_volume.sum()} candles")
        
        # 4. Check feature value ranges
        for f in ['m5_rsi_7', 'm5_rsi_14', 'm15_rsi']:
            if f in features.columns:
                val = last_row.get(f, np.nan)
                if not pd.isna(val) and (val < 0 or val > 100):
                    pair_issues.append(f"{f}={val:.1f} out of range [0,100]")
        
        for f in ['m5_close_position', 'm5_bb_position']:
            if f in features.columns:
                val = last_row.get(f, np.nan)
                if not pd.isna(val) and (val < -1 or val > 2):
                    pair_warnings.append(f"{f}={val:.2f} unusual range")
        
        # Log results for this pair
        if pair_issues:
            issues.append((pair, pair_issues))
            logger.error(f"❌ {pair}: {pair_issues}")
        elif pair_warnings:
            warnings_list.append((pair, pair_warnings))
            logger.warning(f"⚠️ {pair}: {pair_warnings}")
        else:
            logger.info(f"✅ {pair}: All checks passed")
    
    # Summary
    logger.info("\n" + "-" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("-" * 50)
    logger.info(f"Total pairs: {len(ready_pairs)}")
    logger.info(f"✅ Passed: {len(ready_pairs) - len(issues) - len(warnings_list)}")
    logger.info(f"⚠️ Warnings: {len(warnings_list)}")
    logger.info(f"❌ Critical issues: {len(issues)}")
    
    if issues:
        logger.error("\nCRITICAL ISSUES FOUND:")
        for pair, pair_issues in issues:
            logger.error(f"  {pair}: {pair_issues}")
        logger.error("\n⛔ Fix issues before trading!")
        return False
    
    # Show feature values for first pair (BTC) as sample
    logger.info("\n" + "-" * 50)
    logger.info("SAMPLE FEATURE VALUES (BTC/USDT:USDT)")
    logger.info("-" * 50)
    
    sample_pair = 'BTC/USDT:USDT'
    if sample_pair in ready_pairs:
        features = ready_pairs[sample_pair]
        last_row = features.iloc[-1]
        
        # Group features by category
        feature_groups = {
            'ATR (volatility)': ['m5_atr_14_pct', 'm5_atr_ratio', 'm15_atr_pct', 'm5_atr_vs_avg'],
            'Returns %': ['m5_return_1', 'm5_return_5', 'm5_return_10', 'm5_return_20'],
            'RSI (0-100)': ['m5_rsi_7', 'm5_rsi_14', 'm15_rsi'],
            'Position (0-1)': ['m5_close_position', 'm5_bb_position', 'm5_bb_width'],
            'Volume ratios': ['m5_volume_ratio_5', 'm5_volume_ratio_20', 'vol_ratio', 'vol_zscore'],
            'Structure (0/1)': ['m5_higher_high', 'm5_lower_low', 'm5_higher_low', 'm5_lower_high'],
            'S/R (0/1)': ['m5_at_support', 'm5_at_resistance', 'm5_breakout_up', 'm5_breakout_down'],
            'Scores': ['m5_structure_score', 'm15_trend', 'm15_momentum', 'm5_ema_9_dist'],
        }
        
        for group_name, feature_list in feature_groups.items():
            values = []
            for f in feature_list:
                if f in features.columns:
                    val = last_row.get(f, np.nan)
                    if pd.isna(val):
                        values.append(f"{f}=NaN")
                    elif isinstance(val, (int, float)):
                        values.append(f"{f}={val:.3f}")
                    else:
                        values.append(f"{f}={val}")
            if values:
                logger.info(f"  {group_name}: {', '.join(values)}")
        
        # Show raw vs scaled comparison
        if models.get('scaler') is not None:
            logger.info("\n  📊 Scaler applied: StandardScaler")
            X_raw = last_row[models['features']].values.astype(np.float64).reshape(1, -1)
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = models['scaler'].transform(X_raw)
            
            # Show a few examples
            examples = ['m5_atr_14_pct', 'm5_rsi_14', 'm5_return_5', 'vol_ratio']
            for f in examples:
                if f in models['features']:
                    idx = list(models['features']).index(f)
                    logger.info(f"    {f}: raw={X_raw[0][idx]:.4f} → scaled={X_scaled[0][idx]:.4f}")
    
    logger.info("\n✅ ALL DATA QUALITY CHECKS PASSED")
    logger.info("=" * 70)
    return True


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
        
        # Return result for logging (V13: added price and confidence)
        result = {
            'pair': pair,
            'direction': direction,
            'dir_pred': dir_pred,
            'conf': dir_conf,
            'confidence': dir_conf,  # V13: for dynamic risk
            'timing': timing,
            'strength': strength,
            'pred_strength': strength,  # V13: alias for open_position
            'atr': row['atr'].iloc[0] if 'atr' in row.columns else 0.0,
            'price': row['close'].iloc[0],  # V13: required for open_position
            'is_signal': False
        }
        
        # Check if it's a valid signal (V13: stricter thresholds + filters)
        if dir_pred != 1:  # Not sideways
            if dir_conf >= Config.MIN_CONF and timing >= Config.MIN_TIMING and strength >= Config.MIN_STRENGTH:
                # V13: Apply regime filter
                if not check_regime_filter(df):
                    logger.debug(f"  {pair}: Filtered by regime (low volatility)")
                    return result
                
                # V13: Apply MTF confirmation
                if not check_mtf_confirmation(data, direction):
                    logger.debug(f"  {pair}: Filtered by MTF (no trend confirmation)")
                    return result
                
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
    # MEXC WEB API uses browser cookie instead of API keys
    mexc_cookie = secrets['mexc'].get('web_cookie', '')
    if not mexc_cookie:
        logger.error("❌ MEXC web_cookie not found in secrets.yaml!")
        logger.error("   Add 'web_cookie' under 'mexc' section in config/secrets.yaml")
        logger.error("   Get cookie from browser: DevTools > Application > Cookies > futures.mexc.com > u_id")
        return
    
    mexc = MEXCClient(mexc_cookie)
    
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
    
    # ========================================
    # VALIDATION PHASE: Full data quality check
    # ========================================
    if not validate_data_quality(ready_pairs, models, csv_manager):
        logger.error("❌ Data quality validation failed! Fix issues before trading.")
        telegram.send("⛔ <b>V10 Startup Failed</b>", "Data quality validation failed!")
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
            
            # Sync position with exchange (detect external closes)
            has_pos = portfolio.has_position()
            
            # Update position if exists
            if has_pos and portfolio.position:
                pos = portfolio.position
                logger.info(f"📍 Monitoring {pos['pair']} {pos['direction']} | Entry: {pos['entry_price']:.4f} | SL: {pos['stop_loss']:.4f}")
                ticker = binance.fetch_ticker(pos['pair'])
                candles = binance.fetch_ohlcv(pos['pair'], '5m', limit=3)
                if len(candles) >= 2:
                    current_price = ticker['last']
                    pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['direction'] == 'SHORT':
                        pnl_pct = -pnl_pct
                    logger.info(f"📈 Price: {current_price:.4f} | PnL: {pnl_pct:+.2f}%")
                    # Use LAST candle (current, still forming) for trailing - matches backtest!
                    # candles[-1] = current (forming), candles[-2] = last closed
                    # For trailing we need the current candle's high/low to track extremes
                    current_candle_high = candles[-1][2]  # high of current candle
                    current_candle_low = candles[-1][3]   # low of current candle
                    portfolio.update_position(current_price, current_candle_high, current_candle_low)
            
            # Scan for signals using PARALLEL processing (only if no position)
            if not has_pos:
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
                
                # Process BEST signal (sorted by conf * strength)
                if signals_found:
                    # Sort by quality score: confidence * strength
                    signals_found.sort(key=lambda x: x['conf'] * x['strength'], reverse=True)
                    signal_result = signals_found[0]  # Best signal
                    pair = signal_result['pair']
                    
                    logger.info(f"  🎯 Opening BEST signal: {pair} (Conf={signal_result['conf']:.2f} * Str={signal_result['strength']:.1f} = {signal_result['conf'] * signal_result['strength']:.2f})")
                    
                    try:
                        ticker = binance.fetch_ticker(pair)
                        signal = {
                            'pair': pair,
                            'direction': 'LONG' if signal_result['dir_pred'] == 2 else 'SHORT',
                            'price': ticker['last'],
                            'atr': signal_result['atr'],
                            'pred_strength': signal_result['strength'],
                            'confidence': signal_result['conf'],  # V13: for dynamic risk
                            'timing': signal_result['timing'],    # V13: for dynamic risk
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

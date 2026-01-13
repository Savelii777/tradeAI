"""
AI Trading Bot - Constants
Defines all constants used throughout the application.
"""

from enum import Enum, auto


class TimeFrame(str, Enum):
    """Trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalType(str, Enum):
    """Trading signal type."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketRegime(str, Enum):
    """Market regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class TradingSession(str, Enum):
    """Trading sessions."""
    ASIAN = "asian"
    EUROPEAN = "european"
    AMERICAN = "american"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ModelType(str, Enum):
    """ML model types."""
    DIRECTION = "direction"
    STRENGTH = "strength"
    VOLATILITY = "volatility"
    TIMING = "timing"
    EXIT = "exit"


class CandlePattern(str, Enum):
    """Candlestick patterns."""
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    PIN_BAR_BULLISH = "pin_bar_bullish"
    PIN_BAR_BEARISH = "pin_bar_bearish"


# Trading Constants
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = TimeFrame.M5

# Risk Management Constants
MAX_RISK_PER_TRADE = 0.02  # 2%
MAX_POSITION_SIZE = 0.20  # 20%
MAX_DAILY_LOSS = 0.03  # 3%
MAX_WEEKLY_LOSS = 0.07  # 7%
MAX_DRAWDOWN = 0.15  # 15%

# Technical Indicator Periods
EMA_PERIODS = [9, 21, 50, 200]
RSI_PERIODS = [7, 14]
ATR_PERIOD = 14

# Feature Engineering Constants
NORMALIZATION_WINDOW = 500
OUTLIER_STD_THRESHOLD = 3.0

# Model Constants
MIN_DIRECTION_PROBABILITY = 0.60
MIN_EXPECTED_MOVE_ATR = 1.5
MIN_TIMING_SCORE = 0.5

# Time Constants (in seconds)
CANDLE_SECONDS = {
    TimeFrame.M1: 60,
    TimeFrame.M5: 300,
    TimeFrame.M15: 900,
    TimeFrame.H1: 3600,
    TimeFrame.H4: 14400,
    TimeFrame.D1: 86400,
}

# Session Times (UTC)
SESSION_TIMES = {
    TradingSession.ASIAN: {"start": 0, "end": 8},  # 00:00 - 08:00 UTC
    TradingSession.EUROPEAN: {"start": 7, "end": 16},  # 07:00 - 16:00 UTC
    TradingSession.AMERICAN: {"start": 13, "end": 22},  # 13:00 - 22:00 UTC
}

# ============================================================
# FEATURE EXCLUSION PATTERNS
# ============================================================
# These features should be excluded from training and live prediction
# because they cause backtest vs live discrepancy

# Cumsum-dependent features: values depend on data window start position
# In backtest: data starts from 2017 → cumsum accumulates for 8 years
# In live: data starts from last 1000 candles → cumsum is 1000x smaller
# CRITICAL: These cause massive feature drift between backtest and live!
CUMSUM_PATTERNS = [
    # Full feature names (market_structure.py)
    'bars_since_swing_high',  # Used cumsum() - now uses bars_since_true (stable)
    'bars_since_swing_low',   # Used cumsum() - now uses bars_since_true (stable)
    
    # Partial patterns (feature_engine.py)
    'bars_since_swing',       # catches any variant of bars_since_swing
    'consecutive_up',         # groupby().cumsum() - now uses count_consecutive (stable)
    'consecutive_down',       # groupby().cumsum() - now uses count_consecutive (stable)
    
    # Volume cumsum patterns (indicators.py)
    'obv',                    # cumsum() from start of data - REMOVED, use obv_rolling_50
    'obv_ema',                # EMA of cumsum OBV - REMOVED
    'volume_delta_cumsum',    # Old name, now volume_delta_sum_20
]

# Absolute price-based features: values depend on current price level
# In training: price was $500 → m5_ema_200 = 500
# In live: price is $420 → m5_ema_200 = 420 (completely different!)
# Model sees different values and becomes "confused" → low confidence
ABSOLUTE_PRICE_FEATURES = [
    # EMA absolute values (both with and without m5_ prefix)
    'ema_9', 'ema_21', 'ema_50', 'ema_200',  # From indicators.py
    'm5_ema_9', 'm5_ema_21', 'm5_ema_50', 'm5_ema_200',  # From train_mtf.py
    
    # Bollinger Bands absolute levels
    'bb_upper', 'bb_middle', 'bb_lower',  # From indicators.py
    'm5_bb_upper', 'm5_bb_middle', 'm5_bb_lower',  # From train_mtf.py
    
    # Swing prices - absolute values that differ with price level
    'swing_high_price', 'swing_low_price',  # From market_structure.py
    'm5_swing_high_price', 'm5_swing_low_price',  # MTF variants
    
    # Volume MA absolute values (differ between coins and time periods)
    'volume_ma_5', 'volume_ma_10', 'volume_ma_20',  # From indicators.py
    'm5_volume_ma_5', 'm5_volume_ma_10', 'm5_volume_ma_20',  # From train_mtf.py
    
    # ATR absolute values
    'atr_7', 'atr_14', 'atr_21', 'atr_14_ma',  # From indicators.py
    'm5_atr_7', 'm5_atr_14', 'm5_atr_21', 'm5_atr_14_ma',  # From train_mtf.py
    
    # Volume delta/trend absolute metrics (scale varies wildly between coins)
    'volume_delta', 'volume_trend',  # From indicators.py
    'm5_volume_delta', 'm5_volume_trend',  # From train_mtf.py
    
    # MACD features: MACD = EMA_fast - EMA_slow (absolute price difference!)
    # At BTC $25,000: MACD could be $500
    # At BTC $95,000: MACD could be $2,000 - causes Feature Distribution Shift
    'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_change',  # From indicators.py
    'm5_macd', 'm5_macd_signal', 'm5_macd_histogram', 'm5_macd_histogram_change',  # From train_mtf.py
    
    # OBV-based features with absolute scale issues
    'obv_rolling_50',  # Absolute volume sum - varies wildly between coins
    'obv_slope',       # Rate of change of absolute volume - varies between coins
    'volume_delta_sum_20',  # Absolute volume sum
    'm5_obv_rolling_50', 'm5_obv_slope', 'm5_volume_delta_sum_20',  # MTF variants
    
    # Linear regression slope - raw price slope depends on price level
    # At BTC $90K: slope = 100 | At BTC $30K: slope = 33 (3x difference!)
    'linreg_slope_20',  # Raw slope from market_structure.py
    'm5_linreg_slope_20',  # MTF variant
    
    # EMA slopes - derived from absolute EMA values
    # Even though normalized by /ema, the diff() still depends on price scale
    'ema_fast_slope',  # From market_structure.py identify_trend()
    'ema_slow_slope',  # From market_structure.py identify_trend()
    'm5_ema_fast_slope', 'm5_ema_slow_slope',  # MTF variants
    
    # EMA slope features from indicators.py - same issue
    'ema_9_slope', 'ema_21_slope', 'ema_50_slope', 'ema_200_slope',
    'm5_ema_9_slope', 'm5_ema_21_slope', 'm5_ema_50_slope', 'm5_ema_200_slope',
    
    # NOTE: structure_score and m5_structure_score were moved OUT of this list
    # because our new implementation uses rolling sum (stable, not swing-detection based)
    # See CORE_20_FEATURES where m5_structure_score is included
    
    # Market structure trend features that depend on swing detection or absolute values
    'trend_strength',  # ADX-like, derived from ATR (absolute)
    'trend_direction',  # Derived from trend_score which includes structure_score
    'trend_score',  # Includes structure_trend from swing detection
    'structure_trend',  # From structure_score (swing-based)
    'm5_trend_strength', 'm5_trend_direction', 'm5_trend_score', 'm5_structure_trend',  # MTF
    
    # Linear regression trend features
    'linreg_trend',  # Binary but based on linreg_slope_20
    'm5_linreg_trend',  # MTF variant
    
    # Volatility features that can vary based on data window
    'volatility_ratio',  # Ratio of volatility to historical average
    'm5_volatility_ratio',  # MTF variant
]

# Features to exclude from training (in addition to targets and raw OHLCV)
DEFAULT_EXCLUDE_FEATURES = [
    'pair', 'target_dir', 'target_timing', 'target_strength',
    'open', 'high', 'low', 'close', 'volume', 'atr', 'price_change',
    'vol_sma_20', 'm15_volume_ma', 'm15_atr', 'vwap',
]

# ============================================================
# CORE 20 FEATURES (V15) - Absolute minimum for live trading
# ============================================================
# Only 20 features that are:
# 1. 100% relative/normalized (no absolute prices)
# 2. Bounded or stable range
# 3. High predictive power (top importance)
# 4. Work identically in live and backtest
CORE_20_FEATURES = [
    # === ATR % (3) - Most important, always relative ===
    'm5_atr_14_pct',           # ATR as % of price - core volatility
    'm5_atr_ratio',            # ATR vs its moving average
    'm15_atr_pct',             # M15 context volatility
    
    # === RETURNS % (4) - Inherently relative ===
    'm5_return_1',             # 1-bar return %
    'm5_return_5',             # 5-bar return %
    'm5_return_10',            # 10-bar return %
    'm5_return_20',            # 20-bar return %
    
    # === RSI (3) - Bounded 0-100, universal ===
    'm5_rsi_7',                # Fast RSI
    'm5_rsi_14',               # Standard RSI
    'm15_rsi',                 # M15 RSI context
    
    # === POSITION (3) - All 0-1 or normalized ===
    'm5_close_position',       # Where close is in H/L range (0-1)
    'm5_bb_position',          # Position in Bollinger Bands
    'm5_bb_width',             # BB width normalized
    
    # === VOLUME (3) - Ratios, always relative ===
    'm5_volume_ratio_5',       # Volume vs 5-period MA
    'm5_volume_ratio_20',      # Volume vs 20-period MA
    'vol_ratio',               # Current volume ratio
    
    # === STRUCTURE (4) - Binary, universal ===
    'm5_higher_high',          # Made higher high (0/1)
    'm5_lower_low',            # Made lower low (0/1)
    'm5_higher_low',           # Made higher low (0/1) - bullish structure
    'm5_lower_high',           # Made lower high (0/1) - bearish structure
    
    # === SUPPORT/RESISTANCE (4) - Binary/relative ===
    'm5_at_support',           # At support level (0/1)
    'm5_at_resistance',        # At resistance level (0/1)
    'm5_breakout_up',          # Breakout above resistance (0/1)
    'm5_breakout_down',        # Breakout below support (0/1)
    
    # === STRUCTURE SCORE (1) - Composite ===
    'm5_structure_score',      # HH+HL-LH-LL score (rolling 10 bars)
    
    # === M15 CONTEXT (2) - Trend/momentum ===
    'm15_trend',               # M15 trend direction
    'm15_momentum',            # M15 momentum
    
    # === ADDITIONAL STABLE (3) - Proven stable in live ===
    'vol_zscore',              # Volume z-score (relative)
    'm5_ema_9_dist',           # Distance to EMA9 as % (relative)
    'm5_atr_vs_avg',           # ATR vs 50-period average (more stable than MACD)
]
# Total: 30 features - core set with structure, S/R and breakouts

# ============================================================
# ULTRA MINIMAL FEATURES (V14) - Only TOP 40 by importance
# ============================================================
# Based on feature importance analysis from all 3 models
# These features have the highest predictive power and are STABLE
ULTRA_MINIMAL_FEATURES = [
    # === TOP ATR FEATURES (highest importance!) ===
    'm5_atr_7_pct', 'm5_atr_14_pct', 'm5_atr_21_pct',  # ATR as % of price
    'm5_atr_ratio', 'm5_atr_vs_avg',                   # ATR ratios
    'm15_atr_pct',                                      # M15 ATR %
    
    # === SWING STRUCTURE (top importance in direction model) ===
    'm5_swing_high', 'm5_swing_low',                   # Binary swing points
    'm5_higher_high', 'm5_higher_low',                 # Structure patterns
    'm5_lower_high', 'm5_lower_low',
    
    # === RANGE FEATURES ===
    'm5_hl_range_ratio',                               # Range ratio (relative)
    'm5_close_position',                               # Position in range (0-1)
    'm5_bb_width',                                     # BB width (normalized)
    'm5_bb_position',                                  # Position in BB
    
    # === RETURNS (always percentage, inherently stable) ===
    'm5_return_1', 'm5_return_5', 'm5_return_10', 'm5_return_20',
    
    # === TIME FEATURES (cyclical, stable) ===
    'm5_hour_sin', 'm5_hour_cos',
    'm5_day_sin', 'm5_day_cos',
    'm5_session_asian', 'm5_session_european', 'm5_session_american',
    
    # === REGIME (categorical, stable) ===
    'm5_regime_strong_downtrend', 'm5_regime_strong_uptrend',
    'm5_regime_high_volatility',
    
    # === RSI (0-100 bounded, stable) ===
    'm5_rsi_7', 'm5_rsi_14',
    'm5_rsi_7_change', 'm5_rsi_14_change',
    
    # === VOLUME RATIOS (relative, stable) ===
    'm5_volume_ratio_5', 'm5_volume_ratio_10', 'm5_volume_ratio_20',
    'vol_ratio', 'vol_zscore',
    
    # === EMA DISTANCES (percentage, stable) ===
    'm5_ema_9_dist', 'm5_ema_21_dist',
    'm5_ema_50_200_cross',
    
    # === SUPPORT/RESISTANCE (binary/relative) ===
    'm5_at_support', 'm5_at_resistance',
    'm5_dist_to_support', 'm5_dist_to_resistance',
    
    # === M15 CONTEXT (minimal, relative only) ===
    'm15_trend', 'm15_momentum', 'm15_rsi',
]
# Total: ~50 features - ultra minimal and stable!

# ============================================================
# MINIMAL STABLE FEATURES (V13)
# ============================================================
# ONLY relative/normalized features that are STABLE between backtest and live
# These features don't depend on absolute price level or historical window size
#
# Key principles:
# 1. All values are RELATIVE (ratios, percentages, normalized)
# 2. No cumsum-dependent calculations
# 3. No absolute price values (EMA, BB levels, ATR in dollars)
# 4. Remove useless features (importance=0 in analysis)

MINIMAL_STABLE_FEATURES = [
    # === CORE MOMENTUM (RSI, Stoch - always 0-100, stable) ===
    'm5_rsi_7', 'm5_rsi_14',                    # RSI: 0-100 range, stable
    'm5_rsi_7_change', 'm5_rsi_14_change',      # RSI momentum
    'm5_stoch_k', 'm5_stoch_d', 'm5_stoch_cross',  # Stochastic: 0-100
    
    # === PRICE POSITION (relative to recent range) ===
    'm5_close_position',          # Where close is in today's range (0-1)
    'm5_range_position',          # Position in recent range (0-1)
    'm5_bb_position',             # Position in Bollinger Bands (-1 to +1)
    'm5_bb_squeeze',              # BB squeeze (normalized)
    
    # === EMA DISTANCES (relative, not absolute) ===
    'm5_ema_9_dist', 'm5_ema_21_dist', 'm5_ema_50_dist', 'm5_ema_200_dist',  # % distance
    'm5_ema_trend',               # EMA alignment direction
    'm5_price_above_ema_9', 'm5_price_above_ema_21',   # Binary: above/below
    'm5_price_above_ema_50', 'm5_price_above_ema_200',
    'm5_ema_9_21_cross', 'm5_ema_50_200_cross',  # Cross signals
    
    # === VOLATILITY (normalized to ATR) ===
    'm5_atr_7_pct', 'm5_atr_14_pct', 'm5_atr_21_pct',  # ATR as % of price
    'm5_atr_ratio', 'm5_atr_vs_avg',           # ATR relative to average
    'm5_hl_range_ratio',                       # Range ratio (not absolute!)
    
    # === SWING STRUCTURE (binary/relative) ===
    'm5_swing_high', 'm5_swing_low',           # Binary: is swing
    'm5_higher_high', 'm5_higher_low',         # Structure binary
    'm5_lower_high', 'm5_lower_low',
    
    # === RETURNS (always percentage, stable) ===
    'm5_return_1', 'm5_return_5', 'm5_return_10', 'm5_return_20',
    'm5_log_return_1',
    
    # === VOLUME (relative ratios) ===
    'm5_volume_ratio_5', 'm5_volume_ratio_10', 'm5_volume_ratio_20',  # Ratios
    'vol_ratio', 'vol_zscore',                 # Normalized volume
    
    # === MACD (normalized signals only, not absolute values) ===
    'm5_macd_above_zero', 'm5_macd_cross',     # Binary signals
    
    # === TIME FEATURES (cyclical, stable) ===
    'm5_hour_sin', 'm5_hour_cos',              # Hour of day (0-1)
    'm5_day_sin', 'm5_day_cos',                # Day of week (0-1)
    'm5_session_asian', 'm5_session_european', 'm5_session_american',  # Binary
    
    # === REGIME (one-hot encoded, stable) ===
    'm5_regime_strong_uptrend', 'm5_regime_weak_uptrend',
    'm5_regime_strong_downtrend', 'm5_regime_weak_downtrend',
    'm5_regime_ranging', 'm5_regime_high_volatility',
    
    # === SUPPORT/RESISTANCE (relative) ===
    'm5_at_support', 'm5_at_resistance',       # Binary
    'm5_dist_to_support', 'm5_dist_to_resistance',  # Relative distance
    'm5_breakout_up', 'm5_breakout_down',      # Binary
    
    # === M15 CONTEXT (relative only) ===
    'm15_rsi', 'm15_trend', 'm15_momentum',
    'm15_atr_pct',                             # ATR as % (not absolute)
    'm15_volume_ratio',                        # Relative
    
    # === M1 MICRO (relative only) ===
    'm1_momentum_1_last', 'm1_momentum_3_last', 'm1_momentum_5_last',  # Returns %
    'm1_rsi_5_last', 'm1_rsi_9_last',          # RSI 0-100
    'm1_micro_trend_last',                     # Trend direction
    'm1_vwap_dist_last',                       # % distance to VWAP
]

# Total: ~75 features (down from 125)
# All STABLE between backtest and live!

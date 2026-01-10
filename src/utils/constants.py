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
    'bars_since_swing_high',  # swing_high.cumsum() - depends on data start
    'bars_since_swing_low',   # swing_low.cumsum() - depends on data start
    'swing_high_price',       # ffill() from first swing - depends on data start
    'swing_low_price',        # ffill() from first swing - depends on data start
    
    # Partial patterns (feature_engine.py)
    'bars_since_swing',       # catches any variant of bars_since_swing
    'consecutive_up',         # groupby().cumsum() - depends on data start
    'consecutive_down',       # groupby().cumsum() - depends on data start
    
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
    
    # Volume MA absolute values
    'volume_ma_5', 'volume_ma_10', 'volume_ma_20',  # From indicators.py
    'm5_volume_ma_5', 'm5_volume_ma_10', 'm5_volume_ma_20',  # From train_mtf.py
    
    # ATR absolute values
    'atr_7', 'atr_14', 'atr_21', 'atr_14_ma',  # From indicators.py
    'm5_atr_7', 'm5_atr_14', 'm5_atr_21', 'm5_atr_14_ma',  # From train_mtf.py
    
    # Volume delta/trend absolute metrics
    'volume_delta', 'volume_trend',  # From indicators.py
    'm5_volume_delta', 'm5_volume_trend',  # From train_mtf.py
    
    # MACD features: MACD = EMA_fast - EMA_slow (absolute price difference!)
    # At BTC $25,000: MACD could be $500
    # At BTC $95,000: MACD could be $2,000 - causes Feature Distribution Shift
    'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_change',  # From indicators.py
    'm5_macd', 'm5_macd_signal', 'm5_macd_histogram', 'm5_macd_histogram_change',  # From train_mtf.py
]

# Features to exclude from training (in addition to targets and raw OHLCV)
DEFAULT_EXCLUDE_FEATURES = [
    'pair', 'target_dir', 'target_timing', 'target_strength',
    'open', 'high', 'low', 'close', 'volume', 'atr', 'price_change',
    'vol_sma_20', 'm15_volume_ma', 'm15_atr', 'vwap',
]

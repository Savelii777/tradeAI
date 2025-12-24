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

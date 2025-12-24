"""
MTF Signal Confirmation Logic

Implements multi-timeframe confirmation for trade signals.
Only generates signals when all timeframes agree.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class TrendDirection(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


class SignalStrength(Enum):
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NONE = 0


@dataclass
class MTFSignal:
    """Multi-timeframe signal with confirmation."""
    
    # Direction
    direction: int  # 1 = long, -1 = short, 0 = no signal
    
    # Confidence (0-1)
    confidence: float
    
    # Timeframe confirmations
    m15_trend: TrendDirection
    m5_signal: int
    m1_timing: bool
    
    # Signal quality
    strength: SignalStrength
    
    # Risk parameters
    suggested_sl_pct: float
    suggested_tp_pct: float
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is tradeable."""
        return (
            self.direction != 0 and
            self.confidence >= 0.55 and
            self.strength.value >= 2 and
            self.m1_timing
        )
    
    @property
    def alignment_score(self) -> int:
        """Count how many timeframes agree."""
        score = 0
        
        # M15 agrees with direction
        if (self.direction == 1 and self.m15_trend == TrendDirection.BULLISH) or \
           (self.direction == -1 and self.m15_trend == TrendDirection.BEARISH):
            score += 1
        
        # M5 has signal
        if self.m5_signal == self.direction:
            score += 1
        
        # M1 timing is good
        if self.m1_timing:
            score += 1
        
        return score


class MTFConfirmation:
    """
    Multi-Timeframe Signal Confirmation System.
    
    Strategy:
    - M15: Trend context (must be in same direction)
    - M5:  Signal generation (main model prediction)
    - M1:  Entry timing (pullback end, volume confirmation)
    """
    
    def __init__(
        self,
        min_confidence: float = 0.55,
        require_m15_trend: bool = True,
        require_m1_timing: bool = True
    ):
        self.min_confidence = min_confidence
        self.require_m15_trend = require_m15_trend
        self.require_m1_timing = require_m1_timing
        
        # Statistics
        self.stats = {
            'total_signals': 0,
            'filtered_signals': 0,
            'passed_signals': 0
        }
    
    def analyze_m15_trend(self, m15_data: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """
        Analyze M15 trend direction and strength.
        
        Returns:
            Tuple of (direction, strength)
        """
        if len(m15_data) < 21:
            return TrendDirection.NEUTRAL, 0.0
        
        close = m15_data['close']
        
        # EMA crossover
        ema_8 = close.ewm(span=8, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        
        # Current position
        ema_diff = (ema_8.iloc[-1] - ema_21.iloc[-1]) / close.iloc[-1]
        
        # Trend strength (0-1)
        strength = min(abs(ema_diff) * 100, 1.0)
        
        # Additional confirmations
        rsi = self._calculate_rsi(close, 14).iloc[-1]
        
        # Trend direction
        if ema_diff > 0.001 and rsi > 50:  # Bullish
            return TrendDirection.BULLISH, strength
        elif ema_diff < -0.001 and rsi < 50:  # Bearish
            return TrendDirection.BEARISH, strength
        else:
            return TrendDirection.NEUTRAL, strength
    
    def analyze_m5_signal(
        self,
        m5_data: pd.DataFrame,
        model_prediction: int,
        model_confidence: float
    ) -> Tuple[int, float]:
        """
        Validate M5 signal with additional checks.
        
        Returns:
            Tuple of (direction, adjusted_confidence)
        """
        if model_confidence < self.min_confidence:
            return 0, model_confidence
        
        close = m5_data['close']
        
        # Additional M5 confirmations
        rsi = self._calculate_rsi(close, 14).iloc[-1]
        momentum = close.pct_change(5).iloc[-1] * 100
        
        # Adjust confidence based on confluence
        confidence_adj = model_confidence
        
        # RSI confirmation
        if model_prediction == 1 and 30 < rsi < 70:  # Long, RSI not overbought
            confidence_adj += 0.02
        elif model_prediction == -1 and 30 < rsi < 70:  # Short, RSI not oversold
            confidence_adj += 0.02
        
        # Momentum confirmation
        if model_prediction == 1 and momentum > 0:  # Long with positive momentum
            confidence_adj += 0.02
        elif model_prediction == -1 and momentum < 0:  # Short with negative momentum
            confidence_adj += 0.02
        
        # Cap confidence
        confidence_adj = min(confidence_adj, 0.95)
        
        return model_prediction, confidence_adj
    
    def analyze_m1_timing(
        self,
        m1_data: pd.DataFrame,
        direction: int
    ) -> Tuple[bool, Dict]:
        """
        Check M1 for optimal entry timing.
        
        Looks for:
        - End of pullback
        - Volume confirmation
        - Micro-momentum alignment
        
        Returns:
            Tuple of (is_good_timing, details)
        """
        if len(m1_data) < 20:
            return False, {}
        
        close = m1_data['close']
        volume = m1_data['volume']
        
        # Micro RSI
        rsi_5 = self._calculate_rsi(close, 5).iloc[-1]
        
        # Volume confirmation
        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        
        # Micro momentum (last 3 candles)
        momentum_3 = close.pct_change(3).iloc[-1] * 100
        
        # Last candle direction
        last_candle = close.iloc[-1] - close.iloc[-2]
        
        details = {
            'rsi_5': rsi_5,
            'vol_ratio': vol_ratio,
            'momentum_3': momentum_3,
            'last_candle': 'green' if last_candle > 0 else 'red'
        }
        
        # Entry timing logic
        is_good_timing = False
        
        if direction == 1:  # Looking to go LONG
            # Good entry: RSI recovering from oversold, volume spike, micro-momentum turning up
            if (rsi_5 < 60 and  # Not overbought on M1
                momentum_3 > -0.1 and  # Momentum not strongly down
                last_candle > 0):  # Last candle is green
                is_good_timing = True
                
        elif direction == -1:  # Looking to go SHORT
            # Good entry: RSI coming down from overbought, micro-momentum turning down
            if (rsi_5 > 40 and  # Not oversold on M1
                momentum_3 < 0.1 and  # Momentum not strongly up
                last_candle < 0):  # Last candle is red
                is_good_timing = True
        
        # Volume confirmation bonus
        if vol_ratio > 1.5:  # Volume spike
            details['volume_spike'] = True
        
        return is_good_timing, details
    
    def generate_signal(
        self,
        m15_data: pd.DataFrame,
        m5_data: pd.DataFrame,
        m1_data: pd.DataFrame,
        model_prediction: int,
        model_confidence: float,
        volatility: float = 0.02
    ) -> MTFSignal:
        """
        Generate MTF-confirmed signal.
        
        Args:
            m15_data: M15 OHLCV DataFrame
            m5_data: M5 OHLCV DataFrame
            m1_data: M1 OHLCV DataFrame
            model_prediction: Model's direction prediction (1 or -1)
            model_confidence: Model's confidence (0-1)
            volatility: Current volatility for SL/TP calculation
        
        Returns:
            MTFSignal with all confirmations
        """
        self.stats['total_signals'] += 1
        
        # Analyze each timeframe
        m15_trend, m15_strength = self.analyze_m15_trend(m15_data)
        m5_direction, m5_confidence = self.analyze_m5_signal(m5_data, model_prediction, model_confidence)
        m1_timing, m1_details = self.analyze_m1_timing(m1_data, m5_direction)
        
        # Check alignment
        aligned = True
        
        # M15 trend must match direction (if required)
        if self.require_m15_trend:
            if m5_direction == 1 and m15_trend != TrendDirection.BULLISH:
                aligned = False
            elif m5_direction == -1 and m15_trend != TrendDirection.BEARISH:
                aligned = False
        
        # M1 timing must be good (if required)
        if self.require_m1_timing and not m1_timing:
            aligned = False
        
        # Determine final direction
        final_direction = m5_direction if aligned else 0
        
        # Determine signal strength
        if aligned and m5_confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif aligned and m5_confidence >= 0.6:
            strength = SignalStrength.MODERATE
        elif aligned:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE
        
        # Calculate SL/TP based on volatility
        sl_pct = volatility * 1.5  # 1.5x volatility for SL
        tp_pct = volatility * 2.5  # 2.5x volatility for TP (R:R = 1:1.67)
        
        signal = MTFSignal(
            direction=final_direction,
            confidence=m5_confidence,
            m15_trend=m15_trend,
            m5_signal=m5_direction,
            m1_timing=m1_timing,
            strength=strength,
            suggested_sl_pct=sl_pct,
            suggested_tp_pct=tp_pct
        )
        
        # Update stats
        if signal.is_valid:
            self.stats['passed_signals'] += 1
        else:
            self.stats['filtered_signals'] += 1
        
        return signal
    
    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        total = self.stats['total_signals']
        if total > 0:
            pass_rate = self.stats['passed_signals'] / total
        else:
            pass_rate = 0
        
        return {
            **self.stats,
            'pass_rate': pass_rate
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


# ============================================================
# PAIR RANKER
# ============================================================

class PairRanker:
    """
    Dynamic pair ranking based on recent performance.
    
    Re-ranks pairs weekly based on:
    - Accuracy on recent data
    - Profitability
    - Consistency
    """
    
    def __init__(
        self,
        top_n: int = 20,
        min_accuracy: float = 0.52,
        min_trades: int = 50
    ):
        self.top_n = top_n
        self.min_accuracy = min_accuracy
        self.min_trades = min_trades
        
        self.pair_scores: Dict[str, Dict] = {}
    
    def update_pair_score(
        self,
        pair: str,
        accuracy: float,
        profit_factor: float,
        trade_count: int,
        avg_confidence: float
    ):
        """Update pair's score based on recent performance."""
        
        # Calculate composite score
        score = 0.0
        
        # Accuracy contribution (40%)
        if accuracy >= self.min_accuracy:
            score += (accuracy - 0.5) * 4 * 0.4  # Normalize to 0-1
        
        # Profit factor contribution (30%)
        if profit_factor > 1.0:
            score += min((profit_factor - 1.0) * 0.5, 0.3)
        
        # Trade count contribution (15%) - more trades = more reliable
        if trade_count >= self.min_trades:
            score += min(trade_count / 200, 0.15)
        
        # Confidence contribution (15%)
        score += avg_confidence * 0.15
        
        self.pair_scores[pair] = {
            'score': score,
            'accuracy': accuracy,
            'profit_factor': profit_factor,
            'trade_count': trade_count,
            'avg_confidence': avg_confidence
        }
    
    def get_top_pairs(self) -> List[str]:
        """Get top N pairs by composite score."""
        sorted_pairs = sorted(
            self.pair_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Filter by minimum accuracy
        valid_pairs = [
            pair for pair, stats in sorted_pairs
            if stats['accuracy'] >= self.min_accuracy
        ]
        
        return valid_pairs[:self.top_n]
    
    def should_trade_pair(self, pair: str) -> bool:
        """Check if pair meets minimum criteria."""
        if pair not in self.pair_scores:
            return False
        
        stats = self.pair_scores[pair]
        return (
            stats['accuracy'] >= self.min_accuracy and
            stats['trade_count'] >= self.min_trades
        )


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == '__main__':
    # Example usage
    confirmation = MTFConfirmation(
        min_confidence=0.55,
        require_m15_trend=True,
        require_m1_timing=True
    )
    
    # Simulate data
    np.random.seed(42)
    
    m15_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(50) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(50) * 0.5) + 0.5,
        'low': 100 + np.cumsum(np.random.randn(50) * 0.5) - 0.5,
        'volume': np.random.randint(1000, 5000, 50)
    })
    
    m5_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 0.3),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.3) + 0.3,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.3) - 0.3,
        'volume': np.random.randint(500, 2000, 100)
    })
    
    m1_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(200) * 0.1) + 0.1,
        'low': 100 + np.cumsum(np.random.randn(200) * 0.1) - 0.1,
        'volume': np.random.randint(100, 500, 200)
    })
    
    # Generate signal
    signal = confirmation.generate_signal(
        m15_data=m15_data,
        m5_data=m5_data,
        m1_data=m1_data,
        model_prediction=1,  # Model predicts LONG
        model_confidence=0.65
    )
    
    print(f"Signal: {signal}")
    print(f"Valid: {signal.is_valid}")
    print(f"Alignment: {signal.alignment_score}/3")
    print(f"Stats: {confirmation.get_stats()}")

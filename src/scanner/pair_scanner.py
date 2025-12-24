"""
AI Trading Bot - Multi-Pair Scanner
Scans multiple cryptocurrency pairs for high-potential setups.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.features import FeatureEngine
from src.models import EnsembleModel


@dataclass
class ScanResult:
    """Result of scanning a single pair."""
    symbol: str
    timestamp: datetime
    score: float  # Overall opportunity score (0-100)
    direction: int  # 1 for long, -1 for short
    direction_probability: float
    expected_move_atr: float
    volatility: float
    current_price: float
    atr: float
    volume_ratio: float  # Current volume / average volume
    trend_strength: float
    setup_type: str  # Type of detected setup
    timeframe_alignment: bool  # Whether higher timeframes agree
    entry_zone: Tuple[float, float]  # Optimal entry price range
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PairScanner:
    """
    Multi-pair scanner for finding trading opportunities.
    
    Scans multiple cryptocurrency pairs simultaneously to find
    the best setup for "one shot, one target" trading strategy.
    
    Features:
    - Parallel scanning of multiple pairs
    - Multi-timeframe analysis
    - Setup quality scoring
    - Volume and liquidity filtering
    - Trend and momentum analysis
    """
    
    # Default pairs to scan (top liquid pairs)
    DEFAULT_PAIRS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "ATOMUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT",
        "NEARUSDT", "ARBUSDT", "OPUSDT", "FILUSDT", "INJUSDT",
        "SUIUSDT", "SEIUSDT", "TIAUSDT", "JUPUSDT", "WIFUSDT"
    ]
    
    def __init__(
        self,
        model: EnsembleModel,
        feature_engine: FeatureEngine,
        config: Optional[Dict] = None
    ):
        """
        Initialize the pair scanner.
        
        Args:
            model: Trained ensemble model for predictions.
            feature_engine: Feature engineering engine.
            config: Scanner configuration.
        """
        self.model = model
        self.feature_engine = feature_engine
        self.config = config or {}
        
        # Pairs to scan
        self.pairs = self.config.get('pairs', self.DEFAULT_PAIRS)
        
        # Scanning parameters
        self.min_score = self.config.get('min_score', 70)  # Minimum score to consider
        self.min_volume_ratio = self.config.get('min_volume_ratio', 0.8)  # vs average
        self.min_direction_prob = self.config.get('min_direction_probability', 0.65)
        self.min_expected_move = self.config.get('min_expected_move_atr', 2.0)
        self.require_tf_alignment = self.config.get('require_timeframe_alignment', True)
        
        # Timeframes for analysis
        self.primary_timeframe = self.config.get('primary_timeframe', '5m')
        self.confirmation_timeframes = self.config.get('confirmation_timeframes', ['15m', '1h'])
        
        # State
        self._last_scan_results: Dict[str, ScanResult] = {}
        self._scan_history: List[Dict[str, Any]] = []
        
    async def scan_all_pairs(
        self,
        data_collector,  # DataCollector instance
        limit: int = 500
    ) -> List[ScanResult]:
        """
        Scan all configured pairs for trading opportunities.
        
        Args:
            data_collector: Data collector for fetching market data.
            limit: Number of candles to fetch per timeframe.
            
        Returns:
            List of ScanResult objects, sorted by score (best first).
        """
        results = []
        
        # Scan pairs in parallel batches
        batch_size = 5
        for i in range(0, len(self.pairs), batch_size):
            batch = self.pairs[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._scan_pair(symbol, data_collector, limit) for symbol in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, ScanResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Scan error: {result}")
                    
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update state
        self._last_scan_results = {r.symbol: r for r in results}
        self._scan_history.append({
            'timestamp': datetime.utcnow(),
            'pairs_scanned': len(self.pairs),
            'opportunities_found': len([r for r in results if r.score >= self.min_score]),
            'best_score': results[0].score if results else 0
        })
        
        logger.info(f"Scanned {len(self.pairs)} pairs, found {len(results)} opportunities")
        
        return results
        
    async def _scan_pair(
        self,
        symbol: str,
        data_collector,
        limit: int
    ) -> Optional[ScanResult]:
        """
        Scan a single pair for trading opportunity.
        
        Args:
            symbol: Trading pair symbol.
            data_collector: Data collector instance.
            limit: Number of candles to fetch.
            
        Returns:
            ScanResult if opportunity found, None otherwise.
        """
        try:
            # Fetch data for all timeframes
            symbol_ccxt = symbol.replace('USDT', '/USDT')
            
            # Get primary timeframe data
            df_primary = await data_collector.fetch_ohlcv(
                symbol=symbol_ccxt,
                timeframe=self.primary_timeframe,
                limit=limit
            )
            
            if df_primary.empty or len(df_primary) < 100:
                return None
                
            # Get confirmation timeframe data
            df_15m = await data_collector.fetch_ohlcv(
                symbol=symbol_ccxt,
                timeframe='15m',
                limit=200
            )
            
            df_1h = await data_collector.fetch_ohlcv(
                symbol=symbol_ccxt,
                timeframe='1h',
                limit=100
            )
            
            # Generate features
            features = self.feature_engine.generate_all_features(
                df_primary,
                normalize=True,
                additional_timeframes={'15m': df_15m, '1h': df_1h} if not df_15m.empty else None
            )
            
            if features.empty:
                return None
                
            latest_features = features.iloc[[-1]].fillna(0)
            
            # Get model predictions
            predictions = self.model.predict(latest_features)
            
            # Extract predictions
            direction_proba = predictions.get('direction_proba', [0.33, 0.34, 0.33])
            if isinstance(direction_proba, np.ndarray) and direction_proba.ndim == 2:
                direction_proba = direction_proba[0]
                
            strength = predictions.get('strength', 0)
            if isinstance(strength, np.ndarray):
                strength = float(strength[0]) if len(strength) > 0 else 0
                
            timing = predictions.get('timing', 0)
            if isinstance(timing, np.ndarray):
                timing = float(timing[0]) if len(timing) > 0 else 0
                
            # Current market state
            current_price = df_primary['close'].iloc[-1]
            atr = self._calculate_atr(df_primary)
            volume = df_primary['volume'].iloc[-1]
            volume_avg = df_primary['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / volume_avg if volume_avg > 0 else 0
            
            # Determine direction
            max_prob_idx = int(np.argmax(direction_proba))
            max_prob = float(direction_proba[max_prob_idx])
            
            if max_prob_idx == 2:  # Up
                direction = 1
                setup_type = "bullish"
            elif max_prob_idx == 0:  # Down
                direction = -1
                setup_type = "bearish"
            else:  # Sideways
                direction = 0
                setup_type = "neutral"
                
            # Check timeframe alignment
            tf_aligned = self._check_timeframe_alignment(
                df_primary, df_15m, df_1h, direction
            )
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(df_primary)
            
            # Calculate entry zone, stop, and take profit
            stop_distance_pct = 0.003  # 0.3% default, will be adjusted
            
            if direction == 1:  # Long
                entry_low = current_price * 0.999
                entry_high = current_price * 1.001
                stop_loss = current_price * (1 - stop_distance_pct)
                take_profit = current_price * (1 + stop_distance_pct * 3)  # 1:3 RR
            else:  # Short
                entry_low = current_price * 0.999
                entry_high = current_price * 1.001
                stop_loss = current_price * (1 + stop_distance_pct)
                take_profit = current_price * (1 - stop_distance_pct * 3)
                
            risk_reward = abs(take_profit - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 0
            
            # Calculate opportunity score
            score = self._calculate_score(
                direction_prob=max_prob,
                expected_move=strength,
                timing=timing,
                volume_ratio=volume_ratio,
                trend_strength=trend_strength,
                tf_aligned=tf_aligned
            )
            
            # Check minimum thresholds
            if max_prob < self.min_direction_prob:
                score *= 0.5  # Penalize low probability
                
            if volume_ratio < self.min_volume_ratio:
                score *= 0.7  # Penalize low volume
                
            if direction == 0:
                score *= 0.3  # Strong penalize neutral direction
                
            if self.require_tf_alignment and not tf_aligned:
                score *= 0.6  # Penalize misalignment
                
            return ScanResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                score=score,
                direction=direction,
                direction_probability=max_prob,
                expected_move_atr=strength,
                volatility=atr / current_price,
                current_price=current_price,
                atr=atr,
                volume_ratio=volume_ratio,
                trend_strength=trend_strength,
                setup_type=setup_type,
                timeframe_alignment=tf_aligned,
                entry_zone=(entry_low, entry_high),
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                metadata={
                    'direction_proba': list(direction_proba),
                    'timing_score': timing,
                    'predictions': {k: float(v[0]) if isinstance(v, np.ndarray) else float(v) 
                                   for k, v in predictions.items() if k != 'direction_proba'}
                }
            )
            
        except Exception as e:
            logger.debug(f"Error scanning {symbol}: {e}")
            return None
            
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
        
    def _check_timeframe_alignment(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        direction: int
    ) -> bool:
        """Check if higher timeframes agree with the direction."""
        if direction == 0:
            return False
            
        alignments = []
        
        # Check 5m trend
        if len(df_5m) >= 20:
            ema_short = df_5m['close'].ewm(span=9).mean().iloc[-1]
            ema_long = df_5m['close'].ewm(span=21).mean().iloc[-1]
            trend_5m = 1 if ema_short > ema_long else -1
            alignments.append(trend_5m == direction)
            
        # Check 15m trend
        if not df_15m.empty and len(df_15m) >= 20:
            ema_short = df_15m['close'].ewm(span=9).mean().iloc[-1]
            ema_long = df_15m['close'].ewm(span=21).mean().iloc[-1]
            trend_15m = 1 if ema_short > ema_long else -1
            alignments.append(trend_15m == direction)
            
        # Check 1h trend
        if not df_1h.empty and len(df_1h) >= 20:
            ema_short = df_1h['close'].ewm(span=9).mean().iloc[-1]
            ema_long = df_1h['close'].ewm(span=21).mean().iloc[-1]
            trend_1h = 1 if ema_short > ema_long else -1
            alignments.append(trend_1h == direction)
            
        # At least 2 out of 3 must align
        return sum(alignments) >= 2 if alignments else False
        
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)."""
        if len(df) < 50:
            return 0.5
            
        close = df['close']
        
        # ADX-like calculation
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        
        current = close.iloc[-1]
        
        # Count how many EMAs are in order
        bullish_order = (ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1])
        bearish_order = (ema_9.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1])
        
        # Price momentum
        momentum = (current - close.iloc[-20]) / close.iloc[-20]
        
        if bullish_order or bearish_order:
            strength = 0.7 + min(abs(momentum) * 5, 0.3)
        else:
            strength = 0.3 + min(abs(momentum) * 3, 0.3)
            
        return min(max(strength, 0), 1)
        
    def _calculate_score(
        self,
        direction_prob: float,
        expected_move: float,
        timing: float,
        volume_ratio: float,
        trend_strength: float,
        tf_aligned: bool
    ) -> float:
        """
        Calculate overall opportunity score (0-100).
        
        Weights:
        - Direction probability: 30%
        - Expected move: 20%
        - Timing: 15%
        - Volume: 15%
        - Trend strength: 10%
        - Timeframe alignment: 10%
        """
        score = 0
        
        # Direction probability (0-30 points)
        score += (direction_prob - 0.33) / 0.67 * 30
        
        # Expected move (0-20 points)
        move_score = min(expected_move / 3.0, 1.0) * 20
        score += move_score
        
        # Timing (0-15 points)
        score += timing * 15
        
        # Volume (0-15 points)
        volume_score = min(volume_ratio / 1.5, 1.0) * 15
        score += volume_score
        
        # Trend strength (0-10 points)
        score += trend_strength * 10
        
        # Timeframe alignment (0-10 points)
        if tf_aligned:
            score += 10
            
        return max(0, min(100, score))
        
    def get_best_opportunity(self) -> Optional[ScanResult]:
        """Get the best current opportunity from last scan."""
        if not self._last_scan_results:
            return None
            
        valid_results = [
            r for r in self._last_scan_results.values()
            if r.score >= self.min_score and r.direction != 0
        ]
        
        if not valid_results:
            return None
            
        return max(valid_results, key=lambda x: x.score)
        
    def get_top_opportunities(self, n: int = 5) -> List[ScanResult]:
        """Get top N opportunities from last scan."""
        valid_results = [
            r for r in self._last_scan_results.values()
            if r.score >= self.min_score and r.direction != 0
        ]
        
        return sorted(valid_results, key=lambda x: x.score, reverse=True)[:n]
        
    def set_pairs(self, pairs: List[str]) -> None:
        """Set pairs to scan."""
        self.pairs = pairs
        logger.info(f"Updated scan pairs: {len(pairs)} pairs")
        
    def add_pair(self, pair: str) -> None:
        """Add a pair to scan list."""
        if pair not in self.pairs:
            self.pairs.append(pair)
            
    def remove_pair(self, pair: str) -> None:
        """Remove a pair from scan list."""
        if pair in self.pairs:
            self.pairs.remove(pair)
            
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanning statistics."""
        if not self._scan_history:
            return {'total_scans': 0}
            
        return {
            'total_scans': len(self._scan_history),
            'pairs_count': len(self.pairs),
            'last_scan': self._scan_history[-1] if self._scan_history else None,
            'avg_opportunities': np.mean([s['opportunities_found'] for s in self._scan_history[-10:]]),
            'avg_best_score': np.mean([s['best_score'] for s in self._scan_history[-10:]])
        }

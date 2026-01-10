"""
AI Trading Bot - Market Structure Analysis
Analyzes market structure including swings, trends, and support/resistance.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.constants import MarketRegime


class MarketStructure:
    """
    Analyzes market structure for trading decisions.
    
    Includes:
    - Swing high/low detection
    - Support and resistance levels
    - Trend identification
    - Market regime classification
    """
    
    def __init__(
        self,
        swing_period: int = 5,
        sr_lookback: int = 100,
        trend_ema_fast: int = 20,
        trend_ema_slow: int = 50
    ):
        """
        Initialize market structure analyzer.
        
        Args:
            swing_period: Period for swing detection.
            sr_lookback: Lookback for support/resistance.
            trend_ema_fast: Fast EMA for trend.
            trend_ema_slow: Slow EMA for trend.
        """
        self.swing_period = swing_period
        self.sr_lookback = sr_lookback
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        
    def detect_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and lows.
        
        A swing high is a high surrounded by lower highs.
        A swing low is a low surrounded by higher lows.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            DataFrame with swing point indicators.
        """
        swings = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        
        # Detect swing highs
        swing_high = pd.Series(False, index=df.index)
        swing_low = pd.Series(False, index=df.index)
        
        # FIXED: Use larger end buffer to ensure stability when new candles are added
        # Without this, swings near the end of dataset change as new data arrives
        end_buffer = self.swing_period * 3  # 3x buffer for stability
        
        for i in range(self.swing_period, len(df) - end_buffer):
            # Check if current high is highest in window
            window_start = i - self.swing_period
            window_end = i + self.swing_period + 1
            
            if high.iloc[i] == high.iloc[window_start:window_end].max():
                if high.iloc[i] > high.iloc[i - 1] and high.iloc[i] > high.iloc[i + 1]:
                    swing_high.iloc[i] = True
                    
            # Check if current low is lowest in window
            if low.iloc[i] == low.iloc[window_start:window_end].min():
                if low.iloc[i] < low.iloc[i - 1] and low.iloc[i] < low.iloc[i + 1]:
                    swing_low.iloc[i] = True
                    
        swings['swing_high'] = swing_high
        swings['swing_low'] = swing_low
        # FIXED: limit ffill to 200 bars to ensure consistency between live and backtest
        # Without limit, ffill propagates from FIRST swing in dataset, causing divergence
        swings['swing_high_price'] = high.where(swing_high, np.nan).ffill(limit=200)
        swings['swing_low_price'] = low.where(swing_low, np.nan).ffill(limit=200)
        
        # Distance to last swing - use rolling window approach for consistency
        # FIXED: cumsum() depends on data window start, causing backtest/live divergence
        # Instead, calculate bars since last swing using a stable approach
        # that gives the same result regardless of data start position
        
        # Count bars since last True value in swing_high/swing_low
        # This is stable because it only depends on recent data, not the entire history
        def bars_since_true(series: pd.Series, max_lookback: int = 200) -> pd.Series:
            """Count bars since last True value, with max lookback for stability."""
            result = pd.Series(index=series.index, dtype=float)
            last_true_idx = -1
            for i in range(len(series)):
                if series.iloc[i]:
                    last_true_idx = i
                if last_true_idx >= 0:
                    bars = i - last_true_idx
                    result.iloc[i] = min(bars, max_lookback)  # Cap at max_lookback
                else:
                    result.iloc[i] = max_lookback  # No swing found yet
            return result
        
        swings['bars_since_swing_high'] = bars_since_true(swing_high)
        swings['bars_since_swing_low'] = bars_since_true(swing_low)
        
        return swings
        
    def detect_higher_highs_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect higher highs/lows and lower highs/lows.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            DataFrame with HH/HL/LH/LL indicators.
        """
        swings = self.detect_swings(df)
        result = pd.DataFrame(index=df.index)
        
        # Get swing high and low values
        swing_highs = df['high'].where(swings['swing_high'], np.nan)
        swing_lows = df['low'].where(swings['swing_low'], np.nan)
        
        # Previous swing values (limit=200 for live/backtest consistency)
        prev_swing_high = swing_highs.ffill(limit=200).shift(1)
        prev_swing_low = swing_lows.ffill(limit=200).shift(1)
        
        # Detect patterns
        result['higher_high'] = (swing_highs > prev_swing_high) & swings['swing_high']
        result['lower_high'] = (swing_highs < prev_swing_high) & swings['swing_high']
        result['higher_low'] = (swing_lows > prev_swing_low) & swings['swing_low']
        result['lower_low'] = (swing_lows < prev_swing_low) & swings['swing_low']
        
        # Trend structure score
        result['structure_score'] = (
            result['higher_high'].rolling(10).sum() +
            result['higher_low'].rolling(10).sum() -
            result['lower_high'].rolling(10).sum() -
            result['lower_low'].rolling(10).sum()
        )
        
        return result
        
    def find_support_resistance(
        self,
        df: pd.DataFrame,
        num_levels: int = 5
    ) -> Dict[str, List[float]]:
        """
        Find support and resistance levels.
        
        Uses swing points and volume profile to identify key levels.
        
        Args:
            df: DataFrame with OHLCV data.
            num_levels: Number of levels to return.
            
        Returns:
            Dictionary with support and resistance levels.
        """
        # Use recent data for S/R detection
        recent = df.tail(self.sr_lookback)
        swings = self.detect_swings(recent)
        
        # Get swing high and low prices
        swing_high_prices = recent['high'][swings['swing_high']].values
        swing_low_prices = recent['low'][swings['swing_low']].values
        
        current_price = df['close'].iloc[-1]
        
        # Cluster nearby levels
        def cluster_levels(prices: np.ndarray, tolerance: float = 0.005) -> List[float]:
            if len(prices) == 0:
                return []
            sorted_prices = np.sort(prices)
            clusters = []
            current_cluster = [sorted_prices[0]]
            
            for price in sorted_prices[1:]:
                if price / current_cluster[-1] - 1 < tolerance:
                    current_cluster.append(price)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [price]
            clusters.append(np.mean(current_cluster))
            
            return clusters
            
        # Cluster and separate into support/resistance
        all_levels = np.concatenate([swing_high_prices, swing_low_prices])
        clustered = cluster_levels(all_levels)
        
        support = sorted([l for l in clustered if l < current_price], reverse=True)[:num_levels]
        resistance = sorted([l for l in clustered if l > current_price])[:num_levels]
        
        return {
            'support': support,
            'resistance': resistance,
            'nearest_support': support[0] if support else current_price * 0.99,
            'nearest_resistance': resistance[0] if resistance else current_price * 1.01
        }
        
    def calculate_sr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate support/resistance based features.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with S/R features.
        """
        close = df['close']
        
        # Rolling S/R calculation (approximate)
        rolling_high = df['high'].rolling(window=self.sr_lookback).max()
        rolling_low = df['low'].rolling(window=self.sr_lookback).min()
        
        # Distance to rolling extremes
        dist_to_resistance = (rolling_high - close) / close * 100
        dist_to_support = (close - rolling_low) / close * 100
        
        # Position in range
        price_range = rolling_high - rolling_low
        
        feature_dict = {
            'dist_to_resistance': dist_to_resistance,
            'dist_to_support': dist_to_support,
            'range_position': (close - rolling_low) / price_range.replace(0, np.nan),
            'at_resistance': dist_to_resistance < 0.5,
            'at_support': dist_to_support < 0.5,
            'breakout_up': close > rolling_high.shift(1),
            'breakout_down': close < rolling_low.shift(1)
        }
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    def identify_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify current trend using multiple methods.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            DataFrame with trend indicators.
        """
        trend = pd.DataFrame(index=df.index)
        close = df['close']
        
        # EMA-based trend
        ema_fast = close.ewm(span=self.trend_ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.trend_ema_slow, adjust=False).mean()
        
        trend['ema_trend'] = np.where(
            ema_fast > ema_slow, 1,
            np.where(ema_fast < ema_slow, -1, 0)
        )
        
        # Price position relative to EMAs
        trend['above_fast_ema'] = (close > ema_fast).astype(int)
        trend['above_slow_ema'] = (close > ema_slow).astype(int)
        
        # EMA slope (momentum)
        trend['ema_fast_slope'] = ema_fast.diff(5) / ema_fast * 100
        trend['ema_slow_slope'] = ema_slow.diff(5) / ema_slow * 100
        
        # Higher highs/lows trend
        hl_features = self.detect_higher_highs_lows(df)
        trend['structure_trend'] = np.sign(hl_features['structure_score'])
        
        # Linear regression trend
        def linreg_slope(series):
            x = np.arange(len(series))
            if len(series) < 2:
                return 0
            slope, _ = np.polyfit(x, series.values, 1)
            return slope
            
        trend['linreg_slope_20'] = close.rolling(20).apply(linreg_slope, raw=False)
        trend['linreg_trend'] = np.sign(trend['linreg_slope_20'])
        
        # Combined trend score
        trend['trend_score'] = (
            trend['ema_trend'] +
            trend['above_fast_ema'] +
            trend['above_slow_ema'] +
            trend['structure_trend']
        ) / 4
        
        return trend
        
    def classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime.
        
        Regimes:
        - Strong uptrend
        - Weak uptrend
        - Strong downtrend
        - Weak downtrend
        - Ranging/choppy
        - High volatility
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with regime classification.
        """
        regime = pd.DataFrame(index=df.index)
        
        # Get trend information
        trend = self.identify_trend(df)
        trend_score = trend['trend_score']
        
        # Calculate volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        volatility_ma = volatility.rolling(100).mean()
        volatility_ratio = volatility / volatility_ma
        
        # Calculate ADX-like directional strength
        high_change = df['high'].diff()
        low_change = -df['low'].diff()
        
        plus_dm = high_change.where((high_change > low_change) & (high_change > 0), 0)
        minus_dm = low_change.where((low_change > high_change) & (low_change > 0), 0)
        
        atr = self._calculate_atr(df, 14)
        
        plus_di = (plus_dm.ewm(span=14).mean() / atr) * 100
        minus_di = (minus_dm.ewm(span=14).mean() / atr) * 100
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(span=14).mean()
        
        # Classify regime
        regime['trend_strength'] = adx
        regime['volatility_ratio'] = volatility_ratio
        regime['trend_direction'] = trend_score
        
        # Regime classification
        def classify(row):
            if row['volatility_ratio'] > 1.5:
                return MarketRegime.HIGH_VOLATILITY.value
            elif row['trend_strength'] > 25:
                if row['trend_direction'] > 0.25:
                    return MarketRegime.STRONG_UPTREND.value
                elif row['trend_direction'] < -0.25:
                    return MarketRegime.STRONG_DOWNTREND.value
            elif row['trend_strength'] > 15:
                if row['trend_direction'] > 0:
                    return MarketRegime.WEAK_UPTREND.value
                elif row['trend_direction'] < 0:
                    return MarketRegime.WEAK_DOWNTREND.value
            return MarketRegime.RANGING.value
            
        regime['regime'] = regime.apply(classify, axis=1)
        
        # One-hot encode regime
        for r in MarketRegime:
            regime[f'regime_{r.value}'] = (regime['regime'] == r.value).astype(int)
        
        # Drop string column - LightGBM requires numeric types only
        regime = regime.drop(columns=['regime'])
            
        return regime
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR for internal use."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()
        
    def calculate_all_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all market structure features.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with all structure features.
        """
        all_features = pd.DataFrame(index=df.index)
        
        # Swing detection
        swings = self.detect_swings(df)
        all_features = pd.concat([all_features, swings], axis=1)
        
        # Higher highs/lows
        hh_ll = self.detect_higher_highs_lows(df)
        all_features = pd.concat([all_features, hh_ll], axis=1)
        
        # Support/resistance features
        sr_features = self.calculate_sr_features(df)
        all_features = pd.concat([all_features, sr_features], axis=1)
        
        # Trend identification
        trend = self.identify_trend(df)
        all_features = pd.concat([all_features, trend], axis=1)
        
        # Regime classification
        regime = self.classify_regime(df)
        all_features = pd.concat([all_features, regime], axis=1)
        
        return all_features

"""
AI Trading Bot - Candlestick Pattern Recognition
Identifies candlestick patterns for trading signals.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class CandlePatterns:
    """
    Recognizes candlestick patterns for trading analysis.
    
    Patterns include:
    - Single candle: Doji, Hammer, Inverted Hammer, Spinning Top
    - Double candle: Engulfing, Harami, Piercing, Dark Cloud
    - Triple candle: Morning Star, Evening Star, Three White Soldiers, etc.
    """
    
    def __init__(
        self,
        body_threshold: float = 0.1,
        shadow_ratio: float = 2.0
    ):
        """
        Initialize pattern recognizer.
        
        Args:
            body_threshold: Body size threshold as ratio of range (for doji).
            shadow_ratio: Minimum shadow to body ratio for hammers.
        """
        self.body_threshold = body_threshold
        self.shadow_ratio = shadow_ratio
        
    def _calculate_candle_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic candle properties.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            DataFrame with candle properties.
        """
        props = pd.DataFrame(index=df.index)
        
        # Basic measurements
        props['body'] = df['close'] - df['open']
        props['body_abs'] = abs(props['body'])
        props['range'] = df['high'] - df['low']
        props['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        props['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Ratios
        props['body_ratio'] = props['body_abs'] / props['range'].replace(0, np.nan)
        props['upper_shadow_ratio'] = props['upper_shadow'] / props['body_abs'].replace(0, np.nan)
        props['lower_shadow_ratio'] = props['lower_shadow'] / props['body_abs'].replace(0, np.nan)
        
        # Direction
        props['bullish'] = props['body'] > 0
        props['bearish'] = props['body'] < 0
        
        return props
        
    # -------------------- Single Candle Patterns --------------------
    
    def detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Doji patterns (indecision).
        
        A doji has a very small body relative to its range.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (0-1).
        """
        props = self._calculate_candle_properties(df)
        
        # Doji: body is less than threshold of range
        is_doji = props['body_ratio'] < self.body_threshold
        
        # Strength based on how small the body is
        strength = 1 - props['body_ratio'].clip(upper=self.body_threshold) / self.body_threshold
        strength = strength.where(is_doji, 0)
        
        return strength
        
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Hammer patterns (bullish reversal).
        
        A hammer has a long lower shadow and small upper shadow.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (-1 to 1, positive = bullish).
        """
        props = self._calculate_candle_properties(df)
        
        # Hammer criteria
        long_lower_shadow = props['lower_shadow_ratio'] >= self.shadow_ratio
        small_upper_shadow = props['upper_shadow'] < props['body_abs'] * 0.3
        small_body = props['body_ratio'] < 0.4
        
        is_hammer = long_lower_shadow & small_upper_shadow & small_body
        
        # Strength based on shadow ratio
        strength = (props['lower_shadow_ratio'] / self.shadow_ratio).clip(upper=1)
        strength = strength.where(is_hammer, 0)
        
        return strength
        
    def detect_inverted_hammer(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Inverted Hammer patterns.
        
        An inverted hammer has a long upper shadow and small lower shadow.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (0-1).
        """
        props = self._calculate_candle_properties(df)
        
        # Inverted hammer criteria
        long_upper_shadow = props['upper_shadow_ratio'] >= self.shadow_ratio
        small_lower_shadow = props['lower_shadow'] < props['body_abs'] * 0.3
        small_body = props['body_ratio'] < 0.4
        
        is_inv_hammer = long_upper_shadow & small_lower_shadow & small_body
        
        # Strength based on shadow ratio
        strength = (props['upper_shadow_ratio'] / self.shadow_ratio).clip(upper=1)
        strength = strength.where(is_inv_hammer, 0)
        
        return strength
        
    def detect_spinning_top(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Spinning Top patterns (indecision).
        
        A spinning top has a small body with upper and lower shadows.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (0-1).
        """
        props = self._calculate_candle_properties(df)
        
        # Spinning top: small body with shadows on both sides
        small_body = props['body_ratio'] < 0.3
        has_upper_shadow = props['upper_shadow'] > props['body_abs'] * 0.5
        has_lower_shadow = props['lower_shadow'] > props['body_abs'] * 0.5
        
        is_spinning = small_body & has_upper_shadow & has_lower_shadow
        
        # Strength based on body ratio
        strength = 1 - props['body_ratio'].clip(upper=0.3) / 0.3
        strength = strength.where(is_spinning, 0)
        
        return strength
        
    def detect_marubozu(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Marubozu patterns (strong trend).
        
        A marubozu has no or very small shadows (strong conviction).
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (-1 to 1, negative = bearish).
        """
        props = self._calculate_candle_properties(df)
        
        # Marubozu: very small shadows
        small_shadows = (props['upper_shadow'] + props['lower_shadow']) < props['range'] * 0.1
        large_body = props['body_ratio'] > 0.8
        
        is_marubozu = small_shadows & large_body
        
        # Direction and strength
        strength = props['body_ratio'].clip(upper=1)
        strength = strength.where(is_marubozu, 0)
        strength = strength * np.sign(props['body'])
        
        return strength
        
    # -------------------- Double Candle Patterns --------------------
    
    def detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Engulfing patterns.
        
        The second candle's body completely engulfs the first.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (-1 to 1, positive = bullish).
        """
        props = self._calculate_candle_properties(df)
        
        # Current candle engulfs previous
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        
        # Bullish engulfing
        bullish = (
            (props['bullish']) &
            (df['close'] > prev_open) &
            (df['open'] < prev_close) &
            (prev_close < prev_open)  # Previous was bearish
        )
        
        # Bearish engulfing
        bearish = (
            (props['bearish']) &
            (df['close'] < prev_open) &
            (df['open'] > prev_close) &
            (prev_close > prev_open)  # Previous was bullish
        )
        
        # Strength based on size comparison
        size_ratio = props['body_abs'] / props['body_abs'].shift(1).replace(0, np.nan)
        strength = (size_ratio - 1).clip(lower=0, upper=1)
        
        result = pd.Series(0, index=df.index)
        result = result.where(~bullish, strength)
        result = result.where(~bearish, -strength)
        
        return result
        
    def detect_harami(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Harami patterns.
        
        The second candle's body is contained within the first.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (-1 to 1, positive = bullish).
        """
        props = self._calculate_candle_properties(df)
        
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_body = abs(prev_close - prev_open)
        
        # Current body within previous
        contained = (
            (df['open'] > df[['open', 'close']].shift(1).min(axis=1)) &
            (df['open'] < df[['open', 'close']].shift(1).max(axis=1)) &
            (df['close'] > df[['open', 'close']].shift(1).min(axis=1)) &
            (df['close'] < df[['open', 'close']].shift(1).max(axis=1))
        )
        
        # Direction
        bullish = contained & (prev_close < prev_open) & props['bullish']
        bearish = contained & (prev_close > prev_open) & props['bearish']
        
        # Strength based on size ratio
        size_ratio = 1 - props['body_abs'] / prev_body.replace(0, np.nan)
        strength = size_ratio.clip(lower=0, upper=1)
        
        result = pd.Series(0, index=df.index)
        result = result.where(~bullish, strength)
        result = result.where(~bearish, -strength)
        
        return result
        
    # -------------------- Triple Candle Patterns --------------------
    
    def detect_morning_star(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Morning Star patterns (bullish reversal).
        
        Pattern: bearish candle, small candle, bullish candle.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (0-1).
        """
        props = self._calculate_candle_properties(df)
        
        # First candle: bearish with large body
        first_bearish = (df['close'].shift(2) < df['open'].shift(2))
        first_large = props['body_abs'].shift(2) > props['range'].shift(2) * 0.5
        
        # Second candle: small body
        second_small = props['body_ratio'].shift(1) < 0.3
        
        # Third candle: bullish with large body
        third_bullish = props['bullish']
        third_large = props['body_ratio'] > 0.5
        
        # Third candle closes above midpoint of first
        first_midpoint = (df['open'].shift(2) + df['close'].shift(2)) / 2
        third_strong = df['close'] > first_midpoint
        
        is_morning_star = (
            first_bearish & first_large &
            second_small &
            third_bullish & third_large & third_strong
        )
        
        strength = props['body_ratio'].where(is_morning_star, 0)
        return strength
        
    def detect_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Evening Star patterns (bearish reversal).
        
        Pattern: bullish candle, small candle, bearish candle.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            Series with pattern strength (0-1).
        """
        props = self._calculate_candle_properties(df)
        
        # First candle: bullish with large body
        first_bullish = (df['close'].shift(2) > df['open'].shift(2))
        first_large = props['body_abs'].shift(2) > props['range'].shift(2) * 0.5
        
        # Second candle: small body
        second_small = props['body_ratio'].shift(1) < 0.3
        
        # Third candle: bearish with large body
        third_bearish = props['bearish']
        third_large = props['body_ratio'] > 0.5
        
        # Third candle closes below midpoint of first
        first_midpoint = (df['open'].shift(2) + df['close'].shift(2)) / 2
        third_strong = df['close'] < first_midpoint
        
        is_evening_star = (
            first_bullish & first_large &
            second_small &
            third_bearish & third_large & third_strong
        )
        
        strength = props['body_ratio'].where(is_evening_star, 0)
        return strength
        
    # -------------------- Combined Pattern Detection --------------------
    
    def detect_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns.
        
        Args:
            df: DataFrame with OHLC data.
            
        Returns:
            DataFrame with pattern scores.
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Single candle patterns
        patterns['doji'] = self.detect_doji(df)
        patterns['hammer'] = self.detect_hammer(df)
        patterns['inverted_hammer'] = self.detect_inverted_hammer(df)
        patterns['spinning_top'] = self.detect_spinning_top(df)
        patterns['marubozu'] = self.detect_marubozu(df)
        
        # Double candle patterns
        patterns['engulfing'] = self.detect_engulfing(df)
        patterns['harami'] = self.detect_harami(df)
        
        # Triple candle patterns
        patterns['morning_star'] = self.detect_morning_star(df)
        patterns['evening_star'] = self.detect_evening_star(df)
        
        # Aggregate bullish/bearish signals
        bullish_patterns = ['hammer', 'morning_star']
        bearish_patterns = ['inverted_hammer', 'evening_star']
        
        patterns['bullish_pattern_sum'] = patterns[bullish_patterns].sum(axis=1)
        patterns['bearish_pattern_sum'] = patterns[bearish_patterns].sum(axis=1)
        
        # Add engulfing to appropriate category based on sign
        patterns['bullish_pattern_sum'] += patterns['engulfing'].clip(lower=0)
        patterns['bearish_pattern_sum'] += (-patterns['engulfing']).clip(lower=0)
        
        return patterns

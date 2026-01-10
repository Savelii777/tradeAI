"""
AI Trading Bot - Technical Indicators
Calculates technical indicators for feature engineering.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalIndicators:
    """
    Calculates technical indicators for trading analysis.
    
    Includes:
    - Trend indicators (EMA, SMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators
    """
    
    def __init__(self):
        """Initialize the indicator calculator."""
        pass
        
    # -------------------- Trend Indicators --------------------
    
    def ema(
        self,
        series: pd.Series,
        period: int
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price series.
            period: EMA period.
            
        Returns:
            EMA series.
        """
        return series.ewm(span=period, adjust=False).mean()
        
    def sma(
        self,
        series: pd.Series,
        period: int
    ) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            series: Price series.
            period: SMA period.
            
        Returns:
            SMA series.
        """
        return series.rolling(window=period).mean()
        
    def calculate_ema_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [9, 21, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate EMA-based features.
        
        Args:
            df: DataFrame with 'close' column.
            periods: List of EMA periods.
            
        Returns:
            DataFrame with EMA features.
        """
        close = df['close']
        feature_dict = {}
        ema_values = {}  # Store for crossover calculations
        
        for period in periods:
            ema = self.ema(close, period)
            ema_values[period] = ema
            feature_dict[f'ema_{period}'] = ema
            feature_dict[f'ema_{period}_dist'] = (close - ema) / ema * 100  # Distance in %
            feature_dict[f'ema_{period}_slope'] = ema.diff(5) / ema * 100  # 5-period slope
            feature_dict[f'price_above_ema_{period}'] = (close > ema).astype(int)
            
        # EMA crossovers
        if 9 in periods and 21 in periods:
            feature_dict['ema_9_21_cross'] = (
                (ema_values[9] > ema_values[21]).astype(int) -
                (ema_values[9] < ema_values[21]).astype(int)
            )
            
        if 50 in periods and 200 in periods:
            feature_dict['ema_50_200_cross'] = (
                (ema_values[50] > ema_values[200]).astype(int) -
                (ema_values[50] < ema_values[200]).astype(int)
            )
            
        return pd.DataFrame(feature_dict, index=df.index)
        
    # -------------------- Momentum Indicators --------------------
    
    def rsi(
        self,
        series: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series.
            period: RSI period.
            
        Returns:
            RSI series (0-100).
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_rsi_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [7, 14]
    ) -> pd.DataFrame:
        """
        Calculate RSI-based features.
        
        Args:
            df: DataFrame with 'close' column.
            periods: List of RSI periods.
            
        Returns:
            DataFrame with RSI features.
        """
        close = df['close']
        feature_dict = {}
        
        for period in periods:
            rsi = self.rsi(close, period)
            feature_dict[f'rsi_{period}'] = rsi
            feature_dict[f'rsi_{period}_change'] = rsi.diff(5)  # Rate of change
            feature_dict[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
            feature_dict[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
            
            # RSI divergence (simplified)
            price_change = close.diff(10)
            rsi_change = rsi.diff(10)
            feature_dict[f'rsi_{period}_divergence'] = np.sign(price_change) != np.sign(rsi_change)
            
        return pd.DataFrame(feature_dict, index=df.index)
        
    def macd(
        self,
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD.
        
        Args:
            series: Price series.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.
            
        Returns:
            Tuple of (MACD line, signal line, histogram).
        """
        ema_fast = self.ema(series, fast)
        ema_slow = self.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def calculate_macd_features(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD-based features.
        
        Args:
            df: DataFrame with 'close' column.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.
            
        Returns:
            DataFrame with MACD features.
        """
        macd_line, signal_line, histogram = self.macd(df['close'], fast, slow, signal)
        
        feature_dict = {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'macd_histogram_change': histogram.diff(),
            'macd_cross': (
                (macd_line > signal_line).astype(int) -
                (macd_line < signal_line).astype(int)
            ),
            'macd_above_zero': (macd_line > 0).astype(int)
        }
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    def stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with high, low, close columns.
            k_period: %K period.
            d_period: %D period.
            smooth_k: %K smoothing period.
            
        Returns:
            Tuple of (%K, %D).
        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
        
    def calculate_stochastic_features(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Calculate Stochastic-based features.
        
        Args:
            df: DataFrame with OHLC data.
            k_period: %K period.
            d_period: %D period.
            smooth_k: %K smoothing period.
            
        Returns:
            DataFrame with Stochastic features.
        """
        stoch_k, stoch_d = self.stochastic(df, k_period, d_period, smooth_k)
        
        feature_dict = {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_cross': (
                (stoch_k > stoch_d).astype(int) -
                (stoch_k < stoch_d).astype(int)
            ),
            'stoch_overbought': (stoch_k > 80).astype(int),
            'stoch_oversold': (stoch_k < 20).astype(int)
        }
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    # -------------------- Volatility Indicators --------------------
    
    def atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with high, low, close columns.
            period: ATR period.
            
        Returns:
            ATR series.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
        
    def calculate_atr_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [7, 14, 21]
    ) -> pd.DataFrame:
        """
        Calculate ATR-based features.
        
        Args:
            df: DataFrame with OHLC data.
            periods: List of ATR periods.
            
        Returns:
            DataFrame with ATR features.
        """
        feature_dict = {}
        atr_values = {}  # Store for ratio calculations
        
        for period in periods:
            atr = self.atr(df, period)
            atr_values[period] = atr
            feature_dict[f'atr_{period}'] = atr
            feature_dict[f'atr_{period}_pct'] = atr / df['close'] * 100  # ATR as % of price
            
        # ATR ratio (short-term vs long-term)
        if 7 in periods and 21 in periods:
            feature_dict['atr_ratio'] = atr_values[7] / atr_values[21]
            
        # Historical ATR comparison
        if 14 in periods:
            atr_14_ma = atr_values[14].rolling(window=50).mean()
            feature_dict['atr_14_ma'] = atr_14_ma
            feature_dict['atr_vs_avg'] = atr_values[14] / atr_14_ma
        else:
            feature_dict['atr_14_ma'] = np.nan
            feature_dict['atr_vs_avg'] = np.nan
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    def bollinger_bands(
        self,
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            series: Price series.
            period: Moving average period.
            std_dev: Standard deviation multiplier.
            
        Returns:
            Tuple of (upper band, middle band, lower band).
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
        
    def calculate_bollinger_features(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands features.
        
        Args:
            df: DataFrame with 'close' column.
            period: Moving average period.
            std_dev: Standard deviation multiplier.
            
        Returns:
            DataFrame with Bollinger features.
        """
        close = df['close']
        upper, middle, lower = self.bollinger_bands(close, period, std_dev)
        
        bb_width = (upper - lower) / middle * 100  # Width as %
        
        feature_dict = {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': bb_width,
            'bb_position': (close - lower) / (upper - lower),  # 0-1 position
            'bb_squeeze': bb_width < bb_width.rolling(window=50).quantile(0.2)
        }
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    # -------------------- Volume Indicators --------------------
    
    def calculate_volume_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with volume column.
            periods: List of periods for moving averages.
            
        Returns:
            DataFrame with volume features.
        """
        volume = df['volume']
        close = df['close']
        feature_dict = {}
        
        # Volume moving averages and ratios
        for period in periods:
            vol_ma = volume.rolling(window=period).mean()
            feature_dict[f'volume_ma_{period}'] = vol_ma
            feature_dict[f'volume_ratio_{period}'] = volume / vol_ma
            
        # Volume delta (buy vs sell volume - simplified)
        price_change = close.diff()
        volume_delta = np.where(price_change > 0, volume, -volume)
        feature_dict['volume_delta'] = volume_delta
        # FIXED: Use rolling sum with fixed window (stable across data lengths)
        feature_dict['volume_delta_sum_20'] = pd.Series(volume_delta, index=df.index).rolling(window=20).sum()
        
        # FIXED: OBV removed - cumsum() depends on data window start!
        # In backtest: data starts from 2017 → OBV = huge number
        # In live: data starts from last 3000 candles → OBV = much smaller
        # This causes MASSIVE feature drift between backtest and live!
        # 
        # Instead, we use a ROLLING OBV that's stable across data lengths:
        # Rolling OBV: sum of signed volume over last N bars
        obv_rolling = pd.Series(np.sign(price_change) * volume, index=df.index).rolling(window=50).sum()
        feature_dict['obv_rolling_50'] = obv_rolling
        
        # OBV slope (rate of change) - stable feature
        feature_dict['obv_slope'] = obv_rolling.diff(10) / volume.rolling(10).mean()
        
        # Volume trend (slope of volume over last 20 bars)
        # FIXED: Normalize by average volume to make it scale-independent
        vol_mean = volume.rolling(window=20).mean()
        feature_dict['volume_trend'] = volume.rolling(window=20).apply(
            lambda x: np.polyfit(range(len(x)), x / x.mean() if x.mean() > 0 else x, 1)[0]
        )
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    # -------------------- Combined Features --------------------
    
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data.
            config: Configuration dictionary (optional).
            
        Returns:
            DataFrame with all indicator features.
        """
        if config is None:
            config = {
                'ema_periods': [9, 21, 50, 200],
                'rsi_periods': [7, 14],
                'atr_periods': [7, 14, 21],
                'volume_periods': [5, 10, 20],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'stochastic': {'k_period': 14, 'd_period': 3, 'smooth_k': 3},
                'bollinger': {'period': 20, 'std_dev': 2.0}
            }
            
        all_features = pd.DataFrame(index=df.index)
        
        # Trend indicators
        ema_features = self.calculate_ema_features(df, config.get('ema_periods', [9, 21, 50, 200]))
        all_features = pd.concat([all_features, ema_features], axis=1)
        
        # Momentum indicators
        rsi_features = self.calculate_rsi_features(df, config.get('rsi_periods', [7, 14]))
        all_features = pd.concat([all_features, rsi_features], axis=1)
        
        macd_config = config.get('macd', {})
        macd_features = self.calculate_macd_features(
            df,
            macd_config.get('fast', 12),
            macd_config.get('slow', 26),
            macd_config.get('signal', 9)
        )
        all_features = pd.concat([all_features, macd_features], axis=1)
        
        stoch_config = config.get('stochastic', {})
        stoch_features = self.calculate_stochastic_features(
            df,
            stoch_config.get('k_period', 14),
            stoch_config.get('d_period', 3),
            stoch_config.get('smooth_k', 3)
        )
        all_features = pd.concat([all_features, stoch_features], axis=1)
        
        # Volatility indicators
        atr_features = self.calculate_atr_features(df, config.get('atr_periods', [7, 14, 21]))
        all_features = pd.concat([all_features, atr_features], axis=1)
        
        bb_config = config.get('bollinger', {})
        bb_features = self.calculate_bollinger_features(
            df,
            bb_config.get('period', 20),
            bb_config.get('std_dev', 2.0)
        )
        all_features = pd.concat([all_features, bb_features], axis=1)
        
        # Volume indicators
        volume_features = self.calculate_volume_features(df, config.get('volume_periods', [5, 10, 20]))
        all_features = pd.concat([all_features, volume_features], axis=1)
        
        return all_features

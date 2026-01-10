"""
AI Trading Bot - Feature Engine
Orchestrates feature generation from all sources.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.constants import TimeFrame, TradingSession
from src.utils.helpers import encode_cyclical, normalize_series, clip_outliers

from .indicators import TechnicalIndicators
from .patterns import CandlePatterns
from .market_structure import MarketStructure


class FeatureEngine:
    """
    Central feature engineering engine.
    
    Combines:
    - Technical indicators
    - Candlestick patterns
    - Market structure analysis
    - Time-based features
    - Multi-timeframe features
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize the feature engine.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        
        # Initialize component analyzers
        self.indicators = TechnicalIndicators()
        self.patterns = CandlePatterns(
            body_threshold=self.config.get('pattern_body_threshold', 0.1),
            shadow_ratio=self.config.get('pattern_shadow_ratio', 2.0)
        )
        self.market_structure = MarketStructure(
            swing_period=self.config.get('swing_period', 5),
            sr_lookback=self.config.get('sr_lookback', 100)
        )
        
        # Feature normalization settings
        self.normalization_window = self.config.get('normalization_window', 500)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        
        # Feature lists
        self._feature_columns: List[str] = []
        
    def generate_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic OHLCV-derived features.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with OHLCV features.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # High-low range (calculate first for reuse)
        hl_range = (high - low) / close
        hl_range_ma = hl_range.rolling(20).mean()
        
        # Consecutive candle direction
        direction = (close > open_price).astype(int) * 2 - 1
        
        # Helper function to count consecutive values using rolling window (stable approach)
        # FIXED: groupby().cumsum() depends on data window start, causing backtest/live divergence
        def count_consecutive(condition: pd.Series, max_count: int = 20) -> pd.Series:
            """Count consecutive True values, capped at max_count for stability."""
            result = pd.Series(0, index=condition.index)
            count = 0
            for i in range(len(condition)):
                if condition.iloc[i]:
                    count += 1
                    result.iloc[i] = min(count, max_count)
                else:
                    count = 0
                    result.iloc[i] = 0
            return result
        
        feature_dict = {
            # Price changes
            'return_1': close.pct_change(1),
            'return_5': close.pct_change(5),
            'return_10': close.pct_change(10),
            'return_20': close.pct_change(20),
            # Log returns
            'log_return_1': np.log(close / close.shift(1)),
            # High-low range
            'hl_range': hl_range,
            'hl_range_ma': hl_range_ma,
            'hl_range_ratio': hl_range / hl_range_ma,
            # Close position in range
            'close_position': (close - low) / (high - low).replace(0, np.nan),
            # Gap
            'gap': (open_price - close.shift(1)) / close.shift(1),
            # Consecutive candle direction - using stable counting method
            'consecutive_up': count_consecutive(direction == 1),
            'consecutive_down': count_consecutive(direction == -1)
        }
        
        return pd.DataFrame(feature_dict, index=df.index)
        
    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-based features.
        
        Args:
            df: DataFrame with datetime index.
            
        Returns:
            DataFrame with time features.
        """
        features = pd.DataFrame(index=df.index)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex")
            return features
            
        # Hour of day (cyclical encoding)
        hour = df.index.hour
        features['hour_sin'], features['hour_cos'] = zip(*[encode_cyclical(h, 24) for h in hour])
        
        # Day of week (cyclical encoding)
        day = df.index.dayofweek
        features['day_sin'], features['day_cos'] = zip(*[encode_cyclical(d, 7) for d in day])
        
        # Day of month
        features['day_of_month'] = df.index.day
        
        # Trading session
        def get_session(hour):
            if 0 <= hour < 8:
                return TradingSession.ASIAN.value
            elif 7 <= hour < 16:
                return TradingSession.EUROPEAN.value
            else:
                return TradingSession.AMERICAN.value
                
        sessions = [get_session(h) for h in hour]
        features['session'] = sessions
        
        # One-hot encode sessions
        for s in TradingSession:
            features[f'session_{s.value}'] = (features['session'] == s.value).astype(int)
            
        features.drop('session', axis=1, inplace=True)
        
        # Is weekend (crypto trades 24/7 but weekends are different)
        features['is_weekend'] = (day >= 5).astype(int)
        
        return features
        
    def generate_multitimeframe_features(
        self,
        dataframes: Dict[str, pd.DataFrame],
        base_timeframe: str = "5m"
    ) -> pd.DataFrame:
        """
        Generate features from multiple timeframes.
        
        Args:
            dataframes: Dictionary of {timeframe: DataFrame}.
            base_timeframe: Base timeframe to align to.
            
        Returns:
            DataFrame with multi-timeframe features.
        """
        if base_timeframe not in dataframes:
            logger.error(f"Base timeframe {base_timeframe} not in dataframes")
            return pd.DataFrame()
            
        base_df = dataframes[base_timeframe]
        features = pd.DataFrame(index=base_df.index)
        
        for tf, df in dataframes.items():
            if tf == base_timeframe:
                continue
                
            # Calculate trend for each timeframe
            ema_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            
            trend = (ema_20 > ema_50).astype(int) - (ema_20 < ema_50).astype(int)
            
            # Resample to base timeframe
            trend_resampled = trend.reindex(base_df.index, method='ffill')
            features[f'trend_{tf}'] = trend_resampled
            
            # RSI from higher timeframe
            rsi = self.indicators.rsi(df['close'], 14)
            rsi_resampled = rsi.reindex(base_df.index, method='ffill')
            features[f'rsi_{tf}'] = rsi_resampled
            
        # Timeframe alignment score
        trend_cols = [col for col in features.columns if col.startswith('trend_')]
        if trend_cols:
            features['tf_alignment'] = features[trend_cols].mean(axis=1)
            features['tf_consensus'] = (features[trend_cols].abs().mean(axis=1) > 0.5).astype(int)
            
        return features
        
    def normalize_features(
        self,
        features: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize feature values.
        
        Args:
            features: Feature DataFrame.
            method: Normalization method ('zscore', 'minmax').
            
        Returns:
            Normalized DataFrame.
        """
        # Columns to skip (non-numeric or categorical)
        skip_cols = set()
        for col in features.columns:
            # Skip non-numeric columns
            if features[col].dtype == 'object' or features[col].dtype.name == 'category':
                skip_cols.add(col)
            # Skip binary/categorical columns
            elif features[col].nunique() <= 3:
                skip_cols.add(col)
            # Skip session and regime columns
            elif col.startswith('session_') or col.startswith('regime_') or col == 'regime':
                skip_cols.add(col)
        
        # Build normalized columns as dict for efficiency
        normalized_cols = {}
        for col in features.columns:
            if col in skip_cols:
                normalized_cols[col] = features[col]
            else:
                # Rolling normalization
                norm_col = normalize_series(
                    features[col],
                    window=self.normalization_window,
                    method=method
                )
                # Clip outliers
                normalized_cols[col] = clip_outliers(norm_col, self.outlier_threshold)
        
        # Create DataFrame from dict in one operation (avoids fragmentation)
        return pd.DataFrame(normalized_cols, index=features.index)
        
    def select_features(
        self,
        features: pd.DataFrame,
        method: str = 'variance',
        n_features: Optional[int] = None,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Select most important features.
        
        Args:
            features: Feature DataFrame.
            method: Selection method ('variance', 'correlation').
            n_features: Number of features to select.
            threshold: Variance threshold for filtering.
            
        Returns:
            DataFrame with selected features.
        """
        if method == 'variance':
            # Remove low variance features
            variances = features.var()
            high_var_cols = variances[variances > threshold].index.tolist()
            selected = features[high_var_cols]
            
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            selected = features.drop(columns=to_drop)
            
        else:
            selected = features
            
        if n_features and len(selected.columns) > n_features:
            # Keep top n by variance
            variances = selected.var().sort_values(ascending=False)
            selected = selected[variances.head(n_features).index]
            
        logger.info(f"Selected {len(selected.columns)} features from {len(features.columns)}")
        return selected
        
    def generate_all_features(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        additional_timeframes: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Generate all features for a single timeframe.
        
        Args:
            df: DataFrame with OHLCV data.
            normalize: Whether to normalize features.
            additional_timeframes: Additional timeframe data for MTF features.
            
        Returns:
            Complete feature DataFrame.
        """
        all_features = pd.DataFrame(index=df.index)
        
        # Basic OHLCV features
        ohlcv_features = self.generate_ohlcv_features(df)
        all_features = pd.concat([all_features, ohlcv_features], axis=1)
        
        # Technical indicators
        indicator_features = self.indicators.calculate_all_indicators(df)
        all_features = pd.concat([all_features, indicator_features], axis=1)
        
        # Candlestick patterns
        pattern_features = self.patterns.detect_all_patterns(df)
        all_features = pd.concat([all_features, pattern_features], axis=1)
        
        # Market structure
        structure_features = self.market_structure.calculate_all_structure_features(df)
        all_features = pd.concat([all_features, structure_features], axis=1)
        
        # Time features
        time_features = self.generate_time_features(df)
        all_features = pd.concat([all_features, time_features], axis=1)
        
        # Multi-timeframe features
        if additional_timeframes:
            mtf_features = self.generate_multitimeframe_features(
                {**additional_timeframes, '5m': df},
                base_timeframe='5m'
            )
            all_features = pd.concat([all_features, mtf_features], axis=1)
            
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Handle infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        
        # Normalize if requested
        if normalize:
            all_features = self.normalize_features(all_features)
            
        # Store feature columns
        self._feature_columns = all_features.columns.tolist()
        
        return all_features
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self._feature_columns
        
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get features grouped by category.
        
        Returns:
            Dictionary mapping group names to feature lists.
        """
        groups = {
            'ohlcv': [],
            'trend': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'pattern': [],
            'structure': [],
            'time': [],
            'mtf': []
        }
        
        for col in self._feature_columns:
            if any(x in col for x in ['return', 'gap', 'consecutive', 'close_position', 'hl_range']):
                groups['ohlcv'].append(col)
            elif any(x in col for x in ['ema', 'sma', 'trend']):
                groups['trend'].append(col)
            elif any(x in col for x in ['rsi', 'macd', 'stoch']):
                groups['momentum'].append(col)
            elif any(x in col for x in ['atr', 'bb_', 'volatility', 'parkinson']):
                groups['volatility'].append(col)
            elif any(x in col for x in ['volume', 'obv']):
                groups['volume'].append(col)
            elif any(x in col for x in ['doji', 'hammer', 'engulf', 'star', 'harami', 'marubozu', 'spinning', 'pattern']):
                groups['pattern'].append(col)
            elif any(x in col for x in ['swing', 'regime', 'support', 'resistance', 'breakout', 'structure', 'higher', 'lower']):
                groups['structure'].append(col)
            elif any(x in col for x in ['hour', 'day', 'session', 'weekend']):
                groups['time'].append(col)
            elif 'tf_' in col or col.startswith('trend_') or col.startswith('rsi_'):
                groups['mtf'].append(col)
                
        return groups

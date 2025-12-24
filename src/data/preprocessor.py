"""
AI Trading Bot - Data Preprocessor
Prepares raw market data for ML models and technical analysis.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

from .models import Candle


class DataPreprocessor:
    """
    Preprocesses market data for ML models and technical indicators.
    
    Provides methods for converting data formats, resampling timeframes,
    filling gaps, normalization, and outlier handling.
    
    Attributes:
        logger: Logger instance.
    """
    
    # Timeframe to pandas frequency mapping
    TIMEFRAME_MAP = {
        '1m': '1min',
        '3m': '3min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '1d': '1D',
        '3d': '3D',
        '1w': '1W',
        '1M': '1ME',
    }
    
    # Timeframe to seconds mapping
    TIMEFRAME_SECONDS = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '8h': 28800,
        '12h': 43200,
        '1d': 86400,
        '3d': 259200,
        '1w': 604800,
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize DataPreprocessor.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)

    def candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """
        Convert list of Candle objects to pandas DataFrame.
        
        Args:
            candles: List of Candle objects.
            
        Returns:
            DataFrame with timestamp index and OHLCV columns.
        """
        if not candles:
            return pd.DataFrame(
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
        
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df

    def dataframe_to_candles(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> List[Candle]:
        """
        Convert DataFrame back to list of Candle objects.
        
        Args:
            df: DataFrame with OHLCV data.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            
        Returns:
            List of Candle objects.
        """
        candles = []
        
        for timestamp, row in df.iterrows():
            candle = Candle(
                timestamp=timestamp if isinstance(timestamp, datetime) else pd.Timestamp(timestamp).to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                symbol=symbol,
                timeframe=timeframe
            )
            candles.append(candle)
        
        return candles

    def resample_candles(
        self,
        candles: List[Candle],
        target_timeframe: str
    ) -> List[Candle]:
        """
        Resample candles to a different (larger) timeframe.
        
        Args:
            candles: List of Candle objects (smaller timeframe).
            target_timeframe: Target timeframe to resample to.
            
        Returns:
            List of Candle objects in new timeframe.
        """
        if not candles:
            return []
        
        # Get symbol from first candle
        symbol = candles[0].symbol
        
        # Convert to DataFrame
        df = self.candles_to_dataframe(candles)
        
        # Get pandas frequency
        freq = self.TIMEFRAME_MAP.get(target_timeframe)
        if not freq:
            self.logger.warning(f"Unknown timeframe: {target_timeframe}")
            return candles
        
        # Resample OHLCV
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Drop rows with NaN values
        resampled.dropna(inplace=True)
        
        # Convert back to candles
        return self.dataframe_to_candles(resampled, symbol, target_timeframe)

    def fill_gaps(
        self,
        candles: List[Candle],
        timeframe: str
    ) -> List[Candle]:
        """
        Fill gaps in candle data with synthetic candles.
        
        Missing candles are filled with OHLC equal to previous close
        and volume equal to zero.
        
        Args:
            candles: List of Candle objects with possible gaps.
            timeframe: Expected timeframe of candles.
            
        Returns:
            List of Candle objects without gaps.
        """
        if not candles or len(candles) < 2:
            return candles
        
        # Sort by timestamp
        sorted_candles = sorted(candles, key=lambda x: x.timestamp)
        
        # Get expected interval
        interval_seconds = self.TIMEFRAME_SECONDS.get(timeframe)
        if not interval_seconds:
            self.logger.warning(f"Unknown timeframe: {timeframe}")
            return candles
        
        interval = timedelta(seconds=interval_seconds)
        symbol = sorted_candles[0].symbol
        
        filled_candles = [sorted_candles[0]]
        
        for i in range(1, len(sorted_candles)):
            current = sorted_candles[i]
            previous = filled_candles[-1]
            
            # Calculate expected next timestamp
            expected_time = previous.timestamp + interval
            
            # Fill any gaps
            while expected_time < current.timestamp:
                # Create synthetic candle with previous close
                synthetic = Candle(
                    timestamp=expected_time,
                    open=previous.close,
                    high=previous.close,
                    low=previous.close,
                    close=previous.close,
                    volume=0.0,
                    symbol=symbol,
                    timeframe=timeframe
                )
                filled_candles.append(synthetic)
                expected_time += interval
            
            filled_candles.append(current)
        
        if len(filled_candles) > len(candles):
            self.logger.info(
                f"Filled {len(filled_candles) - len(candles)} gaps in {timeframe} data"
            )
        
        return filled_candles

    def normalize(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: str = 'zscore'
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], 'Scaler']:
        """
        Normalize data for ML models.
        
        Args:
            data: NumPy array or DataFrame to normalize.
            method: Normalization method ('minmax', 'zscore', 'robust').
            
        Returns:
            Tuple of (normalized_data, scaler).
            The scaler can be used for inverse transformation.
        """
        scaler = Scaler(method=method)
        normalized = scaler.fit_transform(data)
        return normalized, scaler

    def detect_outliers(
        self,
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers in data.
        
        Args:
            data: NumPy array of values.
            method: Detection method ('iqr' or 'zscore').
            threshold: Threshold for outlier detection.
                      For IQR: multiplier (default 1.5).
                      For zscore: number of standard deviations (default 3).
            
        Returns:
            Boolean array where True indicates an outlier.
        """
        data = np.asarray(data).flatten()
        
        if method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (data < lower) | (data > upper)
            
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros(len(data), dtype=bool)
            z_scores = np.abs((data - mean) / std)
            return z_scores > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def clip_outliers(
        self,
        data: np.ndarray,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> np.ndarray:
        """
        Clip outliers to percentile bounds.
        
        Args:
            data: NumPy array of values.
            lower_percentile: Lower percentile bound (default: 1).
            upper_percentile: Upper percentile bound (default: 99).
            
        Returns:
            Array with clipped values.
        """
        data = np.asarray(data)
        lower = np.percentile(data, lower_percentile)
        upper = np.percentile(data, upper_percentile)
        return np.clip(data, lower, upper)

    def align_multi_timeframe(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align DataFrames from different timeframes to common time range.
        
        Args:
            data_dict: Dictionary mapping timeframe to DataFrame.
            
        Returns:
            Dictionary with aligned DataFrames.
        """
        if not data_dict:
            return {}
        
        # Find common time range
        start_times = []
        end_times = []
        
        for tf, df in data_dict.items():
            if not df.empty:
                start_times.append(df.index.min())
                end_times.append(df.index.max())
        
        if not start_times:
            return data_dict
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        self.logger.debug(
            f"Aligning to common range: {common_start} - {common_end}"
        )
        
        # Align each DataFrame
        aligned = {}
        
        for tf, df in data_dict.items():
            # Filter to common range
            mask = (df.index >= common_start) & (df.index <= common_end)
            aligned_df = df[mask].copy()
            
            # Forward fill for larger timeframes
            aligned_df = aligned_df.ffill()
            
            aligned[tf] = aligned_df
        
        return aligned

    def calculate_returns(
        self,
        prices: Union[np.ndarray, pd.Series],
        method: str = 'log'
    ) -> Union[np.ndarray, pd.Series]:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price array or series.
            method: Return calculation method ('log' or 'simple').
            
        Returns:
            Returns array or series.
        """
        prices = np.asarray(prices)
        
        if method == 'log':
            returns = np.log(prices[1:] / prices[:-1])
        elif method == 'simple':
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        # Prepend NaN for first value
        returns = np.concatenate([[np.nan], returns])
        
        if isinstance(prices, pd.Series):
            return pd.Series(returns, index=prices.index)
        
        return returns

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to DataFrame.
        
        Args:
            df: DataFrame with datetime index.
            
        Returns:
            DataFrame with added time features.
        """
        result = df.copy()
        
        # Extract time components
        result['hour'] = result.index.hour
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        
        # Cyclical encoding for hour
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        
        # Cyclical encoding for day of week
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        return result


class Scaler:
    """
    Scaler for data normalization with inverse transform support.
    
    Attributes:
        method: Normalization method.
    """
    
    def __init__(self, method: str = 'zscore'):
        """
        Initialize Scaler.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'robust').
        """
        self.method = method
        self._params: Dict[str, Any] = {}
        self._is_fitted = False

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> 'Scaler':
        """
        Fit the scaler to data.
        
        Args:
            data: Data to fit scaler on.
            
        Returns:
            Self for chaining.
        """
        if isinstance(data, pd.DataFrame):
            values = data.values
        else:
            values = np.asarray(data)
        
        if self.method == 'minmax':
            self._params['min'] = np.min(values, axis=0)
            self._params['max'] = np.max(values, axis=0)
            # Avoid division by zero
            range_val = self._params['max'] - self._params['min']
            if np.isscalar(range_val) or range_val.ndim == 0:
                range_val = float(range_val) if range_val != 0 else 1.0
            else:
                range_val = np.where(range_val == 0, 1, range_val)
            self._params['range'] = range_val
            
        elif self.method == 'zscore':
            self._params['mean'] = np.mean(values, axis=0)
            self._params['std'] = np.std(values, axis=0)
            # Avoid division by zero
            if np.isscalar(self._params['std']) or self._params['std'].ndim == 0:
                self._params['std'] = float(self._params['std']) if self._params['std'] != 0 else 1.0
            else:
                self._params['std'] = np.where(self._params['std'] == 0, 1, self._params['std'])
            
        elif self.method == 'robust':
            self._params['median'] = np.median(values, axis=0)
            q1 = np.percentile(values, 25, axis=0)
            q3 = np.percentile(values, 75, axis=0)
            iqr = q3 - q1
            if np.isscalar(iqr) or iqr.ndim == 0:
                iqr = float(iqr) if iqr != 0 else 1.0
            else:
                iqr = np.where(iqr == 0, 1, iqr)
            self._params['iqr'] = iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self._is_fitted = True
        return self

    def transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to transform.
            
        Returns:
            Transformed data (same type as input).
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            values = data.values
        else:
            values = np.asarray(data)
        
        if self.method == 'minmax':
            normalized = (values - self._params['min']) / self._params['range']
            
        elif self.method == 'zscore':
            normalized = (values - self._params['mean']) / self._params['std']
            
        elif self.method == 'robust':
            normalized = (values - self._params['median']) / self._params['iqr']
        
        if is_dataframe:
            return pd.DataFrame(normalized, index=data.index, columns=data.columns)
        
        return normalized

    def fit_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit and transform in one step.
        
        Args:
            data: Data to fit and transform.
            
        Returns:
            Transformed data.
        """
        return self.fit(data).transform(data)

    def inverse_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data.
            
        Returns:
            Data in original scale.
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            values = data.values
        else:
            values = np.asarray(data)
        
        if self.method == 'minmax':
            original = values * self._params['range'] + self._params['min']
            
        elif self.method == 'zscore':
            original = values * self._params['std'] + self._params['mean']
            
        elif self.method == 'robust':
            original = values * self._params['iqr'] + self._params['median']
        
        if is_dataframe:
            return pd.DataFrame(original, index=data.index, columns=data.columns)
        
        return original

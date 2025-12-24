"""
AI Trading Bot - Unit Tests for Data Module - Preprocessor
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.models import Candle
from src.data.preprocessor import DataPreprocessor, Scaler


@pytest.fixture
def sample_candles():
    """Create sample candles for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    candles = []
    
    for i in range(100):
        candle = Candle(
            timestamp=base_time + timedelta(hours=i),
            open=50000 + i * 10,
            high=50100 + i * 10,
            low=49900 + i * 10,
            close=50050 + i * 10,
            volume=100 + i,
            symbol="BTC/USDT:USDT",
            timeframe="1h"
        )
        candles.append(candle)
    
    return candles


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return DataPreprocessor()


class TestCandlesToDataframe:
    """Tests for candles_to_dataframe method."""
    
    def test_conversion(self, preprocessor, sample_candles):
        """Test converting candles to DataFrame."""
        df = preprocessor.candles_to_dataframe(sample_candles)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert df.index.name == 'timestamp'
    
    def test_empty_candles(self, preprocessor):
        """Test converting empty list."""
        df = preprocessor.candles_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestDataframeToCandles:
    """Tests for dataframe_to_candles method."""
    
    def test_conversion(self, preprocessor, sample_candles):
        """Test converting DataFrame back to candles."""
        df = preprocessor.candles_to_dataframe(sample_candles)
        result = preprocessor.dataframe_to_candles(df, "BTC/USDT:USDT", "1h")
        
        assert len(result) == len(sample_candles)
        assert result[0].symbol == "BTC/USDT:USDT"
        assert result[0].timeframe == "1h"


class TestResampleCandles:
    """Tests for resample_candles method."""
    
    def test_resample_to_larger_timeframe(self, preprocessor, sample_candles):
        """Test resampling hourly candles to 4h."""
        result = preprocessor.resample_candles(sample_candles, "4h")
        
        # 100 hourly candles should give 25 4-hour candles
        assert len(result) == 25
        assert result[0].timeframe == "4h"
    
    def test_resample_ohlc_aggregation(self, preprocessor):
        """Test that OHLC is correctly aggregated."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        candles = [
            Candle(base_time, 100, 110, 90, 105, 10, "TEST", "1h"),
            Candle(base_time + timedelta(hours=1), 105, 120, 95, 115, 20, "TEST", "1h"),
            Candle(base_time + timedelta(hours=2), 115, 125, 100, 110, 15, "TEST", "1h"),
            Candle(base_time + timedelta(hours=3), 110, 115, 85, 90, 25, "TEST", "1h"),
        ]
        
        result = preprocessor.resample_candles(candles, "4h")
        
        assert len(result) == 1
        assert result[0].open == 100  # First open
        assert result[0].high == 125  # Max high
        assert result[0].low == 85    # Min low
        assert result[0].close == 90  # Last close
        assert result[0].volume == 70  # Sum of volumes
    
    def test_empty_candles(self, preprocessor):
        """Test resampling empty list."""
        result = preprocessor.resample_candles([], "4h")
        assert result == []


class TestFillGaps:
    """Tests for fill_gaps method."""
    
    def test_fill_missing_candles(self, preprocessor):
        """Test filling gaps in candle data."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        # Create candles with a 2-hour gap
        candles = [
            Candle(base_time, 100, 110, 90, 105, 10, "TEST", "1h"),
            Candle(base_time + timedelta(hours=1), 105, 115, 95, 110, 15, "TEST", "1h"),
            # Missing hour 2
            Candle(base_time + timedelta(hours=3), 115, 120, 105, 118, 20, "TEST", "1h"),
        ]
        
        result = preprocessor.fill_gaps(candles, "1h")
        
        assert len(result) == 4  # 3 original + 1 filled
        assert result[2].open == 110  # Previous close
        assert result[2].volume == 0  # Zero volume for synthetic
    
    def test_no_gaps(self, preprocessor, sample_candles):
        """Test that consecutive candles don't get filled."""
        result = preprocessor.fill_gaps(sample_candles, "1h")
        assert len(result) == len(sample_candles)


class TestNormalize:
    """Tests for normalize method."""
    
    def test_zscore_normalization(self, preprocessor):
        """Test z-score normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, scaler = preprocessor.normalize(data, method='zscore')
        
        # Mean should be ~0, std should be ~1
        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01
    
    def test_minmax_normalization(self, preprocessor):
        """Test min-max normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, scaler = preprocessor.normalize(data, method='minmax')
        
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1
    
    def test_robust_normalization(self, preprocessor):
        """Test robust normalization."""
        data = np.array([1, 2, 3, 4, 5, 100])  # With outlier
        normalized, scaler = preprocessor.normalize(data, method='robust')
        
        # Robust should be less affected by outlier
        assert isinstance(normalized, np.ndarray)


class TestDetectOutliers:
    """Tests for detect_outliers method."""
    
    def test_iqr_detection(self, preprocessor):
        """Test IQR outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
        outliers = preprocessor.detect_outliers(data, method='iqr')
        
        assert outliers[-1] == True  # 100 should be detected
        assert sum(outliers[:-1]) == 0  # Others should not be outliers
    
    def test_zscore_detection(self, preprocessor):
        """Test z-score outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = preprocessor.detect_outliers(data, method='zscore', threshold=2)
        
        assert outliers[-1] == True


class TestClipOutliers:
    """Tests for clip_outliers method."""
    
    def test_clip(self, preprocessor):
        """Test clipping outliers."""
        data = np.array([1, 2, 3, 4, 5, 100, -50])
        clipped = preprocessor.clip_outliers(data, lower_percentile=5, upper_percentile=95)
        
        # Values should be clipped
        assert clipped.max() < 100
        assert clipped.min() > -50


class TestAlignMultiTimeframe:
    """Tests for align_multi_timeframe method."""
    
    def test_alignment(self, preprocessor):
        """Test aligning DataFrames."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        # Create DataFrames with different time ranges
        df1 = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=pd.date_range(base, periods=5, freq='1h'))
        
        df2 = pd.DataFrame({
            'close': [200, 201, 202, 203]
        }, index=pd.date_range(base + timedelta(hours=1), periods=4, freq='1h'))
        
        data_dict = {'1h': df1, '1h_delayed': df2}
        aligned = preprocessor.align_multi_timeframe(data_dict)
        
        # Both should have same time range after alignment
        assert aligned['1h'].index.min() >= aligned['1h_delayed'].index.min()
        assert aligned['1h'].index.max() <= aligned['1h_delayed'].index.max()


class TestScaler:
    """Tests for Scaler class."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        scaler = Scaler(method='zscore')
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        transformed = scaler.fit_transform(data)
        
        assert transformed.shape == data.shape
    
    def test_inverse_transform(self):
        """Test inverse transform."""
        scaler = Scaler(method='minmax')
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        transformed = scaler.fit_transform(data)
        restored = scaler.inverse_transform(transformed)
        
        np.testing.assert_array_almost_equal(data, restored)
    
    def test_dataframe_support(self):
        """Test that scaler works with DataFrames."""
        scaler = Scaler(method='zscore')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        transformed = scaler.fit_transform(df)
        
        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == ['a', 'b']
    
    def test_unfitted_error(self):
        """Test error when using unfitted scaler."""
        scaler = Scaler()
        
        with pytest.raises(ValueError):
            scaler.transform(np.array([1, 2, 3]))


class TestCalculateReturns:
    """Tests for calculate_returns method."""
    
    def test_log_returns(self, preprocessor):
        """Test log returns calculation."""
        prices = np.array([100, 105, 110])
        returns = preprocessor.calculate_returns(prices, method='log')
        
        assert np.isnan(returns[0])  # First value is NaN
        assert len(returns) == len(prices)
    
    def test_simple_returns(self, preprocessor):
        """Test simple returns calculation."""
        prices = np.array([100, 105, 110])
        returns = preprocessor.calculate_returns(prices, method='simple')
        
        assert np.isnan(returns[0])
        assert abs(returns[1] - 0.05) < 0.001  # (105-100)/100 = 0.05


class TestAddTimeFeatures:
    """Tests for add_time_features method."""
    
    def test_time_features(self, preprocessor):
        """Test adding time features."""
        base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range(base, periods=3, freq='1h'))
        
        result = preprocessor.add_time_features(df)
        
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'dow_sin' in result.columns
        assert 'dow_cos' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

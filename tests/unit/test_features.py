"""
AI Trading Bot - Unit Tests for Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features import TechnicalIndicators, CandlePatterns, MarketStructure, FeatureEngine


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    np.random.seed(42)
    
    # Generate realistic price data
    price = 50000
    prices = [price]
    for _ in range(199):
        change = np.random.normal(0, 50)
        price = max(1, price + change)
        prices.append(price)
        
    df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 20)) for p in prices],
        'low': [p - abs(np.random.normal(0, 20)) for p in prices],
        'close': [p + np.random.normal(0, 10) for p in prices],
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    }, index=dates)
    
    # Ensure high/low constraints
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""
    
    def test_ema_calculation(self, sample_ohlcv):
        """Test EMA calculation."""
        indicators = TechnicalIndicators()
        ema = indicators.ema(sample_ohlcv['close'], period=20)
        
        assert len(ema) == len(sample_ohlcv)
        assert not ema.isnull().all()
        assert ema.iloc[-1] > 0
        
    def test_rsi_calculation(self, sample_ohlcv):
        """Test RSI calculation."""
        indicators = TechnicalIndicators()
        rsi = indicators.rsi(sample_ohlcv['close'], period=14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all((valid_rsi >= 0) & (valid_rsi <= 100))
        
    def test_macd_calculation(self, sample_ohlcv):
        """Test MACD calculation."""
        indicators = TechnicalIndicators()
        macd_line, signal_line, histogram = indicators.macd(sample_ohlcv['close'])
        
        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal_line) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)
        
    def test_atr_calculation(self, sample_ohlcv):
        """Test ATR calculation."""
        indicators = TechnicalIndicators()
        atr = indicators.atr(sample_ohlcv, period=14)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert all(valid_atr >= 0)
        
    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators()
        upper, middle, lower = indicators.bollinger_bands(sample_ohlcv['close'])
        
        # Upper should be above middle, middle above lower
        valid_idx = ~(upper.isnull() | middle.isnull() | lower.isnull())
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])
        
    def test_all_indicators(self, sample_ohlcv):
        """Test calculate_all_indicators."""
        indicators = TechnicalIndicators()
        features = indicators.calculate_all_indicators(sample_ohlcv)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)
        assert len(features.columns) > 10  # Should have many features


class TestCandlePatterns:
    """Tests for CandlePatterns class."""
    
    def test_doji_detection(self, sample_ohlcv):
        """Test Doji pattern detection."""
        patterns = CandlePatterns()
        doji = patterns.detect_doji(sample_ohlcv)
        
        assert len(doji) == len(sample_ohlcv)
        assert all((doji >= 0) & (doji <= 1))
        
    def test_hammer_detection(self, sample_ohlcv):
        """Test Hammer pattern detection."""
        patterns = CandlePatterns()
        hammer = patterns.detect_hammer(sample_ohlcv)
        
        assert len(hammer) == len(sample_ohlcv)
        
    def test_engulfing_detection(self, sample_ohlcv):
        """Test Engulfing pattern detection."""
        patterns = CandlePatterns()
        engulfing = patterns.detect_engulfing(sample_ohlcv)
        
        assert len(engulfing) == len(sample_ohlcv)
        assert all((engulfing >= -1) & (engulfing <= 1))
        
    def test_all_patterns(self, sample_ohlcv):
        """Test detect_all_patterns."""
        patterns = CandlePatterns()
        result = patterns.detect_all_patterns(sample_ohlcv)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)
        assert 'doji' in result.columns
        assert 'hammer' in result.columns


class TestMarketStructure:
    """Tests for MarketStructure class."""
    
    def test_swing_detection(self, sample_ohlcv):
        """Test swing high/low detection."""
        structure = MarketStructure()
        swings = structure.detect_swings(sample_ohlcv)
        
        assert 'swing_high' in swings.columns
        assert 'swing_low' in swings.columns
        assert swings['swing_high'].dtype == bool
        assert swings['swing_low'].dtype == bool
        
    def test_trend_identification(self, sample_ohlcv):
        """Test trend identification."""
        structure = MarketStructure()
        trend = structure.identify_trend(sample_ohlcv)
        
        assert 'ema_trend' in trend.columns
        assert 'trend_score' in trend.columns
        
    def test_regime_classification(self, sample_ohlcv):
        """Test market regime classification."""
        structure = MarketStructure()
        regime = structure.classify_regime(sample_ohlcv)
        
        assert 'regime' in regime.columns
        assert 'trend_strength' in regime.columns


class TestFeatureEngine:
    """Tests for FeatureEngine class."""
    
    def test_generate_all_features(self, sample_ohlcv):
        """Test full feature generation."""
        engine = FeatureEngine()
        features = engine.generate_all_features(sample_ohlcv, normalize=False)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)
        assert len(features.columns) > 50  # Should have many features
        
    def test_normalized_features(self, sample_ohlcv):
        """Test normalized feature generation."""
        engine = FeatureEngine()
        features = engine.generate_all_features(sample_ohlcv, normalize=True)
        
        # Check that normalized features are bounded (roughly)
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32]:
                valid_values = features[col].dropna()
                if len(valid_values) > 0:
                    # Most normalized values should be within reasonable bounds
                    assert valid_values.abs().quantile(0.95) < 10
                    
    def test_time_features(self, sample_ohlcv):
        """Test time-based feature generation."""
        engine = FeatureEngine()
        time_features = engine.generate_time_features(sample_ohlcv)
        
        assert 'hour_sin' in time_features.columns
        assert 'hour_cos' in time_features.columns
        assert 'day_sin' in time_features.columns
        
    def test_ohlcv_features(self, sample_ohlcv):
        """Test OHLCV feature generation."""
        engine = FeatureEngine()
        features = engine.generate_ohlcv_features(sample_ohlcv)
        
        assert 'return_1' in features.columns
        assert 'hl_range' in features.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

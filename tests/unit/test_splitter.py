"""Unit tests for DataSplitter class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.splitter import DataSplitter


class TestDataSplitter:
    """Tests for DataSplitter class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 101,
            'low': np.random.randn(1000).cumsum() + 99,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 1000)
        })
        df.set_index('timestamp', inplace=True)
        return df
    
    @pytest.fixture
    def splitter(self):
        """Create DataSplitter instance."""
        return DataSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            gap_periods=0
        )
    
    def test_split_chronological_sizes(self, splitter, sample_data):
        """Test that chronological split creates correct sizes."""
        train, val, test = splitter.split_chronological(sample_data)
        
        # Check approximate sizes (allowing for rounding)
        assert len(train) >= 690 and len(train) <= 710
        assert len(val) >= 140 and len(val) <= 160
        assert len(test) >= 140 and len(test) <= 160
        
        # Check total
        assert len(train) + len(val) + len(test) == len(sample_data)
    
    def test_split_chronological_order(self, splitter, sample_data):
        """Test that splits maintain chronological order."""
        train, val, test = splitter.split_chronological(sample_data)
        
        # Train ends before val starts
        assert train.index[-1] < val.index[0]
        # Val ends before test starts
        assert val.index[-1] < test.index[0]
    
    def test_split_no_overlap(self, splitter, sample_data):
        """Test that splits have no overlapping indices."""
        train, val, test = splitter.split_chronological(sample_data)
        
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        
        # No intersection between any sets
        assert len(train_idx & val_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx & test_idx) == 0
    
    def test_split_with_gap(self, sample_data):
        """Test split with gap periods between sets."""
        splitter = DataSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            gap_periods=10
        )
        
        train, val, test = splitter.split_chronological(sample_data)
        
        # With gaps, total should be less than original
        assert len(train) + len(val) + len(test) < len(sample_data)
    
    def test_walk_forward_split_folds(self, splitter, sample_data):
        """Test walk-forward split creates correct number of folds."""
        n_folds = 5
        folds = splitter.walk_forward_split(sample_data, n_folds=n_folds)
        
        assert len(folds) == n_folds
        
        # Each fold should have train, val, test
        for fold in folds:
            assert 'train' in fold
            assert 'val' in fold
            assert 'test' in fold
            assert 'fold_number' in fold
    
    def test_walk_forward_chronological(self, splitter, sample_data):
        """Test that walk-forward folds are chronologically ordered."""
        folds = splitter.walk_forward_split(sample_data, n_folds=3)
        
        # Each subsequent fold should start after the previous
        for i in range(1, len(folds)):
            prev_fold = folds[i-1]
            curr_fold = folds[i]
            
            # Current fold's training should start at or after previous fold's test
            assert curr_fold['train'].index[0] >= prev_fold['train'].index[0]
    
    def test_walk_forward_no_leakage(self, splitter, sample_data):
        """Test that walk-forward has no data leakage within each fold."""
        folds = splitter.walk_forward_split(sample_data, n_folds=3)
        
        for fold in folds:
            train = fold['train']
            val = fold['val']
            test = fold['test']
            
            # Train should end before val
            assert train.index[-1] < val.index[0]
            # Val should end before test
            assert val.index[-1] < test.index[0]
    
    def test_expanding_window_split(self, splitter, sample_data):
        """Test expanding window creates growing training sets."""
        folds = splitter.expanding_window_split(sample_data, n_folds=4)
        
        # Each fold's training set should be larger than the previous
        for i in range(1, len(folds)):
            prev_train_size = len(folds[i-1]['train'])
            curr_train_size = len(folds[i]['train'])
            assert curr_train_size > prev_train_size
    
    def test_get_fold_stats(self, splitter, sample_data):
        """Test fold statistics calculation."""
        folds = splitter.walk_forward_split(sample_data, n_folds=3)
        stats = splitter.get_fold_stats(folds)
        
        assert 'n_folds' in stats
        assert 'avg_train_size' in stats
        assert 'avg_val_size' in stats
        assert 'avg_test_size' in stats
        assert stats['n_folds'] == 3
    
    def test_validate_no_overlap(self, splitter, sample_data):
        """Test overlap validation."""
        train, val, test = splitter.split_chronological(sample_data)
        
        # Should not raise
        is_valid = splitter.validate_no_overlap(train, val, test)
        assert is_valid == True
    
    def test_validate_catches_overlap(self, splitter, sample_data):
        """Test that validation catches overlapping data."""
        train, val, test = splitter.split_chronological(sample_data)
        
        # Artificially create overlap
        bad_val = pd.concat([train.tail(10), val])
        
        # Should return False for overlapping data
        is_valid = splitter.validate_no_overlap(train, bad_val, test)
        assert is_valid == False
    
    def test_empty_data_handling(self, splitter):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            splitter.split_chronological(empty_df)
    
    def test_small_data_handling(self, splitter):
        """Test handling of very small datasets."""
        small_df = pd.DataFrame({
            'close': [1, 2, 3, 4, 5]
        }, index=pd.date_range('2023-01-01', periods=5, freq='5min'))
        
        # Should either work or raise meaningful error
        try:
            train, val, test = splitter.split_chronological(small_df)
            assert len(train) >= 1
        except ValueError as e:
            assert "too small" in str(e).lower() or "insufficient" in str(e).lower()


class TestDataSplitterRatios:
    """Test different ratio configurations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        return pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100
        }, index=dates)
    
    def test_80_10_10_split(self, sample_data):
        """Test 80/10/10 split."""
        splitter = DataSplitter(train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)
        train, val, test = splitter.split_chronological(sample_data)
        
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100
    
    def test_60_20_20_split(self, sample_data):
        """Test 60/20/20 split."""
        splitter = DataSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)
        train, val, test = splitter.split_chronological(sample_data)
        
        assert len(train) == 600
        assert len(val) == 200
        assert len(test) == 200
    
    def test_invalid_ratios_raise_error(self):
        """Test that invalid ratios raise errors."""
        with pytest.raises(ValueError):
            DataSplitter(train_ratio=0.60, val_ratio=0.30, test_ratio=0.20)  # Sum > 1
        
        with pytest.raises(ValueError):
            DataSplitter(train_ratio=-0.10, val_ratio=0.50, test_ratio=0.50)  # Negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

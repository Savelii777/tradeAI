"""Unit tests for ModelValidator class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.validator import ModelValidator


class TestModelValidator:
    """Tests for ModelValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create ModelValidator with default thresholds."""
        return ModelValidator()
    
    @pytest.fixture
    def strict_validator(self):
        """Create ModelValidator with strict thresholds."""
        return ModelValidator(thresholds={
            'max_train_val_gap': 0.10,
            'max_realistic_winrate': 0.55,
            'max_realistic_monthly_return': 0.50,
            'max_sharpe_ratio': 2.0
        })
    
    # Test overfitting detection
    def test_no_overfitting_when_metrics_close(self, validator):
        """Test that similar train/val metrics don't trigger overfitting."""
        is_overfit, metrics = validator.check_overfitting(
            train_score=0.75,
            val_score=0.70,
            train_loss=0.25,
            val_loss=0.30
        )
        
        assert is_overfit == False
    
    def test_detects_overfitting_large_gap(self, validator):
        """Test detection of overfitting from large train/val gap."""
        is_overfit, metrics = validator.check_overfitting(
            train_score=0.95,
            val_score=0.55,
            train_loss=0.05,
            val_loss=0.45
        )
        
        assert is_overfit == True
        assert 'train_val_gap' in metrics
        assert metrics['train_val_gap'] > 0.20
    
    def test_overfitting_threshold_respected(self, strict_validator):
        """Test that custom thresholds are respected."""
        # This gap is OK for default but not for strict
        is_overfit, _ = strict_validator.check_overfitting(
            train_score=0.80,
            val_score=0.65,  # Gap of 0.15
            train_loss=0.20,
            val_loss=0.35
        )
        
        assert is_overfit == True  # 0.15 > 0.10 strict threshold
    
    # Test realistic result validation
    def test_realistic_results_pass(self, validator):
        """Test that realistic results pass validation."""
        realistic_results = {
            'win_rate': 55.0,  # 55% win rate
            'total_return': 25.0,  # 25% monthly
            'sharpe_ratio': 1.5,
            'profit_factor': 1.3,
            'max_drawdown': 15.0
        }
        
        is_realistic, warnings = validator.validate_results(realistic_results)
        assert is_realistic == True
        assert len(warnings) == 0
    
    def test_unrealistic_winrate_detected(self, validator):
        """Test that unrealistic win rates are flagged."""
        unrealistic = {
            'win_rate': 85.0,  # 85% - too high
            'total_return': 25.0,
            'sharpe_ratio': 1.5,
            'profit_factor': 1.3,
            'max_drawdown': 10.0
        }
        
        is_realistic, warnings = validator.validate_results(unrealistic)
        assert is_realistic == False
        assert any('win_rate' in w.lower() or 'win rate' in w.lower() for w in warnings)
    
    def test_unrealistic_return_detected(self, validator):
        """Test that unrealistic returns are flagged."""
        unrealistic = {
            'win_rate': 55.0,
            'total_return': 500.0,  # 500% - way too high
            'sharpe_ratio': 1.5,
            'profit_factor': 1.3,
            'max_drawdown': 5.0
        }
        
        is_realistic, warnings = validator.validate_results(unrealistic)
        assert is_realistic == False
        assert any('return' in w.lower() for w in warnings)
    
    def test_unrealistic_sharpe_detected(self, validator):
        """Test that unrealistic Sharpe ratios are flagged."""
        unrealistic = {
            'win_rate': 55.0,
            'total_return': 25.0,
            'sharpe_ratio': 10.0,  # Impossibly high
            'profit_factor': 1.3,
            'max_drawdown': 15.0
        }
        
        is_realistic, warnings = validator.validate_results(unrealistic)
        assert is_realistic == False
        assert any('sharpe' in w.lower() for w in warnings)
    
    def test_multiple_warnings_accumulated(self, validator):
        """Test that multiple issues generate multiple warnings."""
        very_unrealistic = {
            'win_rate': 90.0,
            'total_return': 1000.0,
            'sharpe_ratio': 15.0,
            'profit_factor': 10.0,
            'max_drawdown': 1.0
        }
        
        is_realistic, warnings = validator.validate_results(very_unrealistic)
        assert is_realistic == False
        assert len(warnings) >= 3  # Multiple issues flagged
    
    # Test data leakage validation
    def test_validate_no_leakage_chronological(self, validator):
        """Test leakage detection for chronological data."""
        import pandas as pd
        
        train_idx = pd.date_range('2023-01-01', periods=100, freq='5min')
        val_idx = pd.date_range('2023-01-01 08:20:00', periods=50, freq='5min')  # After train
        test_idx = pd.date_range('2023-01-01 12:30:00', periods=50, freq='5min')  # After val
        
        is_valid, message = validator.validate_no_leakage(train_idx, val_idx, test_idx)
        assert is_valid == True
    
    def test_validate_catches_leakage(self, validator):
        """Test that leakage is caught when data overlaps."""
        import pandas as pd
        
        train_idx = pd.date_range('2023-01-01', periods=100, freq='5min')
        val_idx = pd.date_range('2023-01-01 07:00:00', periods=50, freq='5min')  # Overlaps with train
        test_idx = pd.date_range('2023-01-01 12:00:00', periods=50, freq='5min')
        
        is_valid, message = validator.validate_no_leakage(train_idx, val_idx, test_idx)
        assert is_valid == False
        assert 'overlap' in message.lower() or 'leakage' in message.lower()
    
    # Test fold stability validation
    def test_fold_stability_stable(self, validator):
        """Test stable fold results pass validation."""
        fold_results = [
            {'accuracy': 0.60, 'val_accuracy': 0.58},
            {'accuracy': 0.62, 'val_accuracy': 0.59},
            {'accuracy': 0.59, 'val_accuracy': 0.57},
            {'accuracy': 0.61, 'val_accuracy': 0.60},
        ]
        
        is_stable, metrics = validator.validate_fold_stability(fold_results)
        assert is_stable == True
    
    def test_fold_stability_unstable(self, validator):
        """Test unstable fold results are flagged."""
        fold_results = [
            {'accuracy': 0.90, 'val_accuracy': 0.85},
            {'accuracy': 0.50, 'val_accuracy': 0.45},
            {'accuracy': 0.75, 'val_accuracy': 0.40},
            {'accuracy': 0.95, 'val_accuracy': 0.30},
        ]
        
        is_stable, metrics = validator.validate_fold_stability(fold_results)
        assert is_stable == False
        assert 'std' in metrics or 'variance' in str(metrics).lower()
    
    # Test cross-instrument validation
    def test_cross_instrument_generalization(self, validator):
        """Test cross-instrument generalization check."""
        # Model should perform similarly across instruments
        instrument_results = {
            'BTCUSDT': {'accuracy': 0.60, 'return': 15.0},
            'ETHUSDT': {'accuracy': 0.58, 'return': 12.0},
            'BNBUSDT': {'accuracy': 0.62, 'return': 18.0}
        }
        
        is_general, metrics = validator.cross_instrument_test(instrument_results)
        assert is_general == True
    
    def test_cross_instrument_overfit_to_one(self, validator):
        """Test detection when model overfits to one instrument."""
        instrument_results = {
            'BTCUSDT': {'accuracy': 0.90, 'return': 100.0},  # Much better on BTC
            'ETHUSDT': {'accuracy': 0.52, 'return': 5.0},
            'BNBUSDT': {'accuracy': 0.48, 'return': -2.0}
        }
        
        is_general, metrics = validator.cross_instrument_test(instrument_results)
        assert is_general == False
    
    # Test validation report generation
    def test_generate_validation_report(self, validator):
        """Test validation report generation."""
        train_metrics = {'accuracy': 0.75, 'loss': 0.25}
        val_metrics = {'accuracy': 0.70, 'loss': 0.30}
        test_results = {
            'win_rate': 55.0,
            'total_return': 20.0,
            'sharpe_ratio': 1.2,
            'profit_factor': 1.2,
            'max_drawdown': 12.0
        }
        
        report = validator.generate_validation_report(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_results=test_results
        )
        
        assert 'overfitting_detected' in report
        assert 'realistic_results' in report
        assert 'warnings' in report
        assert 'metrics' in report
    
    def test_report_flags_all_issues(self, validator):
        """Test that report captures all validation issues."""
        train_metrics = {'accuracy': 0.95, 'loss': 0.05}  # Overfit
        val_metrics = {'accuracy': 0.50, 'loss': 0.50}
        test_results = {
            'win_rate': 90.0,  # Unrealistic
            'total_return': 500.0,  # Unrealistic
            'sharpe_ratio': 10.0,  # Unrealistic
            'profit_factor': 5.0,
            'max_drawdown': 2.0
        }
        
        report = validator.generate_validation_report(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_results=test_results
        )
        
        assert report['overfitting_detected'] == True
        assert report['realistic_results'] == False
        assert len(report['warnings']) >= 3


class TestValidatorEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def validator(self):
        return ModelValidator()
    
    def test_missing_result_keys(self, validator):
        """Test handling of missing keys in results."""
        incomplete_results = {
            'win_rate': 55.0
            # Missing other keys
        }
        
        # Should handle gracefully, not crash
        is_realistic, warnings = validator.validate_results(incomplete_results)
        # Either returns True (ignoring missing) or flags as issue
        assert isinstance(is_realistic, bool)
    
    def test_none_values_handled(self, validator):
        """Test handling of None values."""
        results_with_none = {
            'win_rate': None,
            'total_return': 25.0,
            'sharpe_ratio': 1.5,
            'profit_factor': 1.3,
            'max_drawdown': 10.0
        }
        
        # Should not crash
        try:
            is_realistic, warnings = validator.validate_results(results_with_none)
            assert isinstance(is_realistic, bool)
        except (TypeError, ValueError):
            pass  # Acceptable to raise on None values
    
    def test_negative_values(self, validator):
        """Test handling of negative values."""
        negative_results = {
            'win_rate': 45.0,
            'total_return': -20.0,  # Loss
            'sharpe_ratio': -0.5,
            'profit_factor': 0.8,
            'max_drawdown': 25.0
        }
        
        # Should handle losses/negative returns
        is_realistic, warnings = validator.validate_results(negative_results)
        # Negative returns are realistic (losses happen)
        assert isinstance(is_realistic, bool)
    
    def test_zero_trades(self, validator):
        """Test handling of zero trades scenario."""
        no_trades = {
            'win_rate': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0
        }
        
        is_realistic, warnings = validator.validate_results(no_trades)
        # Zero trades might be flagged as suspicious
        assert isinstance(is_realistic, bool)
    
    def test_custom_thresholds_override_defaults(self):
        """Test that custom thresholds fully override defaults."""
        custom_thresholds = {
            'max_train_val_gap': 0.50,  # Very permissive
            'max_realistic_winrate': 0.95,
            'max_realistic_monthly_return': 10.0,
            'max_sharpe_ratio': 20.0
        }
        
        validator = ModelValidator(thresholds=custom_thresholds)
        
        # These would fail with default thresholds
        unrealistic_for_default = {
            'win_rate': 85.0,
            'total_return': 500.0,
            'sharpe_ratio': 15.0,
            'profit_factor': 5.0,
            'max_drawdown': 5.0
        }
        
        is_realistic, warnings = validator.validate_results(unrealistic_for_default)
        # With permissive thresholds, should pass
        assert is_realistic == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
AI Trading Bot - Unit Tests for Risk Management
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.risk import RiskLimits, DrawdownController


class TestRiskLimits:
    """Tests for RiskLimits class."""
    
    def test_initialization(self):
        """Test default initialization."""
        limits = RiskLimits()
        
        assert limits.max_risk_per_trade == 0.02
        assert limits.max_daily_loss == 0.03
        assert limits.max_drawdown == 0.15
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = {
            'max_risk_per_trade': 0.01,
            'max_daily_loss': 0.02
        }
        limits = RiskLimits(config)
        
        assert limits.max_risk_per_trade == 0.01
        assert limits.max_daily_loss == 0.02
        
    def test_trade_allowed_normal(self):
        """Test normal trade allowance."""
        limits = RiskLimits()
        limits.update_balance(10000)
        
        result = limits.check_trade_allowed(
            account_balance=10000,
            trade_risk=100,  # 1% risk
            position_value=1000  # 10% position
        )
        
        assert result['allowed'] == True
        
    def test_trade_blocked_high_risk(self):
        """Test trade blocked due to high risk."""
        limits = RiskLimits()
        limits.update_balance(10000)
        
        result = limits.check_trade_allowed(
            account_balance=10000,
            trade_risk=500,  # 5% risk (exceeds 2% limit)
            position_value=1000
        )
        
        assert result['allowed'] == False
        assert 'trade_risk' in str(result['reasons'])
        
    def test_trade_blocked_large_position(self):
        """Test trade blocked due to large position."""
        limits = RiskLimits()
        limits.update_balance(10000)
        
        result = limits.check_trade_allowed(
            account_balance=10000,
            trade_risk=100,
            position_value=3000  # 30% position (exceeds 20% limit)
        )
        
        assert result['allowed'] == False
        assert 'position_size' in str(result['reasons'])
        
    def test_trade_blocked_low_balance(self):
        """Test trade blocked due to low balance."""
        limits = RiskLimits({'min_balance': 100})
        
        result = limits.check_trade_allowed(
            account_balance=50,  # Below minimum
            trade_risk=1,
            position_value=10
        )
        
        assert result['allowed'] == False
        
    def test_consecutive_loss_cooldown(self):
        """Test cooldown after consecutive losses."""
        limits = RiskLimits({'max_consecutive_losses': 3})
        limits.update_balance(10000)
        
        # Record losses
        for _ in range(3):
            limits.record_trade_result(pnl=-100, is_win=False)
            
        # Check if in cooldown
        status = limits.get_status()
        assert status['in_cooldown'] == True
        
    def test_daily_limit(self):
        """Test daily loss limit."""
        limits = RiskLimits({'max_daily_loss': 0.03})
        limits.update_balance(10000)
        
        # Record losses exceeding daily limit
        limits.record_trade_result(pnl=-350, is_win=False)  # 3.5% loss
        
        result = limits.check_trade_allowed(
            account_balance=10000,
            trade_risk=100,
            position_value=1000
        )
        
        assert result['allowed'] == False


class TestDrawdownController:
    """Tests for DrawdownController class."""
    
    def test_initialization(self):
        """Test initialization."""
        controller = DrawdownController()
        
        assert controller.warning_threshold == 0.05
        assert controller.critical_threshold == 0.10
        assert controller.max_threshold == 0.15
        
    def test_equity_tracking(self):
        """Test equity curve tracking."""
        controller = DrawdownController()
        
        controller.update(10000)
        controller.update(10100)
        controller.update(10050)
        
        assert controller._peak_equity == 10100
        
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        controller = DrawdownController()
        
        controller.update(10000)
        controller.update(10000)
        controller.update(9500)  # 5% drawdown
        
        status = controller.update(9500)
        
        assert abs(status['current_drawdown'] - 0.05) < 0.001
        
    def test_warning_level(self):
        """Test warning level trigger."""
        controller = DrawdownController({'warning_threshold': 0.05})
        
        controller.update(10000)
        status = controller.update(9400)  # 6% drawdown
        
        assert status['level'] == 'warning'
        assert status['position_multiplier'] < 1.0
        
    def test_critical_level(self):
        """Test critical level trigger."""
        controller = DrawdownController({'critical_threshold': 0.10})
        
        controller.update(10000)
        status = controller.update(8900)  # 11% drawdown
        
        assert status['level'] == 'critical'
        
    def test_emergency_stop(self):
        """Test emergency stop trigger."""
        controller = DrawdownController({'max_threshold': 0.15})
        
        controller.update(10000)
        status = controller.update(8400)  # 16% drawdown
        
        assert status['level'] == 'emergency'
        assert status['emergency_stop'] == True
        assert status['position_multiplier'] == 0
        
    def test_position_multiplier_normal(self):
        """Test position multiplier in normal conditions."""
        controller = DrawdownController()
        
        controller.update(10000)
        controller.update(9800)  # 2% drawdown (normal)
        
        multiplier = controller.get_position_multiplier()
        assert multiplier == 1.0
        
    def test_recovery_tracking(self):
        """Test recovery from drawdown."""
        controller = DrawdownController()
        
        controller.update(10000)
        controller.update(9000)  # Drawdown
        controller.update(10000)  # Recovery
        controller.update(10100)  # New high
        
        assert controller._peak_equity == 10100
        assert controller._current_drawdown == 0
        
    def test_equity_curve(self):
        """Test equity curve retrieval."""
        controller = DrawdownController()
        
        controller.update(10000)
        controller.update(10100)
        controller.update(10050)
        
        curve = controller.get_equity_curve()
        
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 3
        assert 'equity' in curve.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

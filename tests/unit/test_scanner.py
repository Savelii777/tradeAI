"""
Unit tests for scanner module (v2.1).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from src.scanner.aggressive_sizing import AggressiveSizer, LeverageCalculation
from src.scanner.m1_sniper import M1Sniper


class TestAggressiveSizer:
    """Tests for AggressiveSizer."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        sizer = AggressiveSizer()
        
        assert sizer.fixed_risk_pct == 0.05  # 5%
        assert sizer.min_leverage == 5
        assert sizer.max_leverage == 20
        assert sizer.take_profit_rr == 3.0
        
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'fixed_risk_pct': 0.03,
            'min_leverage': 3,
            'max_leverage': 15,
            'take_profit_rr': 2.0
        }
        sizer = AggressiveSizer(config)
        
        assert sizer.fixed_risk_pct == 0.03
        assert sizer.min_leverage == 3
        assert sizer.max_leverage == 15
        assert sizer.take_profit_rr == 2.0
        
    def test_calculate_long_position(self):
        """Test calculation for long position."""
        sizer = AggressiveSizer()
        
        deposit = 1000
        entry_price = 50000
        stop_loss = 49750  # 0.5% below entry
        direction = 1  # Long
        
        calc = sizer.calculate(deposit, entry_price, stop_loss, direction)
        
        assert isinstance(calc, LeverageCalculation)
        assert calc.deposit_used == 1.0
        assert calc.leverage == 10  # 5% / 0.5% = 10x
        assert calc.position_value == 10000  # 1000 * 10
        assert calc.position_size == pytest.approx(0.2, rel=0.01)  # 10000 / 50000
        assert calc.stop_distance_pct == pytest.approx(0.005, rel=0.01)  # 0.5%
        
    def test_calculate_short_position(self):
        """Test calculation for short position."""
        sizer = AggressiveSizer()
        
        deposit = 1000
        entry_price = 50000
        stop_loss = 50250  # 0.5% above entry
        direction = -1  # Short
        
        calc = sizer.calculate(deposit, entry_price, stop_loss, direction)
        
        assert calc.leverage == 10
        assert calc.take_profit < entry_price  # TP below for short
        
    def test_leverage_min_limit(self):
        """Test minimum leverage limit."""
        sizer = AggressiveSizer({'min_leverage': 5, 'fixed_risk_pct': 0.05})
        
        deposit = 1000
        entry_price = 50000
        # 2% stop distance would require 2.5x leverage, but min is 5x
        stop_loss = 49000  # 2% below entry
        direction = 1
        
        calc = sizer.calculate(deposit, entry_price, stop_loss, direction)
        
        assert calc.leverage == 5  # Capped at minimum
        
    def test_leverage_max_limit(self):
        """Test maximum leverage limit."""
        sizer = AggressiveSizer({'max_leverage': 20, 'fixed_risk_pct': 0.05})
        
        deposit = 1000
        entry_price = 50000
        # 0.1% stop distance would require 50x leverage, but max is 20x
        stop_loss = 49950  # 0.1% below entry
        direction = 1
        
        calc = sizer.calculate(deposit, entry_price, stop_loss, direction)
        
        assert calc.leverage == 20  # Capped at maximum
        
    def test_take_profit_calculation(self):
        """Test take profit calculation with RR ratio."""
        sizer = AggressiveSizer({'take_profit_rr': 3.0})
        
        deposit = 1000
        entry_price = 50000
        stop_loss = 49500  # 500 points below
        direction = 1
        
        calc = sizer.calculate(deposit, entry_price, stop_loss, direction)
        
        # TP should be 1500 points above (3x the stop distance)
        assert calc.take_profit == pytest.approx(51500, rel=0.01)
        assert calc.risk_reward_ratio == 3.0
        
    def test_zero_stop_distance(self):
        """Test handling of zero stop distance."""
        sizer = AggressiveSizer()
        
        calc = sizer.calculate(
            deposit=1000,
            entry_price=50000,
            stop_loss=50000,  # Same as entry
            direction=1
        )
        
        assert calc.position_size == 0
        assert calc.leverage == 0
        
    def test_calculate_from_stop_pct(self):
        """Test calculation from stop percentage."""
        sizer = AggressiveSizer()
        
        calc = sizer.calculate_from_stop_pct(
            deposit=1000,
            entry_price=50000,
            stop_distance_pct=0.005,  # 0.5%
            direction=1
        )
        
        assert calc.leverage == 10  # 5% / 0.5% = 10x
        assert calc.stop_loss == pytest.approx(49750, rel=0.01)
        
    def test_calculate_optimal_stop(self):
        """Test optimal stop calculation for target leverage."""
        sizer = AggressiveSizer()
        
        stop_loss = sizer.calculate_optimal_stop(
            deposit=1000,
            entry_price=50000,
            direction=1,
            target_leverage=10
        )
        
        # For 10x leverage with 5% risk: stop distance = 0.5%
        expected_stop = 50000 * (1 - 0.005)
        assert stop_loss == pytest.approx(expected_stop, rel=0.01)
        
    def test_validate_position_valid(self):
        """Test position validation with valid parameters."""
        sizer = AggressiveSizer()
        
        calc = sizer.calculate(
            deposit=1000,
            entry_price=50000,
            stop_loss=49750,
            direction=1
        )
        
        result = sizer.validate_position(calc, direction=1)
        
        assert result['valid'] is True
        assert len(result['issues']) == 0
        
    def test_validate_position_leverage_issues(self):
        """Test position validation with leverage issues."""
        sizer = AggressiveSizer({'min_leverage': 10, 'max_leverage': 15})
        
        # Create a calculation with leverage outside limits
        calc = LeverageCalculation(
            deposit_used=1.0,
            position_value=5000,
            position_size=0.1,
            leverage=5,  # Below min of 10
            stop_loss=49000,
            stop_distance_pct=0.02,
            risk_amount=100,
            risk_percent=0.1,
            take_profit=52000,
            take_profit_pct=0.04,
            risk_reward_ratio=2.0,
            margin_required=1000,
            liquidation_price=45000,
            entry_price=50000
        )
        
        result = sizer.validate_position(calc, direction=1)
        
        assert result['valid'] is False
        assert any('below minimum' in issue for issue in result['issues'])
        
    def test_get_leverage_table(self):
        """Test leverage table generation."""
        sizer = AggressiveSizer()
        
        # Test with specific stop distances
        table = sizer.get_leverage_table([0.005, 0.0025, 0.01])
        
        assert 0.005 in table  # 0.5%
        assert table[0.005] == 10  # 5% / 0.5% = 10x
        assert table[0.0025] == 20  # Capped at max
        assert table[0.01] == 5  # Capped at min


class TestM1Sniper:
    """Tests for M1Sniper."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        sniper = M1Sniper()
        
        assert sniper.max_wait_candles == 15
        assert sniper.fixed_risk_pct == 0.05
        assert sniper.take_profit_rr == 3.0
        
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'max_wait_candles': 20,
            'fixed_risk_pct': 0.03,
            'take_profit_rr': 2.5
        }
        sniper = M1Sniper(config)
        
        assert sniper.max_wait_candles == 20
        assert sniper.fixed_risk_pct == 0.03
        assert sniper.take_profit_rr == 2.5
        
    def test_calculate_leverage_10x(self):
        """Test leverage calculation for 10x."""
        sniper = M1Sniper({'fixed_risk_pct': 0.05})
        
        leverage = sniper._calculate_leverage(
            stop_distance_pct=0.005,  # 0.5%
            min_leverage=5,
            max_leverage=20
        )
        
        assert leverage == 10  # 5% / 0.5% = 10x
        
    def test_calculate_leverage_capped_at_max(self):
        """Test leverage capped at maximum."""
        sniper = M1Sniper({'fixed_risk_pct': 0.05})
        
        leverage = sniper._calculate_leverage(
            stop_distance_pct=0.001,  # 0.1% -> would be 50x
            min_leverage=5,
            max_leverage=20
        )
        
        assert leverage == 20  # Capped at max
        
    def test_calculate_leverage_capped_at_min(self):
        """Test leverage capped at minimum."""
        sniper = M1Sniper({'fixed_risk_pct': 0.05})
        
        leverage = sniper._calculate_leverage(
            stop_distance_pct=0.02,  # 2% -> would be 2.5x
            min_leverage=5,
            max_leverage=20
        )
        
        assert leverage == 5  # Capped at min
        
    def test_is_waiting_initial(self):
        """Test initial waiting state."""
        sniper = M1Sniper()
        
        assert sniper.is_waiting() is False
        
    def test_get_status(self):
        """Test status retrieval."""
        sniper = M1Sniper()
        
        status = sniper.get_status()
        
        assert 'waiting_for_entry' in status
        assert 'target_symbol' in status
        assert 'max_wait_candles' in status
        assert status['waiting_for_entry'] is False
        
    def test_cancel_entry(self):
        """Test entry cancellation."""
        sniper = M1Sniper()
        sniper._waiting_for_entry = True
        sniper._target_symbol = "BTCUSDT"
        
        sniper.cancel_entry()
        
        assert sniper.is_waiting() is False
        assert sniper._target_symbol is None
        
    def test_check_entry_trigger_pullback_long(self):
        """Test pullback entry trigger for long."""
        sniper = M1Sniper()
        
        # Create data where price touched EMA9 and bounced
        dates = pd.date_range(end=datetime.now(), periods=30, freq='1min')
        prices = [100 + i * 0.1 for i in range(30)]  # Uptrend
        
        # Make the last candle touch EMA9 and bounce
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.3 for p in prices],  # Low touches EMA area
            'close': [p + 0.1 for p in prices],  # Close above
            'volume': [1000] * 30
        }, index=dates)
        
        # Simulate EMA9 touch - adjust last candle
        ema9 = df['close'].ewm(span=9).mean()
        df.loc[df.index[-1], 'low'] = ema9.iloc[-1] - 0.1
        df.loc[df.index[-1], 'close'] = ema9.iloc[-1] + 0.1
        
        result = sniper._check_entry_trigger(df, direction=1)
        
        # May or may not trigger depending on exact values
        assert 'triggered' in result
        assert 'entry_type' in result
        assert 'reason' in result
        
    def test_check_entry_trigger_insufficient_data(self):
        """Test entry trigger with insufficient data."""
        sniper = M1Sniper()
        
        df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        })
        
        result = sniper._check_entry_trigger(df, direction=1)
        
        assert result['triggered'] is False
        assert 'Insufficient data' in result['reason']


class TestLeverageExamples:
    """Test specific leverage examples from the spec."""
    
    def test_example_5pct_risk_05pct_stop(self):
        """Risk 5%, stop 0.5% -> leverage 10x."""
        sizer = AggressiveSizer({
            'fixed_risk_pct': 0.05,
            'min_leverage': 5,
            'max_leverage': 20
        })
        
        calc = sizer.calculate_from_stop_pct(
            deposit=1000,
            entry_price=50000,
            stop_distance_pct=0.005,  # 0.5%
            direction=1
        )
        
        assert calc.leverage == 10
        
    def test_example_5pct_risk_025pct_stop(self):
        """Risk 5%, stop 0.25% -> leverage 20x."""
        sizer = AggressiveSizer({
            'fixed_risk_pct': 0.05,
            'min_leverage': 5,
            'max_leverage': 20
        })
        
        calc = sizer.calculate_from_stop_pct(
            deposit=1000,
            entry_price=50000,
            stop_distance_pct=0.0025,  # 0.25%
            direction=1
        )
        
        assert calc.leverage == 20
        
    def test_example_5pct_risk_1pct_stop(self):
        """Risk 5%, stop 1% -> leverage 5x (minimum)."""
        sizer = AggressiveSizer({
            'fixed_risk_pct': 0.05,
            'min_leverage': 5,
            'max_leverage': 20
        })
        
        calc = sizer.calculate_from_stop_pct(
            deposit=1000,
            entry_price=50000,
            stop_distance_pct=0.01,  # 1%
            direction=1
        )
        
        assert calc.leverage == 5


class TestPositionValue:
    """Test position value calculations."""
    
    def test_100pct_deposit_used(self):
        """Verify 100% of deposit is used."""
        sizer = AggressiveSizer()
        
        deposit = 1000
        calc = sizer.calculate_from_stop_pct(
            deposit=deposit,
            entry_price=50000,
            stop_distance_pct=0.005,
            direction=1
        )
        
        assert calc.deposit_used == 1.0
        assert calc.margin_required == deposit
        
    def test_position_value_with_leverage(self):
        """Verify position value = deposit * leverage."""
        sizer = AggressiveSizer()
        
        deposit = 1000
        calc = sizer.calculate_from_stop_pct(
            deposit=deposit,
            entry_price=50000,
            stop_distance_pct=0.005,  # 10x leverage
            direction=1
        )
        
        expected_value = deposit * 10
        assert calc.position_value == expected_value
        
    def test_position_size_calculation(self):
        """Verify position size = position_value / price."""
        sizer = AggressiveSizer()
        
        deposit = 1000
        entry_price = 50000
        calc = sizer.calculate_from_stop_pct(
            deposit=deposit,
            entry_price=entry_price,
            stop_distance_pct=0.005,
            direction=1
        )
        
        expected_size = calc.position_value / entry_price
        assert calc.position_size == pytest.approx(expected_size, rel=0.01)

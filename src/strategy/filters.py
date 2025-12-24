"""
AI Trading Bot - Signal Filters
Filters trading signals based on various conditions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.constants import MarketRegime, TradingSession


class SignalFilters:
    """
    Applies various filters to trading signals.
    
    Filter types:
    - Volatility filters
    - Liquidity filters
    - Timeframe consistency filters
    - Time-based filters
    - Event filters
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize signal filters.
        
        Args:
            config: Filter configuration.
        """
        self.config = config or {}
        
        # Volatility thresholds
        self.max_atr_multiplier = self.config.get('max_atr_multiplier', 3.0)
        self.min_atr_multiplier = self.config.get('min_atr_multiplier', 0.3)
        
        # Liquidity thresholds
        self.min_volume_percentile = self.config.get('min_volume_percentile', 20)
        
        # Time settings
        self.avoid_first_minutes = self.config.get('avoid_first_minutes', 5)
        self.avoid_last_minutes = self.config.get('avoid_last_minutes', 5)
        
        # State
        self._filter_stats: Dict[str, int] = {}
        
    def apply_all_filters(
        self,
        current_data: Dict[str, Any],
        market_state: Dict[str, Any],
        additional_conditions: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Apply all filters and return result.
        
        Args:
            current_data: Current market data.
            market_state: Current market state indicators.
            additional_conditions: Additional filter conditions.
            
        Returns:
            Dictionary with filter results.
        """
        results = {
            'passed': True,
            'filters_applied': [],
            'filters_failed': [],
            'reasons': []
        }
        
        # Volatility filter
        vol_result = self.check_volatility(
            current_data.get('atr', 0),
            current_data.get('atr_avg', 0)
        )
        results['filters_applied'].append('volatility')
        if not vol_result['passed']:
            results['passed'] = False
            results['filters_failed'].append('volatility')
            results['reasons'].append(vol_result['reason'])
            
        # Liquidity filter
        liq_result = self.check_liquidity(
            current_data.get('volume', 0),
            current_data.get('volume_avg', 0)
        )
        results['filters_applied'].append('liquidity')
        if not liq_result['passed']:
            results['passed'] = False
            results['filters_failed'].append('liquidity')
            results['reasons'].append(liq_result['reason'])
            
        # Time filter
        time_result = self.check_time(current_data.get('timestamp'))
        results['filters_applied'].append('time')
        if not time_result['passed']:
            results['passed'] = False
            results['filters_failed'].append('time')
            results['reasons'].append(time_result['reason'])
            
        # Regime filter
        if 'regime' in market_state:
            regime_result = self.check_regime(market_state['regime'])
            results['filters_applied'].append('regime')
            if not regime_result['passed']:
                results['passed'] = False
                results['filters_failed'].append('regime')
                results['reasons'].append(regime_result['reason'])
                
        # Timeframe consistency filter
        if 'timeframe_trends' in market_state:
            tf_result = self.check_timeframe_consistency(
                market_state['timeframe_trends'],
                current_data.get('signal_direction', 0)
            )
            results['filters_applied'].append('timeframe_consistency')
            if not tf_result['passed']:
                results['passed'] = False
                results['filters_failed'].append('timeframe_consistency')
                results['reasons'].append(tf_result['reason'])
                
        # Update statistics
        for filter_name in results['filters_applied']:
            if filter_name not in self._filter_stats:
                self._filter_stats[filter_name] = {'applied': 0, 'blocked': 0}
            self._filter_stats[filter_name]['applied'] += 1
            
        for filter_name in results['filters_failed']:
            self._filter_stats[filter_name]['blocked'] += 1
            
        return results
        
    def check_volatility(
        self,
        current_atr: float,
        average_atr: float
    ) -> Dict[str, Any]:
        """
        Check if volatility is within acceptable range.
        
        Args:
            current_atr: Current ATR value.
            average_atr: Historical average ATR.
            
        Returns:
            Filter result dictionary.
        """
        if average_atr == 0:
            return {'passed': True, 'reason': None}
            
        atr_ratio = current_atr / average_atr
        
        if atr_ratio > self.max_atr_multiplier:
            return {
                'passed': False,
                'reason': f'ATR ratio {atr_ratio:.2f} > max {self.max_atr_multiplier}'
            }
            
        if atr_ratio < self.min_atr_multiplier:
            return {
                'passed': False,
                'reason': f'ATR ratio {atr_ratio:.2f} < min {self.min_atr_multiplier}'
            }
            
        return {'passed': True, 'reason': None, 'atr_ratio': atr_ratio}
        
    def check_liquidity(
        self,
        current_volume: float,
        average_volume: float
    ) -> Dict[str, Any]:
        """
        Check if liquidity is sufficient.
        
        Args:
            current_volume: Current trading volume.
            average_volume: Historical average volume.
            
        Returns:
            Filter result dictionary.
        """
        if average_volume == 0:
            return {'passed': True, 'reason': None}
            
        volume_ratio = current_volume / average_volume * 100
        
        if volume_ratio < self.min_volume_percentile:
            return {
                'passed': False,
                'reason': f'Volume percentile {volume_ratio:.1f}% < min {self.min_volume_percentile}%'
            }
            
        return {'passed': True, 'reason': None, 'volume_ratio': volume_ratio}
        
    def check_time(
        self,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check if current time is suitable for trading.
        
        Args:
            timestamp: Current timestamp.
            
        Returns:
            Filter result dictionary.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        minute = timestamp.minute
        
        # Avoid first/last minutes of hour (for hourly candle strategies)
        if minute < self.avoid_first_minutes:
            return {
                'passed': False,
                'reason': f'Avoiding first {self.avoid_first_minutes} minutes of hour'
            }
            
        if minute >= 60 - self.avoid_last_minutes:
            return {
                'passed': False,
                'reason': f'Avoiding last {self.avoid_last_minutes} minutes of hour'
            }
            
        return {'passed': True, 'reason': None}
        
    def check_regime(
        self,
        regime: str
    ) -> Dict[str, Any]:
        """
        Check if market regime is suitable for trading.
        
        Args:
            regime: Current market regime.
            
        Returns:
            Filter result dictionary.
        """
        # Block trading in high volatility regime
        if regime == MarketRegime.HIGH_VOLATILITY.value:
            return {
                'passed': False,
                'reason': 'High volatility regime - avoiding trades'
            }
            
        return {'passed': True, 'reason': None, 'regime': regime}
        
    def check_timeframe_consistency(
        self,
        timeframe_trends: Dict[str, int],
        signal_direction: int
    ) -> Dict[str, Any]:
        """
        Check if signal is consistent across timeframes.
        
        Args:
            timeframe_trends: Dictionary of {timeframe: trend_direction}.
            signal_direction: Direction of the trading signal.
            
        Returns:
            Filter result dictionary.
        """
        if not timeframe_trends:
            return {'passed': True, 'reason': None}
            
        # Check for contradicting trends
        conflicting = []
        for tf, trend in timeframe_trends.items():
            if trend != 0 and trend != signal_direction:
                conflicting.append(tf)
                
        if len(conflicting) >= len(timeframe_trends) / 2:
            return {
                'passed': False,
                'reason': f'Conflicting trends in timeframes: {conflicting}'
            }
            
        # Check alignment score
        aligned = sum(1 for t in timeframe_trends.values() if t == signal_direction)
        alignment_ratio = aligned / len(timeframe_trends)
        
        if alignment_ratio < 0.5:
            return {
                'passed': False,
                'reason': f'Low timeframe alignment: {alignment_ratio:.1%}'
            }
            
        return {
            'passed': True,
            'reason': None,
            'alignment_ratio': alignment_ratio
        }
        
    def check_spread(
        self,
        spread: float,
        atr: float,
        max_spread_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check if spread is acceptable.
        
        Args:
            spread: Current bid-ask spread.
            atr: Current ATR value.
            max_spread_ratio: Maximum spread as ratio of ATR.
            
        Returns:
            Filter result dictionary.
        """
        if atr == 0:
            return {'passed': True, 'reason': None}
            
        spread_ratio = spread / atr
        
        if spread_ratio > max_spread_ratio:
            return {
                'passed': False,
                'reason': f'Spread ratio {spread_ratio:.3f} > max {max_spread_ratio}'
            }
            
        return {'passed': True, 'reason': None, 'spread_ratio': spread_ratio}
        
    def get_filter_stats(self) -> Dict[str, Dict]:
        """Get filter application statistics."""
        return self._filter_stats
        
    def reset_stats(self) -> None:
        """Reset filter statistics."""
        self._filter_stats = {}

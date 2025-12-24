"""
AI Trading Bot - Decision Engine
Central decision-making system for trading.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models import EnsembleModel
from src.features import FeatureEngine
from src.utils.constants import SignalType, MarketRegime

from .signals import SignalGenerator, TradingSignal
from .filters import SignalFilters
from .position_sizing import PositionSizer


@dataclass
class TradingDecision:
    """Complete trading decision with all context."""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold', 'close'
    symbol: str
    signal: Optional[TradingSignal]
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    filters_passed: bool
    filter_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    metadata: Dict[str, Any]


class DecisionEngine:
    """
    Central decision-making engine.
    
    Orchestrates:
    - Feature generation
    - Model predictions
    - Signal generation
    - Filter application
    - Position sizing
    - Final decision making
    """
    
    def __init__(
        self,
        model: EnsembleModel,
        feature_engine: FeatureEngine,
        config: Optional[Dict] = None
    ):
        """
        Initialize decision engine.
        
        Args:
            model: Trained ensemble model.
            feature_engine: Feature engineering engine.
            config: Configuration dictionary.
        """
        self.model = model
        self.feature_engine = feature_engine
        self.config = config or {}
        
        # Initialize components
        self.signal_generator = SignalGenerator(self.config.get('signals', {}))
        self.filters = SignalFilters(self.config.get('filters', {}))
        self.position_sizer = PositionSizer(self.config.get('position_sizing', {}))
        
        # Decision history
        self._decision_history: List[TradingDecision] = []
        
        # Thresholds
        self.min_score_to_trade = self.config.get('min_score_to_trade', 0.3)
        
    def make_decision(
        self,
        market_data: pd.DataFrame,
        account_state: Dict[str, Any],
        current_position: Optional[Dict] = None,
        additional_timeframes: Optional[Dict[str, pd.DataFrame]] = None
    ) -> TradingDecision:
        """
        Make a trading decision based on current market state.
        
        Args:
            market_data: Recent OHLCV data.
            account_state: Current account information.
            current_position: Current open position (if any).
            additional_timeframes: Data from other timeframes.
            
        Returns:
            TradingDecision object.
        """
        timestamp = datetime.utcnow()
        
        # Generate features
        features = self.feature_engine.generate_all_features(
            market_data,
            normalize=True,
            additional_timeframes=additional_timeframes
        )
        
        if features.empty:
            return self._create_hold_decision(timestamp, "Feature generation failed")
            
        # Get latest features (single row)
        latest_features = features.iloc[[-1]]
        
        # Handle any remaining NaN values
        latest_features = latest_features.fillna(0)
        
        # Get model predictions
        try:
            predictions = self.model.predict(latest_features)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return self._create_hold_decision(timestamp, f"Prediction failed: {e}")
            
        # Get current market state
        current_price = market_data['close'].iloc[-1]
        atr = self._calculate_atr(market_data)
        volume_avg = market_data['volume'].rolling(20).mean().iloc[-1]
        
        # Generate signal
        signal = self.signal_generator.generate_signal(
            predictions=predictions,
            current_price=current_price,
            atr=atr,
            symbol=self.config.get('symbol', 'BTCUSDT'),
            timestamp=timestamp
        )
        
        if signal is None:
            return self._create_hold_decision(timestamp, "No signal generated")
            
        # Apply filters
        market_state = self._get_market_state(features, predictions)
        
        filter_results = self.filters.apply_all_filters(
            current_data={
                'atr': atr,
                'atr_avg': market_data['close'].rolling(100).apply(
                    lambda x: self._calculate_atr_from_series(x)
                ).iloc[-1] if len(market_data) >= 100 else atr,
                'volume': market_data['volume'].iloc[-1],
                'volume_avg': volume_avg,
                'timestamp': timestamp,
                'signal_direction': signal.direction
            },
            market_state=market_state
        )
        
        if not filter_results['passed']:
            return self._create_hold_decision(
                timestamp,
                f"Filters failed: {filter_results['reasons']}",
                signal=signal,
                filter_results=filter_results
            )
            
        # Calculate position size
        position_result = self.position_sizer.calculate_position_size(
            account_balance=account_state.get('balance', 10000),
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            signal_confidence=signal.confidence,
            volatility=atr / current_price,
            win_rate=account_state.get('win_rate'),
            avg_win_loss_ratio=account_state.get('profit_factor')
        )
        
        # Check if already in position
        if current_position:
            return self._handle_existing_position(
                timestamp, signal, current_position, current_price, atr
            )
            
        # Create trading decision
        action = 'buy' if signal.signal_type == SignalType.BUY else 'sell'
        
        decision = TradingDecision(
            timestamp=timestamp,
            action=action,
            symbol=signal.symbol,
            signal=signal,
            position_size=position_result['position_size'],
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            filters_passed=True,
            filter_results=filter_results,
            risk_assessment=position_result,
            metadata={
                'predictions': {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in predictions.items()},
                'market_state': market_state
            }
        )
        
        self._decision_history.append(decision)
        logger.info(f"Decision: {action} {decision.symbol}, "
                   f"size={position_result['position_size']:.4f}, "
                   f"confidence={signal.confidence:.3f}")
        
        return decision
        
    def _create_hold_decision(
        self,
        timestamp: datetime,
        reason: str,
        signal: Optional[TradingSignal] = None,
        filter_results: Optional[Dict] = None
    ) -> TradingDecision:
        """Create a hold decision."""
        return TradingDecision(
            timestamp=timestamp,
            action='hold',
            symbol=self.config.get('symbol', 'BTCUSDT'),
            signal=signal,
            position_size=0,
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            confidence=0,
            filters_passed=filter_results['passed'] if filter_results else True,
            filter_results=filter_results or {},
            risk_assessment={},
            metadata={'reason': reason}
        )
        
    def _handle_existing_position(
        self,
        timestamp: datetime,
        signal: TradingSignal,
        current_position: Dict,
        current_price: float,
        atr: float
    ) -> TradingDecision:
        """Handle decision when already in a position."""
        position_side = current_position.get('side', 'long')
        signal_direction = signal.direction
        
        # Check for exit signal (opposite direction with high confidence)
        should_close = False
        reason = ""
        
        if position_side == 'long' and signal_direction == -1:
            should_close = True
            reason = "Exit signal: bearish signal while in long position"
        elif position_side == 'short' and signal_direction == 1:
            should_close = True
            reason = "Exit signal: bullish signal while in short position"
            
        if should_close:
            return TradingDecision(
                timestamp=timestamp,
                action='close',
                symbol=signal.symbol,
                signal=signal,
                position_size=current_position.get('size', 0),
                entry_price=current_position.get('entry_price', 0),
                stop_loss=0,
                take_profit=0,
                confidence=signal.confidence,
                filters_passed=True,
                filter_results={},
                risk_assessment={},
                metadata={'reason': reason}
            )
            
        # Otherwise hold
        return self._create_hold_decision(
            timestamp,
            "Already in position - holding"
        )
        
    def _calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> float:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
        
    def _calculate_atr_from_series(self, prices: pd.Series) -> float:
        """Calculate simplified ATR from price series."""
        returns = prices.pct_change().dropna()
        return returns.std() * prices.iloc[-1] if len(returns) > 0 else 0
        
    def _get_market_state(
        self,
        features: pd.DataFrame,
        predictions: Dict
    ) -> Dict[str, Any]:
        """Extract market state from features and predictions."""
        state = {}
        
        # Get regime if available
        regime_cols = [col for col in features.columns if col.startswith('regime_')]
        if regime_cols:
            for col in regime_cols:
                if features[col].iloc[-1] == 1:
                    state['regime'] = col.replace('regime_', '')
                    break
                    
        # Get timeframe trends
        trend_cols = [col for col in features.columns if col.startswith('trend_')]
        if trend_cols:
            state['timeframe_trends'] = {
                col.replace('trend_', ''): int(features[col].iloc[-1])
                for col in trend_cols
            }
            
        return state
        
    def update_after_trade(
        self,
        is_win: bool,
        pnl: float
    ) -> None:
        """Update state after trade completion."""
        self.position_sizer.update_after_trade(is_win)
        
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics."""
        if not self._decision_history:
            return {'total_decisions': 0}
            
        buys = sum(1 for d in self._decision_history if d.action == 'buy')
        sells = sum(1 for d in self._decision_history if d.action == 'sell')
        holds = sum(1 for d in self._decision_history if d.action == 'hold')
        
        return {
            'total_decisions': len(self._decision_history),
            'buys': buys,
            'sells': sells,
            'holds': holds,
            'action_rate': (buys + sells) / len(self._decision_history)
        }

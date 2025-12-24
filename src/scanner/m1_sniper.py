"""
AI Trading Bot - M1 Sniper
Precise entry execution on M1 (1-minute) timeframe.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SniperEntry:
    """Result of sniper entry attempt."""
    success: bool
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    stop_distance_pct: float  # Stop distance as percentage
    leverage: int
    position_size: float  # In base currency
    position_value: float  # In quote currency (USDT)
    risk_amount: float  # Amount at risk
    risk_percent: float  # Risk as percentage of deposit
    entry_type: str  # 'market', 'limit', 'pullback'
    trigger_reason: str
    metadata: Dict[str, Any]


class M1Sniper:
    """
    Precision entry executor on M1 timeframe.
    
    After the scanner identifies a good setup, the sniper
    waits for the optimal micro-entry on the 1-minute chart.
    
    Entry strategies:
    - Pullback to EMA/support
    - Breakout confirmation
    - Momentum burst
    - Engulfing candle
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize M1 Sniper.
        
        Args:
            config: Sniper configuration.
        """
        self.config = config or {}
        
        # Entry parameters
        self.max_wait_candles = self.config.get('max_wait_candles', 15)  # Max candles to wait
        self.pullback_threshold = self.config.get('pullback_threshold', 0.3)  # % pullback
        self.momentum_threshold = self.config.get('momentum_threshold', 0.15)  # % move
        
        # Stop loss parameters
        self.default_stop_pct = self.config.get('default_stop_pct', 0.003)  # 0.3%
        self.min_stop_pct = self.config.get('min_stop_pct', 0.002)  # 0.2%
        self.max_stop_pct = self.config.get('max_stop_pct', 0.005)  # 0.5%
        
        # Risk parameters
        self.fixed_risk_pct = self.config.get('fixed_risk_pct', 0.05)  # 5% of deposit
        
        # Take profit parameters
        self.take_profit_rr = self.config.get('take_profit_rr', 3.0)  # Risk:Reward ratio
        
        # State
        self._waiting_for_entry = False
        self._target_symbol: Optional[str] = None
        self._target_direction: Optional[int] = None
        self._wait_start_time: Optional[datetime] = None
        self._candles_waited = 0
        
    async def snipe_entry(
        self,
        symbol: str,
        direction: int,
        data_collector,
        scan_result,  # ScanResult from scanner
        account_balance: float,
        min_leverage: int = 5,
        max_leverage: int = 20
    ) -> Optional[SniperEntry]:
        """
        Attempt to get optimal entry on M1 timeframe.
        
        Args:
            symbol: Trading pair symbol.
            direction: 1 for long, -1 for short.
            data_collector: Data collector for fetching M1 data.
            scan_result: Original scan result.
            account_balance: Current account balance.
            min_leverage: Minimum leverage to use.
            max_leverage: Maximum leverage to use.
            
        Returns:
            SniperEntry if entry found, None if timeout/failed.
        """
        self._waiting_for_entry = True
        self._target_symbol = symbol
        self._target_direction = direction
        self._wait_start_time = datetime.utcnow()
        self._candles_waited = 0
        
        symbol_ccxt = symbol.replace('USDT', '/USDT')
        
        logger.info(f"M1 Sniper active for {symbol}, direction: {'LONG' if direction == 1 else 'SHORT'}")
        
        try:
            while self._candles_waited < self.max_wait_candles:
                # Fetch M1 data
                df = await data_collector.fetch_ohlcv(
                    symbol=symbol_ccxt,
                    timeframe='1m',
                    limit=100
                )
                
                if df.empty:
                    await asyncio.sleep(5)
                    continue
                    
                # Check for entry trigger
                entry_trigger = self._check_entry_trigger(df, direction)
                
                if entry_trigger['triggered']:
                    # Calculate entry parameters
                    current_price = df['close'].iloc[-1]
                    atr = self._calculate_atr_m1(df)
                    
                    # Calculate stop loss
                    stop_info = self._calculate_stop_loss(
                        df, direction, current_price, atr
                    )
                    
                    stop_loss = stop_info['stop_loss']
                    stop_distance_pct = stop_info['stop_distance_pct']
                    
                    # Calculate leverage based on stop distance
                    leverage = self._calculate_leverage(
                        stop_distance_pct,
                        min_leverage,
                        max_leverage
                    )
                    
                    # Calculate position size (100% of balance)
                    position_value = account_balance * leverage
                    position_size = position_value / current_price
                    
                    # Calculate take profit
                    tp_distance = abs(current_price - stop_loss) * self.take_profit_rr
                    if direction == 1:
                        take_profit = current_price + tp_distance
                    else:
                        take_profit = current_price - tp_distance
                        
                    # Risk amount
                    risk_amount = account_balance * self.fixed_risk_pct
                    
                    entry = SniperEntry(
                        success=True,
                        symbol=symbol,
                        direction=direction,
                        entry_price=current_price,
                        entry_time=datetime.utcnow(),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        stop_distance_pct=stop_distance_pct,
                        leverage=leverage,
                        position_size=position_size,
                        position_value=position_value,
                        risk_amount=risk_amount,
                        risk_percent=self.fixed_risk_pct,
                        entry_type=entry_trigger['entry_type'],
                        trigger_reason=entry_trigger['reason'],
                        metadata={
                            'atr': atr,
                            'candles_waited': self._candles_waited,
                            'scan_score': scan_result.score if scan_result else 0,
                            'stop_info': stop_info
                        }
                    )
                    
                    logger.info(f"Sniper entry found: {symbol} {entry.entry_type} @ {current_price:.4f}, "
                               f"leverage={leverage}x, size={position_size:.4f}")
                    
                    return entry
                    
                # Wait for next candle
                await asyncio.sleep(10)  # Check every 10 seconds
                self._candles_waited += 1
                
            # Timeout - no entry found
            logger.warning(f"M1 Sniper timeout for {symbol} after {self.max_wait_candles} candles")
            return None
            
        finally:
            self._waiting_for_entry = False
            self._target_symbol = None
            self._target_direction = None
            
    def _check_entry_trigger(
        self,
        df: pd.DataFrame,
        direction: int
    ) -> Dict[str, Any]:
        """
        Check for entry trigger on M1 data.
        
        Returns dict with 'triggered', 'entry_type', and 'reason'.
        """
        if len(df) < 20:
            return {'triggered': False, 'entry_type': None, 'reason': 'Insufficient data'}
            
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        current_price = close.iloc[-1]
        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        
        # Calculate EMAs
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        
        # 1. Pullback entry - price touched EMA and bouncing
        if direction == 1:  # Long
            # Price touched EMA9 or EMA21 and bouncing up
            if low.iloc[-1] <= ema_9.iloc[-1] and close.iloc[-1] > ema_9.iloc[-1]:
                return {
                    'triggered': True,
                    'entry_type': 'pullback',
                    'reason': 'Pullback to EMA9 with bounce'
                }
            if low.iloc[-1] <= ema_21.iloc[-1] and close.iloc[-1] > ema_21.iloc[-1]:
                return {
                    'triggered': True,
                    'entry_type': 'pullback',
                    'reason': 'Pullback to EMA21 with bounce'
                }
        else:  # Short
            if high.iloc[-1] >= ema_9.iloc[-1] and close.iloc[-1] < ema_9.iloc[-1]:
                return {
                    'triggered': True,
                    'entry_type': 'pullback',
                    'reason': 'Pullback to EMA9 with rejection'
                }
            if high.iloc[-1] >= ema_21.iloc[-1] and close.iloc[-1] < ema_21.iloc[-1]:
                return {
                    'triggered': True,
                    'entry_type': 'pullback',
                    'reason': 'Pullback to EMA21 with rejection'
                }
                
        # 2. Engulfing candle
        body_current = abs(close.iloc[-1] - df['open'].iloc[-1])
        body_prev = abs(prev_close - df['open'].iloc[-2])
        
        if direction == 1:  # Bullish engulfing
            if (close.iloc[-1] > df['open'].iloc[-1] and  # Current is green
                prev_close < df['open'].iloc[-2] and  # Previous was red
                body_current > body_prev * 1.5):  # Current body is larger
                return {
                    'triggered': True,
                    'entry_type': 'engulfing',
                    'reason': 'Bullish engulfing candle'
                }
        else:  # Bearish engulfing
            if (close.iloc[-1] < df['open'].iloc[-1] and  # Current is red
                prev_close > df['open'].iloc[-2] and  # Previous was green
                body_current > body_prev * 1.5):
                return {
                    'triggered': True,
                    'entry_type': 'engulfing',
                    'reason': 'Bearish engulfing candle'
                }
                
        # 3. Momentum burst - strong candle in direction
        move_pct = (current_price - prev_close) / prev_close
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        if direction == 1 and move_pct > self.momentum_threshold:
            if volume.iloc[-1] > avg_volume * 1.5:
                return {
                    'triggered': True,
                    'entry_type': 'momentum',
                    'reason': f'Momentum burst: +{move_pct:.2%} with volume'
                }
        elif direction == -1 and move_pct < -self.momentum_threshold:
            if volume.iloc[-1] > avg_volume * 1.5:
                return {
                    'triggered': True,
                    'entry_type': 'momentum',
                    'reason': f'Momentum burst: {move_pct:.2%} with volume'
                }
                
        # 4. Breakout confirmation - break of recent high/low
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        
        if direction == 1 and current_price > recent_high:
            return {
                'triggered': True,
                'entry_type': 'breakout',
                'reason': f'Breakout above {recent_high:.4f}'
            }
        elif direction == -1 and current_price < recent_low:
            return {
                'triggered': True,
                'entry_type': 'breakout',
                'reason': f'Breakdown below {recent_low:.4f}'
            }
            
        return {'triggered': False, 'entry_type': None, 'reason': 'Waiting for entry signal'}
        
    def _calculate_atr_m1(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR on M1 data."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
        
    def _calculate_stop_loss(
        self,
        df: pd.DataFrame,
        direction: int,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal stop loss.
        
        Uses combination of:
        - Recent swing points
        - ATR-based distance
        - Recent candle structure
        """
        # Find recent swing low/high
        low = df['low']
        high = df['high']
        
        if direction == 1:  # Long - stop below recent low
            recent_low = low.iloc[-5:].min()
            swing_stop = recent_low * 0.999  # Slightly below
            
            # ATR-based stop
            atr_stop = current_price - atr * 1.5
            
            # Use the tighter of the two, but not too tight
            stop_loss = max(swing_stop, atr_stop)
            
        else:  # Short - stop above recent high
            recent_high = high.iloc[-5:].max()
            swing_stop = recent_high * 1.001  # Slightly above
            
            atr_stop = current_price + atr * 1.5
            
            stop_loss = min(swing_stop, atr_stop)
            
        # Calculate stop distance as percentage
        stop_distance_pct = abs(current_price - stop_loss) / current_price
        
        # Apply limits
        if stop_distance_pct < self.min_stop_pct:
            stop_distance_pct = self.min_stop_pct
            if direction == 1:
                stop_loss = current_price * (1 - stop_distance_pct)
            else:
                stop_loss = current_price * (1 + stop_distance_pct)
                
        elif stop_distance_pct > self.max_stop_pct:
            stop_distance_pct = self.max_stop_pct
            if direction == 1:
                stop_loss = current_price * (1 - stop_distance_pct)
            else:
                stop_loss = current_price * (1 + stop_distance_pct)
                
        return {
            'stop_loss': stop_loss,
            'stop_distance_pct': stop_distance_pct,
            'method': 'swing_atr_hybrid'
        }
        
    def _calculate_leverage(
        self,
        stop_distance_pct: float,
        min_leverage: int,
        max_leverage: int
    ) -> int:
        """
        Calculate leverage based on stop distance.
        
        Formula: leverage = risk_pct / stop_distance_pct
        
        Example:
        - Risk 5%, stop at 0.5% -> leverage = 10x
        - Risk 5%, stop at 0.25% -> leverage = 20x
        """
        # Calculate required leverage for fixed risk
        required_leverage = self.fixed_risk_pct / stop_distance_pct
        
        # Apply limits
        leverage = int(min(max(required_leverage, min_leverage), max_leverage))
        
        logger.debug(f"Leverage calculation: risk={self.fixed_risk_pct:.1%}, "
                    f"stop={stop_distance_pct:.3%}, leverage={leverage}x")
        
        return leverage
        
    def cancel_entry(self) -> None:
        """Cancel current entry attempt."""
        self._waiting_for_entry = False
        self._target_symbol = None
        self._target_direction = None
        logger.info("Sniper entry cancelled")
        
    def is_waiting(self) -> bool:
        """Check if sniper is waiting for entry."""
        return self._waiting_for_entry
        
    def get_status(self) -> Dict[str, Any]:
        """Get current sniper status."""
        return {
            'waiting_for_entry': self._waiting_for_entry,
            'target_symbol': self._target_symbol,
            'target_direction': 'LONG' if self._target_direction == 1 else 'SHORT' if self._target_direction == -1 else None,
            'candles_waited': self._candles_waited,
            'max_wait_candles': self.max_wait_candles,
            'wait_start_time': self._wait_start_time.isoformat() if self._wait_start_time else None
        }

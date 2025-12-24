"""
AI Trading Bot - Position Manager
Manages open positions and their lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncio
from loguru import logger

from src.utils.constants import OrderSide, PositionSide
from .order_manager import OrderManager, Order, OrderStatus


@dataclass
class Position:
    """Position representation."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    stop_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    trailing_stop_active: bool = False
    trailing_stop_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        return self.closed_at is None
        
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100


class PositionManager:
    """
    Manages position lifecycle.
    
    Features:
    - Position tracking
    - Stop-loss management
    - Take-profit management
    - Trailing stop implementation
    - Break-even adjustment
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        config: Optional[Dict] = None
    ):
        """
        Initialize position manager.
        
        Args:
            order_manager: Order manager instance.
            config: Configuration dictionary.
        """
        self.order_manager = order_manager
        self.config = config or {}
        
        # Position settings
        self.trailing_activation_atr = self.config.get('trailing_stop_activation_atr', 1.0)
        self.trailing_distance_atr = self.config.get('trailing_stop_distance_atr', 1.0)
        self.breakeven_atr = self.config.get('move_to_breakeven_atr', 1.0)
        
        # Position storage
        self._positions: Dict[str, Position] = {}
        self._position_counter = 0
        
    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"POS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._position_counter:04d}"
        
    async def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        metadata: Optional[Dict] = None
    ) -> Position:
        """
        Open a new position with stop and take profit.
        
        Args:
            symbol: Trading symbol.
            side: Position side (long/short).
            quantity: Position size.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            atr: Current ATR for trailing stop calculations.
            metadata: Additional metadata.
            
        Returns:
            Created position.
        """
        position = Position(
            id=self._generate_position_id(),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        position.metadata['atr'] = atr
        
        self._positions[position.id] = position
        
        # Place stop loss order
        stop_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
        
        stop_order = await self.order_manager.create_stop_order(
            symbol=symbol,
            side=stop_side,
            quantity=quantity,
            stop_price=stop_loss
        )
        position.stop_order_id = stop_order.id
        
        # Place take profit order
        tp_order = await self.order_manager.create_limit_order(
            symbol=symbol,
            side=stop_side,
            quantity=quantity,
            price=take_profit,
            timeout=None,  # No timeout for TP orders
            convert_to_market=False
        )
        position.take_profit_order_id = tp_order.id
        
        logger.info(f"Position opened: {position.id} {side.value} {quantity} {symbol} "
                   f"@ {entry_price}, SL={stop_loss}, TP={take_profit}")
        
        return position
        
    async def close_position(
        self,
        position_id: str,
        reason: str = "manual",
        price: Optional[float] = None
    ) -> Optional[Position]:
        """
        Close an open position.
        
        Args:
            position_id: Position ID to close.
            reason: Reason for closing.
            price: Exit price (if known).
            
        Returns:
            Closed position.
        """
        if position_id not in self._positions:
            logger.warning(f"Position {position_id} not found")
            return None
            
        position = self._positions[position_id]
        
        if not position.is_open:
            logger.warning(f"Position {position_id} already closed")
            return position
            
        # Cancel existing stop and take profit orders
        if position.stop_order_id:
            await self.order_manager.cancel_order(position.stop_order_id)
            
        if position.take_profit_order_id:
            await self.order_manager.cancel_order(position.take_profit_order_id)
            
        # Place market close order
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        
        close_order = await self.order_manager.create_market_order(
            symbol=position.symbol,
            side=close_side,
            quantity=position.quantity,
            reduce_only=True
        )
        
        # Update position
        exit_price = price or close_order.average_price
        position.current_price = exit_price
        position.closed_at = datetime.utcnow()
        position.metadata['close_reason'] = reason
        position.metadata['close_order_id'] = close_order.id
        
        # Calculate realized PnL
        if position.side == PositionSide.LONG:
            position.realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            position.realized_pnl = (position.entry_price - exit_price) * position.quantity
            
        logger.info(f"Position closed: {position.id} @ {exit_price}, "
                   f"PnL={position.realized_pnl:.2f} ({position.pnl_percent:.2f}%), "
                   f"reason={reason}")
        
        return position
        
    async def update_position(
        self,
        position_id: str,
        current_price: float,
        atr: Optional[float] = None
    ) -> Optional[Position]:
        """
        Update position with current price and check for actions.
        
        Args:
            position_id: Position ID.
            current_price: Current market price.
            atr: Current ATR value.
            
        Returns:
            Updated position.
        """
        if position_id not in self._positions:
            return None
            
        position = self._positions[position_id]
        
        if not position.is_open:
            return position
            
        position.current_price = current_price
        atr = atr or position.metadata.get('atr', 0)
        
        # Calculate unrealized PnL
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            profit_atr = (current_price - position.entry_price) / atr if atr > 0 else 0
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            profit_atr = (position.entry_price - current_price) / atr if atr > 0 else 0
            
        # Check for break-even move
        if not position.metadata.get('breakeven_applied', False):
            if profit_atr >= self.breakeven_atr:
                await self._move_to_breakeven(position)
                position.metadata['breakeven_applied'] = True
                
        # Check for trailing stop activation
        if not position.trailing_stop_active:
            if profit_atr >= self.trailing_activation_atr:
                position.trailing_stop_active = True
                await self._update_trailing_stop(position, current_price, atr)
                logger.info(f"Trailing stop activated for {position_id}")
                
        # Update trailing stop if active
        elif position.trailing_stop_active:
            await self._update_trailing_stop(position, current_price, atr)
            
        return position
        
    async def _move_to_breakeven(self, position: Position) -> None:
        """Move stop loss to breakeven."""
        # Cancel existing stop order
        if position.stop_order_id:
            await self.order_manager.cancel_order(position.stop_order_id)
            
        # Place new stop at entry price (plus small buffer for fees)
        buffer = position.entry_price * 0.001  # 0.1% buffer
        
        if position.side == PositionSide.LONG:
            new_stop = position.entry_price + buffer
        else:
            new_stop = position.entry_price - buffer
            
        stop_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        
        new_stop_order = await self.order_manager.create_stop_order(
            symbol=position.symbol,
            side=stop_side,
            quantity=position.quantity,
            stop_price=new_stop
        )
        
        position.stop_loss = new_stop
        position.stop_order_id = new_stop_order.id
        
        logger.info(f"Stop moved to breakeven for {position.id}: {new_stop}")
        
    async def _update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        atr: float
    ) -> None:
        """Update trailing stop price."""
        trailing_distance = atr * self.trailing_distance_atr
        
        if position.side == PositionSide.LONG:
            new_stop = current_price - trailing_distance
            # Only move stop up, never down
            if position.trailing_stop_price is None or new_stop > position.trailing_stop_price:
                if new_stop > position.stop_loss:
                    await self._update_stop_order(position, new_stop)
                    position.trailing_stop_price = new_stop
        else:
            new_stop = current_price + trailing_distance
            # Only move stop down, never up
            if position.trailing_stop_price is None or new_stop < position.trailing_stop_price:
                if new_stop < position.stop_loss:
                    await self._update_stop_order(position, new_stop)
                    position.trailing_stop_price = new_stop
                    
    async def _update_stop_order(self, position: Position, new_stop: float) -> None:
        """Update stop order to new price."""
        # Cancel existing stop
        if position.stop_order_id:
            await self.order_manager.cancel_order(position.stop_order_id)
            
        # Place new stop
        stop_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        
        new_stop_order = await self.order_manager.create_stop_order(
            symbol=position.symbol,
            side=stop_side,
            quantity=position.quantity,
            stop_price=new_stop
        )
        
        position.stop_loss = new_stop
        position.stop_order_id = new_stop_order.id
        
        logger.debug(f"Stop updated for {position.id}: {new_stop}")
        
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self._positions.get(position_id)
        
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions."""
        positions = [p for p in self._positions.values() if p.is_open]
        
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
            
        return positions
        
    def get_closed_positions(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Position]:
        """Get closed positions."""
        positions = [p for p in self._positions.values() if not p.is_open]
        
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
            
        # Sort by close time, most recent first
        positions.sort(key=lambda p: p.closed_at or datetime.min, reverse=True)
        
        return positions[:limit]
        
    def get_position_stats(self) -> Dict[str, Any]:
        """Get position statistics."""
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions()
        
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_realized = sum(p.realized_pnl for p in closed_positions)
        
        wins = sum(1 for p in closed_positions if p.realized_pnl > 0)
        losses = sum(1 for p in closed_positions if p.realized_pnl < 0)
        
        return {
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'total_unrealized_pnl': total_unrealized,
            'total_realized_pnl': total_realized,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(closed_positions) if closed_positions else 0
        }

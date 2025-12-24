"""
AI Trading Bot - Order Manager
Manages order placement and tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import asyncio
from loguru import logger

from src.utils.constants import OrderSide, OrderType


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    average_price: float = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """
    Manages order lifecycle.
    
    Features:
    - Order creation and submission
    - Order tracking and updates
    - Limit order timeout handling
    - Order cancellation
    """
    
    def __init__(
        self,
        exchange_api,
        config: Optional[Dict] = None
    ):
        """
        Initialize order manager.
        
        Args:
            exchange_api: Exchange API interface.
            config: Configuration dictionary.
        """
        self.exchange = exchange_api
        self.config = config or {}
        
        # Order settings
        self.default_order_type = OrderType(
            self.config.get('default_order_type', 'limit')
        )
        self.limit_order_timeout = self.config.get('limit_order_timeout', 10)
        self.slippage_tolerance = self.config.get('slippage_tolerance', 0.001)
        
        # Order storage
        self._orders: Dict[str, Order] = {}
        self._order_callbacks: List[Callable] = []
        
        # Order ID counter
        self._order_counter = 0
        
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._order_counter:04d}"
        
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False
    ) -> Order:
        """
        Create and submit a market order.
        
        Args:
            symbol: Trading symbol.
            side: Order side (buy/sell).
            quantity: Order quantity.
            reduce_only: If True, only reduce position.
            
        Returns:
            Created order.
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            metadata={'reduce_only': reduce_only}
        )
        
        self._orders[order.id] = order
        
        try:
            # Submit to exchange
            result = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.value,
                amount=quantity,
                params={'reduceOnly': reduce_only}
            )
            
            order.exchange_id = result.get('id')
            order.status = OrderStatus.FILLED
            order.filled_quantity = result.get('filled', quantity)
            order.average_price = result.get('average', result.get('price', 0))
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Market order filled: {order.id} {side.value} {quantity} {symbol} "
                       f"@ {order.average_price}")
            
            await self._notify_callbacks(order)
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)
            logger.error(f"Market order failed: {e}")
            
        return order
        
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timeout: Optional[int] = None,
        convert_to_market: bool = True
    ) -> Order:
        """
        Create and submit a limit order.
        
        Args:
            symbol: Trading symbol.
            side: Order side.
            quantity: Order quantity.
            price: Limit price.
            timeout: Timeout in seconds before cancellation.
            convert_to_market: Convert to market if not filled.
            
        Returns:
            Created order.
        """
        timeout = timeout or self.limit_order_timeout
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            metadata={'timeout': timeout, 'convert_to_market': convert_to_market}
        )
        
        self._orders[order.id] = order
        
        try:
            # Submit to exchange
            result = await self.exchange.create_limit_order(
                symbol=symbol,
                side=side.value,
                amount=quantity,
                price=price
            )
            
            order.exchange_id = result.get('id')
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Limit order submitted: {order.id} {side.value} {quantity} {symbol} "
                       f"@ {price}")
            
            # Start timeout monitoring
            asyncio.create_task(
                self._monitor_limit_order(order, timeout, convert_to_market)
            )
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)
            logger.error(f"Limit order failed: {e}")
            
        return order
        
    async def create_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None
    ) -> Order:
        """
        Create and submit a stop order.
        
        Args:
            symbol: Trading symbol.
            side: Order side.
            quantity: Order quantity.
            stop_price: Stop trigger price.
            limit_price: Limit price (if stop-limit).
            
        Returns:
            Created order.
        """
        order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP_MARKET
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price
        )
        
        self._orders[order.id] = order
        
        try:
            params = {'stopPrice': stop_price}
            
            if limit_price:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop_limit',
                    side=side.value,
                    amount=quantity,
                    price=limit_price,
                    params=params
                )
            else:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=side.value,
                    amount=quantity,
                    params=params
                )
                
            order.exchange_id = result.get('id')
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Stop order submitted: {order.id} {side.value} {quantity} {symbol} "
                       f"stop @ {stop_price}")
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)
            logger.error(f"Stop order failed: {e}")
            
        return order
        
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel.
            
        Returns:
            True if cancelled successfully.
        """
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found")
            return False
            
        order = self._orders[order_id]
        
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PENDING]:
            logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status})")
            return False
            
        try:
            if order.exchange_id:
                await self.exchange.cancel_order(order.exchange_id, order.symbol)
                
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Order cancelled: {order_id}")
            await self._notify_callbacks(order)
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
            
    async def _monitor_limit_order(
        self,
        order: Order,
        timeout: int,
        convert_to_market: bool
    ) -> None:
        """Monitor limit order and handle timeout."""
        await asyncio.sleep(timeout)
        
        # Check if order is still open
        if order.status == OrderStatus.SUBMITTED:
            logger.warning(f"Limit order {order.id} timed out after {timeout}s")
            
            # Cancel the order
            await self.cancel_order(order.id)
            
            # Convert to market if requested
            if convert_to_market and order.filled_quantity < order.quantity:
                remaining = order.quantity - order.filled_quantity
                logger.info(f"Converting remaining {remaining} to market order")
                
                await self.create_market_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=remaining
                )
                
    async def update_order_status(self, order_id: str) -> Optional[Order]:
        """
        Fetch and update order status from exchange.
        
        Args:
            order_id: Order ID to update.
            
        Returns:
            Updated order.
        """
        if order_id not in self._orders:
            return None
            
        order = self._orders[order_id]
        
        if not order.exchange_id:
            return order
            
        try:
            result = await self.exchange.fetch_order(order.exchange_id, order.symbol)
            
            status_map = {
                'open': OrderStatus.SUBMITTED,
                'closed': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'cancelled': OrderStatus.CANCELLED,
                'expired': OrderStatus.EXPIRED,
                'rejected': OrderStatus.REJECTED
            }
            
            order.status = status_map.get(result.get('status', ''), order.status)
            order.filled_quantity = result.get('filled', order.filled_quantity)
            order.average_price = result.get('average', order.average_price)
            order.updated_at = datetime.utcnow()
            
            await self._notify_callbacks(order)
            
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            
        return order
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
        
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        open_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        orders = [o for o in self._orders.values() if o.status in open_statuses]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
            
        return orders
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for order updates."""
        self._order_callbacks.append(callback)
        
    async def _notify_callbacks(self, order: Order) -> None:
        """Notify all registered callbacks."""
        for callback in self._order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Callback error: {e}")

"""
AI Trading Bot - Exchange API Wrapper
Provides unified interface to exchange APIs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncio
import ccxt.async_support as ccxt
from loguru import logger


class ExchangeAPI:
    """
    Unified exchange API interface.
    
    Wraps ccxt library to provide consistent interface
    across different exchanges.
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize exchange API.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance').
            api_key: API key.
            api_secret: API secret.
            testnet: Use testnet if True.
            config: Additional configuration.
        """
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.config = config or {}
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': self.config.get('market_type', 'spot'),
                'adjustForTimeDifference': True
            }
        })
        
        self._initialized = False
        self._markets = {}
        
    async def initialize(self) -> None:
        """Initialize connection and load markets."""
        try:
            await self.exchange.load_markets()
            self._markets = self.exchange.markets
            self._initialized = True
            logger.info(f"Exchange {self.exchange_id} initialized "
                       f"({'testnet' if self.testnet else 'live'})")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
            
    async def close(self) -> None:
        """Close exchange connection."""
        await self.exchange.close()
        logger.info("Exchange connection closed")
        
    # -------------------- Account Methods --------------------
    
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account balance.
        
        Args:
            currency: Specific currency to get balance for.
            
        Returns:
            Balance information.
        """
        try:
            balance = await self.exchange.fetch_balance()
            
            if currency:
                return {
                    'currency': currency,
                    'free': balance.get(currency, {}).get('free', 0),
                    'used': balance.get(currency, {}).get('used', 0),
                    'total': balance.get(currency, {}).get('total', 0)
                }
                
            return balance
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise
            
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions (for futures).
        
        Args:
            symbol: Symbol to filter by.
            
        Returns:
            List of positions.
        """
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                positions = await self.exchange.fetch_positions([symbol] if symbol else None)
                return [p for p in positions if float(p.get('contracts', 0)) != 0]
            return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
            
    # -------------------- Order Methods --------------------
    
    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create an order.
        
        Args:
            symbol: Trading symbol.
            type: Order type (market, limit, etc.).
            side: Order side (buy, sell).
            amount: Order amount.
            price: Order price (for limit orders).
            params: Additional parameters.
            
        Returns:
            Order result.
        """
        params = params or {}
        
        try:
            result = await self.exchange.create_order(
                symbol=symbol,
                type=type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            logger.info(f"Order created: {type} {side} {amount} {symbol} "
                       f"@ {price or 'market'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise
            
    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a limit order."""
        return await self.create_order(symbol, 'limit', side, amount, price, params)
        
    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a market order."""
        return await self.create_order(symbol, 'market', side, amount, None, params)
        
    async def cancel_order(
        self,
        order_id: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID.
            symbol: Trading symbol.
            
        Returns:
            Cancellation result.
        """
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise
            
    async def fetch_order(
        self,
        order_id: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Fetch order details.
        
        Args:
            order_id: Order ID.
            symbol: Trading symbol.
            
        Returns:
            Order details.
        """
        try:
            return await self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Failed to fetch order: {e}")
            raise
            
    async def fetch_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch open orders.
        
        Args:
            symbol: Symbol to filter by.
            
        Returns:
            List of open orders.
        """
        try:
            return await self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []
            
    async def cancel_all_orders(
        self,
        symbol: str
    ) -> List[Dict]:
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            List of cancelled orders.
        """
        try:
            if hasattr(self.exchange, 'cancel_all_orders'):
                return await self.exchange.cancel_all_orders(symbol)
            else:
                # Fallback: cancel one by one
                open_orders = await self.fetch_open_orders(symbol)
                results = []
                for order in open_orders:
                    result = await self.cancel_order(order['id'], symbol)
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            raise
            
    # -------------------- Market Data Methods --------------------
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Ticker data.
        """
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {e}")
            raise
            
    async def get_current_price(self, symbol: str) -> float:
        """
        Get current price.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Current price.
        """
        ticker = await self.fetch_ticker(symbol)
        return ticker.get('last', 0)
        
    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch order book.
        
        Args:
            symbol: Trading symbol.
            limit: Depth limit.
            
        Returns:
            Order book data.
        """
        try:
            return await self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            raise
            
    # -------------------- Utility Methods --------------------
    
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get market information.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Market information.
        """
        if symbol not in self._markets:
            logger.warning(f"Market {symbol} not found")
            return {}
            
        market = self._markets[symbol]
        
        return {
            'symbol': symbol,
            'base': market.get('base'),
            'quote': market.get('quote'),
            'tick_size': market.get('precision', {}).get('price', 8),
            'lot_size': market.get('precision', {}).get('amount', 8),
            'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0),
            'min_cost': market.get('limits', {}).get('cost', {}).get('min', 0)
        }
        
    async def set_leverage(
        self,
        symbol: str,
        leverage: int
    ) -> None:
        """
        Set leverage for a symbol (futures only).
        
        Args:
            symbol: Trading symbol.
            leverage: Leverage value.
        """
        try:
            if hasattr(self.exchange, 'set_leverage'):
                await self.exchange.set_leverage(leverage, symbol)
                logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")
            
    async def set_margin_mode(
        self,
        symbol: str,
        mode: str = "isolated"
    ) -> None:
        """
        Set margin mode (futures only).
        
        Args:
            symbol: Trading symbol.
            mode: Margin mode ('isolated' or 'cross').
        """
        try:
            if hasattr(self.exchange, 'set_margin_mode'):
                await self.exchange.set_margin_mode(mode, symbol)
                logger.info(f"Margin mode set to {mode} for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to set margin mode: {e}")

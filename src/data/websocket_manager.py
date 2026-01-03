"""
AI Trading Bot - WebSocket Manager
Manages WebSocket connections for real-time data streaming.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import logging
import uuid

from .models import Candle, OrderBook, OrderBookLevel, Trade
from .exceptions import (
    WebSocketError,
    WebSocketConnectionError,
    WebSocketSubscriptionError,
)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time market data.
    
    Supports subscriptions to candles, order book, and trade streams.
    Handles automatic reconnection on connection loss.
    
    Attributes:
        exchange_name: Name of the exchange.
        logger: Logger instance.
    """
    
    # WebSocket endpoints by exchange
    WS_ENDPOINTS = {
        'binance': 'wss://fstream.binance.com/ws',
        'binance_testnet': 'wss://stream.binancefuture.com/ws',
        'mexc': 'wss://wbs.mexc.com/ws',
        'bybit': 'wss://stream.bybit.com/v5/public/linear',
        'okx': 'wss://ws.okx.com:8443/ws/v5/public',
    }
    
    def __init__(
        self,
        exchange_name: str,
        api_credentials: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize WebSocketManager.
        
        Args:
            exchange_name: Name of the exchange (e.g., "binance", "mexc").
            api_credentials: Optional API credentials for private streams.
            logger: Optional logger instance.
        """
        self.exchange_name = exchange_name.lower()
        self.api_credentials = api_credentials or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection state
        self._ws = None
        self._connected = False
        self._reconnecting = False
        
        # Subscriptions
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, Callable] = {}
        
        # Message handling
        self._message_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        
        # Reconnection settings
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._ping_interval = 30
        
        # Get WebSocket URL
        self._ws_url = self.WS_ENDPOINTS.get(
            self.exchange_name,
            self.WS_ENDPOINTS.get('binance')
        )

    async def connect(self) -> None:
        """
        Establish WebSocket connection.
        
        Raises:
            WebSocketConnectionError: If connection fails.
        """
        try:
            import websockets
            
            self.logger.info(f"Connecting to {self.exchange_name} WebSocket...")
            
            self._ws = await websockets.connect(
                self._ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            
            self._connected = True
            self._reconnect_delay = 1.0  # Reset delay on successful connect
            
            # Start message handler
            self._message_task = asyncio.create_task(self._message_loop())
            
            # Start ping task
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            self.logger.info(f"Connected to {self.exchange_name} WebSocket")
            
        except ImportError:
            self.logger.error("websockets package not installed")
            raise WebSocketConnectionError(
                "websockets package required: pip install websockets"
            )
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            raise WebSocketConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close the WebSocket connection and cleanup."""
        self.logger.info("Disconnecting WebSocket...")
        
        self._connected = False
        
        # Cancel tasks
        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        
        # Clear subscriptions
        self._subscriptions.clear()
        self._callbacks.clear()
        
        self.logger.info("WebSocket disconnected")

    async def subscribe_candles(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Candle], None]
    ) -> str:
        """
        Subscribe to candle/kline updates.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "1h").
            callback: Callback function that receives Candle objects.
            
        Returns:
            Subscription ID.
            
        Raises:
            WebSocketSubscriptionError: If subscription fails.
        """
        sub_id = str(uuid.uuid4())
        
        try:
            # Format subscription message based on exchange
            stream_name = self._format_stream_name('kline', symbol, timeframe)
            message = self._format_subscribe_message(stream_name)
            
            # Store subscription info
            self._subscriptions[sub_id] = {
                'type': 'candle',
                'symbol': symbol,
                'timeframe': timeframe,
                'stream': stream_name,
            }
            self._callbacks[sub_id] = callback
            
            # Send subscription
            if self._ws and self._connected:
                await self._ws.send(json.dumps(message))
                self.logger.info(
                    f"Subscribed to candles: {symbol} {timeframe} (id={sub_id[:8]})"
                )
            
            return sub_id
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to candles: {e}")
            raise WebSocketSubscriptionError(f"Subscription failed: {e}")

    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[OrderBook], None]
    ) -> str:
        """
        Subscribe to order book updates.
        
        Args:
            symbol: Trading pair symbol.
            callback: Callback function that receives OrderBook objects.
            
        Returns:
            Subscription ID.
            
        Raises:
            WebSocketSubscriptionError: If subscription fails.
        """
        sub_id = str(uuid.uuid4())
        
        try:
            stream_name = self._format_stream_name('depth', symbol)
            message = self._format_subscribe_message(stream_name)
            
            self._subscriptions[sub_id] = {
                'type': 'orderbook',
                'symbol': symbol,
                'stream': stream_name,
            }
            self._callbacks[sub_id] = callback
            
            if self._ws and self._connected:
                await self._ws.send(json.dumps(message))
                self.logger.info(
                    f"Subscribed to orderbook: {symbol} (id={sub_id[:8]})"
                )
            
            return sub_id
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to orderbook: {e}")
            raise WebSocketSubscriptionError(f"Subscription failed: {e}")

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None]
    ) -> str:
        """
        Subscribe to trade stream.
        
        Args:
            symbol: Trading pair symbol.
            callback: Callback function that receives Trade objects.
            
        Returns:
            Subscription ID.
            
        Raises:
            WebSocketSubscriptionError: If subscription fails.
        """
        sub_id = str(uuid.uuid4())
        
        try:
            stream_name = self._format_stream_name('trade', symbol)
            message = self._format_subscribe_message(stream_name)
            
            self._subscriptions[sub_id] = {
                'type': 'trade',
                'symbol': symbol,
                'stream': stream_name,
            }
            self._callbacks[sub_id] = callback
            
            if self._ws and self._connected:
                await self._ws.send(json.dumps(message))
                self.logger.info(
                    f"Subscribed to trades: {symbol} (id={sub_id[:8]})"
                )
            
            return sub_id
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to trades: {e}")
            raise WebSocketSubscriptionError(f"Subscription failed: {e}")

    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from a data stream.
        
        Args:
            subscription_id: ID returned from subscribe_* method.
        """
        if subscription_id not in self._subscriptions:
            return
            
        sub_info = self._subscriptions[subscription_id]
        
        try:
            # Send unsubscribe message
            message = self._format_unsubscribe_message(sub_info['stream'])
            
            if self._ws and self._connected:
                await self._ws.send(json.dumps(message))
                
            self.logger.info(
                f"Unsubscribed: {sub_info['type']} {sub_info.get('symbol', '')} "
                f"(id={subscription_id[:8]})"
            )
            
        except Exception as e:
            self.logger.warning(f"Error unsubscribing: {e}")
        finally:
            # Remove from tracking
            self._subscriptions.pop(subscription_id, None)
            self._callbacks.pop(subscription_id, None)

    def _format_stream_name(
        self,
        stream_type: str,
        symbol: str,
        timeframe: str = None
    ) -> str:
        """Format stream name based on exchange protocol."""
        # Normalize symbol
        symbol_clean = symbol.replace('/', '').replace(':USDT', '').lower()
        
        if self.exchange_name in ('binance', 'binance_testnet'):
            if stream_type == 'kline':
                return f"{symbol_clean}@kline_{timeframe}"
            elif stream_type == 'depth':
                return f"{symbol_clean}@depth20@100ms"
            elif stream_type == 'trade':
                return f"{symbol_clean}@trade"
                
        elif self.exchange_name == 'mexc':
            if stream_type == 'kline':
                return f"spot@public.kline.v3.api@{symbol_clean.upper()}@{timeframe}"
            elif stream_type == 'depth':
                return f"spot@public.limit.depth.v3.api@{symbol_clean.upper()}@20"
            elif stream_type == 'trade':
                return f"spot@public.deals.v3.api@{symbol_clean.upper()}"
                
        elif self.exchange_name == 'bybit':
            if stream_type == 'kline':
                return f"kline.{timeframe}.{symbol_clean.upper()}"
            elif stream_type == 'depth':
                return f"orderbook.50.{symbol_clean.upper()}"
            elif stream_type == 'trade':
                return f"publicTrade.{symbol_clean.upper()}"
        
        # Default format
        return f"{stream_type}.{symbol_clean}.{timeframe or ''}"

    def _format_subscribe_message(self, stream: str) -> dict:
        """Format subscribe message based on exchange."""
        if self.exchange_name in ('binance', 'binance_testnet'):
            return {
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": int(time.time() * 1000)
            }
        elif self.exchange_name == 'mexc':
            return {
                "method": "SUBSCRIPTION",
                "params": [stream]
            }
        elif self.exchange_name == 'bybit':
            return {
                "op": "subscribe",
                "args": [stream]
            }
        elif self.exchange_name == 'okx':
            return {
                "op": "subscribe",
                "args": [{"channel": stream}]
            }
        
        return {"subscribe": stream}

    def _format_unsubscribe_message(self, stream: str) -> dict:
        """Format unsubscribe message based on exchange."""
        if self.exchange_name in ('binance', 'binance_testnet'):
            return {
                "method": "UNSUBSCRIBE",
                "params": [stream],
                "id": int(time.time() * 1000)
            }
        elif self.exchange_name == 'mexc':
            return {
                "method": "UNSUBSCRIPTION",
                "params": [stream]
            }
        elif self.exchange_name == 'bybit':
            return {
                "op": "unsubscribe",
                "args": [stream]
            }
        
        return {"unsubscribe": stream}

    async def _message_loop(self) -> None:
        """Main loop for receiving and processing messages."""
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                await self._process_message(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._connected:
                    self.logger.error(f"WebSocket error: {e}")
                    await self._handle_reconnect()
                break

    async def _process_message(self, raw_message: str) -> None:
        """
        Parse and route incoming message to appropriate callback.
        
        Args:
            raw_message: Raw JSON message from WebSocket.
        """
        try:
            data = json.loads(raw_message)
            
            # Skip ping/pong and result messages
            if isinstance(data, dict):
                if 'result' in data or 'ping' in data or 'pong' in data:
                    return
                if data.get('e') == 'ping' or data.get('op') == 'pong':
                    return
            
            # Route message to appropriate handler
            for sub_id, sub_info in self._subscriptions.items():
                if self._matches_subscription(data, sub_info):
                    callback = self._callbacks.get(sub_id)
                    if callback:
                        obj = self._parse_message(data, sub_info['type'], sub_info.get('symbol', ''))
                        if obj:
                            try:
                                callback(obj)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")
                    break
                    
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON message: {raw_message[:100]}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def _matches_subscription(self, data: dict, sub_info: dict) -> bool:
        """Check if message matches a subscription."""
        stream = sub_info.get('stream', '').lower()
        
        # Check various message formats
        if 's' in data:  # Binance format
            symbol = data['s'].lower()
            if symbol in stream:
                return True
                
        if 'stream' in data:
            if data['stream'].lower() == stream:
                return True
                
        if 'channel' in data:
            if data['channel'].lower() in stream:
                return True
        
        return False

    def _parse_message(
        self,
        data: dict,
        msg_type: str,
        symbol: str
    ) -> Optional[Any]:
        """Parse message into appropriate data model."""
        try:
            if msg_type == 'candle':
                return self._parse_candle(data, symbol)
            elif msg_type == 'orderbook':
                return self._parse_orderbook(data, symbol)
            elif msg_type == 'trade':
                return self._parse_trade(data, symbol)
        except Exception as e:
            self.logger.warning(f"Parse error for {msg_type}: {e}")
        return None

    def _parse_candle(self, data: dict, symbol: str) -> Optional[Candle]:
        """Parse candle data from message."""
        kline = data.get('k', data.get('data', data))
        
        if not isinstance(kline, dict):
            return None
            
        try:
            timestamp = kline.get('t') or kline.get('start') or data.get('T')
            if timestamp:
                ts = datetime.fromtimestamp(
                    timestamp / 1000 if timestamp > 1e10 else timestamp,
                    tz=timezone.utc
                )
            else:
                ts = datetime.now(timezone.utc)
            
            return Candle(
                timestamp=ts,
                open=float(kline.get('o', kline.get('open', 0))),
                high=float(kline.get('h', kline.get('high', 0))),
                low=float(kline.get('l', kline.get('low', 0))),
                close=float(kline.get('c', kline.get('close', 0))),
                volume=float(kline.get('v', kline.get('volume', 0))),
                symbol=symbol,
                timeframe=kline.get('i', kline.get('interval', '1m'))
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse candle: {e}")
            return None

    def _parse_orderbook(self, data: dict, symbol: str) -> Optional[OrderBook]:
        """Parse order book data from message."""
        try:
            bids_raw = data.get('b', data.get('bids', []))
            asks_raw = data.get('a', data.get('asks', []))
            
            bids = [
                OrderBookLevel(price=float(b[0]), quantity=float(b[1]))
                for b in bids_raw
            ]
            asks = [
                OrderBookLevel(price=float(a[0]), quantity=float(a[1]))
                for a in asks_raw
            ]
            
            timestamp = data.get('T') or data.get('ts')
            if timestamp:
                ts = datetime.fromtimestamp(
                    timestamp / 1000 if timestamp > 1e10 else timestamp,
                    tz=timezone.utc
                )
            else:
                ts = datetime.now(timezone.utc)
            
            return OrderBook(
                symbol=symbol,
                timestamp=ts,
                bids=bids,
                asks=asks
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse orderbook: {e}")
            return None

    def _parse_trade(self, data: dict, symbol: str) -> Optional[Trade]:
        """Parse trade data from message."""
        try:
            trade_data = data.get('data', data)
            
            timestamp = trade_data.get('T') or trade_data.get('t') or trade_data.get('ts')
            if timestamp:
                ts = datetime.fromtimestamp(
                    timestamp / 1000 if timestamp > 1e10 else timestamp,
                    tz=timezone.utc
                )
            else:
                ts = datetime.now(timezone.utc)
            
            # Determine side
            side = 'buy'
            if trade_data.get('m') is True or trade_data.get('side', '').lower() == 'sell':
                side = 'sell'
            elif trade_data.get('S', '').lower() == 'sell':
                side = 'sell'
            
            # Parse price - try multiple fields
            price = float(trade_data.get('p', 0) or trade_data.get('price', 0) or 0)
            quantity = float(trade_data.get('q', 0) or trade_data.get('qty', 0) or trade_data.get('size', 0) or 0)
            
            # If price is still 0, it's a heartbeat/status message - ignore silently
            if price == 0:
                return None
            
            return Trade(
                timestamp=ts,
                symbol=symbol,
                price=price,
                quantity=quantity,
                side=side
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse trade: {e}")
            return None

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._connected:
            try:
                await asyncio.sleep(self._ping_interval)
                if self._ws and self._connected:
                    await self._ws.ping()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Ping error: {e}")

    async def _handle_reconnect(self) -> None:
        """Handle automatic reconnection on connection loss."""
        if self._reconnecting:
            return
            
        self._reconnecting = True
        self.logger.warning("Connection lost, attempting to reconnect...")
        
        # Store current subscriptions
        saved_subscriptions = dict(self._subscriptions)
        saved_callbacks = dict(self._callbacks)
        
        while self._connected:
            try:
                # Wait before reconnecting
                await asyncio.sleep(self._reconnect_delay)
                
                # Try to connect
                await self.connect()
                
                # Restore subscriptions
                for sub_id, sub_info in saved_subscriptions.items():
                    try:
                        message = self._format_subscribe_message(sub_info['stream'])
                        await self._ws.send(json.dumps(message))
                        self._subscriptions[sub_id] = sub_info
                        self._callbacks[sub_id] = saved_callbacks.get(sub_id)
                    except Exception as e:
                        self.logger.warning(f"Failed to restore subscription: {e}")
                
                self.logger.info("Reconnected and restored subscriptions")
                self._reconnecting = False
                return
                
            except Exception as e:
                self.logger.warning(
                    f"Reconnection failed: {e}, retrying in {self._reconnect_delay}s"
                )
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
        
        self._reconnecting = False

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None

    @property
    def subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self._subscriptions)

#!/usr/bin/env python3
"""
MEXC Futures WEB API - Bypass maintenance mode
Uses internal WEB API instead of public OpenAPI

Based on: https://github.com/biberhund/MEXC_Future_Order_API_Maintenance_Bypass

FULL FUNCTIONALITY:
- Place market/limit orders
- Cancel orders
- Change leverage
- Place stop loss / take profit
- Update/move stop loss (trailing)
- Close positions
- Get positions, orders, assets
"""

import hashlib
import json
import time
import argparse
from typing import Optional, Dict, Any, List
from curl_cffi import requests


# ============================================================
# WEB API CORE
# ============================================================

def md5(value: str) -> str:
    """Calculate MD5 hash of string"""
    return hashlib.md5(value.encode('utf-8')).hexdigest()


def mexc_crypto(key: str, obj: dict) -> dict:
    """
    Generate MEXC WEB API signature
    
    Args:
        key: WEB cookie (u_id from browser)
        obj: Request body as dict
    
    Returns:
        dict with 'time' and 'sign'
    """
    date_now = str(int(time.time() * 1000))
    g = md5(key + date_now)[7:]  # Take substring from position 7
    s = json.dumps(obj, separators=(',', ':'))  # Compact JSON
    sign = md5(date_now + s + g)
    return {'time': date_now, 'sign': sign}


def make_request(key: str, obj: dict, url: str, method: str = 'POST') -> dict:
    """
    Make authenticated request to MEXC WEB API
    
    Args:
        key: WEB cookie (u_id)
        obj: Request body
        url: API endpoint URL
        method: HTTP method (POST or GET)
    
    Returns:
        Response JSON
    """
    signature = mexc_crypto(key, obj)
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'x-mxc-sign': signature['sign'],
        'x-mxc-nonce': signature['time'],
        'x-kl-ajax-request': 'Ajax_Request',
        'Authorization': key,
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        'Origin': 'https://futures.mexc.com',
        'Referer': 'https://futures.mexc.com/'
    }
    
    try:
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=obj, impersonate="chrome")
        else:
            # For GET with parameters, add them to URL
            if obj:
                params = '&'.join([f"{k}={v}" for k, v in obj.items()])
                url = f"{url}?{params}"
            response = requests.get(url, headers=headers, impersonate="chrome")
        return response.json()
    except Exception as e:
        return {'error': str(e), 'success': False}


# ============================================================
# ACCOUNT FUNCTIONS
# ============================================================

def get_assets(key: str) -> dict:
    """Get account assets"""
    url = 'https://futures.mexc.com/api/v1/private/account/assets'
    return make_request(key, {}, url, method='GET')


def get_usdt_balance(key: str) -> float:
    """Get USDT balance"""
    result = get_assets(key)
    if result.get('success'):
        for asset in result.get('data', []):
            if asset.get('currency') == 'USDT':
                return float(asset.get('availableBalance', 0))
    return 0.0


# ============================================================
# POSITION FUNCTIONS
# ============================================================

def get_open_positions(key: str, symbol: str = None) -> dict:
    """Get open positions"""
    obj = {}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/position/open_positions'
    return make_request(key, obj, url, method='GET')


def get_position_for_symbol(key: str, symbol: str) -> Optional[dict]:
    """Get position for specific symbol"""
    result = get_open_positions(key, symbol)
    if result.get('success'):
        positions = result.get('data', [])
        for pos in positions:
            if pos.get('symbol') == symbol:
                return pos
    return None


# ============================================================
# LEVERAGE FUNCTIONS
# ============================================================

def get_leverage(key: str, symbol: str) -> dict:
    """Get current leverage for symbol"""
    obj = {'symbol': symbol}
    url = 'https://futures.mexc.com/api/v1/private/position/leverage'
    return make_request(key, obj, url, method='GET')


def change_leverage(key: str, symbol: str, leverage: int, open_type: int = 1, 
                    position_type: int = 1) -> dict:
    """
    Change leverage for a symbol
    
    Args:
        key: WEB cookie
        symbol: Trading pair (e.g., BTC_USDT)
        leverage: New leverage value (1-200)
        open_type: 1=isolated, 2=cross
        position_type: 1=long, 2=short
    
    Returns:
        Response from API
    """
    obj = {
        "symbol": symbol,
        "leverage": leverage,
        "openType": open_type,
        "positionType": position_type
    }
    url = 'https://futures.mexc.com/api/v1/private/position/change_leverage'
    return make_request(key, obj, url)


# ============================================================
# ORDER FUNCTIONS
# ============================================================

def place_order(key: str, symbol: str, side: int, vol: int, leverage: int, 
                price: float = None, order_type: int = 5, open_type: int = 1,
                stop_loss_price: float = None, take_profit_price: float = None) -> dict:
    """
    Place a futures order
    
    Args:
        key: WEB cookie (u_id)
        symbol: Trading pair (e.g., BTC_USDT)
        side: 1=open long, 2=close short, 3=open short, 4=close long
        vol: Volume (number of contracts)
        leverage: Leverage multiplier
        price: Order price (required for limit orders, None for market)
        order_type: 1=limit, 2=post only, 3=IOC, 4=FOK, 5=market
        open_type: 1=isolated, 2=cross
        stop_loss_price: Optional stop loss price
        take_profit_price: Optional take profit price
    
    Returns:
        Response from API
    """
    obj = {
        "symbol": symbol,
        "side": side,
        "openType": open_type,
        "type": str(order_type),
        "vol": vol,
        "leverage": leverage,
        "priceProtect": "0"
    }
    
    if price is not None and order_type == 1:
        obj["price"] = price
    
    if stop_loss_price is not None:
        obj["stopLossPrice"] = stop_loss_price
        obj["lossTrend"] = 1  # 1=last price
    
    if take_profit_price is not None:
        obj["takeProfitPrice"] = take_profit_price
        obj["profitTrend"] = 1  # 1=last price
    
    url = 'https://futures.mexc.com/api/v1/private/order/create'
    return make_request(key, obj, url)


def get_open_orders(key: str, symbol: str = None) -> dict:
    """Get open orders"""
    obj = {'page_num': 1, 'page_size': 100}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/order/list/open_orders'
    return make_request(key, obj, url, method='GET')


def cancel_order(key: str, order_ids: List[int]) -> dict:
    """
    Cancel orders by ID
    
    Args:
        key: WEB cookie
        order_ids: List of order IDs to cancel
    
    Returns:
        Response from API
    """
    url = 'https://futures.mexc.com/api/v1/private/order/cancel'
    return make_request(key, order_ids, url)


def cancel_all_orders(key: str, symbol: str = None) -> dict:
    """
    Cancel all orders (optionally for specific symbol)
    
    Args:
        key: WEB cookie
        symbol: Optional symbol to cancel orders for
    """
    obj = {}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/order/cancel_all'
    return make_request(key, obj, url)


# ============================================================
# PLAN ORDER FUNCTIONS (Stop Loss, Take Profit, Trailing)
# ============================================================

def place_plan_order(key: str, symbol: str, side: int, vol: int, leverage: int,
                     trigger_price: float, price: float = None, order_type: int = 5,
                     open_type: int = 1, trigger_type: int = 1, trend: int = 1,
                     execute_cycle: int = 1) -> dict:
    """
    Place a plan order (conditional order - triggers at trigger_price)
    
    Args:
        key: WEB cookie
        symbol: Trading pair
        side: 1=open long, 2=close short, 3=open short, 4=close long
        vol: Volume in contracts
        leverage: Leverage
        trigger_price: Price at which order triggers
        price: Execution price (None for market)
        order_type: 1=limit, 5=market
        open_type: 1=isolated, 2=cross
        trigger_type: 1=greater than or equal, 2=less than or equal
        trend: 1=last price, 2=fair price, 3=index price
        execute_cycle: 1=24h, 2=7days
    """
    obj = {
        "symbol": symbol,
        "side": side,
        "openType": open_type,
        "vol": vol,
        "leverage": leverage,
        "triggerPrice": trigger_price,
        "triggerType": trigger_type,
        "executeCycle": execute_cycle,
        "orderType": order_type,
        "trend": trend
    }
    
    if price is not None and order_type == 1:
        obj["price"] = price
    
    url = 'https://futures.mexc.com/api/v1/private/planorder/place/v2'
    return make_request(key, obj, url)


def get_plan_orders(key: str, symbol: str = None, states: str = "1") -> dict:
    """
    Get plan orders
    
    Args:
        key: WEB cookie
        symbol: Optional symbol filter
        states: Order states (1=untriggered, 2=canceled, 3=executed, 4=invalidated, 5=failed)
    """
    now = int(time.time() * 1000)
    week_ago = now - (7 * 24 * 60 * 60 * 1000)
    
    obj = {
        'page_num': 1,
        'page_size': 100,
        'start_time': week_ago,
        'end_time': now,
        'states': states
    }
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/planorder/list/orders'
    return make_request(key, obj, url, method='GET')


def cancel_plan_order(key: str, symbol: str, order_id: int) -> dict:
    """Cancel a specific plan order"""
    obj = [{"symbol": symbol, "orderId": str(order_id)}]
    url = 'https://futures.mexc.com/api/v1/private/planorder/cancel'
    return make_request(key, obj, url)


def cancel_all_plan_orders(key: str, symbol: str = None) -> dict:
    """Cancel all plan orders"""
    obj = {}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/planorder/cancel_all'
    return make_request(key, obj, url)


# ============================================================
# STOP ORDER FUNCTIONS (TP/SL by Position)
# ============================================================

def place_stop_order(key: str, position_id: int, vol: int,
                     stop_loss_price: float = None, take_profit_price: float = None,
                     loss_trend: int = 1, profit_trend: int = 1) -> dict:
    """
    Place TP/SL order by position
    
    Args:
        key: WEB cookie
        position_id: Position ID
        vol: Volume to close
        stop_loss_price: Stop loss price (optional)
        take_profit_price: Take profit price (optional)
        loss_trend: SL price type (1=last, 2=fair, 3=index)
        profit_trend: TP price type (1=last, 2=fair, 3=index)
    """
    obj = {
        "positionId": position_id,
        "vol": vol,
        "lossTrend": loss_trend,
        "profitTrend": profit_trend
    }
    
    if stop_loss_price is not None:
        obj["stopLossPrice"] = stop_loss_price
    
    if take_profit_price is not None:
        obj["takeProfitPrice"] = take_profit_price
    
    url = 'https://futures.mexc.com/api/v1/private/stoporder/place'
    return make_request(key, obj, url)


def get_stop_orders(key: str, symbol: str = None, is_finished: int = 0) -> dict:
    """
    Get TP/SL orders
    
    Args:
        is_finished: 0=active, 1=finished
    """
    now = int(time.time() * 1000)
    week_ago = now - (7 * 24 * 60 * 60 * 1000)
    
    obj = {
        'page_num': 1,
        'page_size': 100,
        'start_time': week_ago,
        'end_time': now,
        'is_finished': is_finished
    }
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/stoporder/list/orders'
    return make_request(key, obj, url, method='GET')


def cancel_stop_order(key: str, stop_order_id: int) -> dict:
    """Cancel a specific stop order"""
    obj = [{"stopPlanOrderId": stop_order_id}]
    url = 'https://futures.mexc.com/api/v1/private/stoporder/cancel'
    return make_request(key, obj, url)


def cancel_all_stop_orders(key: str, symbol: str = None, position_id: int = None) -> dict:
    """Cancel all stop orders"""
    obj = {}
    if symbol:
        obj['symbol'] = symbol
    if position_id:
        obj['positionId'] = position_id
    url = 'https://futures.mexc.com/api/v1/private/stoporder/cancel_all'
    return make_request(key, obj, url)


def update_stop_order(key: str, stop_order_id: int, 
                      stop_loss_price: float = None, take_profit_price: float = None,
                      loss_trend: int = 1, profit_trend: int = 1) -> dict:
    """
    Update/move an existing stop order (trailing stop)
    
    Args:
        key: WEB cookie
        stop_order_id: ID of stop order to update
        stop_loss_price: New stop loss price
        take_profit_price: New take profit price
        loss_trend: SL price type
        profit_trend: TP price type
    """
    obj = {
        "stopPlanOrderId": stop_order_id,
        "lossTrend": loss_trend,
        "profitTrend": profit_trend
    }
    
    if stop_loss_price is not None:
        obj["stopLossPrice"] = stop_loss_price
    
    if take_profit_price is not None:
        obj["takeProfitPrice"] = take_profit_price
    
    url = 'https://futures.mexc.com/api/v1/private/stoporder/change_plan_price'
    return make_request(key, obj, url)


# ============================================================
# TRAILING ORDER FUNCTIONS
# ============================================================

def place_trailing_order(key: str, symbol: str, side: int, vol: int, leverage: int,
                         back_type: int, back_value: float, active_price: float = None,
                         trend: int = 1, open_type: int = 1, position_mode: int = 1) -> dict:
    """
    Place a trailing stop order
    
    Args:
        key: WEB cookie
        symbol: Trading pair
        side: 1=open long, 2=close short, 3=open short, 4=close long
        vol: Volume
        leverage: Leverage
        back_type: 1=percentage, 2=absolute value
        back_value: Callback value (e.g., 0.02 for 2% if back_type=1)
        active_price: Activation price (optional)
        trend: Price type (1=last, 2=fair, 3=index)
        open_type: 1=isolated, 2=cross
        position_mode: 1=hedge, 2=one-way
    """
    obj = {
        "symbol": symbol,
        "leverage": leverage,
        "side": side,
        "vol": vol,
        "openType": open_type,
        "trend": trend,
        "backType": back_type,
        "backValue": back_value,
        "positionMode": position_mode
    }
    
    if active_price is not None:
        obj["activePrice"] = active_price
    
    url = 'https://futures.mexc.com/api/v1/private/trackorder/place'
    return make_request(key, obj, url)


def get_trailing_orders(key: str, symbol: str = None, states: str = "0,1") -> dict:
    """Get trailing orders"""
    obj = {'states': states}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/trackorder/list/orders'
    return make_request(key, obj, url, method='GET')


def cancel_trailing_order(key: str, symbol: str, order_id: int) -> dict:
    """Cancel trailing order"""
    obj = {"symbol": symbol, "trackOrderId": order_id}
    url = 'https://futures.mexc.com/api/v1/private/trackorder/cancel'
    return make_request(key, obj, url)


# ============================================================
# CLOSE POSITION FUNCTIONS
# ============================================================

def close_position(key: str, symbol: str, position_id: int = None, vol: int = None) -> dict:
    """
    Close a position (market order)
    
    Args:
        key: WEB cookie
        symbol: Trading pair
        position_id: Optional position ID
        vol: Volume to close (None = close all)
    """
    # First get position info
    pos = get_position_for_symbol(key, symbol)
    if not pos:
        return {'success': False, 'error': 'No position found'}
    
    position_type = pos.get('positionType', 1)  # 1=long, 2=short
    hold_vol = int(pos.get('holdVol', 0))
    leverage = pos.get('leverage', 1)
    
    # Determine close side: 4=close long, 2=close short
    close_side = 4 if position_type == 1 else 2
    close_vol = vol if vol else hold_vol
    
    return place_order(
        key=key,
        symbol=symbol,
        side=close_side,
        vol=close_vol,
        leverage=leverage,
        order_type=5  # Market
    )


def close_all_positions(key: str) -> dict:
    """Close all open positions"""
    url = 'https://futures.mexc.com/api/v1/private/position/close_all'
    return make_request(key, {}, url)


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def get_order_history(key: str, symbol: str = None, page: int = 1, page_size: int = 20) -> dict:
    """Get historical orders"""
    obj = {'page_num': page, 'page_size': page_size}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/order/list/history_orders'
    return make_request(key, obj, url, method='GET')


def get_position_history(key: str, symbol: str = None, page: int = 1, page_size: int = 20) -> dict:
    """Get historical positions"""
    obj = {'page_num': page, 'page_size': page_size}
    if symbol:
        obj['symbol'] = symbol
    url = 'https://futures.mexc.com/api/v1/private/position/list/history_positions'
    return make_request(key, obj, url, method='GET')


# ============================================================
# TEST ALL FUNCTIONS
# ============================================================

def test_all_functions(key: str, symbol: str = "DOGE_USDT"):
    """Test all API functions"""
    print("=" * 70)
    print("MEXC WEB API - FULL FUNCTIONALITY TEST")
    print("=" * 70)
    print()
    
    tests = []
    
    # 1. Get Assets
    print("1. Testing get_assets...")
    result = get_assets(key)
    success = result.get('success', False)
    tests.append(('get_assets', success))
    if success:
        balance = get_usdt_balance(key)
        print(f"   ✅ Success! USDT Balance: ${balance:.2f}")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # 2. Get Positions
    print("2. Testing get_open_positions...")
    result = get_open_positions(key)
    success = result.get('success', False)
    tests.append(('get_open_positions', success))
    if success:
        positions = result.get('data', [])
        print(f"   ✅ Success! Open positions: {len(positions)}")
        for pos in positions[:3]:
            print(f"      - {pos.get('symbol')}: {pos.get('holdVol')} @ {pos.get('holdAvgPrice')}")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # 3. Get Leverage
    print(f"3. Testing get_leverage for {symbol}...")
    result = get_leverage(key, symbol)
    success = result.get('success', False)
    tests.append(('get_leverage', success))
    if success:
        data = result.get('data', [])
        if data:
            print(f"   ✅ Success! Current leverage: {data[0].get('leverage')}x")
        else:
            print(f"   ✅ Success! No leverage data (no position)")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # 4. Change Leverage
    print(f"4. Testing change_leverage for {symbol} to 10x...")
    result = change_leverage(key, symbol, leverage=10, position_type=1)
    success = result.get('success', False)
    tests.append(('change_leverage (long)', success))
    if success:
        print(f"   ✅ Success! Leverage changed to 10x (long)")
    else:
        print(f"   ❌ Failed: {result}")
    
    result = change_leverage(key, symbol, leverage=10, position_type=2)
    success = result.get('success', False)
    tests.append(('change_leverage (short)', success))
    if success:
        print(f"   ✅ Success! Leverage changed to 10x (short)")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # 5. Place Order
    print(f"5. Testing place_order (LONG {symbol}, 1 contract, 10x)...")
    result = place_order(key, symbol, side=1, vol=1, leverage=10, order_type=5)
    success = result.get('success', False)
    tests.append(('place_order', success))
    if success:
        order_id = result.get('data', {}).get('orderId')
        print(f"   ✅ Success! Order ID: {order_id}")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # Wait for order to fill
    time.sleep(2)
    
    # 6. Get Position
    print(f"6. Testing get_position_for_symbol {symbol}...")
    pos = get_position_for_symbol(key, symbol)
    tests.append(('get_position_for_symbol', pos is not None))
    if pos:
        position_id = pos.get('positionId')
        hold_vol = pos.get('holdVol')
        entry_price = pos.get('holdAvgPrice')
        print(f"   ✅ Success! Position ID: {position_id}, Vol: {hold_vol}, Entry: {entry_price}")
    else:
        print(f"   ❌ No position found")
        position_id = None
    print()
    
    # 7. Place Stop Order (if position exists)
    if position_id:
        print(f"7. Testing place_stop_order for position {position_id}...")
        # Calculate SL at 5% below entry, round to 5 decimal places
        sl_price = round(float(entry_price) * 0.95, 5)
        result = place_stop_order(key, position_id=int(position_id), vol=int(hold_vol), 
                                  stop_loss_price=sl_price)
        success = result.get('success', False)
        tests.append(('place_stop_order', success))
        if success:
            print(f"   ✅ Success! Stop order placed at {sl_price:.6f}")
        else:
            print(f"   ❌ Failed: {result}")
        print()
        
        # 8. Get Stop Orders
        print("8. Testing get_stop_orders...")
        result = get_stop_orders(key, symbol)
        success = result.get('success', False)
        tests.append(('get_stop_orders', success))
        if success:
            orders = result.get('data', [])
            print(f"   ✅ Success! Stop orders: {len(orders)}")
        else:
            print(f"   ❌ Failed: {result}")
        print()
        
        # 9. Cancel All Stop Orders
        print("9. Testing cancel_all_stop_orders...")
        result = cancel_all_stop_orders(key, symbol)
        success = result.get('success', False)
        tests.append(('cancel_all_stop_orders', success))
        if success:
            print(f"   ✅ Success! All stop orders cancelled")
        else:
            print(f"   ❌ Failed: {result}")
        print()
        
        # 10. Close Position
        print(f"10. Testing close_position for {symbol}...")
        result = close_position(key, symbol)
        success = result.get('success', False)
        tests.append(('close_position', success))
        if success:
            print(f"   ✅ Success! Position closed")
        else:
            print(f"   ❌ Failed: {result}")
        print()
    
    # 11. Get Order History
    print("11. Testing get_order_history...")
    result = get_order_history(key, symbol)
    success = result.get('success', False)
    tests.append(('get_order_history', success))
    if success:
        print(f"   ✅ Success!")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # 12. Get Plan Orders
    print("12. Testing get_plan_orders...")
    result = get_plan_orders(key)
    success = result.get('success', False)
    tests.append(('get_plan_orders', success))
    if success:
        print(f"   ✅ Success!")
    else:
        print(f"   ❌ Failed: {result}")
    print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s in tests if s)
    total = len(tests)
    
    for name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MEXC Futures WEB API - Full Functionality')
    parser.add_argument('--key', type=str, help='WEB cookie (u_id)', required=True)
    parser.add_argument('--action', type=str, 
                        choices=['test', 'test_all', 'order', 'close', 'cancel', 'positions', 
                                'assets', 'leverage', 'stop', 'trailing', 'plan'],
                        default='test', help='Action to perform')
    parser.add_argument('--symbol', type=str, default='DOGE_USDT', help='Trading pair')
    parser.add_argument('--side', type=int, default=1, help='1=open long, 3=open short, 4=close long, 2=close short')
    parser.add_argument('--vol', type=int, default=1, help='Volume (contracts)')
    parser.add_argument('--leverage', type=int, default=10, help='Leverage')
    parser.add_argument('--price', type=float, help='Price for limit order')
    parser.add_argument('--type', type=int, default=5, help='Order type: 1=limit, 5=market')
    parser.add_argument('--sl', type=float, help='Stop loss price')
    parser.add_argument('--tp', type=float, help='Take profit price')
    parser.add_argument('--trigger', type=float, help='Trigger price for plan orders')
    parser.add_argument('--back-type', type=int, default=1, help='Trailing: 1=percent, 2=absolute')
    parser.add_argument('--back-value', type=float, default=0.02, help='Trailing callback value')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MEXC Futures WEB API")
    print("=" * 60)
    print(f"Action: {args.action}")
    print(f"Key: {args.key[:20]}..." if len(args.key) > 20 else f"Key: {args.key}")
    print()
    
    if args.action == 'test':
        print("Testing connection with get_assets...")
        result = get_assets(args.key)
        print(f"Result: {json.dumps(result, indent=2)}")
        
    elif args.action == 'test_all':
        test_all_functions(args.key, args.symbol)
        
    elif args.action == 'assets':
        result = get_assets(args.key)
        print(f"Assets: {json.dumps(result, indent=2)}")
        
    elif args.action == 'positions':
        result = get_open_positions(args.key, args.symbol if args.symbol != 'DOGE_USDT' else None)
        print(f"Positions: {json.dumps(result, indent=2)}")
        
    elif args.action == 'leverage':
        # Change leverage for both long and short
        result1 = change_leverage(args.key, args.symbol, args.leverage, position_type=1)
        result2 = change_leverage(args.key, args.symbol, args.leverage, position_type=2)
        print(f"Long leverage: {json.dumps(result1, indent=2)}")
        print(f"Short leverage: {json.dumps(result2, indent=2)}")
        
    elif args.action == 'order':
        print(f"Placing order: {args.symbol} Side={args.side} x{args.leverage}")
        print(f"Volume: {args.vol}, Type: {'Market' if args.type == 5 else 'Limit'}")
        
        result = place_order(
            key=args.key,
            symbol=args.symbol,
            side=args.side,
            vol=args.vol,
            leverage=args.leverage,
            price=args.price,
            order_type=args.type,
            stop_loss_price=args.sl,
            take_profit_price=args.tp
        )
        print(f"Result: {json.dumps(result, indent=2)}")
        
    elif args.action == 'close':
        print(f"Closing position: {args.symbol}")
        result = close_position(args.key, args.symbol)
        print(f"Result: {json.dumps(result, indent=2)}")
        
    elif args.action == 'cancel':
        print(f"Cancelling all orders for: {args.symbol}")
        result1 = cancel_all_orders(args.key, args.symbol)
        result2 = cancel_all_plan_orders(args.key, args.symbol)
        result3 = cancel_all_stop_orders(args.key, args.symbol)
        print(f"Cancel orders: {json.dumps(result1, indent=2)}")
        print(f"Cancel plan orders: {json.dumps(result2, indent=2)}")
        print(f"Cancel stop orders: {json.dumps(result3, indent=2)}")
        
    elif args.action == 'stop':
        # Place stop order for existing position
        pos = get_position_for_symbol(args.key, args.symbol)
        if pos:
            position_id = int(pos.get('positionId'))
            hold_vol = int(pos.get('holdVol'))
            result = place_stop_order(
                args.key, position_id, hold_vol,
                stop_loss_price=args.sl,
                take_profit_price=args.tp
            )
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print("No position found for this symbol")
            
    elif args.action == 'trailing':
        result = place_trailing_order(
            key=args.key,
            symbol=args.symbol,
            side=args.side,
            vol=args.vol,
            leverage=args.leverage,
            back_type=args.back_type,
            back_value=args.back_value
        )
        print(f"Result: {json.dumps(result, indent=2)}")
        
    elif args.action == 'plan':
        if args.trigger:
            result = place_plan_order(
                key=args.key,
                symbol=args.symbol,
                side=args.side,
                vol=args.vol,
                leverage=args.leverage,
                trigger_price=args.trigger,
                price=args.price,
                order_type=args.type
            )
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print("--trigger price required for plan orders")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MEXC API Test Script
Тестирует все функции которые используются в live trading:
1. Подключение к API
2. Получение баланса
3. Открытие позиции (LONG/SHORT)
4. Установка стоп-лосса
5. Закрытие позиции
6. Получение открытых позиций

ВАЖНО: Используй маленький размер позиции для тестов!
"""
import sys
import time
import hmac
import hashlib
import requests
from pathlib import Path

# Load secrets
import yaml
secrets_path = Path(__file__).parent.parent / 'config' / 'secrets.yaml'
with open(secrets_path) as f:
    secrets = yaml.safe_load(f)

API_KEY = secrets['mexc']['api_key']
API_SECRET = secrets['mexc']['api_secret']
BASE_URL = 'https://contract.mexc.com'

def generate_signature(secret: str, sign_str: str) -> str:
    return hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()

def mexc_request(method: str, endpoint: str, params: dict = None, max_retries: int = 3):
    """Make authenticated MEXC API request with retry logic"""
    import json as json_lib
    if params is None:
        params = {}
    
    timestamp = int(time.time() * 1000)
    
    # For GET: use sorted params string
    # For POST: use JSON string (as per MEXC docs)
    if method == 'GET':
        params['timestamp'] = timestamp
        sorted_params = sorted(params.items())
        params_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
    else:  # POST
        params_str = json_lib.dumps(params, separators=(',', ':'))
    
    sign_str = f"{API_KEY}{timestamp}{params_str}"
    signature = generate_signature(API_SECRET, sign_str)
    
    headers = {
        'ApiKey': API_KEY,
        'Request-Time': str(timestamp),
        'Signature': signature,
        'Content-Type': 'application/json'
    }
    
    url = f"{BASE_URL}{endpoint}"
    
    for attempt in range(max_retries):
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=15)
            elif method == 'POST':
                # POST should send JSON body, not query params
                response = requests.post(url, json=params, headers=headers, timeout=15)
            else:
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"  ⏱️ Timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    return None

def test_connectivity():
    """Test basic API connectivity"""
    print("\n" + "="*60)
    print("1. TESTING CONNECTIVITY")
    print("="*60)
    
    try:
        r = requests.get(f"{BASE_URL}/api/v1/contract/ping", timeout=10)
        print(f"  ✅ Ping: {r.status_code} - {r.json()}")
        return True
    except Exception as e:
        print(f"  ❌ Ping failed: {e}")
        return False

def test_account_balance():
    """Test getting account balance"""
    print("\n" + "="*60)
    print("2. TESTING ACCOUNT BALANCE")
    print("="*60)
    
    result = mexc_request('GET', '/api/v1/private/account/assets', {})
    
    if result and result.get('success'):
        for asset in result.get('data', []):
            if asset['currency'] == 'USDT':
                avail = float(asset.get('availableBalance', 0))
                frozen = float(asset.get('frozenBalance', 0))
                print(f"  ✅ USDT Balance:")
                print(f"     Available: ${avail:,.2f}")
                print(f"     Frozen:    ${frozen:,.2f}")
                return avail
    else:
        print(f"  ❌ Failed to get balance: {result}")
    return 0

def test_open_positions():
    """Test getting open positions"""
    print("\n" + "="*60)
    print("3. TESTING OPEN POSITIONS")
    print("="*60)
    
    result = mexc_request('GET', '/api/v1/private/position/open_positions', {})
    
    if result and result.get('success'):
        positions = result.get('data', [])
        if positions:
            print(f"  ✅ Found {len(positions)} open positions:")
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                side = 'LONG' if pos.get('positionType') == 1 else 'SHORT'
                vol = pos.get('holdVol', 0)
                pnl = pos.get('unrealisedPnl', 0)
                print(f"     {symbol} {side}: {vol} contracts | PnL: ${pnl}")
        else:
            print("  ✅ No open positions")
        return positions
    else:
        print(f"  ❌ Failed to get positions: {result}")
    return []

def test_contract_info(symbol: str = 'PIPPIN_USDT'):
    """Test getting contract info"""
    print("\n" + "="*60)
    print(f"4. TESTING CONTRACT INFO ({symbol})")
    print("="*60)
    
    result = mexc_request('GET', '/api/v1/contract/detail', {'symbol': symbol})
    
    if result and result.get('success'):
        data = result.get('data', {})
        print(f"  ✅ Contract {symbol}:")
        print(f"     Min Volume: {data.get('minVol', 'N/A')}")
        print(f"     Max Volume: {data.get('maxVol', 'N/A')}")
        print(f"     Contract Size: {data.get('contractSize', 'N/A')}")
        print(f"     Price Precision: {data.get('priceScale', 'N/A')}")
        print(f"     Volume Precision: {data.get('volScale', 'N/A')}")
        return data
    else:
        print(f"  ❌ Failed to get contract info: {result}")
    return None

def test_ticker(symbol: str = 'PIPPIN_USDT'):
    """Test getting current price"""
    print("\n" + "="*60)
    print(f"5. TESTING TICKER ({symbol})")
    print("="*60)
    
    try:
        r = requests.get(f"{BASE_URL}/api/v1/contract/ticker", params={'symbol': symbol}, timeout=10)
        result = r.json()
        
        if result.get('success'):
            data = result.get('data', {})
            print(f"  ✅ {symbol} Ticker:")
            print(f"     Last Price: {data.get('lastPrice', 'N/A')}")
            print(f"     Bid: {data.get('bid1', 'N/A')}")
            print(f"     Ask: {data.get('ask1', 'N/A')}")
            print(f"     24h Volume: {data.get('volume24', 'N/A')}")
            return float(data.get('lastPrice', 0))
        else:
            print(f"  ❌ Failed: {result}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    return 0

def test_place_order(symbol: str = 'PIPPIN_USDT', side: int = 1, volume: int = 10, 
                     leverage: int = 5, dry_run: bool = True):
    """
    Test placing an order
    side: 1=Open Long, 2=Close Short, 3=Open Short, 4=Close Long
    type: 5=Market, 1=Limit
    """
    print("\n" + "="*60)
    print(f"6. TESTING ORDER PLACEMENT {'(DRY RUN)' if dry_run else '(REAL!)'}")
    print("="*60)
    
    params = {
        'symbol': symbol,
        'price': 0,  # Market order
        'vol': volume,
        'leverage': leverage,
        'side': side,
        'type': 5,  # Market
        'openType': 1  # Isolated margin
    }
    
    side_name = {1: 'Open LONG', 2: 'Close Short', 3: 'Open SHORT', 4: 'Close Long'}
    
    print(f"  Order params:")
    print(f"     Symbol: {symbol}")
    print(f"     Side: {side_name.get(side, side)}")
    print(f"     Volume: {volume} contracts")
    print(f"     Leverage: {leverage}x")
    print(f"     Type: Market")
    
    if dry_run:
        print(f"\n  ⚠️ DRY RUN - Order NOT placed")
        print(f"     To place real order, run with --execute flag")
        return None
    
    result = mexc_request('POST', '/api/v1/private/order/create', params)
    
    if result and result.get('success'):
        order_id = result.get('data')
        print(f"\n  ✅ Order placed successfully!")
        print(f"     Order ID: {order_id}")
        return order_id
    else:
        print(f"\n  ❌ Order failed: {result}")
    return None

def test_stop_order(symbol: str = 'PIPPIN_USDT', side: int = 4, volume: int = 10,
                    stop_price: float = 0.01, leverage: int = 5, dry_run: bool = True):
    """
    Test placing a stop-loss order
    side: 2=Close Short (stop for long), 4=Close Long (stop for short)
    """
    print("\n" + "="*60)
    print(f"7. TESTING STOP ORDER {'(DRY RUN)' if dry_run else '(REAL!)'}")
    print("="*60)
    
    params = {
        'symbol': symbol,
        'vol': volume,
        'side': side,
        'triggerPrice': stop_price,
        'triggerType': 1,  # 1=Mark price, 2=Fair price, 3=Last price
        'executeCycle': 1,  # 1=Always
        'orderType': 5,  # Market
        'trend': 2 if side == 4 else 1  # 1=for short, 2=for long (price falling)
    }
    
    print(f"  Stop order params:")
    print(f"     Symbol: {symbol}")
    print(f"     Side: {'Close Long' if side == 4 else 'Close Short'}")
    print(f"     Volume: {volume} contracts")
    print(f"     Trigger Price: {stop_price}")
    
    if dry_run:
        print(f"\n  ⚠️ DRY RUN - Stop order NOT placed")
        return None
    
    result = mexc_request('POST', '/api/v1/private/planorder/place/v2', params)
    
    if result and result.get('success'):
        order_id = result.get('data')
        print(f"\n  ✅ Stop order placed!")
        print(f"     Order ID: {order_id}")
        return order_id
    else:
        print(f"\n  ❌ Stop order failed: {result}")
    return None

def test_close_position(symbol: str = 'PIPPIN_USDT', position_id: int = None, 
                        dry_run: bool = True):
    """Test closing a position"""
    print("\n" + "="*60)
    print(f"8. TESTING CLOSE POSITION {'(DRY RUN)' if dry_run else '(REAL!)'}")
    print("="*60)
    
    if position_id is None:
        print("  ⚠️ No position_id provided")
        return None
    
    params = {
        'symbol': symbol,
        'positionId': position_id
    }
    
    print(f"  Close params:")
    print(f"     Symbol: {symbol}")
    print(f"     Position ID: {position_id}")
    
    if dry_run:
        print(f"\n  ⚠️ DRY RUN - Position NOT closed")
        return None
    
    result = mexc_request('POST', '/api/v1/private/position/close', params)
    
    if result and result.get('success'):
        print(f"\n  ✅ Position closed!")
        return True
    else:
        print(f"\n  ❌ Close failed: {result}")
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test MEXC API functionality')
    parser.add_argument('--symbol', default='PIPPIN_USDT', help='Symbol to test (default: PIPPIN_USDT)')
    parser.add_argument('--volume', type=int, default=10, help='Volume for test orders (default: 10)')
    parser.add_argument('--leverage', type=int, default=5, help='Leverage for test orders (default: 5)')
    parser.add_argument('--execute', action='store_true', help='Actually execute orders (WARNING: REAL MONEY!)')
    parser.add_argument('--long', action='store_true', help='Open LONG position')
    parser.add_argument('--short', action='store_true', help='Open SHORT position')
    parser.add_argument('--close', action='store_true', help='Close all positions for symbol')
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    print("="*60)
    print("MEXC API TEST SCRIPT")
    print("="*60)
    if args.execute:
        print("⚠️  WARNING: EXECUTE MODE - REAL ORDERS WILL BE PLACED!")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Cancelled.")
            return
    else:
        print("ℹ️  DRY RUN MODE - No real orders will be placed")
    print("="*60)
    
    # Basic tests
    if not test_connectivity():
        print("\n❌ Connectivity failed, stopping.")
        return
    
    balance = test_account_balance()
    positions = test_open_positions()
    contract = test_contract_info(args.symbol)
    price = test_ticker(args.symbol)
    
    # Order tests
    if args.long:
        test_place_order(args.symbol, side=1, volume=args.volume, 
                        leverage=args.leverage, dry_run=dry_run)
        if not dry_run and price:
            # Place stop loss at -5%
            stop_price = price * 0.95
            time.sleep(1)
            test_stop_order(args.symbol, side=4, volume=args.volume,
                           stop_price=stop_price, leverage=args.leverage, dry_run=dry_run)
    
    if args.short:
        test_place_order(args.symbol, side=3, volume=args.volume,
                        leverage=args.leverage, dry_run=dry_run)
        if not dry_run and price:
            # Place stop loss at +5%
            stop_price = price * 1.05
            time.sleep(1)
            test_stop_order(args.symbol, side=2, volume=args.volume,
                           stop_price=stop_price, leverage=args.leverage, dry_run=dry_run)
    
    if args.close and positions:
        for pos in positions:
            if pos.get('symbol') == args.symbol:
                test_close_position(args.symbol, pos.get('positionId'), dry_run=dry_run)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()

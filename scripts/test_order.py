#!/usr/bin/env python3
"""Test MEXC order placement with proper signature."""

import yaml
import hmac
import hashlib
import time
import json
import requests

def main():
    with open('config/secrets.yaml') as f:
        secrets = yaml.safe_load(f)

    api_key = secrets['mexc']['api_key']
    api_secret = secrets['mexc']['api_secret']

    timestamp = int(time.time() * 1000)
    
    # Попробуем BTC_USDT с лимитным ордером (низкая цена - не исполнится)
    params = {
        'symbol': 'BTC_USDT',
        'price': 50000,  # Низкая цена - ордер не исполнится
        'vol': 1,
        'leverage': 5,
        'side': 1,  # Open long
        'type': 1,  # Limit order
        'openType': 1  # Isolated
    }
    
    params_str = json.dumps(params, separators=(',', ':'))
    sign_str = f'{api_key}{timestamp}{params_str}'
    signature = hmac.new(api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()

    headers = {
        'ApiKey': api_key,
        'Request-Time': str(timestamp),
        'Signature': signature,
        'Content-Type': 'application/json'
    }

    print(f"Testing order on BTC_USDT...")
    print(f"Params: {params}")
    print(f"Timestamp: {timestamp}")
    print()
    
    try:
        r = requests.post(
            'https://contract.mexc.com/api/v1/private/order/create',
            json=params, 
            headers=headers, 
            timeout=30  # Longer timeout
        )
        print(f'Status: {r.status_code}')
        print(f'Response: {r.text}')
    except requests.exceptions.Timeout:
        print('TIMEOUT after 30 seconds')
    except Exception as e:
        print(f'Error: {type(e).__name__}: {e}')

if __name__ == '__main__':
    main()

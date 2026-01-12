#!/usr/bin/env python3
"""Test MEXC order with proper signature according to docs."""

import yaml
import hmac
import hashlib
import time
import json
import requests

def main():
    # 1. Get server time first
    print("1. Getting server time...")
    r = requests.get('https://contract.mexc.com/api/v1/contract/ping', timeout=10)
    server_time = r.json()['data']
    local_time = int(time.time() * 1000)
    diff = local_time - server_time
    print(f"   Server time: {server_time}")
    print(f"   Local time:  {local_time}")
    print(f"   Difference:  {diff}ms")
    
    # Use server time to avoid time sync issues
    timestamp = server_time
    
    # 2. Load credentials
    with open('config/secrets.yaml') as f:
        secrets = yaml.safe_load(f)
    api_key = secrets['mexc']['api_key']
    api_secret = secrets['mexc']['api_secret']
    
    print(f"\n2. API Key: {api_key[:10]}...")
    
    # 3. Build order params
    params = {
        'symbol': 'BTC_USDT',
        'price': 50000,
        'vol': 1,
        'leverage': 5,
        'side': 1,
        'type': 1,
        'openType': 1
    }
    
    # For POST: JSON string (no sorting)
    params_str = json.dumps(params, separators=(',', ':'))
    
    # Signature string: accessKey + timestamp + parameterString
    sign_str = f'{api_key}{timestamp}{params_str}'
    
    print(f"\n3. Signature components:")
    print(f"   accessKey: {api_key}")
    print(f"   timestamp: {timestamp}")
    print(f"   params:    {params_str}")
    print(f"   sign_str:  {sign_str[:50]}...")
    
    # HMAC-SHA256
    signature = hmac.new(
        api_secret.encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"   signature: {signature}")
    
    # 4. Headers
    headers = {
        'ApiKey': api_key,
        'Request-Time': str(timestamp),
        'Signature': signature,
        'Content-Type': 'application/json'
    }
    
    print(f"\n4. Headers: {headers}")
    print(f"\n5. Sending POST request...")
    
    try:
        r = requests.post(
            'https://contract.mexc.com/api/v1/private/order/create',
            json=params,
            headers=headers,
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.text}")
    except requests.exceptions.Timeout:
        print("   TIMEOUT after 30 seconds")
    except Exception as e:
        print(f"   Error: {type(e).__name__}: {e}")

if __name__ == '__main__':
    main()

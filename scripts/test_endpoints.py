#!/usr/bin/env python3
"""Test different MEXC POST endpoints to find which ones work."""

import yaml
import hmac
import hashlib
import time
import json
import requests

def make_request(api_key, api_secret, method, endpoint, params=None):
    """Make authenticated request to MEXC."""
    timestamp = int(time.time() * 1000)
    
    if params is None:
        params = {}
    
    if method == 'GET':
        params['timestamp'] = timestamp
        sorted_params = sorted(params.items())
        params_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
    else:  # POST
        params_str = json.dumps(params, separators=(',', ':'))
    
    sign_str = f'{api_key}{timestamp}{params_str}'
    signature = hmac.new(api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
    
    headers = {
        'ApiKey': api_key,
        'Request-Time': str(timestamp),
        'Signature': signature,
        'Content-Type': 'application/json'
    }
    
    url = f'https://contract.mexc.com{endpoint}'
    
    try:
        if method == 'GET':
            r = requests.get(url, params=params, headers=headers, timeout=10)
        else:
            r = requests.post(url, json=params, headers=headers, timeout=10)
        return r.status_code, r.text[:200]
    except requests.exceptions.Timeout:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)

def main():
    with open('config/secrets.yaml') as f:
        secrets = yaml.safe_load(f)
    api_key = secrets['mexc']['api_key']
    api_secret = secrets['mexc']['api_secret']
    
    print("Testing MEXC API endpoints...\n")
    
    # Test endpoints
    tests = [
        # GET endpoints (should work)
        ('GET', '/api/v1/private/account/assets', {}),
        ('GET', '/api/v1/private/position/open_positions', {}),
        ('GET', '/api/v1/private/order/list/open_orders', {'page_num': 1, 'page_size': 10}),
        
        # POST endpoints
        ('POST', '/api/v1/private/account/asset/analysis/v3', 
         {'startTime': int(time.time()*1000) - 86400000, 'endTime': int(time.time()*1000)}),
        
        # Order endpoint
        ('POST', '/api/v1/private/order/create', 
         {'symbol': 'BTC_USDT', 'price': 50000, 'vol': 1, 'leverage': 5, 'side': 1, 'type': 1, 'openType': 1}),
    ]
    
    for method, endpoint, params in tests:
        print(f"{method} {endpoint}")
        status, response = make_request(api_key, api_secret, method, endpoint, params)
        if status:
            print(f"   Status: {status}")
            print(f"   Response: {response}")
        else:
            print(f"   Result: {response}")
        print()

if __name__ == '__main__':
    main()

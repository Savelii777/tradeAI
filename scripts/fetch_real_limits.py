#!/usr/bin/env python3
"""
Fetch real MEXC position limits using authenticated API.
"""

import yaml
import requests
import time
import hmac
import hashlib
from pathlib import Path

# Load API keys
config_dir = Path(__file__).parent.parent / 'config'
with open(config_dir / 'secrets.yaml') as f:
    secrets = yaml.safe_load(f)

api_key = secrets.get('mexc', {}).get('api_key', '')
api_secret = secrets.get('mexc', {}).get('api_secret', '')

print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

base_url = "https://contract.mexc.com"

def sign_request(params, secret):
    query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return signature

def make_auth_request(endpoint, extra_params=None):
    timestamp = str(int(time.time() * 1000))
    params = {'timestamp': timestamp}
    if extra_params:
        params.update(extra_params)
    
    signature = sign_request(params, api_secret)
    
    headers = {
        'ApiKey': api_key,
        'Request-Time': timestamp,
        'Signature': signature,
        'Content-Type': 'application/json'
    }
    
    url = base_url + endpoint
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    return resp

# Our pairs
pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'DOGE_USDT']

print("\n" + "="*80)
print("Trying to get risk limit tiers...")
print("="*80)

# Try different endpoints
endpoints_to_try = [
    '/api/v1/private/account/risk_limit',
    '/api/v1/private/position/leverage',  
    '/api/v1/contract/risk_limit',
    '/api/v1/contract/leverage_bracket',
]

for endpoint in endpoints_to_try:
    try:
        resp = make_auth_request(endpoint)
        print(f"\n{endpoint}: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Success! Response: {str(data)[:800]}")
        else:
            print(f"Response: {resp.text[:200]}")
    except Exception as e:
        print(f"{endpoint}: ERROR - {e}")

# Try with symbol parameter
print("\n" + "="*80)
print("Trying with symbol parameter...")
print("="*80)

for symbol in ['BTC_USDT']:
    endpoints_with_symbol = [
        f'/api/v1/contract/risk_limit/{symbol}',
        f'/api/v1/private/position/risk_limit',
    ]
    
    for endpoint in endpoints_with_symbol:
        try:
            if 'private' in endpoint:
                resp = make_auth_request(endpoint, {'symbol': symbol})
            else:
                resp = requests.get(base_url + endpoint, timeout=10)
            print(f"\n{endpoint}: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Success! Response: {str(data)[:800]}")
            else:
                print(f"Response: {resp.text[:200]}")
        except Exception as e:
            print(f"{endpoint}: ERROR - {e}")

# Try public contract detail with leverages field
print("\n" + "="*80)
print("Getting leverage tiers from contract detail...")
print("="*80)

for symbol in ['BTC_USDT', 'ETH_USDT']:
    try:
        resp = requests.get(f"{base_url}/api/v1/contract/detail/{symbol}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('success') and data.get('data'):
                d = data['data']
                print(f"\n{symbol}:")
                print(f"  maxVol: {d.get('maxVol')}")
                print(f"  maxLeverage: {d.get('maxLeverage')}")
                # Check for tier info
                for key in d.keys():
                    if 'tier' in key.lower() or 'bracket' in key.lower() or 'limit' in key.lower():
                        print(f"  {key}: {d[key]}")
                # Print all keys to see what's available
                print(f"  Available keys: {list(d.keys())}")
    except Exception as e:
        print(f"{symbol}: ERROR - {e}")

#!/usr/bin/env python3
"""
Fetch REAL MEXC position limits with proper authentication.
"""

import yaml
import requests
import time
import hmac
import hashlib
import json
from pathlib import Path

config_dir = Path(__file__).parent.parent / 'config'

with open(config_dir / 'secrets.yaml') as f:
    secrets = yaml.safe_load(f)

api_key = secrets['mexc']['api_key']
api_secret = secrets['mexc']['api_secret']

base_url = 'https://contract.mexc.com'

def make_auth_request(endpoint):
    timestamp = str(int(time.time() * 1000))
    to_sign = api_key + timestamp
    signature = hmac.new(api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        'Content-Type': 'application/json',
        'ApiKey': api_key,
        'Request-Time': timestamp,
        'Signature': signature
    }
    return requests.get(base_url + endpoint, headers=headers, timeout=10)

# Our pairs
pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'DOGE_USDT',
         'BNB_USDT', 'ADA_USDT', 'AVAX_USDT', 'LINK_USDT', 'SUI_USDT',
         'TRX_USDT', 'DOT_USDT', 'LTC_USDT', 'BCH_USDT', 'UNI_USDT',
         'APT_USDT', 'ATOM_USDT', 'NEAR_USDT', 'AAVE_USDT', 'ZEC_USDT']

print("="*80)
print("Searching for risk limit endpoints...")
print("="*80)

# Try different endpoints
test_endpoints = [
    '/api/v1/private/account/risk_limit',
    '/api/v1/private/position/risk_limit', 
    '/api/v1/private/account/leverage_bracket',
    '/api/v1/private/order/open_orders/BTC_USDT',  # known working format
]

for ep in test_endpoints:
    try:
        resp = make_auth_request(ep)
        print(f"\n{ep}:")
        print(f"  Status: {resp.status_code}")
        data = resp.json() if resp.status_code == 200 else {}
        if data.get('success'):
            print(f"  SUCCESS! Data: {str(data.get('data', ''))[:500]}")
        else:
            print(f"  Failed: {data.get('message', resp.text[:100])}")
    except Exception as e:
        print(f"  Error: {e}")

# Get public contract info and look at ALL fields
print("\n" + "="*80)
print("Checking public contract/detail for all fields...")
print("="*80)

resp = requests.get(f"{base_url}/api/v1/contract/detail", timeout=10)
if resp.status_code == 200:
    data = resp.json()
    if data.get('success') and data.get('data'):
        # Find BTC
        for contract in data['data']:
            if contract.get('symbol') == 'BTC_USDT':
                print("\nBTC_USDT all fields:")
                for key, val in contract.items():
                    print(f"  {key}: {val}")
                break

# Check leverage tier info from UI endpoint if available
print("\n" + "="*80)
print("Trying web API endpoints...")
print("="*80)

web_endpoints = [
    'https://futures.mexc.com/api/v1/contract/risk_reverse/BTC_USDT',
    'https://www.mexc.com/api/platform/contract/leverage/BTC_USDT',
]

for ep in web_endpoints:
    try:
        resp = requests.get(ep, timeout=10)
        print(f"\n{ep}:")
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  Data: {resp.text[:500]}")
    except Exception as e:
        print(f"  Error: {e}")

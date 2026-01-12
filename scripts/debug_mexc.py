#!/usr/bin/env python3
"""Debug MEXC API order placement"""
import requests
import hmac
import hashlib
import time
import json
import yaml

# Load secrets
with open('config/secrets.yaml') as f:
    secrets = yaml.safe_load(f)

API_KEY = secrets['mexc']['api_key']
API_SECRET = secrets['mexc']['api_secret']
BASE_URL = 'https://contract.mexc.com'

def sign(secret, msg):
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

# Test order placement
params = {
    'symbol': 'PIPPIN_USDT',
    'price': 0,
    'vol': 10,
    'leverage': 5,
    'side': 1,
    'type': 5,
    'openType': 1
}

timestamp = int(time.time() * 1000)
params_json = json.dumps(params, separators=(',', ':'))
sign_str = f'{API_KEY}{timestamp}{params_json}'
signature = sign(API_SECRET, sign_str)

headers = {
    'ApiKey': API_KEY,
    'Request-Time': str(timestamp),
    'Signature': signature,
    'Content-Type': 'application/json'
}

url = f'{BASE_URL}/api/v1/private/order/create'

print('Request details:')
print(f'  URL: {url}')
print(f'  Method: POST')
print(f'  Timestamp: {timestamp}')
print(f'  Body: {params_json}')
print()

try:
    t0 = time.time()
    r = requests.post(url, json=params, headers=headers, timeout=10)
    elapsed = time.time() - t0
    print(f'Response ({elapsed:.2f}s):')
    print(f'  Status: {r.status_code}')
    print(f'  Body: {r.text}')
except requests.exceptions.Timeout:
    print('TIMEOUT after 10 seconds')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')

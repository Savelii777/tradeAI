#!/usr/bin/env python3
"""
Fetch REAL MEXC position limits from riskLimitCustom field.
"""

import requests
import json
from pathlib import Path

config_dir = Path(__file__).parent.parent / 'config'
base_url = 'https://contract.mexc.com'

# Our pairs
pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'DOGE_USDT',
         'BNB_USDT', 'ADA_USDT', 'AVAX_USDT', 'LINK_USDT', 'SUI_USDT',
         'TRX_USDT', 'DOT_USDT', 'LTC_USDT', 'BCH_USDT', 'UNI_USDT',
         'APT_USDT', 'ATOM_USDT', 'NEAR_USDT', 'AAVE_USDT', 'ZEC_USDT']

print("="*100)
print("MEXC REAL POSITION LIMITS (from riskLimitCustom)")
print("="*100)

# Get all contracts
resp = requests.get(f"{base_url}/api/v1/contract/detail", timeout=10)
contracts = {}
if resp.status_code == 200:
    data = resp.json()
    if data.get('success') and data.get('data'):
        for c in data['data']:
            contracts[c.get('symbol')] = c

# Get current prices
price_resp = requests.get(f"{base_url}/api/v1/contract/ticker", timeout=10)
prices = {}
if price_resp.status_code == 200:
    pdata = price_resp.json()
    if pdata.get('success') and pdata.get('data'):
        for t in pdata['data']:
            prices[t.get('symbol')] = float(t.get('lastPrice', 0))

results = {}

for symbol in pairs:
    if symbol not in contracts:
        print(f"{symbol}: NOT FOUND")
        continue
    
    c = contracts[symbol]
    contract_size = float(c.get('contractSize', 1))
    price = prices.get(symbol, 0)
    risk_tiers = c.get('riskLimitCustom', [])
    
    # If no custom tiers, use default maxVol with max leverage
    if not risk_tiers:
        max_vol = c.get('maxVol', 0)
        max_lev = c.get('maxLeverage', 100)
        risk_tiers = [{'level': 1, 'maxVol': max_vol, 'maxLeverage': max_lev, 'mmr': 0.01, 'imr': 0.02}]
    
    print(f"\n{symbol} (contractSize={contract_size}, price=${price:.2f}):")
    print(f"  {'Level':<6} {'Max Contracts':<15} {'Max USDT':<18} {'Max Lev':<10} {'MMR':<8} {'IMR':<8}")
    print(f"  {'-'*65}")
    
    tier_data = []
    for tier in risk_tiers:
        level = tier.get('level', 0)
        max_vol = tier.get('maxVol', 0)  # in contracts
        max_lev = tier.get('maxLeverage', 0)
        mmr = tier.get('mmr', 0)
        imr = tier.get('imr', 0)
        
        # Convert to USDT
        max_usdt = max_vol * contract_size * price
        
        tier_data.append({
            'level': level,
            'max_contracts': max_vol,
            'max_usdt': max_usdt,
            'max_leverage': max_lev,
            'mmr': mmr,
            'imr': imr
        })
        
        def fmt(val):
            if val >= 1_000_000_000:
                return f"${val/1_000_000_000:.2f}B"
            elif val >= 1_000_000:
                return f"${val/1_000_000:.1f}M"
            elif val >= 1_000:
                return f"${val/1000:.0f}K"
            else:
                return f"${val:.0f}"
        
        print(f"  {level:<6} {max_vol:<15,} {fmt(max_usdt):<18} {max_lev}x{'':<7} {mmr:<8} {imr:<8}")
    
    results[symbol] = {
        'contract_size': contract_size,
        'price': price,
        'tiers': tier_data
    }

# Save results
output_file = config_dir / 'mexc_risk_tiers.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n\nSaved to {output_file}")

# Summary table for quick reference
print("\n" + "="*100)
print("SUMMARY: Max Position at Different Leverage Levels")
print("="*100)
print(f"{'Symbol':<12} {'@ 500x':<15} {'@ 200x':<15} {'@ 100x':<15} {'@ 50x':<15} {'@ 20x':<15} {'@ 10x':<15}")
print("-"*100)

for symbol in pairs:
    if symbol not in results:
        continue
    
    tiers = results[symbol]['tiers']
    
    def get_limit_at_lev(target_lev):
        # Find the tier where max_leverage matches or is closest to target_lev
        # Lower leverage = higher tier = higher limit
        best_tier = None
        for tier in sorted(tiers, key=lambda x: x['max_leverage'], reverse=True):
            if tier['max_leverage'] >= target_lev:
                best_tier = tier
            else:
                break
        if best_tier:
            return best_tier['max_usdt']
        # If target_lev is lower than all tiers, return the highest limit (last tier)
        return max(tiers, key=lambda x: x['max_usdt'])['max_usdt'] if tiers else 0
    
    def fmt(val):
        if val >= 1_000_000_000:
            return f"${val/1_000_000_000:.1f}B"
        elif val >= 1_000_000:
            return f"${val/1_000_000:.0f}M"
        elif val >= 1_000:
            return f"${val/1000:.0f}K"
        else:
            return f"${val:.0f}"
    
    limits = [get_limit_at_lev(lev) for lev in [500, 200, 100, 50, 20, 10]]
    print(f"{symbol:<12} {fmt(limits[0]):<15} {fmt(limits[1]):<15} {fmt(limits[2]):<15} {fmt(limits[3]):<15} {fmt(limits[4]):<15} {fmt(limits[5]):<15}")

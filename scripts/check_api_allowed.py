#!/usr/bin/env python3
"""Check which MEXC contracts have API trading allowed."""

import requests

def main():
    r = requests.get('https://contract.mexc.com/api/v1/contract/detail', timeout=15)
    data = r.json()
    
    if not data.get('success'):
        print("Error:", data)
        return
    
    usdt_m = []
    for c in data['data']:
        if c.get('settleCoin') == 'USDT':
            usdt_m.append({
                'symbol': c['symbol'],
                'apiAllowed': c.get('apiAllowed'),
                'state': c.get('state'),
            })
    
    print(f"Total USDT-M contracts: {len(usdt_m)}")
    print()
    
    api_true = [x for x in usdt_m if x['apiAllowed'] == True]
    api_false = [x for x in usdt_m if x['apiAllowed'] == False]
    
    print(f"apiAllowed=True: {len(api_true)}")
    print(f"apiAllowed=False: {len(api_false)}")
    print()
    
    if api_true:
        print("Contracts with API allowed:")
        for c in api_true[:20]:
            print(f"  {c['symbol']}")
    
    print()
    print("Sample contracts (BTC, ETH, DOGE, PIPPIN):")
    for sym in ['BTC_USDT', 'ETH_USDT', 'DOGE_USDT', 'PIPPIN_USDT']:
        found = [x for x in usdt_m if x['symbol'] == sym]
        if found:
            print(f"  {sym}: apiAllowed={found[0]['apiAllowed']}, state={found[0]['state']}")
        else:
            print(f"  {sym}: NOT FOUND")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fetch MEXC Futures position limits with current prices
"""
import requests
import json
from pathlib import Path

def main():
    # Get ticker prices
    url = 'https://contract.mexc.com/api/v1/contract/ticker'
    resp = requests.get(url, timeout=10)
    data = resp.json()

    if not data.get('success'):
        print("Failed to fetch tickers")
        return
    
    tickers = {t['symbol']: float(t.get('lastPrice', 0)) for t in data['data']}
    
    # Load our limits
    config_dir = Path(__file__).parent.parent / "config"
    with open(config_dir / 'mexc_limits_fetched.json') as f:
        limits = json.load(f)
    
    # Calculate real limits
    print('='*80)
    print('MEXC FUTURES POSITION LIMITS (with prices)')
    print('='*80)
    print(f"{'Symbol':<15} {'Max Vol':<12} {'Contract':<10} {'Price':<12} {'Max Position':<15}")
    print('-'*80)
    
    results = []
    for symbol, info in limits.items():
        if symbol in tickers:
            price = tickers[symbol]
            max_vol = info.get('max_vol', 0)
            contract_size = info.get('contract_size', 1)
            max_leverage = info.get('max_leverage', 0)
            
            # Max position = max_vol * contract_size * price
            max_pos = max_vol * contract_size * price
            
            results.append({
                'symbol': symbol,
                'max_vol': max_vol,
                'contract_size': contract_size,
                'price': price,
                'max_position': max_pos,
                'max_leverage': max_leverage
            })
    
    # Sort by max position
    results.sort(key=lambda x: x['max_position'], reverse=True)
    
    # Print top 40
    for r in results[:40]:
        if r['max_position'] >= 1_000_000:
            pos_str = f"${r['max_position']/1_000_000:.1f}M"
        else:
            pos_str = f"${r['max_position']/1000:.0f}K"
        print(f"{r['symbol']:<15} {r['max_vol']:<12} {r['contract_size']:<10} ${r['price']:<10.2f} {pos_str:<15}")
    
    # Check our pairs
    print(f"\n{'='*80}")
    print("OUR PAIRS vs MEXC LIMITS:")
    print(f"{'='*80}")
    
    pairs_file = config_dir / "pairs_20.json"
    if pairs_file.exists():
        with open(pairs_file) as f:
            our_pairs = json.load(f)['pairs']
        
        for pair in our_pairs:
            symbol = pair['symbol'].replace('/USDT:USDT', '_USDT').replace('/', '')
            our_limit = pair.get('mexc_limit', 0)
            
            # Find in results
            mexc_limit = 0
            for r in results:
                if r['symbol'] == symbol:
                    mexc_limit = r['max_position']
                    break
            
            if mexc_limit > 0:
                if mexc_limit >= our_limit:
                    status = "✅"
                else:
                    status = "⚠️ LOWER!"
                    
                mexc_str = f"${mexc_limit/1_000_000:.1f}M" if mexc_limit >= 1_000_000 else f"${mexc_limit/1000:.0f}K"
                our_str = f"${our_limit/1_000_000:.1f}M" if our_limit >= 1_000_000 else f"${our_limit/1000:.0f}K"
                print(f"{status} {pair['symbol']:<20} MEXC: {mexc_str:<12} Our: {our_str}")
            else:
                print(f"❓ {pair['symbol']:<20} NOT FOUND")
    
    # Save updated limits
    for r in results:
        if r['symbol'] in limits:
            limits[r['symbol']]['max_position_usdt'] = r['max_position']
            limits[r['symbol']]['last_price'] = r['price']
    
    with open(config_dir / 'mexc_limits_fetched.json', 'w') as f:
        json.dump(limits, f, indent=2)
    print(f"\n✅ Updated config/mexc_limits_fetched.json")


if __name__ == "__main__":
    main()

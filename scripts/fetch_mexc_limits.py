#!/usr/bin/env python3
"""
Fetch MEXC Futures position limits for all pairs
"""
import requests
import json
from pathlib import Path

def fetch_mexc_limits():
    """Fetch position limits from MEXC API"""
    
    # MEXC Futures API - contract details
    url = "https://contract.mexc.com/api/v1/contract/detail"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') and data.get('data'):
            contracts = data['data']
            
            limits = {}
            for contract in contracts:
                symbol = contract.get('symbol', '')
                
                # Get max position value (in quote currency)
                max_vol = contract.get('maxVol', 0)  # Max volume in contracts
                contract_size = contract.get('contractSize', 1)
                last_price = contract.get('lastPrice', 0)
                
                # Calculate max position in USDT
                if last_price and contract_size:
                    max_position_usdt = float(max_vol) * float(contract_size) * float(last_price)
                else:
                    max_position_usdt = 0
                
                # Also get leverage info
                max_leverage = contract.get('maxLeverage', 0)
                
                limits[symbol] = {
                    'max_position_usdt': max_position_usdt,
                    'max_vol': max_vol,
                    'contract_size': contract_size,
                    'max_leverage': max_leverage,
                    'last_price': last_price
                }
            
            return limits
    except Exception as e:
        print(f"Error fetching from contract API: {e}")
    
    return None


def fetch_risk_limits():
    """Fetch risk limits specifically"""
    
    # Try risk limit endpoint
    url = "https://contract.mexc.com/api/v1/contract/risk_limit"
    
    all_limits = {}
    
    # First get all symbols
    detail_url = "https://contract.mexc.com/api/v1/contract/detail"
    try:
        response = requests.get(detail_url, timeout=10)
        data = response.json()
        
        if data.get('success') and data.get('data'):
            symbols = [c.get('symbol') for c in data['data'] if 'USDT' in c.get('symbol', '')]
            
            print(f"Found {len(symbols)} USDT pairs")
            
            # Get risk limits for each (or batch if possible)
            for symbol in symbols[:50]:  # First 50 for test
                try:
                    resp = requests.get(f"{url}?symbol={symbol}", timeout=5)
                    rdata = resp.json()
                    
                    if rdata.get('success') and rdata.get('data'):
                        levels = rdata['data']
                        # Get max position from highest level
                        if levels:
                            max_level = levels[-1]  # Last level usually has highest position
                            all_limits[symbol] = {
                                'max_position': max_level.get('maxVol', 0),
                                'levels': len(levels)
                            }
                except:
                    pass
                    
    except Exception as e:
        print(f"Error: {e}")
    
    return all_limits


def main():
    print("Fetching MEXC Futures contract details...")
    
    limits = fetch_mexc_limits()
    
    if limits:
        # Sort by max position
        sorted_limits = sorted(limits.items(), 
                              key=lambda x: x[1].get('max_position_usdt', 0), 
                              reverse=True)
        
        print(f"\n{'='*80}")
        print(f"MEXC FUTURES POSITION LIMITS")
        print(f"{'='*80}")
        print(f"{'Symbol':<20} {'Max Position (USDT)':<25} {'Max Leverage':<15} {'Contract Size'}")
        print(f"{'-'*80}")
        
        for symbol, info in sorted_limits[:50]:  # Top 50
            max_pos = info.get('max_position_usdt', 0)
            max_lev = info.get('max_leverage', 0)
            cs = info.get('contract_size', 0)
            
            if max_pos > 0:
                if max_pos >= 1_000_000:
                    pos_str = f"${max_pos/1_000_000:.1f}M"
                elif max_pos >= 1000:
                    pos_str = f"${max_pos/1000:.0f}K"
                else:
                    pos_str = f"${max_pos:.0f}"
                    
                print(f"{symbol:<20} {pos_str:<25} {max_lev}x{'':<10} {cs}")
        
        # Save to file
        output = Path(__file__).parent.parent / "config" / "mexc_limits_fetched.json"
        with open(output, 'w') as f:
            json.dump(limits, f, indent=2)
        print(f"\n✅ Saved all {len(limits)} pairs to {output}")
        
        # Also check our pairs
        print(f"\n{'='*80}")
        print("CHECKING OUR PAIRS:")
        print(f"{'='*80}")
        
        pairs_file = Path(__file__).parent.parent / "config" / "pairs_20.json"
        if pairs_file.exists():
            with open(pairs_file) as f:
                our_pairs = json.load(f)['pairs']
            
            for pair in our_pairs:
                symbol = pair['symbol'].replace('/USDT:USDT', '_USDT').replace('/', '')
                
                if symbol in limits:
                    info = limits[symbol]
                    max_pos = info.get('max_position_usdt', 0)
                    our_limit = pair.get('mexc_limit', 'N/A')
                    
                    status = "✅" if max_pos >= 1_000_000 else "⚠️"
                    print(f"{status} {pair['symbol']:<20} MEXC: ${max_pos:,.0f} | Our config: {our_limit}")
                else:
                    print(f"❓ {pair['symbol']:<20} NOT FOUND in MEXC")
    else:
        print("Failed to fetch limits")
        
        # Try alternative
        print("\nTrying risk limits endpoint...")
        risk_limits = fetch_risk_limits()
        if risk_limits:
            for sym, info in list(risk_limits.items())[:20]:
                print(f"{sym}: {info}")


if __name__ == "__main__":
    main()

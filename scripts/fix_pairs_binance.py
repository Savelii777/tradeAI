
import ccxt
import json
import sys

def main():
    # 1. Load the original pairs
    with open('config/pairs_list.json', 'r') as f:
        data = json.load(f)
    
    # Take top 20
    top_20 = data['pairs'][:20]
    print(f"Loaded {len(top_20)} pairs from config/pairs_list.json")

    # 2. Connect to Binance Futures
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'}
    })
    markets = exchange.load_markets()
    print(f"Loaded {len(markets)} markets from Binance Futures")

    # 3. Map symbols
    # Binance Futures symbols are usually like 'BTC/USDT:USDT' or 'BTC/USDT'
    # But some base currencies are different (e.g. 1000PEPE instead of PEPE)
    
    valid_pairs = []
    
    for p in top_20:
        original_symbol = p['symbol'] # e.g. "PEPE/USDT:USDT"
        base = p['base'] # "PEPE"
        quote = p['quote'] # "USDT"
        
        # Try to find a matching market
        # 1. Direct match
        if original_symbol in markets:
            valid_pairs.append(p)
            print(f"✓ {original_symbol} found")
            continue
            
        # 2. Try standard format BASE/QUOTE:QUOTE
        standard = f"{base}/{quote}:{quote}"
        if standard in markets:
            p['symbol'] = standard
            valid_pairs.append(p)
            print(f"✓ {original_symbol} -> {standard}")
            continue

        # 3. Try 1000 prefix (common for memes)
        meme_base = f"1000{base}"
        meme_symbol = f"{meme_base}/{quote}:{quote}"
        if meme_symbol in markets:
            p['symbol'] = meme_symbol
            p['base'] = meme_base
            valid_pairs.append(p)
            print(f"✓ {original_symbol} -> {meme_symbol}")
            continue
            
        # 4. Try special mappings
        mappings = {
            'TONCOIN': 'TON',
            'ASTER': 'ASTR',
            'LUNA': 'LUNA2'
        }
        
        if base in mappings:
            new_base = mappings[base]
            new_symbol = f"{new_base}/{quote}:{quote}"
            if new_symbol in markets:
                p['symbol'] = new_symbol
                p['base'] = new_base
                valid_pairs.append(p)
                print(f"✓ {original_symbol} -> {new_symbol}")
                continue
        
        print(f"✗ {original_symbol} NOT FOUND on Binance Futures")

    # 4. Save to new file
    output = {'count': len(valid_pairs), 'pairs': valid_pairs}
    with open('config/pairs_20.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(valid_pairs)} valid pairs to config/pairs_20.json")

if __name__ == "__main__":
    main()

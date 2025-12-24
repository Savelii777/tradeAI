#!/usr/bin/env python3
"""
Fetch trading pairs from MEXC exchange.
Filters by volume, excludes stablecoins, saves to config/pairs_list.json
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import ccxt.async_support as ccxt
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Stablecoins to exclude
STABLECOINS = {
    'USDC', 'TUSD', 'DAI', 'BUSD', 'USDP', 'GUSD', 'FRAX', 'LUSD',
    'USDD', 'CUSD', 'SUSD', 'USTC', 'EURC', 'PYUSD', 'FDUSD'
}


async def fetch_pairs(
    min_volume_usd: float = 10_000_000,
    max_pairs: int = 100,
    quote_asset: str = 'USDT',
    market_type: str = 'swap'  # futures
) -> List[Dict[str, Any]]:
    """
    Fetch and filter trading pairs from MEXC.
    
    Args:
        min_volume_usd: Minimum 24h volume in USD
        max_pairs: Maximum number of pairs to return
        quote_asset: Quote asset (USDT)
        market_type: Market type (swap for futures)
    
    Returns:
        List of pair info dicts
    """
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {
            'defaultType': market_type,
        }
    })
    
    try:
        logger.info(f"Connecting to MEXC {market_type} market...")
        
        # Load markets
        await exchange.load_markets()
        logger.info(f"Loaded {len(exchange.markets)} markets")
        
        # Fetch tickers for volume data
        logger.info("Fetching tickers for volume data...")
        tickers = await exchange.fetch_tickers()
        
        # Filter pairs
        filtered_pairs = []
        stats = {
            'total': 0,
            'wrong_quote': 0,
            'inactive': 0,
            'stablecoin': 0,
            'low_volume': 0,
            'no_ticker': 0,
            'passed': 0
        }
        
        for symbol, market in exchange.markets.items():
            stats['total'] += 1
            
            # Check quote asset
            if market.get('quote') != quote_asset:
                stats['wrong_quote'] += 1
                continue
            
            # Check if active
            if not market.get('active', True):
                stats['inactive'] += 1
                continue
            
            # Check if swap/futures
            if market.get('type') != market_type:
                continue
            
            # Get base asset
            base = market.get('base', '')
            
            # Exclude stablecoins
            if base in STABLECOINS:
                stats['stablecoin'] += 1
                continue
            
            # Get ticker for volume
            ticker = tickers.get(symbol)
            if not ticker:
                stats['no_ticker'] += 1
                continue
            
            # Calculate 24h volume in USD
            quote_volume = ticker.get('quoteVolume', 0) or 0
            
            # Filter by volume
            if quote_volume < min_volume_usd:
                stats['low_volume'] += 1
                continue
            
            stats['passed'] += 1
            
            filtered_pairs.append({
                'symbol': symbol,
                'base': base,
                'quote': market.get('quote'),
                'volume_24h_usd': quote_volume,
                'last_price': ticker.get('last', 0),
                'change_24h': ticker.get('percentage', 0),
                'type': market_type
            })
        
        # Sort by volume descending
        filtered_pairs.sort(key=lambda x: x['volume_24h_usd'], reverse=True)
        
        # Limit to max_pairs
        filtered_pairs = filtered_pairs[:max_pairs]
        
        # Log statistics
        logger.info("=" * 60)
        logger.info("PAIR FILTERING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total markets scanned: {stats['total']}")
        logger.info(f"Filtered - wrong quote asset: {stats['wrong_quote']}")
        logger.info(f"Filtered - inactive: {stats['inactive']}")
        logger.info(f"Filtered - stablecoin: {stats['stablecoin']}")
        logger.info(f"Filtered - low volume (<${min_volume_usd:,.0f}): {stats['low_volume']}")
        logger.info(f"Filtered - no ticker data: {stats['no_ticker']}")
        logger.info(f"Pairs passed all filters: {stats['passed']}")
        logger.info(f"Selected top {len(filtered_pairs)} pairs by volume")
        logger.info("=" * 60)
        
        return filtered_pairs
        
    finally:
        await exchange.close()


def save_pairs(pairs: List[Dict[str, Any]], output_path: str = 'config/pairs_list.json'):
    """Save pairs list to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'count': len(pairs),
        'pairs': pairs,
        'symbols': [p['symbol'] for p in pairs]
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved {len(pairs)} pairs to {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch trading pairs from MEXC')
    parser.add_argument('--min-volume', type=float, default=10_000_000,
                        help='Minimum 24h volume in USD (default: 10M)')
    parser.add_argument('--max-pairs', type=int, default=100,
                        help='Maximum number of pairs (default: 100)')
    parser.add_argument('--quote', type=str, default='USDT',
                        help='Quote asset (default: USDT)')
    parser.add_argument('--output', type=str, default='config/pairs_list.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    logger.info(f"Fetching pairs with min volume ${args.min_volume:,.0f}")
    
    pairs = asyncio.run(fetch_pairs(
        min_volume_usd=args.min_volume,
        max_pairs=args.max_pairs,
        quote_asset=args.quote
    ))
    
    if pairs:
        save_pairs(pairs, args.output)
        
        # Print top 10
        print("\nTop 10 pairs by volume:")
        print("-" * 60)
        for i, p in enumerate(pairs[:10], 1):
            vol_m = p['volume_24h_usd'] / 1_000_000
            print(f"{i:2}. {p['symbol']:<20} Volume: ${vol_m:>10.1f}M")
    else:
        logger.error("No pairs found matching criteria")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

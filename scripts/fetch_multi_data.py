#!/usr/bin/env python3
"""
Fetch historical data for multiple trading pairs.
Downloads OHLCV data and saves to data/candles/ directory.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional
import time

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MultiPairDataFetcher:
    """Fetches data for multiple pairs with rate limiting."""
    
    def __init__(
        self,
        exchange_id: str = 'mexc',
        max_concurrent: int = 5,
        data_dir: str = 'data/candles'
    ):
        self.exchange_id = exchange_id
        self.max_concurrent = max_concurrent
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.exchange = None
        
        self.stats = {
            'success': 0,
            'failed': 0,
            'total_candles': 0,
            'errors': []
        }
    
    async def init_exchange(self):
        """Initialize exchange connection."""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        await self.exchange.load_markets()
    
    async def close_exchange(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        days: int,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single symbol with retry logic.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (1m, 5m, 1h, etc)
            days: Number of days to fetch
            max_retries: Maximum number of retries per request
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        async with self.semaphore:
            try:
                # Calculate timestamps
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=days)
                since = int(start_time.timestamp() * 1000)
                
                # Timeframe to milliseconds
                tf_ms = {
                    '1m': 60000,
                    '5m': 300000,
                    '15m': 900000,
                    '1h': 3600000,
                    '4h': 14400000,
                    '1d': 86400000
                }
                
                limit = 1000  # Limit per request
                all_ohlcv = []
                current_since = since
                req_count = 0
                consecutive_errors = 0
                
                logger.info(f"  → {symbol} {timeframe}: Starting download...")

                while True:
                    # Retry logic for each request
                    ohlcv = None
                    for retry in range(max_retries):
                        try:
                            ohlcv = await self.exchange.fetch_ohlcv(
                                symbol,
                                timeframe,
                                since=current_since,
                                limit=limit
                            )
                            consecutive_errors = 0  # Reset on success
                            break  # Success - exit retry loop
                        except Exception as e:
                            consecutive_errors += 1
                            if retry < max_retries - 1:
                                wait_time = (retry + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                                logger.warning(f"  ⚠️ {symbol} {timeframe}: Retry {retry + 1}/{max_retries} after error: {str(e)[:50]}...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"  ✗ {symbol} {timeframe}: Failed after {max_retries} retries: {str(e)[:100]}")
                                # If too many consecutive errors, stop trying
                                if consecutive_errors >= 5:
                                    logger.error(f"  ✗ {symbol} {timeframe}: Too many consecutive errors, stopping")
                                    break
                    
                    if ohlcv is None:
                        # Failed after all retries - continue with what we have
                        if all_ohlcv:
                            logger.warning(f"  ⚠️ {symbol} {timeframe}: Continuing with {len(all_ohlcv)} candles fetched so far")
                        break
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    req_count += 1
                    
                    # Log progress every 10 requests (approx every 10k candles)
                    if req_count % 10 == 0:
                        logger.info(f"  → {symbol} {timeframe}: Fetched {len(all_ohlcv)} candles...")

                    # Check if we got all data
                    last_ts = ohlcv[-1][0]
                    if last_ts >= end_time.timestamp() * 1000:
                        break
                    
                    if len(ohlcv) < limit:
                        break
                    
                    current_since = last_ts + tf_ms.get(timeframe, 300000)
                    
                    # Small delay between requests to avoid rate limiting
                    await asyncio.sleep(0.05)
                
                if not all_ohlcv:
                    return None
                
                logger.info(f"  ✓ {symbol} {timeframe}: Done ({len(all_ohlcv)} candles)")

                # Convert to DataFrame
                df = pd.DataFrame(all_ohlcv, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe}: {e}")
                return None
    
    async def fetch_pair(
        self,
        symbol: str,
        days: int,
        timeframes: List[str]
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch all timeframes for a single pair.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days
            timeframes: List of timeframes to fetch
        
        Returns:
            Dict mapping timeframe to DataFrame
        """
        results = {}
        
        for tf in timeframes:
            df = await self.fetch_ohlcv(symbol, tf, days)
            results[tf] = df
            
            if df is not None:
                # Save to file
                safe_symbol = symbol.replace('/', '_').replace(':', '_')
                filename = self.data_dir / f"{safe_symbol}_{tf}.csv"
                df.to_csv(filename)
        
        return results
    
    async def fetch_all_pairs(
        self,
        symbols: List[str],
        days: int = 30,
        timeframes: List[str] = ['5m', '1h', '4h']
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for all pairs in parallel.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days to fetch
            timeframes: List of timeframes
        
        Returns:
            Dict mapping symbol to {timeframe: DataFrame}
        """
        await self.init_exchange()
        
        try:
            all_data = {}
            total = len(symbols)
            tasks = []

            # Create a wrapper to handle the result and symbol association
            async def fetch_wrapper(sym):
                try:
                    return sym, await self.fetch_pair(sym, days, timeframes)
                except Exception as e:
                    logger.error(f"Error in fetch_wrapper for {sym}: {e}")
                    return sym, None

            # Create tasks for all pairs
            for symbol in symbols:
                tasks.append(fetch_wrapper(symbol))
            
            logger.info(f"Starting parallel download for {total} pairs (max_concurrent={self.semaphore._value})")
            
            # Process as they complete
            completed = 0
            for task in asyncio.as_completed(tasks):
                symbol, data = await task
                completed += 1
                
                if data:
                    # Check results
                    main_tf = timeframes[0]
                    if data.get(main_tf) is not None:
                        all_data[symbol] = data
                        candles = len(data[main_tf])
                        self.stats['success'] += 1
                        self.stats['total_candles'] += candles
                        logger.info(f"[{completed}/{total}]  ✓ {symbol}: {candles} candles")
                    else:
                        self.stats['failed'] += 1
                        self.stats['errors'].append(f"{symbol}: No data")
                        logger.warning(f"[{completed}/{total}]  ✗ {symbol}: No data received")
                else:
                    self.stats['failed'] += 1
                    self.stats['errors'].append(f"{symbol}: Failed to fetch")
                    logger.error(f"[{completed}/{total}]  ✗ {symbol}: Failed")

            return all_data
            
        finally:
            await self.close_exchange()
    
    def print_stats(self):
        """Print download statistics."""
        print("\n" + "=" * 60)
        print("DOWNLOAD STATISTICS")
        print("=" * 60)
        print(f"Pairs successful: {self.stats['success']}")
        print(f"Pairs failed: {self.stats['failed']}")
        print(f"Total candles downloaded: {self.stats['total_candles']:,}")
        print(f"Data directory: {self.data_dir}")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for err in self.stats['errors'][:10]:
                print(f"  - {err}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more")
        
        print("=" * 60)


def load_pairs_list(path: str = 'config/pairs_list.json') -> List[str]:
    """Load pairs list from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    # Handle new format (list of objects)
    if 'pairs' in data:
        return [p['symbol'] for p in data['pairs']]
    
    # Handle old format (list of strings)
    return data.get('symbols', [])


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch data for multiple pairs')
    parser.add_argument('--pairs-file', type=str, default='config/pairs_list.json',
                        help='Path to pairs list JSON')
    parser.add_argument('--pairs', type=str, nargs='+', default=None,
                        help='Specific pairs to fetch (e.g., DOT/USDT:USDT POL/USDT:USDT)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to fetch (default: 30)')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['5m', '1h', '4h'],
                        help='Timeframes to fetch (default: 5m 1h 4h)')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Max concurrent requests (default: 5)')
    parser.add_argument('--output-dir', type=str, default='data/candles',
                        help='Output directory for CSV files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of pairs to fetch')
    
    args = parser.parse_args()
    
    # Load pairs - use --pairs if specified, otherwise load from file
    if args.pairs:
        symbols = args.pairs
        logger.info(f"Using {len(symbols)} pairs from command line: {symbols}")
    else:
        try:
            symbols = load_pairs_list(args.pairs_file)
            if args.limit:
                symbols = symbols[:args.limit]
            logger.info(f"Loaded {len(symbols)} pairs from {args.pairs_file}")
        except FileNotFoundError:
            logger.error(f"Pairs file not found: {args.pairs_file}")
            logger.info("Run 'python scripts/fetch_pairs.py' first")
            return 1
    
    if not symbols:
        logger.error("No pairs found")
        return 1
    
    # Create fetcher
    fetcher = MultiPairDataFetcher(
        max_concurrent=args.max_concurrent,
        data_dir=args.output_dir
    )
    
    # Fetch data
    logger.info(f"Fetching {len(symbols)} pairs, {args.days} days, timeframes: {args.timeframes}")
    start_time = time.time()
    
    asyncio.run(fetcher.fetch_all_pairs(
        symbols=symbols,
        days=args.days,
        timeframes=args.timeframes
    ))
    
    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f} seconds")
    
    # Print stats
    fetcher.print_stats()
    
    return 0 if fetcher.stats['success'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

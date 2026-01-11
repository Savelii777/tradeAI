#!/usr/bin/env python3
"""
Fetch historical data and save to Parquet.
Designed for walk-forward validation (30+ days of data).
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger


class ParquetDataFetcher:
    """Fetches data for multiple pairs and saves to Parquet."""
    
    def __init__(self, data_dir: str = 'data/candles', max_concurrent: int = 3):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.exchange = None
    
    async def init_exchange(self):
        """Initialize exchange (Binance futures)."""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        await self.exchange.load_markets()
    
    async def close_exchange(self):
        if self.exchange:
            await self.exchange.close()
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with pagination."""
        async with self.semaphore:
            try:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=days)
                since = int(start_time.timestamp() * 1000)
                
                tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}
                all_ohlcv = []
                current_since = since
                
                logger.info(f"  → {symbol} {timeframe}: Fetching...")
                
                while True:
                    try:
                        ohlcv = await self.exchange.fetch_ohlcv(
                            symbol, timeframe, since=current_since, limit=1000
                        )
                    except Exception as e:
                        logger.warning(f"  ⚠️ {symbol} {timeframe}: {e}")
                        await asyncio.sleep(2)
                        try:
                            ohlcv = await self.exchange.fetch_ohlcv(
                                symbol, timeframe, since=current_since, limit=1000
                            )
                        except:
                            break
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    last_ts = ohlcv[-1][0]
                    
                    if last_ts >= end_time.timestamp() * 1000 or len(ohlcv) < 1000:
                        break
                    
                    current_since = last_ts + tf_ms.get(timeframe, 300000)
                    await asyncio.sleep(0.05)
                
                if not all_ohlcv:
                    return None
                
                logger.info(f"  ✓ {symbol} {timeframe}: {len(all_ohlcv)} candles")
                
                df = pd.DataFrame(all_ohlcv, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error {symbol} {timeframe}: {e}")
                return None
    
    async def fetch_and_save_pair(self, symbol: str, days: int, timeframes: List[str]):
        """Fetch all timeframes and save to Parquet."""
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        
        for tf in timeframes:
            df = await self.fetch_ohlcv(symbol, tf, days)
            if df is not None:
                # Save to Parquet
                parquet_path = self.data_dir / f"{safe_symbol}_{tf}.parquet"
                df.to_parquet(parquet_path)
                
                # Also save to CSV for backup
                csv_path = self.data_dir / f"{safe_symbol}_{tf}.csv"
                df.to_csv(csv_path)
                
                logger.info(f"  💾 Saved {parquet_path.name}")
    
    async def fetch_all(self, symbols: List[str], days: int, timeframes: List[str]):
        """Fetch all symbols."""
        await self.init_exchange()
        try:
            for i, symbol in enumerate(symbols, 1):
                logger.info(f"[{i}/{len(symbols)}] {symbol}")
                await self.fetch_and_save_pair(symbol, days, timeframes)
        finally:
            await self.close_exchange()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=35, help='Days to fetch (default: 35)')
    parser.add_argument('--pairs', type=str, default='config/pairs_list.json')
    args = parser.parse_args()
    
    # Load pairs
    with open(args.pairs) as f:
        data = json.load(f)
    symbols = [p['symbol'] for p in data.get('pairs', [])]
    
    if not symbols:
        logger.error("No pairs found!")
        return 1
    
    logger.info(f"Fetching {len(symbols)} pairs, {args.days} days, timeframes: 1m, 5m, 15m")
    
    fetcher = ParquetDataFetcher()
    asyncio.run(fetcher.fetch_all(symbols, args.days, ['1m', '5m', '15m']))
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

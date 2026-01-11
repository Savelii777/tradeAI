#!/usr/bin/env python3
"""Quick data integrity check after live run."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'
pairs = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'LINK', 'ADA', 'AVAX', 'LTC', 'NEAR']

print('='*70)
print('ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ ПОСЛЕ LIVE ЗАПУСКА')
print('='*70)

all_ok = True
for pair in pairs:
    pair_name = f'{pair}_USDT_USDT'
    parquet_path = DATA_DIR / f'{pair_name}_5m.parquet'
    
    if not parquet_path.exists():
        print(f'❌ {pair} 5m: FILE NOT FOUND')
        all_ok = False
        continue
    
    df = pd.read_parquet(parquet_path)
    dups = df.index.duplicated().sum()
    mono = df.index.is_monotonic_increasing
    start = df.index[0]
    end = df.index[-1]
    days = (end - start).days
    
    # Check start date is preserved (2025-08-14)
    start_ok = start.date().isoformat() == '2025-08-14'
    ok = dups == 0 and mono and start_ok
    if not ok: 
        all_ok = False
    
    status = '✅' if ok else '❌'
    print(f'{status} {pair}: {len(df):,} rows | {start.date()} → {end.date()} ({days}d) | dups={dups} mono={mono}')

print()
print('='*70)
if all_ok:
    print('ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅ Данные не повреждены!')
else:
    print('ЕСТЬ ПРОБЛЕМЫ ❌')
print('='*70)

#!/usr/bin/env python3
"""Check data for walk-forward periods."""
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

data_dir = Path('data/candles')

m5 = pd.read_parquet(data_dir / 'BTC_USDT_USDT_5m.parquet')
print(f'BTC M5: {m5.index.min()} to {m5.index.max()}')
print(f'Total candles: {len(m5)}')

periods = [
    ('Period_1', datetime(2025, 12, 8, tzinfo=timezone.utc), datetime(2025, 12, 18, tzinfo=timezone.utc),
                 datetime(2025, 12, 19, tzinfo=timezone.utc), datetime(2025, 12, 24, tzinfo=timezone.utc)),
    ('Period_2', datetime(2025, 12, 12, tzinfo=timezone.utc), datetime(2025, 12, 22, tzinfo=timezone.utc),
                 datetime(2025, 12, 23, tzinfo=timezone.utc), datetime(2025, 12, 28, tzinfo=timezone.utc)),
    ('Period_3', datetime(2025, 12, 16, tzinfo=timezone.utc), datetime(2025, 12, 26, tzinfo=timezone.utc),
                 datetime(2025, 12, 27, tzinfo=timezone.utc), datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)),
]

print("\nWalk-forward periods check:")
for name, tr_start, tr_end, te_start, te_end in periods:
    train = m5[(m5.index >= tr_start) & (m5.index < tr_end)]
    test = m5[(m5.index >= te_start) & (m5.index <= te_end)]
    print(f'{name}: train={len(train)}, test={len(test)}')

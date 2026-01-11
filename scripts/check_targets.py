#!/usr/bin/env python3
"""Check target distribution"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from train_v3_dynamic import create_targets_v1

# Load BTC
pair = 'BTC'
m5 = pd.read_parquet(f'data/candles/{pair}_USDT_USDT_5m.parquet')  

# Get training period
m5_train = m5[(m5.index >= '2025-09-14') & (m5.index <= '2025-11-12')]
print(f'Training data: {len(m5_train)} 5m candles')

# Generate targets
df = create_targets_v1(m5_train.copy())

print()
print('Target distribution (BTC):')
print(df['target_dir'].value_counts())

total = len(df['target_dir'].dropna())
print()
for v in [0, 1, 2]:
    pct = 100 * (df['target_dir'] == v).sum() / total
    label = {0: 'SHORT', 1: 'SIDEWAYS', 2: 'LONG'}[v]
    print(f'  {label}: {pct:.1f}%')

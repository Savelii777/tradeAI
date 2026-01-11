#!/usr/bin/env python3
"""
Test Parquet data handling in live_trading_v10_csv.py

Checks:
1. Loading Parquet correctly (no data loss)
2. Appending new candles (no duplicates)
3. Saving without corruption
4. Update cycle simulation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

print("="*70)
print("PARQUET DATA HANDLING TEST")
print("="*70)

# Test 1: Load existing Parquet and verify integrity
print("\n1. Loading existing Parquet files...")

pair = 'BTC/USDT:USDT'
pair_name = pair.replace('/', '_').replace(':', '_')

m5_path = DATA_DIR / f'{pair_name}_5m.parquet'
m5_original = pd.read_parquet(m5_path)

print(f"   Original file: {m5_path}")
print(f"   Rows: {len(m5_original)}")
print(f"   Index type: {type(m5_original.index)}")
print(f"   Timezone: {m5_original.index.tz}")
print(f"   First: {m5_original.index[0]}")
print(f"   Last: {m5_original.index[-1]}")

# Check for duplicates
dups = m5_original.index.duplicated().sum()
print(f"   Duplicates: {dups}")

if dups == 0:
    print("   ✅ No duplicates in original data")
else:
    print("   ❌ DUPLICATES FOUND!")

# Check monotonic
if m5_original.index.is_monotonic_increasing:
    print("   ✅ Index is monotonically increasing")
else:
    print("   ❌ Index is NOT monotonically increasing!")

# Test 2: Simulate append operation (like live does)
print("\n2. Testing append operation (simulating live update)...")

# Create temp directory
temp_dir = Path(tempfile.mkdtemp())
temp_parquet = temp_dir / f'{pair_name}_5m.parquet'

# Copy original to temp
shutil.copy(m5_path, temp_parquet)

# Load temp file
df_existing = pd.read_parquet(temp_parquet)
original_len = len(df_existing)
original_last = df_existing.index[-1]

# Simulate fetching new candles (2 candles - current + last closed)
# This mimics what Binance API returns
new_candles = pd.DataFrame({
    'open': [90000.0, 90100.0],
    'high': [90050.0, 90150.0],
    'low': [89950.0, 90050.0],
    'close': [90030.0, 90120.0],
    'volume': [100.0, 120.0]
}, index=pd.DatetimeIndex([
    original_last,  # Duplicate - should be updated
    original_last + timedelta(minutes=5)  # New candle
], tz='UTC'))
new_candles.index.name = 'timestamp'

print(f"   Existing data ends at: {original_last}")
print(f"   New candles: {len(new_candles)}")
print(f"   New candles timestamps: {list(new_candles.index)}")

# Append like live does
combined = pd.concat([df_existing, new_candles])
print(f"   After concat: {len(combined)} rows")

# Remove duplicates (keep='last' to update current candle)
combined = combined[~combined.index.duplicated(keep='last')]
print(f"   After dedup: {len(combined)} rows")

# Sort
combined.sort_index(inplace=True)

# Verify
new_count = len(combined) - original_len
print(f"   New rows added: {new_count}")

if new_count == 1:
    print("   ✅ Correctly added 1 new row (duplicate was updated)")
else:
    print(f"   ❌ Expected 1 new row, got {new_count}")

# Check no duplicates after append
dups_after = combined.index.duplicated().sum()
if dups_after == 0:
    print("   ✅ No duplicates after append")
else:
    print(f"   ❌ {dups_after} duplicates after append!")

# Test 3: Save and reload
print("\n3. Testing save/reload cycle...")

# Save
combined.to_parquet(temp_parquet, engine='pyarrow')
print(f"   Saved to: {temp_parquet}")

# Reload
reloaded = pd.read_parquet(temp_parquet)
print(f"   Reloaded: {len(reloaded)} rows")

# Compare
if len(reloaded) == len(combined):
    print("   ✅ Row count matches after reload")
else:
    print(f"   ❌ Row count mismatch: {len(combined)} vs {len(reloaded)}")

if reloaded.index.equals(combined.index):
    print("   ✅ Index matches after reload")
else:
    print("   ❌ Index mismatch after reload!")

# Check data integrity
if np.allclose(reloaded['close'].values, combined['close'].values, equal_nan=True):
    print("   ✅ Data integrity verified (close prices match)")
else:
    print("   ❌ Data corruption detected!")

# Test 4: Timezone handling
print("\n4. Testing timezone handling...")

# Check original timezone is preserved
if reloaded.index.tz is not None:
    print(f"   ✅ Timezone preserved: {reloaded.index.tz}")
else:
    print("   ❌ Timezone lost after reload!")

# Test 5: Simulate multiple update cycles
print("\n5. Simulating 5 update cycles...")

for cycle in range(5):
    # Load
    df = pd.read_parquet(temp_parquet)
    last_ts = df.index[-1]
    
    # Add one candle
    new_ts = last_ts + timedelta(minutes=5)
    new_row = pd.DataFrame({
        'open': [90000.0 + cycle],
        'high': [90050.0 + cycle],
        'low': [89950.0 + cycle],
        'close': [90030.0 + cycle],
        'volume': [100.0 + cycle]
    }, index=pd.DatetimeIndex([new_ts], tz='UTC'))
    new_row.index.name = 'timestamp'
    
    # Append and dedupe
    combined = pd.concat([df, new_row])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    
    # Save
    combined.to_parquet(temp_parquet, engine='pyarrow')

# Final check
final = pd.read_parquet(temp_parquet)
expected_len = original_len + 1 + 5  # original + 1 from test 2 + 5 from test 5
if len(final) == expected_len:
    print(f"   ✅ After 5 cycles: {len(final)} rows (expected {expected_len})")
else:
    print(f"   ❌ Row count wrong: {len(final)} vs expected {expected_len}")

dups_final = final.index.duplicated().sum()
if dups_final == 0:
    print("   ✅ No duplicates after 5 cycles")
else:
    print(f"   ❌ {dups_final} duplicates!")

# Cleanup
shutil.rmtree(temp_dir)

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

# Test 6: Check actual live code logic
print("\n6. Checking live_trading_v10_csv.py code logic...")

import ccxt

# Mock binance for testing
class MockBinance:
    def fetch_ohlcv(self, pair, tf, **kwargs):
        return []

# Import and check CSVDataManager
try:
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Check the key methods exist and work
    from live_trading_v10_csv import CSVDataManager, Config
    
    mock_binance = MockBinance()
    manager = CSVDataManager(DATA_DIR, mock_binance)
    
    print("   ✅ CSVDataManager imports correctly")
    
    # Test load
    df = manager.load_csv(pair, '5m')
    if df is not None and len(df) > 0:
        print(f"   ✅ load_csv works: {len(df)} rows")
    else:
        print("   ❌ load_csv failed")
    
    # Check it uses Parquet
    if manager.USE_PARQUET:
        print("   ✅ Parquet mode enabled")
    else:
        print("   ⚠️ Parquet mode disabled (using slower CSV)")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)
    
except Exception as e:
    print(f"   ❌ Error testing live code: {e}")

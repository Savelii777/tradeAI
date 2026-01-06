#!/usr/bin/env python3
"""
Debug: Compare Binance API data vs CSV data for the SAME timestamps.
Check for timezone issues, data format differences, and prediction discrepancies.
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import joblib
import ccxt
from datetime import datetime, timezone, timedelta
from train_mtf import MTFFeatureEngine

def add_volume_features(df):
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    df["vol_zscore"] = (df["volume"] - df["vol_sma_20"]) / df["volume"].rolling(20).std()
    df["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["price_vs_vwap"] = df["close"] / df["vwap"] - 1
    df["vol_momentum"] = df["volume"].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

print("="*80)
print("DEBUG: Binance API vs CSV Data Comparison")
print("="*80)

# 1. Check system time vs exchange time
print("\n1. TIME CHECK:")
print("-"*40)
now_local = datetime.now()
now_utc = datetime.now(timezone.utc)
print(f"   Local time:  {now_local}")
print(f"   UTC time:    {now_utc}")
print(f"   Timezone offset: {now_local.astimezone().tzinfo}")

# Get Binance server time
binance = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
server_time_ms = binance.fetch_time()
server_time = datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc)
print(f"   Binance time: {server_time}")
time_diff = abs((now_utc - server_time).total_seconds())
print(f"   Diff from local: {time_diff:.1f} seconds")
if time_diff > 5:
    print("   ⚠️ WARNING: Time difference > 5 seconds!")
else:
    print("   ✅ Time sync OK")

# 2. Compare OHLCV data
print("\n2. OHLCV DATA COMPARISON:")
print("-"*40)

pair = "HYPE/USDT:USDT"
pair_csv = "HYPE_USDT"

# Load CSV data
csv_5m = pd.read_csv(f"../data/candles/{pair_csv}_USDT_5m.csv", parse_dates=["timestamp"], index_col="timestamp")
csv_5m.index = csv_5m.index.tz_localize('UTC') if csv_5m.index.tz is None else csv_5m.index

print(f"\n   CSV 5m data:")
print(f"   - Range: {csv_5m.index.min()} to {csv_5m.index.max()}")
print(f"   - Total rows: {len(csv_5m)}")
print(f"   - Index timezone: {csv_5m.index.tz}")

# Fetch Binance data
print(f"\n   Fetching Binance 5m data...")
binance_candles = binance.fetch_ohlcv(pair, '5m', limit=100)
binance_5m = pd.DataFrame(binance_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
binance_5m['timestamp'] = pd.to_datetime(binance_5m['timestamp'], unit='ms', utc=True)
binance_5m.set_index('timestamp', inplace=True)

print(f"\n   Binance 5m data:")
print(f"   - Range: {binance_5m.index.min()} to {binance_5m.index.max()}")
print(f"   - Total rows: {len(binance_5m)}")
print(f"   - Index timezone: {binance_5m.index.tz}")

# Find overlapping timestamps
overlap_start = max(csv_5m.index.min(), binance_5m.index.min())
overlap_end = min(csv_5m.index.max(), binance_5m.index.max())
print(f"\n   Overlap period: {overlap_start} to {overlap_end}")

# Compare last N overlapping candles
print("\n3. CANDLE-BY-CANDLE COMPARISON (last 10 overlapping):")
print("-"*80)

common_times = sorted(set(csv_5m.index) & set(binance_5m.index))
if len(common_times) == 0:
    print("   ⚠️ NO OVERLAPPING TIMESTAMPS!")
    print(f"   CSV ends at: {csv_5m.index.max()}")
    print(f"   Binance starts at: {binance_5m.index.min()}")
else:
    print(f"   Found {len(common_times)} overlapping candles")
    print(f"\n   {'Timestamp':<25} {'CSV Close':>12} {'Binance Close':>14} {'Diff':>10} {'Match':>8}")
    print("   " + "-"*75)
    
    mismatches = 0
    for ts in common_times[-10:]:
        csv_close = csv_5m.loc[ts, 'close']
        bin_close = binance_5m.loc[ts, 'close']
        diff = abs(csv_close - bin_close)
        match = "✅" if diff < 0.001 else "❌"
        if diff >= 0.001:
            mismatches += 1
        print(f"   {ts}  {csv_close:>12.4f}  {bin_close:>14.4f}  {diff:>10.6f}  {match:>8}")
    
    if mismatches == 0:
        print("\n   ✅ All candles match perfectly!")
    else:
        print(f"\n   ⚠️ {mismatches} candles have price differences!")

# 4. Compare feature generation
print("\n4. FEATURE GENERATION COMPARISON:")
print("-"*80)

if len(common_times) > 0:
    # Load models
    dir_model = joblib.load("../models/v8_improved/direction_model.joblib")
    feature_cols = joblib.load("../models/v8_improved/feature_names.joblib")
    engine = MTFFeatureEngine()
    
    # We need more data for features, so let's use the last common timestamp
    target_time = common_times[-1]
    print(f"\n   Target timestamp: {target_time}")
    
    # Prepare CSV features (using CSV data only)
    csv_1m = pd.read_csv(f"../data/candles/{pair_csv}_USDT_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
    csv_15m = pd.read_csv(f"../data/candles/{pair_csv}_USDT_15m.csv", parse_dates=["timestamp"], index_col="timestamp")
    csv_1m.index = csv_1m.index.tz_localize('UTC') if csv_1m.index.tz is None else csv_1m.index
    csv_15m.index = csv_15m.index.tz_localize('UTC') if csv_15m.index.tz is None else csv_15m.index
    
    # Filter to before target_time + warmup
    warmup = target_time - timedelta(hours=48)
    csv_1m_f = csv_1m[(csv_1m.index >= warmup) & (csv_1m.index <= target_time)]
    csv_5m_f = csv_5m[(csv_5m.index >= warmup) & (csv_5m.index <= target_time)]
    csv_15m_f = csv_15m[(csv_15m.index >= warmup) & (csv_15m.index <= target_time)]
    
    # Prepare Binance features (fetch more data)
    print("   Fetching more Binance data for features...")
    bin_1m = pd.DataFrame(binance.fetch_ohlcv(pair, '1m', limit=1000), 
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    bin_1m['timestamp'] = pd.to_datetime(bin_1m['timestamp'], unit='ms', utc=True)
    bin_1m.set_index('timestamp', inplace=True)
    
    bin_5m = pd.DataFrame(binance.fetch_ohlcv(pair, '5m', limit=1000), 
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    bin_5m['timestamp'] = pd.to_datetime(bin_5m['timestamp'], unit='ms', utc=True)
    bin_5m.set_index('timestamp', inplace=True)
    
    bin_15m = pd.DataFrame(binance.fetch_ohlcv(pair, '15m', limit=1000), 
                           columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    bin_15m['timestamp'] = pd.to_datetime(bin_15m['timestamp'], unit='ms', utc=True)
    bin_15m.set_index('timestamp', inplace=True)
    
    # Filter Binance data to same period
    bin_1m_f = bin_1m[(bin_1m.index >= warmup) & (bin_1m.index <= target_time)]
    bin_5m_f = bin_5m[(bin_5m.index >= warmup) & (bin_5m.index <= target_time)]
    bin_15m_f = bin_15m[(bin_15m.index >= warmup) & (bin_15m.index <= target_time)]
    
    print(f"\n   Data sizes:")
    print(f"   CSV:     1m={len(csv_1m_f)}, 5m={len(csv_5m_f)}, 15m={len(csv_15m_f)}")
    print(f"   Binance: 1m={len(bin_1m_f)}, 5m={len(bin_5m_f)}, 15m={len(bin_15m_f)}")
    
    def prepare_features(m1, m5, m15):
        ft = engine.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        ft = ft.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return ft
    
    # Generate features
    try:
        csv_features = prepare_features(csv_1m_f, csv_5m_f, csv_15m_f)
        bin_features = prepare_features(bin_1m_f, bin_5m_f, bin_15m_f)
        
        # Get features at target time
        if target_time in csv_features.index and target_time in bin_features.index:
            csv_row = csv_features.loc[[target_time]]
            bin_row = bin_features.loc[[target_time]]
            
            # Compare key features
            print(f"\n   Feature comparison at {target_time}:")
            print(f"   {'Feature':<30} {'CSV':>15} {'Binance':>15} {'Diff':>12}")
            print("   " + "-"*75)
            
            key_features = ['close', 'atr', 'm5_rsi_14', 'm5_macd', 'm5_bb_width', 'm5_atr_14', 'vol_ratio']
            for feat in key_features:
                if feat in csv_row.columns and feat in bin_row.columns:
                    csv_val = csv_row[feat].iloc[0]
                    bin_val = bin_row[feat].iloc[0]
                    diff = abs(csv_val - bin_val)
                    match = "✅" if diff < 0.01 else ("⚠️" if diff < 0.1 else "❌")
                    print(f"   {feat:<30} {csv_val:>15.6f} {bin_val:>15.6f} {diff:>12.6f} {match}")
            
            # Compare predictions
            print("\n   MODEL PREDICTIONS:")
            X_csv = csv_row[feature_cols].values.astype(np.float64)
            X_bin = bin_row[feature_cols].values.astype(np.float64)
            X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=0.0, neginf=0.0)
            X_bin = np.nan_to_num(X_bin, nan=0.0, posinf=0.0, neginf=0.0)
            
            proba_csv = dir_model.predict_proba(X_csv)[0]
            proba_bin = dir_model.predict_proba(X_bin)[0]
            
            pred_csv = np.argmax(proba_csv)
            pred_bin = np.argmax(proba_bin)
            
            dir_map = {0: 'SHORT', 1: 'SIDEWAYS', 2: 'LONG'}
            
            print(f"\n   CSV prediction:     {dir_map[pred_csv]} (conf={max(proba_csv):.2f})")
            print(f"                       DOWN={proba_csv[0]:.3f} SIDE={proba_csv[1]:.3f} UP={proba_csv[2]:.3f}")
            print(f"\n   Binance prediction: {dir_map[pred_bin]} (conf={max(proba_bin):.2f})")
            print(f"                       DOWN={proba_bin[0]:.3f} SIDE={proba_bin[1]:.3f} UP={proba_bin[2]:.3f}")
            
            if pred_csv == pred_bin:
                print("\n   ✅ Predictions MATCH!")
            else:
                print("\n   ❌ Predictions DIFFER!")
                
        else:
            print(f"   ⚠️ Target time {target_time} not found in features")
            
    except Exception as e:
        print(f"   ❌ Error generating features: {e}")
        import traceback
        traceback.print_exc()

# 5. Check live script's data fetching logic
print("\n5. LIVE SCRIPT DATA FETCH SIMULATION:")
print("-"*80)

# Simulate exactly what live script does
LOOKBACK = 3000
TIMEFRAMES = ['1m', '5m', '15m']

print(f"\n   Simulating live script fetch (LOOKBACK={LOOKBACK})...")

data = {}
for tf in TIMEFRAMES:
    all_candles = []
    limit_per_request = 1000
    
    # First request
    candles = binance.fetch_ohlcv(pair, tf, limit=min(limit_per_request, LOOKBACK))
    all_candles.extend(candles)
    
    # Need more?
    if len(all_candles) < LOOKBACK:
        tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[tf]
        oldest_ts = all_candles[0][0]
        since_ts = oldest_ts - (limit_per_request * tf_ms)
        
        more_candles = binance.fetch_ohlcv(pair, tf, since=since_ts, limit=limit_per_request)
        seen = {c[0] for c in all_candles}
        new = [c for c in more_candles if c[0] not in seen]
        all_candles = new + all_candles
    
    all_candles = sorted(all_candles, key=lambda x: x[0])
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    data[tf] = df
    print(f"   {tf}: {len(df)} candles, from {df.index.min()} to {df.index.max()}")

# Check last candle timestamp vs current time
last_5m = data['5m'].index.max()
now = datetime.now(timezone.utc)
lag = (now - last_5m).total_seconds() / 60

print(f"\n   Last 5m candle: {last_5m}")
print(f"   Current time:   {now}")
print(f"   Lag: {lag:.1f} minutes")

if lag > 10:
    print("   ⚠️ Data lag is high!")
else:
    print("   ✅ Data is fresh")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

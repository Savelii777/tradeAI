#!/usr/bin/env python3
"""
Check the distribution of direction predictions in the training data.
This helps understand if the model is biased towards SIDEWAYS.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

def check_target_distribution():
    """Check distribution of target_dir in training data"""
    data_dir = Path("data/candles")
    
    # Load pairs
    import json
    pairs_file = Path("config/pairs_20.json")
    if not pairs_file.exists():
        pairs_file = Path("config/pairs_list.json")
    
    with open(pairs_file) as f:
        pairs_data = json.load(f)
    pairs = [p['symbol'] for p in pairs_data['pairs'][:20]]
    
    print("="*70)
    print("CHECKING TARGET DISTRIBUTION IN TRAINING DATA")
    print("="*70)
    
    all_targets = []
    
    # Load recent 90 days of data (same as training)
    now = datetime.now()
    train_start = now - timedelta(days=90)
    
    for pair in pairs:
        pair_name = pair.replace('/', '_').replace(':', '_')
        
        try:
            m1 = pd.read_csv(data_dir / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m5 = pd.read_csv(data_dir / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m15 = pd.read_csv(data_dir / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
        except FileNotFoundError:
            continue
        
        # Filter to training period
        m1 = m1[(m1.index >= train_start) & (m1.index < now)]
        m5 = m5[(m5.index >= train_start) & (m5.index < now)]
        m15 = m15[(m15.index >= train_start) & (m15.index < now)]
        
        if len(m5) < 500:
            continue
        
        # Prepare features and targets (same as train_v3_dynamic.py)
        mtf_fe = MTFFeatureEngine()
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        
        # Add volume features
        ft['vol_sma_20'] = ft['volume'].rolling(20).mean()
        ft['vol_ratio'] = ft['volume'] / ft['vol_sma_20']
        ft['vol_zscore'] = (ft['volume'] - ft['vol_sma_20']) / ft['volume'].rolling(20).std()
        ft['vwap'] = (ft['close'] * ft['volume']).rolling(20).sum() / ft['volume'].rolling(20).sum()
        ft['price_vs_vwap'] = ft['close'] / ft['vwap'] - 1
        ft['vol_momentum'] = ft['volume'].pct_change(5)
        
        # Calculate ATR
        high = ft['high']
        low = ft['low']
        close = ft['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        ft['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        # Create targets (same as train_v3_dynamic.py)
        LOOKAHEAD = 12
        rolling_vol = ft['close'].pct_change().rolling(window=100, min_periods=50).std()
        rolling_vol = rolling_vol.shift(1)
        threshold = np.maximum(rolling_vol, 0.005)
        future_return = ft['close'].pct_change(LOOKAHEAD).shift(-LOOKAHEAD)
        
        target_dir = np.where(
            future_return > threshold, 2,
            np.where(future_return < -threshold, 0, 1)
        )
        
        ft['target_dir'] = target_dir
        ft = ft.dropna(subset=['target_dir'])
        
        all_targets.extend(ft['target_dir'].tolist())
    
    if len(all_targets) == 0:
        print("❌ No targets found")
        return
    
    # Count distribution
    targets_array = np.array(all_targets)
    down_count = np.sum(targets_array == 0)
    sideways_count = np.sum(targets_array == 1)
    up_count = np.sum(targets_array == 2)
    total = len(targets_array)
    
    print(f"\n📊 TARGET DISTRIBUTION:")
    print(f"   Total samples: {total:,}")
    print(f"   DOWN (0):     {down_count:6,} ({down_count/total*100:.1f}%)")
    print(f"   SIDEWAYS (1): {sideways_count:6,} ({sideways_count/total*100:.1f}%)")
    print(f"   UP (2):       {up_count:6,} ({up_count/total*100:.1f}%)")
    
    print(f"\n💡 INTERPRETATION:")
    if sideways_count / total > 0.6:
        print("   ⚠️  WARNING: More than 60% of targets are SIDEWAYS!")
        print("   → Model may be biased towards predicting SIDEWAYS")
        print("   → Consider adjusting threshold in create_targets_v1()")
    elif sideways_count / total > 0.5:
        print("   ⚠️  CAUTION: More than 50% of targets are SIDEWAYS")
        print("   → Model may favor SIDEWAYS predictions")
    else:
        print("   ✅ Distribution looks balanced")
    
    print("="*70)

if __name__ == '__main__':
    check_target_distribution()


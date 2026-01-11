#!/usr/bin/env python3
"""Quick check of model confidence distribution on real data."""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

def main():
    # Load models
    model_dir = Path(__file__).parent.parent / "models" / "v8_improved"
    dir_model = joblib.load(model_dir / 'direction_model.joblib')
    timing_model = joblib.load(model_dir / 'timing_model.joblib')
    strength_model = joblib.load(model_dir / 'strength_model.joblib')
    features = joblib.load(model_dir / 'feature_names.joblib')
    
    print(f"Model uses {len(features)} features")
    print(f"Features: {features[:5]} ... {features[-5:]}")
    
    # Load BTC data
    data_dir = Path(__file__).parent.parent / "data" / "candles"
    pair = 'BTC_USDT_USDT'
    
    m1 = pd.read_parquet(data_dir / f'{pair}_1m.parquet')
    m5 = pd.read_parquet(data_dir / f'{pair}_5m.parquet')
    m15 = pd.read_parquet(data_dir / f'{pair}_15m.parquet')
    
    print(f"\nLoaded BTC data: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
    
    # Feature engine
    mtf_fe = MTFFeatureEngine()
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    
    # Add volume features
    ft['vol_sma_20'] = m5['volume'].rolling(20).mean()
    ft['vol_ratio'] = m5['volume'] / ft['vol_sma_20']
    
    # Add ATR
    tr = pd.concat([
        m5['high'] - m5['low'],
        abs(m5['high'] - m5['close'].shift()),
        abs(m5['low'] - m5['close'].shift())
    ], axis=1).max(axis=1)
    ft['atr'] = tr.ewm(span=14, adjust=False).mean()
    
    ft = ft.dropna()
    print(f"Total feature rows: {len(ft)}")
    
    # Check last 2000 rows
    last_n = 2000
    data = ft.tail(last_n).copy()
    
    # Fill missing features
    missing = []
    for f in features:
        if f not in data.columns:
            data[f] = 0.0
            missing.append(f)
    
    if missing:
        print(f"\n⚠️ Missing features: {missing}")
    
    X = data[features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    
    # Get predictions
    probas = dir_model.predict_proba(X)
    max_probs = np.max(probas, axis=1)
    timing = timing_model.predict(X)
    strength = strength_model.predict(X)
    
    print(f"\n=== Confidence distribution (last {last_n} candles) ===")
    print(f"Min: {max_probs.min():.3f}")
    print(f"Max: {max_probs.max():.3f}")
    print(f"Mean: {max_probs.mean():.3f}")
    print(f"Median: {np.median(max_probs):.3f}")
    print(f"> 0.50: {(max_probs > 0.50).sum()} ({(max_probs > 0.50).mean()*100:.1f}%)")
    print(f"> 0.55: {(max_probs > 0.55).sum()} ({(max_probs > 0.55).mean()*100:.1f}%)")
    print(f"> 0.57: {(max_probs > 0.57).sum()} ({(max_probs > 0.57).mean()*100:.1f}%)")
    print(f"> 0.60: {(max_probs > 0.60).sum()} ({(max_probs > 0.60).mean()*100:.1f}%)")
    
    print(f"\n=== Timing distribution ===")
    print(f"Min: {timing.min():.3f}, Max: {timing.max():.3f}, Mean: {timing.mean():.3f}")
    print(f"> 1.0: {(timing > 1.0).sum()} ({(timing > 1.0).mean()*100:.1f}%)")
    print(f"> 1.3: {(timing > 1.3).sum()} ({(timing > 1.3).mean()*100:.1f}%)")
    
    print(f"\n=== Strength distribution ===")
    print(f"Min: {strength.min():.3f}, Max: {strength.max():.3f}, Mean: {strength.mean():.3f}")
    print(f"> 1.5: {(strength > 1.5).sum()} ({(strength > 1.5).mean()*100:.1f}%)")
    print(f"> 1.7: {(strength > 1.7).sum()} ({(strength > 1.7).mean()*100:.1f}%)")
    
    # Combined filter
    pass_all = (max_probs > 0.57) & (timing > 1.3) & (strength > 1.7)
    print(f"\n=== Combined filter (conf>0.57 & tim>1.3 & str>1.7) ===")
    print(f"Passing: {pass_all.sum()} ({pass_all.mean()*100:.1f}%)")
    
    # Lower threshold test
    pass_lower = (max_probs > 0.50) & (timing > 1.0) & (strength > 1.4)
    print(f"\n=== Lower threshold (conf>0.50 & tim>1.0 & str>1.4) ===")
    print(f"Passing: {pass_lower.sum()} ({pass_lower.mean()*100:.1f}%)")
    
    # Show some high confidence examples
    high_conf_idx = np.where(max_probs > 0.55)[0]
    if len(high_conf_idx) > 0:
        print(f"\n=== High confidence examples ===")
        for i in high_conf_idx[-5:]:
            print(f"  idx={i}: conf={max_probs[i]:.3f}, tim={timing[i]:.2f}, str={strength[i]:.2f}")

if __name__ == "__main__":
    main()

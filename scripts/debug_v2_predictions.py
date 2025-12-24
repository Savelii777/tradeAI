#!/usr/bin/env python3
"""
Debug script to analyze V2 model predictions distribution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import ccxt
from datetime import datetime, timedelta

from src.features.feature_engine import FeatureEngine


def main():
    """Analyze model predictions distribution."""
    print("\n" + "="*60)
    print("V2 MODEL PREDICTIONS ANALYSIS")
    print("="*60)
    
    # Load model
    model_path = "models/mtf_v2/ensemble_v2.pkl"
    print(f"\nLoading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully!")
    
    # Fetch data
    exchange = ccxt.binance({'enableRateLimit': True})
    pair = "BTC/USDT:USDT"
    timeframe = "5m"
    days = 7
    
    print(f"\nFetching {days} days of {timeframe} data for {pair}...")
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    limit = days * 24 * 12  # 5m candles per day
    
    ohlcv = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetched {len(df)} candles")
    
    # Generate features
    print("\nGenerating features...")
    feature_engine = FeatureEngine()
    features = feature_engine.generate_all_features(df)
    features = features.dropna()
    print(f"Generated {len(features.columns)} features for {len(features)} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(features)
    
    # Analyze direction probabilities
    print("\n" + "-"*60)
    print("DIRECTION PROBABILITIES ANALYSIS")
    print("-"*60)
    
    direction_proba = predictions.get('direction_proba', None)
    if direction_proba is not None:
        p_down = direction_proba[:, 0]
        p_sideways = direction_proba[:, 1]
        p_up = direction_proba[:, 2]
        
        print(f"\nP(DOWN):")
        print(f"  Mean: {np.mean(p_down):.4f}")
        print(f"  Std:  {np.std(p_down):.4f}")
        print(f"  Min:  {np.min(p_down):.4f}")
        print(f"  Max:  {np.max(p_down):.4f}")
        print(f"  > 0.30: {np.sum(p_down > 0.30) / len(p_down) * 100:.1f}%")
        print(f"  > 0.35: {np.sum(p_down > 0.35) / len(p_down) * 100:.1f}%")
        print(f"  > 0.40: {np.sum(p_down > 0.40) / len(p_down) * 100:.1f}%")
        
        print(f"\nP(SIDEWAYS):")
        print(f"  Mean: {np.mean(p_sideways):.4f}")
        print(f"  Std:  {np.std(p_sideways):.4f}")
        print(f"  Min:  {np.min(p_sideways):.4f}")
        print(f"  Max:  {np.max(p_sideways):.4f}")
        
        print(f"\nP(UP):")
        print(f"  Mean: {np.mean(p_up):.4f}")
        print(f"  Std:  {np.std(p_up):.4f}")
        print(f"  Min:  {np.min(p_up):.4f}")
        print(f"  Max:  {np.max(p_up):.4f}")
        print(f"  > 0.30: {np.sum(p_up > 0.30) / len(p_up) * 100:.1f}%")
        print(f"  > 0.35: {np.sum(p_up > 0.35) / len(p_up) * 100:.1f}%")
        print(f"  > 0.40: {np.sum(p_up > 0.40) / len(p_up) * 100:.1f}%")
        
        # Predicted classes
        predicted_classes = np.argmax(direction_proba, axis=1)
        print(f"\n\nPREDICTED CLASS DISTRIBUTION:")
        print(f"  Down (0):     {np.sum(predicted_classes == 0) / len(predicted_classes) * 100:.1f}%")
        print(f"  Sideways (1): {np.sum(predicted_classes == 1) / len(predicted_classes) * 100:.1f}%")
        print(f"  Up (2):       {np.sum(predicted_classes == 2) / len(predicted_classes) * 100:.1f}%")
    
    # Analyze timing
    print("\n" + "-"*60)
    print("TIMING ANALYSIS")
    print("-"*60)
    
    timing = predictions.get('timing', None)
    if timing is not None:
        print(f"  Mean: {np.mean(timing):.4f}")
        print(f"  Std:  {np.std(timing):.4f}")
        print(f"  Min:  {np.min(timing):.4f}")
        print(f"  Max:  {np.max(timing):.4f}")
        print(f"  > 0.30: {np.sum(timing > 0.30) / len(timing) * 100:.1f}%")
        print(f"  > 0.40: {np.sum(timing > 0.40) / len(timing) * 100:.1f}%")
        print(f"  > 0.50: {np.sum(timing > 0.50) / len(timing) * 100:.1f}%")
    
    # Analyze strength
    print("\n" + "-"*60)
    print("STRENGTH ANALYSIS")
    print("-"*60)
    
    strength = predictions.get('strength', None)
    if strength is not None:
        print(f"  Mean: {np.mean(strength):.4f}")
        print(f"  Std:  {np.std(strength):.4f}")
        print(f"  Min:  {np.min(strength):.4f}")
        print(f"  Max:  {np.max(strength):.4f}")
        print(f"  > 0.25: {np.sum(strength > 0.25) / len(strength) * 100:.1f}%")
        print(f"  > 0.30: {np.sum(strength > 0.30) / len(strength) * 100:.1f}%")
    else:
        print("  NOT AVAILABLE - strength_model not trained!")
    
    # Check trained status
    print("\n" + "-"*60)
    print("MODEL TRAINING STATUS")
    print("-"*60)
    print(f"  direction_model._is_trained: {model.direction_model._is_trained}")
    print(f"  strength_model._is_trained: {model.strength_model._is_trained}")
    print(f"  timing_model._is_trained: {model.timing_model._is_trained}")
    print(f"  volatility_model._is_trained: {model.volatility_model._is_trained}")
    
    # Signal eligibility check
    print("\n" + "-"*60)
    print("SIGNAL ELIGIBILITY (current thresholds)")
    print("-"*60)
    
    min_direction_prob = 0.32
    min_timing_score = 0.15
    min_strength_score = 0.20
    
    print(f"\nThresholds: direction>{min_direction_prob}, timing>{min_timing_score}, strength>{min_strength_score}")
    
    # Simulate signal generation like ensemble_v2.py does
    if direction_proba is not None and timing is not None:
        # Strength defaults to 1.0 if not available
        if strength is None:
            strength_vals = np.ones(len(timing))
            print("\n** strength_model not trained - using default 1.0 **")
        else:
            strength_vals = strength
        
        # Potential long signals (mimicking get_trading_signal logic)
        long_direction_ok = (p_up > min_direction_prob) & (p_up > p_down)
        long_timing_ok = timing >= min_timing_score
        long_strength_ok = strength_vals >= min_strength_score
        
        long_all_ok = long_direction_ok & long_timing_ok & long_strength_ok
        
        print(f"\nLONG SIGNALS:")
        print(f"  Direction OK (p_up > {min_direction_prob} AND p_up > p_down): {np.sum(long_direction_ok) / len(p_up) * 100:.1f}%")
        print(f"  Timing OK (>= {min_timing_score}):                            {np.sum(long_timing_ok) / len(timing) * 100:.1f}%")
        print(f"  Strength OK (>= {min_strength_score}):                        {np.sum(long_strength_ok) / len(strength_vals) * 100:.1f}%")
        print(f"  ALL OK (eligible):                                    {np.sum(long_all_ok) / len(p_up) * 100:.1f}%")
        print(f"  Eligible count:                                       {np.sum(long_all_ok)} of {len(p_up)}")
        
        # Potential short signals
        short_direction_ok = (p_down > min_direction_prob) & (p_down > p_up)
        short_timing_ok = timing >= min_timing_score
        short_strength_ok = strength_vals >= min_strength_score
        
        short_all_ok = short_direction_ok & short_timing_ok & short_strength_ok
        
        print(f"\nSHORT SIGNALS:")
        print(f"  Direction OK (p_down > {min_direction_prob} AND p_down > p_up): {np.sum(short_direction_ok) / len(p_down) * 100:.1f}%")
        print(f"  Timing OK (>= {min_timing_score}):                             {np.sum(short_timing_ok) / len(timing) * 100:.1f}%")
        print(f"  Strength OK (>= {min_strength_score}):                         {np.sum(short_strength_ok) / len(strength_vals) * 100:.1f}%")
        print(f"  ALL OK (eligible):                                     {np.sum(short_all_ok) / len(p_down) * 100:.1f}%")
        print(f"  Eligible count:                                        {np.sum(short_all_ok)} of {len(p_down)}")
        
        print(f"\nTOTAL ELIGIBLE (long + short): {np.sum(long_all_ok) + np.sum(short_all_ok)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

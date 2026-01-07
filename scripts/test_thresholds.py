#!/usr/bin/env python3
"""
Тест разных порогов для понимания количества сигналов
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def test_thresholds():
    print("=" * 70)
    print("ТЕСТ РАЗНЫХ ПОРОГОВ")
    print("=" * 70)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features_list = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    mtf_fe = MTFFeatureEngine()
    
    pair = "BTC/USDT:USDT"
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    m1 = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m5 = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m15 = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    # Generate features
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft = ft.dropna()
    
    # Predict for ALL rows
    X = np.zeros((len(ft), len(features_list)))
    for i, f in enumerate(features_list):
        if f in ft.columns:
            X[:, i] = ft[f].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba = models['direction'].predict_proba(X)
    timing = models['timing'].predict(X)
    strength = models['strength'].predict(X)
    
    dir_pred = np.argmax(dir_proba, axis=1)
    dir_conf = np.max(dir_proba, axis=1)
    
    ft['direction'] = dir_pred
    ft['conf'] = dir_conf
    ft['timing'] = timing
    ft['strength'] = strength
    
    # Разные комбинации порогов
    threshold_configs = [
        {"name": "Current Live", "conf": 0.50, "timing": 0.8, "strength": 1.4},
        {"name": "backtest_v2", "conf": 0.32, "timing": 0.15, "strength": 0.2},
        {"name": "Lower conf", "conf": 0.40, "timing": 0.8, "strength": 1.4},
        {"name": "No timing/str", "conf": 0.50, "timing": 0.0, "strength": 0.0},
        {"name": "Very Low", "conf": 0.35, "timing": 0.5, "strength": 0.8},
    ]
    
    print(f"\nВсего свечей: {len(ft)}")
    print(f"Период: {ft.index[0]} → {ft.index[-1]}")
    
    print("\n" + "-" * 80)
    print(f"{'Config':<20} {'Conf':<8} {'Timing':<8} {'Strength':<10} {'Signals':<10} {'% of total'}")
    print("-" * 80)
    
    for cfg in threshold_configs:
        signals = ft[
            (ft['direction'] != 1) &  # Not SIDEWAYS
            (ft['conf'] >= cfg['conf']) &
            (ft['timing'] >= cfg['timing']) &
            (ft['strength'] >= cfg['strength'])
        ]
        pct = 100 * len(signals) / len(ft)
        
        longs = len(signals[signals['direction'] == 2])
        shorts = len(signals[signals['direction'] == 0])
        
        print(f"{cfg['name']:<20} {cfg['conf']:<8.2f} {cfg['timing']:<8.2f} {cfg['strength']:<10.2f} {len(signals):<10} {pct:.3f}% (L:{longs}, S:{shorts})")
    
    # Показать распределение confidence для не-SIDEWAYS
    non_sideways = ft[ft['direction'] != 1]
    print(f"\n📊 Распределение для не-SIDEWAYS ({len(non_sideways)} свечей):")
    print(f"   Confidence: min={non_sideways['conf'].min():.3f}, max={non_sideways['conf'].max():.3f}, median={non_sideways['conf'].median():.3f}")
    print(f"   Timing: min={non_sideways['timing'].min():.2f}, max={non_sideways['timing'].max():.2f}, median={non_sideways['timing'].median():.2f}")
    print(f"   Strength: min={non_sideways['strength'].min():.2f}, max={non_sideways['strength'].max():.2f}, median={non_sideways['strength'].median():.2f}")
    
    # Проверяем что мешает пройти фильтры
    print("\n📊 ЧТО МЕШАЕТ ПРОЙТИ ФИЛЬТРЫ (для не-SIDEWAYS):")
    low_conf = len(non_sideways[non_sideways['conf'] < 0.50])
    low_timing = len(non_sideways[non_sideways['timing'] < 0.8])
    low_strength = len(non_sideways[non_sideways['strength'] < 1.4])
    
    print(f"   Conf < 0.50: {low_conf} ({100*low_conf/len(non_sideways):.1f}%)")
    print(f"   Timing < 0.8: {low_timing} ({100*low_timing/len(non_sideways):.1f}%)")
    print(f"   Strength < 1.4: {low_strength} ({100*low_strength/len(non_sideways):.1f}%)")


if __name__ == "__main__":
    test_thresholds()

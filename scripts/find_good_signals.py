#!/usr/bin/env python3
"""
Поиск моментов когда были реальные LONG/SHORT сигналы с высоким confidence
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def find_good_signals():
    print("=" * 80)
    print("ПОИСК ХОРОШИХ СИГНАЛОВ В ИСТОРИИ")
    print("=" * 80)
    
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
    
    # Load full data
    m1 = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m5 = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m15 = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    print(f"\n📊 Данные: {m5.index[0]} → {m5.index[-1]}")
    
    # Generate features for ALL data
    print("\n⏳ Генерация фичей для всего периода...")
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna()
    
    print(f"   Всего свечей с фичами: {len(ft)}")
    
    # Predict for ALL rows
    print("\n⏳ Предсказания для всех свечей...")
    
    # Prepare features matrix
    X = np.zeros((len(ft), len(features_list)))
    for i, f in enumerate(features_list):
        if f in ft.columns:
            X[:, i] = ft[f].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Predict
    dir_proba = models['direction'].predict_proba(X)
    timing = models['timing'].predict(X)
    strength = models['strength'].predict(X)
    
    dir_pred = np.argmax(dir_proba, axis=1)
    dir_conf = np.max(dir_proba, axis=1)
    
    # Add to dataframe
    ft['direction'] = dir_pred  # 0=SHORT, 1=SIDEWAYS, 2=LONG
    ft['conf'] = dir_conf
    ft['timing'] = timing
    ft['strength'] = strength
    
    # Filter for good signals
    good_signals = ft[
        (ft['direction'] != 1) &  # Not SIDEWAYS
        (ft['conf'] >= 0.50) &
        (ft['timing'] >= 0.8) &
        (ft['strength'] >= 1.4)
    ]
    
    print(f"\n🎯 ХОРОШИЕ СИГНАЛЫ (CONF>=0.50, T>=0.8, S>=1.4):")
    print(f"   Всего: {len(good_signals)} из {len(ft)} ({100*len(good_signals)/len(ft):.2f}%)")
    
    # Split by direction
    longs = good_signals[good_signals['direction'] == 2]
    shorts = good_signals[good_signals['direction'] == 0]
    
    print(f"   LONG: {len(longs)}")
    print(f"   SHORT: {len(shorts)}")
    
    # Show distribution by date
    print("\n📅 РАСПРЕДЕЛЕНИЕ ПО ДНЯМ:")
    good_signals_daily = good_signals.groupby(good_signals.index.date).size()
    
    print(f"   Дней с сигналами: {len(good_signals_daily)}")
    print(f"   Среднее сигналов/день: {good_signals_daily.mean():.1f}")
    
    # Show last 30 days
    print("\n📅 ПОСЛЕДНИЕ 30 ДНЕЙ:")
    last_30_days = good_signals_daily.tail(30)
    for date, count in last_30_days.items():
        day_signals = good_signals[good_signals.index.date == date]
        long_count = len(day_signals[day_signals['direction'] == 2])
        short_count = len(day_signals[day_signals['direction'] == 0])
        print(f"   {date}: {count:3d} сигналов (L:{long_count:2d}, S:{short_count:2d})")
    
    # When was the last good signal?
    if len(good_signals) > 0:
        print(f"\n⏰ ПОСЛЕДНИЙ ХОРОШИЙ СИГНАЛ:")
        last_sig = good_signals.iloc[-1]
        direction = ['SHORT', 'SIDEWAYS', 'LONG'][int(last_sig['direction'])]
        print(f"   Время: {good_signals.index[-1]}")
        print(f"   Направление: {direction}")
        print(f"   Confidence: {last_sig['conf']:.3f}")
        print(f"   Timing: {last_sig['timing']:.2f}")
        print(f"   Strength: {last_sig['strength']:.2f}")
    
    # What's the current state?
    print(f"\n📊 ТЕКУЩЕЕ СОСТОЯНИЕ (последняя свеча):")
    last = ft.iloc[-1]
    direction = ['SHORT', 'SIDEWAYS', 'LONG'][int(last['direction'])]
    print(f"   Время: {ft.index[-1]}")
    print(f"   Направление: {direction}")
    print(f"   Confidence: {last['conf']:.3f}")
    print(f"   Timing: {last['timing']:.2f}")
    print(f"   Strength: {last['strength']:.2f}")
    
    # Distribution of directions
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ (весь период):")
    dir_counts = ft['direction'].value_counts()
    total = len(ft)
    for d, count in sorted(dir_counts.items()):
        name = ['SHORT', 'SIDEWAYS', 'LONG'][d]
        print(f"   {name}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    find_good_signals()

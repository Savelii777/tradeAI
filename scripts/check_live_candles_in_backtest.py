#!/usr/bin/env python3
"""
Проверить те же свечи, которые проверялись на лайве, но на бектесте (CSV данные).
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from train_v3_dynamic import add_volume_features, calculate_atr

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

# Свечи из лога, которые дали LONG/SHORT (но были отклонены)
LIVE_CANDLES = [
    {'time': '2026-01-05 23:35:25', 'pair': 'AVAX/USDT:USDT', 'direction': 'LONG', 'conf': 0.36, 'timing': 2.38, 'strength': 1.9},
    {'time': '2026-01-05 23:40:29', 'pair': 'AVAX/USDT:USDT', 'direction': 'LONG', 'conf': 0.37, 'timing': 2.29, 'strength': 1.8},
    {'time': '2026-01-05 23:50:43', 'pair': '1000PEPE/USDT:USDT', 'direction': 'SHORT', 'conf': 0.35, 'timing': 2.90, 'strength': 2.1},
    {'time': '2026-01-06 00:00:43', 'pair': '1000PEPE/USDT:USDT', 'direction': 'SHORT', 'conf': 0.35, 'timing': 2.98, 'strength': 2.2},
    {'time': '2026-01-06 00:05:49', 'pair': '1000PEPE/USDT:USDT', 'direction': 'SHORT', 'conf': 0.35, 'timing': 3.01, 'strength': 2.3},
    {'time': '2026-01-06 00:10:23', 'pair': 'AVAX/USDT:USDT', 'direction': 'LONG', 'conf': 0.37, 'timing': 2.40, 'strength': 1.9},
    {'time': '2026-01-06 00:15:21', 'pair': 'AVAX/USDT:USDT', 'direction': 'LONG', 'conf': 0.37, 'timing': 2.18, 'strength': 1.8},
    {'time': '2026-01-06 00:20:25', 'pair': 'AVAX/USDT:USDT', 'direction': 'LONG', 'conf': 0.37, 'timing': 2.18, 'strength': 1.7},
    {'time': '2026-01-06 00:25:25', 'pair': 'AVAX/USDT:USDT', 'direction': 'LONG', 'conf': 0.37, 'timing': 2.27, 'strength': 1.8},
]

def load_models():
    """Load trained models."""
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }

def find_candle_in_csv(pair_name: str, target_time: datetime):
    """Найти свечу в CSV данных, ближайшую к target_time."""
    pair_file = pair_name.replace('/', '_').replace(':', '_')
    
    try:
        m5 = pd.read_csv(DATA_DIR / f"{pair_file}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"❌ CSV не найден для {pair_name}")
        return None
    
    # Localize to UTC if needed
    if m5.index.tz is None:
        m5.index = m5.index.tz_localize('UTC')
    
    # Найти ближайшую свечу (5m свечи закрываются на :00, :05, :10, etc.)
    # target_time - это время проверки на лайве, нужно найти закрытую свечу перед этим
    # На лайве проверяется последняя закрытая свеча (df.iloc[[-2]])
    # Значит, если проверка в 23:35:25, то проверяется свеча, закрывшаяся в 23:30:00 или 23:35:00
    
    # Округляем до ближайшей 5-минутной свечи (вниз)
    target_5m = target_time.replace(second=0, microsecond=0)
    target_5m = target_5m - timedelta(minutes=target_5m.minute % 5)
    
    # Ищем свечу, закрывшуюся ДО target_time (последняя закрытая)
    # Свеча закрывается в :00, :05, :10, etc.
    # Если проверка в 23:35:25, то последняя закрытая свеча - 23:30:00
    if target_5m.minute % 5 == 0:
        # Если точно на границе, берем предыдущую
        candle_time = target_5m - timedelta(minutes=5)
    else:
        candle_time = target_5m
    
    # Ищем в диапазоне ±10 минут
    start_time = candle_time - timedelta(minutes=10)
    end_time = candle_time + timedelta(minutes=10)
    
    mask = (m5.index >= start_time) & (m5.index <= end_time)
    candidates = m5[mask]
    
    if len(candidates) == 0:
        print(f"⚠️  Не найдено свечей в диапазоне {start_time} - {end_time}")
        return None
    
    # Берем ближайшую к candle_time
    idx = candidates.index.get_indexer([candle_time], method='nearest')[0]
    if idx == -1:
        return None
    
    candle_idx = candidates.index[idx]
    return candle_idx

def check_candle_in_backtest(pair_name: str, target_time_str: str, models: dict, mtf_fe: MTFFeatureEngine):
    """Проверить конкретную свечу на бектесте."""
    print(f"\n{'='*70}")
    print(f"Проверка: {pair_name} @ {target_time_str}")
    print(f"{'='*70}")
    
    # Парсим время
    target_time = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
    target_time = target_time.replace(tzinfo=timezone.utc)
    
    # Найти свечу в CSV
    candle_time = find_candle_in_csv(pair_name, target_time)
    if candle_time is None:
        print(f"❌ Свеча не найдена в CSV")
        return None
    
    print(f"✅ Найдена свеча в CSV: {candle_time}")
    
    # Загрузить данные для этой пары
    pair_file = pair_name.replace('/', '_').replace(':', '_')
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_file}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_file}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_file}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"❌ CSV файлы не найдены")
        return None
    
    # Localize to UTC
    for df in [m1, m5, m15]:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
    
    # Нужно загрузить достаточно данных ДО этой свечи для расчета фичей
    # Берем данные за последние 1500 свечей 5m до этой свечи
    end_time = candle_time
    start_time = end_time - timedelta(hours=1500 * 5 / 60)  # 1500 свечей * 5 минут
    
    m1_subset = m1[(m1.index >= start_time) & (m1.index <= end_time)]
    m5_subset = m5[(m5.index >= start_time) & (m5.index <= end_time)]
    m15_subset = m15[(m15.index >= start_time) & (m15.index <= end_time)]
    
    if len(m5_subset) < 100:
        print(f"⚠️  Недостаточно данных: {len(m5_subset)} свечей")
        return None
    
    # Подготовить фичи
    df = mtf_fe.align_timeframes(m1_subset, m5_subset, m15_subset)
    df = df.join(m5_subset[['open', 'high', 'low', 'close', 'volume']])
    df = add_volume_features(df)
    df['atr'] = calculate_atr(df)
    
    # Проверить, есть ли нужная свеча
    if candle_time not in df.index:
        # Найти ближайшую
        idx = df.index.get_indexer([candle_time], method='nearest')[0]
        if idx == -1:
            print(f"❌ Свеча не найдена в фичах")
            return None
        candle_time = df.index[idx]
        print(f"⚠️  Используется ближайшая свеча: {candle_time}")
    
    # Получить фичи для этой свечи
    feature_cols = models['features']
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"❌ Отсутствуют фичи: {missing[:5]}...")
        return None
    
    row = df.loc[[candle_time]]
    X = row[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    # Предсказания
    dir_proba = models['direction'].predict_proba(X)
    dir_conf = float(np.max(dir_proba))
    dir_pred = int(np.argmax(dir_proba))
    timing_pred = float(models['timing'].predict(X)[0])
    strength_pred = float(models['strength'].predict(X)[0])
    
    direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
    
    print(f"\n📊 Результаты на бектесте (CSV):")
    print(f"   Направление: {direction_str}")
    print(f"   Уверенность: {dir_conf:.3f}")
    print(f"   Timing: {timing_pred:.2f} ATR")
    print(f"   Strength: {strength_pred:.1f}")
    
    print(f"\n📊 Результаты на лайве (из лога):")
    # Найти в LIVE_CANDLES
    live_candle = next((c for c in LIVE_CANDLES if c['time'] == target_time_str and c['pair'] == pair_name), None)
    if live_candle:
        print(f"   Направление: {live_candle['direction']}")
        print(f"   Уверенность: {live_candle['conf']:.3f}")
        print(f"   Timing: {live_candle['timing']:.2f} ATR")
        print(f"   Strength: {live_candle['strength']:.1f}")
    
    # Проверить фильтры
    MIN_CONF = 0.5
    MIN_TIMING = 0.8
    MIN_STRENGTH = 1.4
    
    print(f"\n📊 Проверка фильтров (Conf >= {MIN_CONF}, Timing >= {MIN_TIMING}, Strength >= {MIN_STRENGTH}):")
    passes_conf = dir_conf >= MIN_CONF
    passes_timing = timing_pred >= MIN_TIMING
    passes_strength = strength_pred >= MIN_STRENGTH
    
    print(f"   Conf: {dir_conf:.3f} >= {MIN_CONF}? {'✅' if passes_conf else '❌'}")
    print(f"   Timing: {timing_pred:.2f} >= {MIN_TIMING}? {'✅' if passes_timing else '❌'}")
    print(f"   Strength: {strength_pred:.1f} >= {MIN_STRENGTH}? {'✅' if passes_strength else '❌'}")
    
    if dir_pred != 1 and passes_conf and passes_timing and passes_strength:
        print(f"\n✅ СИГНАЛ НАЙДЕН НА БЕКТЕСТЕ!")
        return True
    else:
        print(f"\n❌ Сигнал НЕ прошел фильтры на бектесте")
        return False

if __name__ == "__main__":
    print("="*70)
    print("ПРОВЕРКА СВЕЧЕЙ ИЗ ЛАЙВА НА БЕКТЕСТЕ")
    print("="*70)
    
    models = load_models()
    mtf_fe = MTFFeatureEngine()
    
    results = []
    for candle in LIVE_CANDLES:
        result = check_candle_in_backtest(
            candle['pair'],
            candle['time'],
            models,
            mtf_fe
        )
        results.append({
            'candle': candle,
            'result': result
        })
    
    print(f"\n\n{'='*70}")
    print("ИТОГИ")
    print(f"{'='*70}")
    
    signals_found = sum(1 for r in results if r['result'] is True)
    print(f"Сигналов найдено на бектесте: {signals_found} из {len(LIVE_CANDLES)}")
    
    if signals_found > 0:
        print(f"\n⚠️  НА БЕКТЕСТЕ ЕСТЬ СИГНАЛЫ, КОТОРЫХ НЕТ НА ЛАЙВЕ!")
        print(f"   Это означает проблему с данными или расчетом фичей на лайве.")
    else:
        print(f"\n✅ На бектесте тоже нет сигналов для этих свечей.")
        print(f"   Это означает, что проблема не в данных, а в том, что модель редко дает высокую уверенность.")


#!/usr/bin/env python3
"""
Сравнить бектест и лайв для ОДНОГО И ТОГО ЖЕ периода (который есть в CSV).
Проверить последние N свечей из CSV на бектесте и сравнить с тем, что было бы на лайве.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib
import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from train_v3_dynamic import add_volume_features, calculate_atr

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

MIN_CONF = 0.5
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

def load_models():
    """Load trained models."""
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }

def check_backtest_signals(pair_name: str, models: dict, mtf_fe: MTFFeatureEngine, last_n_candles: int = 100):
    """Проверить последние N свечей на бектесте (CSV)."""
    print(f"\n{'='*70}")
    print(f"БЕКТЕСТ (CSV): {pair_name}")
    print(f"{'='*70}")
    
    pair_file = pair_name.replace('/', '_').replace(':', '_')
    try:
        m1 = pd.read_csv(DATA_DIR / f"{pair_file}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m5 = pd.read_csv(DATA_DIR / f"{pair_file}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
        m15 = pd.read_csv(DATA_DIR / f"{pair_file}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"❌ CSV не найден")
        return None
    
    # Localize to UTC
    for df in [m1, m5, m15]:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
    
    print(f"Данные: {m5.index[0]} - {m5.index[-1]} ({len(m5)} свечей)")
    
    # Берем последние N свечей
    m5_recent = m5.tail(last_n_candles)
    if len(m5_recent) < 50:
        print(f"❌ Недостаточно данных")
        return None
    
    # Нужно достаточно данных для фичей (1500 свечей до последней)
    end_time = m5_recent.index[-1]
    start_time = end_time - timedelta(hours=1500 * 5 / 60)
    
    m1_subset = m1[(m1.index >= start_time) & (m1.index <= end_time)]
    m5_subset = m5[(m5.index >= start_time) & (m5.index <= end_time)]
    m15_subset = m15[(m15.index >= start_time) & (m15.index <= end_time)]
    
    # Подготовить фичи
    df = mtf_fe.align_timeframes(m1_subset, m5_subset, m15_subset)
    df = df.join(m5_subset[['open', 'high', 'low', 'close', 'volume']])
    df = add_volume_features(df)
    df['atr'] = calculate_atr(df)
    
    feature_cols = models['features']
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"❌ Отсутствуют фичи: {missing[:5]}...")
        return None
    
    df = df[feature_cols + ['close', 'atr']].dropna()
    
    # Проверить последние N свечей
    signals = []
    for candle_time in m5_recent.index:
        if candle_time not in df.index:
            continue
        
        row = df.loc[[candle_time]]
        X = row[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Предсказания
        dir_proba = models['direction'].predict_proba(X)
        dir_conf = float(np.max(dir_proba))
        dir_pred = int(np.argmax(dir_proba))
        timing_pred = float(models['timing'].predict(X)[0])
        strength_pred = float(models['strength'].predict(X)[0])
        
        # Проверка фильтров
        if dir_pred != 1:  # Не SIDEWAYS
            passes_conf = dir_conf >= MIN_CONF
            passes_timing = timing_pred >= MIN_TIMING
            passes_strength = strength_pred >= MIN_STRENGTH
            
            if passes_conf and passes_timing and passes_strength:
                direction_str = 'LONG' if dir_pred == 2 else 'SHORT'
                signals.append({
                    'time': candle_time,
                    'direction': direction_str,
                    'conf': dir_conf,
                    'timing': timing_pred,
                    'strength': strength_pred
                })
    
    print(f"\n📊 Проверено свечей: {len(m5_recent)}")
    print(f"✅ Сигналов найдено: {len(signals)}")
    
    if signals:
        print(f"\nСигналы:")
        for sig in signals:
            print(f"   {sig['time']} | {sig['direction']} | Conf: {sig['conf']:.3f} | Timing: {sig['timing']:.2f} | Strength: {sig['strength']:.1f}")
    
    return signals

def check_live_signals(pair_name: str, models: dict, mtf_fe: MTFFeatureEngine, target_time: datetime):
    """Проверить свечу на лайве (API) для того же времени, что и в CSV."""
    print(f"\n{'='*70}")
    print(f"ЛАЙВ (API): {pair_name} @ {target_time}")
    print(f"{'='*70}")
    
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    LOOKBACK = 1500
    
    # Загрузить данные с API
    data = {}
    for tf in ['1m', '5m', '15m']:
        try:
            # Вычисляем since для получения данных до target_time
            hours_needed = {
                '1m': LOOKBACK / 60,
                '5m': LOOKBACK * 5 / 60,
                '15m': LOOKBACK * 15 / 60
            }
            since_time = target_time - timedelta(hours=hours_needed[tf] + 1)
            since_ms = int(since_time.timestamp() * 1000)
            
            candles = binance.fetch_ohlcv(pair_name, tf, since=since_ms, limit=LOOKBACK)
            
            if not candles:
                print(f"❌ Не получены данные для {tf}")
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index().tail(LOOKBACK)
            
            data[tf] = df
            print(f"✅ {tf}: {len(df)} свечей")
        except Exception as e:
            print(f"❌ Ошибка загрузки {tf}: {e}")
            return None
    
    # Подготовить фичи
    df = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
    df = df.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
    df = add_volume_features(df)
    df['atr'] = calculate_atr(df)
    
    feature_cols = models['features']
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"❌ Отсутствуют фичи: {missing[:5]}...")
        return None
    
    # Найти свечу, ближайшую к target_time
    if target_time not in df.index:
        idx = df.index.get_indexer([target_time], method='nearest')[0]
        if idx == -1:
            print(f"❌ Свеча не найдена")
            return None
        target_time = df.index[idx]
        print(f"⚠️  Используется ближайшая свеча: {target_time}")
    
    row = df.loc[[target_time]]
    X = row[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    # Предсказания
    dir_proba = models['direction'].predict_proba(X)
    dir_conf = float(np.max(dir_proba))
    dir_pred = int(np.argmax(dir_proba))
    timing_pred = float(models['timing'].predict(X)[0])
    strength_pred = float(models['strength'].predict(X)[0])
    
    direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
    
    print(f"\n📊 Результаты:")
    print(f"   Направление: {direction_str}")
    print(f"   Уверенность: {dir_conf:.3f}")
    print(f"   Timing: {timing_pred:.2f} ATR")
    print(f"   Strength: {strength_pred:.1f}")
    
    # Проверка фильтров
    if dir_pred != 1:
        passes_conf = dir_conf >= MIN_CONF
        passes_timing = timing_pred >= MIN_TIMING
        passes_strength = strength_pred >= MIN_STRENGTH
        
        print(f"\n📊 Фильтры:")
        print(f"   Conf >= {MIN_CONF}? {'✅' if passes_conf else '❌'} ({dir_conf:.3f})")
        print(f"   Timing >= {MIN_TIMING}? {'✅' if passes_timing else '❌'} ({timing_pred:.2f})")
        print(f"   Strength >= {MIN_STRENGTH}? {'✅' if passes_strength else '❌'} ({strength_pred:.1f})")
        
        if passes_conf and passes_timing and passes_strength:
            print(f"\n✅ СИГНАЛ НАЙДЕН НА ЛАЙВЕ!")
            return True
        else:
            print(f"\n❌ Сигнал НЕ прошел фильтры на лайве")
            return False
    else:
        print(f"\n❌ SIDEWAYS - сигнала нет")
        return False

if __name__ == "__main__":
    print("="*70)
    print("СРАВНЕНИЕ БЕКТЕСТА И ЛАЙВА ДЛЯ ОДНОГО ПЕРИОДА")
    print("="*70)
    
    models = load_models()
    mtf_fe = MTFFeatureEngine()
    
    # Проверить несколько пар
    pairs = ['AVAX/USDT:USDT', 'UNI/USDT:USDT', '1000PEPE/USDT:USDT']
    
    for pair in pairs:
        # 1. Проверить бектест (CSV) - последние 100 свечей
        backtest_signals = check_backtest_signals(pair, models, mtf_fe, last_n_candles=100)
        
        if backtest_signals and len(backtest_signals) > 0:
            # 2. Проверить первую найденную свечу на лайве
            first_signal = backtest_signals[0]
            print(f"\n\n🔍 Проверяю эту же свечу на лайве...")
            live_result = check_live_signals(pair, models, mtf_fe, first_signal['time'])
            
            if live_result:
                print(f"\n✅ ОДИНАКОВО: И на бектесте, и на лайве есть сигнал!")
            else:
                print(f"\n⚠️  РАЗНИЦА: На бектесте есть сигнал, на лайве - нет!")
                print(f"   Это означает проблему с данными или расчетом фичей на лайве.")
        
        print(f"\n{'='*70}\n")


#!/usr/bin/env python3
"""
ЭКСПЕРИМЕНТ: Влияние количества данных на предсказания

Проблема: В бэктесте данные с 2017 года (много свечей), в лайве - только 1500.
Гипотеза: Индикаторы с rolling window зависят от количества данных.

Этот скрипт:
1. Загружает 5000 свечей с Binance
2. Вычисляет фичи для ОДНОЙ свечи, используя разное кол-во истории:
   - Все 5000 свечей (как бэктест)
   - Последние 1500 свечей (как лайв)
   - Последние 500 свечей (ещё меньше)
3. Сравнивает предсказания модели

Если предсказания РАЗНЫЕ - значит проблема в зависимости от длины данных!
"""

import sys
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# CONFIG
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_list.json"


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


def fetch_candles(exchange, pair: str, timeframe: str, total_needed: int) -> pd.DataFrame:
    """Загружает много свечей"""
    all_candles = []
    limit = 1000
    
    candles = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
    all_candles = candles
    
    while len(all_candles) < total_needed:
        oldest = all_candles[0][0]
        tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[timeframe]
        since = oldest - limit * tf_ms
        
        candles = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        if not candles:
            break
        
        new = [c for c in candles if c[0] < oldest]
        if not new:
            break
        
        all_candles = new + all_candles
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df


def build_features(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Строит фичи"""
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna()
    return ft


def predict_with_models(row: pd.Series, models: dict, feature_names: list) -> dict:
    """Делает предсказание для одной строки"""
    # Fill missing
    for f in feature_names:
        if f not in row:
            row[f] = 0.0
    
    X = pd.DataFrame([row[feature_names].values], columns=feature_names).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba = models['direction'].predict_proba(X)
    dir_pred = int(np.argmax(dir_proba))
    dir_conf = float(np.max(dir_proba))
    timing = float(models['timing'].predict(X)[0])
    strength = float(models['strength'].predict(X)[0])
    
    return {
        'direction': ['SHORT', 'SIDEWAYS', 'LONG'][dir_pred],
        'dir_pred': dir_pred,
        'confidence': dir_conf,
        'timing': timing,
        'strength': strength,
        'proba': dir_proba[0].tolist()
    }


def main():
    print("="*70)
    print("ЭКСПЕРИМЕНТ: Влияние количества данных на предсказания")
    print("="*70)
    
    # Load models
    print("\n📦 Загрузка моделей...")
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    feature_names = joblib.load(MODEL_DIR / 'feature_names.joblib')
    print(f"   Модель: {len(feature_names)} фичей")
    
    # Pairs
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:10]]
    
    # Init
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    # Тестовые размеры окон
    WINDOW_SIZES = [
        (5000, 5000, 2000, "ПОЛНЫЙ (как бэктест)"),
        (3000, 3000, 1000, "СРЕДНИЙ"),
        (1500, 1500, 500, "ЛАЙВ (текущий)"),
        (500, 500, 200, "МИНИМАЛЬНЫЙ"),
    ]
    
    print(f"\n📊 Тестируем {len(WINDOW_SIZES)} размеров окна данных")
    print("   Цель: найти зависимость confidence от количества данных\n")
    
    all_results = []
    
    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"📊 {pair}")
        print("="*60)
        
        # Загружаем максимум данных
        print("   Загружаем данные...")
        try:
            full_m1 = fetch_candles(binance, pair, '1m', 5000)
            full_m5 = fetch_candles(binance, pair, '5m', 5000)
            full_m15 = fetch_candles(binance, pair, '15m', 2000)
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            continue
        
        print(f"   M1: {len(full_m1)}, M5: {len(full_m5)}, M15: {len(full_m15)}")
        
        # Общий индекс - используем ЗАКРЫТУЮ свечу (предпоследнюю)
        target_time = full_m5.index[-2]  # Закрытая свеча
        print(f"   Целевая свеча: {target_time.strftime('%Y-%m-%d %H:%M')} UTC")
        
        results = []
        
        for m1_size, m5_size, m15_size, label in WINDOW_SIZES:
            # Обрезаем данные
            m1_cut = full_m1.tail(m1_size)
            m5_cut = full_m5.tail(m5_size)
            m15_cut = full_m15.tail(m15_size)
            
            # Проверяем что target_time есть в данных
            if target_time not in m5_cut.index:
                print(f"   ❌ {label}: target_time не в данных, пропускаем")
                continue
            
            try:
                features = build_features(m1_cut, m5_cut, m15_cut, mtf_fe)
                
                if target_time not in features.index:
                    # Находим ближайший индекс
                    idx = features.index.get_indexer([target_time], method='nearest')[0]
                    row = features.iloc[idx]
                    actual_time = features.index[idx]
                else:
                    row = features.loc[target_time]
                    actual_time = target_time
                
                pred = predict_with_models(row, models, feature_names)
                
                results.append({
                    'window': label,
                    'm5_size': len(m5_cut),
                    **pred
                })
                
                print(f"   {label:20s} | {pred['direction']:8s} | Conf={pred['confidence']:.3f} | T={pred['timing']:.2f} | S={pred['strength']:.2f}")
                
            except Exception as e:
                print(f"   ❌ {label}: {e}")
                continue
        
        if len(results) >= 2:
            # Сравниваем первый (полный) и последний (минимальный)
            conf_diff = abs(results[0]['confidence'] - results[-1]['confidence'])
            if conf_diff > 0.05:
                print(f"\n   ⚠️  ЗНАЧИТЕЛЬНОЕ РАСХОЖДЕНИЕ CONFIDENCE: {conf_diff:.3f}")
            else:
                print(f"\n   ✅ Confidence стабилен (diff={conf_diff:.3f})")
            
            all_results.append({
                'pair': pair,
                'full_conf': results[0]['confidence'],
                'live_conf': results[-1]['confidence'] if len(results) > 2 else results[-1]['confidence'],
                'diff': conf_diff
            })
    
    # Итоговая статистика
    print("\n" + "="*70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*70)
    
    if all_results:
        avg_full = np.mean([r['full_conf'] for r in all_results])
        avg_live = np.mean([r['live_conf'] for r in all_results])
        avg_diff = np.mean([r['diff'] for r in all_results])
        
        print(f"\n   Среднее Confidence (ПОЛНЫЙ):  {avg_full:.3f}")
        print(f"   Среднее Confidence (ЛАЙВ):    {avg_live:.3f}")
        print(f"   Среднее различие:             {avg_diff:.3f}")
        
        if avg_diff > 0.05:
            print(f"\n   🚨 КРИТИЧЕСКАЯ ПРОБЛЕМА!")
            print(f"   Confidence СИЛЬНО зависит от количества данных!")
            print(f"   Решение: увеличить LOOKBACK в лайв сканере или пересмотреть фичи")
        else:
            print(f"\n   ✅ Confidence стабилен относительно количества данных")
            print(f"   Проблема низкого confidence в другом месте")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
СТАТИСТИКА ПРЕДСКАЗАНИЙ ЗА 24 ЧАСА

Проверяем:
1. Сколько % свечей предсказаны как LONG/SHORT/SIDEWAYS?
2. Каково распределение confidence?
3. Сколько свечей прошли бы thresholds?

Это покажет РЕАЛЬНУЮ картину модели.
"""

import sys
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_list.json"

# Thresholds
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


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


def main():
    print("="*70)
    print("СТАТИСТИКА ПРЕДСКАЗАНИЙ ЗА 24 ЧАСА")
    print("="*70)
    
    # Load models
    print("\n📦 Загрузка моделей...")
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    feature_names = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:20]]
    
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    now = datetime.now(timezone.utc)
    print(f"📅 Время: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"📊 Анализируем {len(pairs)} пар за последние 24 часа\n")
    
    all_predictions = []
    valid_signals = []
    
    for pair in pairs[:10]:  # Первые 10 пар для скорости
        print(f"   Загружаем {pair}...", end=" ", flush=True)
        
        try:
            m1 = fetch_candles(binance, pair, '1m', 2000)
            m5 = fetch_candles(binance, pair, '5m', 2000)
            m15 = fetch_candles(binance, pair, '15m', 700)
            
            # Build features
            ft = mtf_fe.align_timeframes(m1, m5, m15)
            ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
            
            # Fill missing features
            for f in feature_names:
                if f not in ft.columns:
                    ft[f] = 0.0
            
            # Filter to last 24 hours (288 M5 candles)
            cutoff = now - timedelta(hours=24)
            ft_24h = ft[ft.index >= cutoff]
            
            if len(ft_24h) < 10:
                print(f"недостаточно данных ({len(ft_24h)})")
                continue
            
            # Predict all
            X = ft_24h[feature_names].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            dir_proba = models['direction'].predict_proba(X)
            dir_preds = np.argmax(dir_proba, axis=1)
            dir_confs = np.max(dir_proba, axis=1)
            timings = models['timing'].predict(X)
            strengths = models['strength'].predict(X)
            
            for i, idx in enumerate(ft_24h.index):
                pred = {
                    'pair': pair,
                    'time': idx,
                    'direction': ['SHORT', 'SIDEWAYS', 'LONG'][dir_preds[i]],
                    'dir_pred': dir_preds[i],
                    'conf': dir_confs[i],
                    'timing': timings[i],
                    'strength': strengths[i]
                }
                all_predictions.append(pred)
                
                # Check if passes thresholds
                if dir_preds[i] != 1:  # Not SIDEWAYS
                    if dir_confs[i] >= MIN_CONF and timings[i] >= MIN_TIMING and strengths[i] >= MIN_STRENGTH:
                        valid_signals.append(pred)
            
            print(f"OK ({len(ft_24h)} свечей)")
            
        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue
    
    # Статистика
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*70)
    
    total = len(all_predictions)
    if total == 0:
        print("❌ Нет данных для анализа")
        return
    
    # Распределение направлений
    directions = Counter([p['direction'] for p in all_predictions])
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ (всего {total} свечей):")
    for d in ['LONG', 'SIDEWAYS', 'SHORT']:
        count = directions.get(d, 0)
        pct = count / total * 100
        print(f"   {d:10s}: {count:5d} ({pct:5.1f}%)")
    
    # Распределение confidence для NON-SIDEWAYS
    non_sideways = [p for p in all_predictions if p['dir_pred'] != 1]
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ CONFIDENCE (только LONG/SHORT, всего {len(non_sideways)}):")
    
    if non_sideways:
        confs = [p['conf'] for p in non_sideways]
        conf_bins = [0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
        hist, _ = np.histogram(confs, bins=conf_bins)
        
        for i in range(len(conf_bins)-1):
            pct = hist[i] / len(non_sideways) * 100 if non_sideways else 0
            bar = "█" * int(pct / 2)
            print(f"   {conf_bins[i]:.2f}-{conf_bins[i+1]:.2f}: {hist[i]:4d} ({pct:5.1f}%) {bar}")
    
    # Сколько прошли thresholds
    print(f"\n📊 ВАЛИДНЫЕ СИГНАЛЫ (conf>={MIN_CONF}, timing>={MIN_TIMING}, strength>={MIN_STRENGTH}):")
    print(f"   Всего: {len(valid_signals)} из {len(non_sideways)} LONG/SHORT ({len(valid_signals)/max(1,len(non_sideways))*100:.1f}%)")
    
    if valid_signals:
        print(f"\n   📌 Первые 20 валидных сигналов:")
        for sig in valid_signals[:20]:
            print(f"      {sig['time'].strftime('%m-%d %H:%M')} {sig['pair']:20s} {sig['direction']:6s} "
                  f"C={sig['conf']:.3f} T={sig['timing']:.2f} S={sig['strength']:.2f}")
    
    # Почему не проходят?
    if non_sideways:
        print(f"\n📊 ПРИЧИНЫ ОТКЛОНЕНИЯ СИГНАЛОВ:")
        conf_fail = sum(1 for p in non_sideways if p['conf'] < MIN_CONF)
        timing_fail = sum(1 for p in non_sideways if p['timing'] < MIN_TIMING)
        strength_fail = sum(1 for p in non_sideways if p['strength'] < MIN_STRENGTH)
        
        print(f"   conf < {MIN_CONF}:     {conf_fail:4d} ({conf_fail/len(non_sideways)*100:.1f}%)")
        print(f"   timing < {MIN_TIMING}:   {timing_fail:4d} ({timing_fail/len(non_sideways)*100:.1f}%)")
        print(f"   strength < {MIN_STRENGTH}: {strength_fail:4d} ({strength_fail/len(non_sideways)*100:.1f}%)")
    
    # Выводы
    print("\n" + "="*70)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("="*70)
    
    sideways_pct = directions.get('SIDEWAYS', 0) / total * 100
    
    if sideways_pct > 70:
        print(f"""
   ⚠️  РЫНОК В ГЛУБОКОМ БОКОВИКЕ ({sideways_pct:.1f}% SIDEWAYS)
   
   Это НОРМАЛЬНОЕ поведение модели! Она правильно определяет что:
   - Нет явного тренда
   - Нет хороших точек входа
   - Лучше подождать
   
   РЕКОМЕНДАЦИИ:
   1. НЕ понижать thresholds - это приведёт к плохим трейдам
   2. Добавить более волатильные пары (мемкоины, новые листинги)
   3. Ждать волатильность (обычно приходит с новостями)
   4. Сканировать 24/7 - хорошие моменты редки, но profitable
   
   💡 В бэктесте было ~14 сделок/день на 20 пар
      = 0.7 сделки на пару в день
      = 1 сигнал каждые ~34 часа на пару
        """)
    else:
        print(f"""
   ✅ Рынок более активный ({sideways_pct:.1f}% SIDEWAYS)
   
   Если всё равно мало сигналов, попробуй:
   1. Понизить MIN_CONF до 0.45
   2. Понизить MIN_STRENGTH до 1.2
   3. Проверить конкретные пары где были сигналы
        """)


if __name__ == '__main__':
    main()

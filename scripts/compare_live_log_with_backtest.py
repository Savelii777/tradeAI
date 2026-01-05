#!/usr/bin/env python3
"""
Получить свечи за период из лога лайва и сравнить с бектестом.
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

def get_candles_from_api(pair: str, target_time: datetime, binance: ccxt.Exchange, lookback: int = 1500):
    """Получить свечи через API для конкретного времени."""
    data = {}
    
    for tf in ['1m', '5m', '15m']:
        try:
            # Вычисляем since для получения данных до target_time
            hours_needed = {
                '1m': lookback / 60,
                '5m': lookback * 5 / 60,
                '15m': lookback * 15 / 60
            }
            since_time = target_time - timedelta(hours=hours_needed[tf] + 1)
            since_ms = int(since_time.timestamp() * 1000)
            
            candles = binance.fetch_ohlcv(pair, tf, since=since_ms, limit=lookback)
            
            if not candles:
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index().tail(lookback)
            
            data[tf] = df
        except Exception as e:
            print(f"❌ Ошибка загрузки {tf}: {e}")
            return None
    
    return data

def check_signal(pair: str, candle_time: datetime, data: dict, models: dict, mtf_fe: MTFFeatureEngine):
    """Проверить сигнал для конкретной свечи."""
    # Подготовить фичи (ТОЧНО КАК В ЛАЙВЕ)
    df = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
    df = df.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
    df = add_volume_features(df)
    df['atr'] = calculate_atr(df)
    df = df.dropna()  # Важно! Как в prepare_features
    
    feature_cols = models['features']
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        return None
    
    # В ЛАЙВЕ используется df.iloc[[-2]] - предпоследняя свеча
    # Но нам нужно найти свечу по времени из лога
    # Найдем индекс свечи, которая соответствует времени из лога
    if candle_time not in df.index:
        # Ищем ближайшую
        idx = df.index.get_indexer([candle_time], method='nearest')[0]
        if idx == -1:
            return None
        actual_time = df.index[idx]
        # Проверим, что это правильная свеча (в пределах 5 минут)
        if abs((actual_time - candle_time).total_seconds()) > 300:
            return None
        candle_time = actual_time
    
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
    
    # Проверка фильтров
    passes = False
    if dir_pred != 1:
        passes = (dir_conf >= MIN_CONF) and (timing_pred >= MIN_TIMING) and (strength_pred >= MIN_STRENGTH)
    
    return {
        'time': candle_time,
        'direction': direction_str,
        'conf': dir_conf,
        'timing': timing_pred,
        'strength': strength_pred,
        'passes': passes
    }

def parse_live_log():
    """Парсить лог и извлечь все проверенные свечи."""
    log_file = Path(__file__).parent.parent / "logs" / "live_trading.log"
    
    candles = []
    current_pair = None
    current_candle_time = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Ищем строки с парами
            if 'Checking ' in line and '/USDT:USDT' in line:
                parts = line.split('Checking ')
                if len(parts) > 1:
                    current_pair = parts[1].split('...')[0].strip()
            
            # Ищем строки с временем свечи
            if 'Candle @' in line:
                parts = line.split('Candle @ ')
                if len(parts) > 1:
                    time_str = parts[1].split('+')[0].strip()
                    try:
                        current_candle_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                        current_candle_time = current_candle_time.replace(tzinfo=timezone.utc)
                    except:
                        pass
            
            # Ищем предсказания LONG/SHORT
            if '→ LONG' in line or '→ SHORT' in line:
                if current_pair and current_candle_time:
                    parts = line.split('→ ')
                    if len(parts) > 1:
                        direction = parts[1].split(' |')[0].strip()
                        # Извлечь conf, timing, strength
                        conf = None
                        timing = None
                        strength = None
                        
                        if 'Conf:' in line:
                            conf_part = line.split('Conf: ')[1].split(' |')[0]
                            conf = float(conf_part)
                        if 'Timing:' in line:
                            timing_part = line.split('Timing: ')[1].split(' ATR')[0]
                            timing = float(timing_part)
                        if 'Strength:' in line:
                            strength_part = line.split('Strength: ')[1].split()[0]
                            strength = float(strength_part)
                        
                        candles.append({
                            'pair': current_pair,
                            'time': current_candle_time,
                            'direction': direction,
                            'conf': conf,
                            'timing': timing,
                            'strength': strength
                        })
    
    return candles

if __name__ == "__main__":
    print("="*70)
    print("СРАВНЕНИЕ СВЕЧЕЙ ИЗ ЛОГА ЛАЙВА С БЕКТЕСТОМ")
    print("="*70)
    
    # Парсить лог
    print("\n📖 Парсинг лога...")
    live_candles = parse_live_log()
    print(f"✅ Найдено {len(live_candles)} предсказаний LONG/SHORT в логе")
    
    if not live_candles:
        print("❌ Не найдено предсказаний в логе")
        sys.exit(1)
    
    # Загрузить модели
    models = load_models()
    mtf_fe = MTFFeatureEngine()
    
    # Инициализировать Binance
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Группировать по парам
    pairs_candles = {}
    for candle in live_candles:
        pair = candle['pair']
        if pair not in pairs_candles:
            pairs_candles[pair] = []
        pairs_candles[pair].append(candle)
    
    print(f"\n📊 Проверка {len(pairs_candles)} пар...")
    
    results = []
    
    for pair, candles_list in pairs_candles.items():
        print(f"\n{'='*70}")
        print(f"Пара: {pair} ({len(candles_list)} свечей)")
        print(f"{'='*70}")
        
        # Берем первую свечу для получения данных
        first_candle = candles_list[0]
        target_time = first_candle['time']
        
        print(f"Загрузка данных через API для {target_time}...")
        data = get_candles_from_api(pair, target_time, binance)
        
        if not data:
            print(f"❌ Не удалось загрузить данные")
            continue
        
        print(f"✅ Данные загружены: M1={len(data['1m'])}, M5={len(data['5m'])}, M15={len(data['15m'])}")
        
        # Проверить каждую свечу
        for candle in candles_list:
            print(f"\n  Свеча: {candle['time']}")
            print(f"  Лайв: {candle['direction']} | Conf: {candle['conf']:.3f} | Timing: {candle['timing']:.2f} | Strength: {candle['strength']:.1f}")
            
            result = check_signal(pair, candle['time'], data, models, mtf_fe)
            
            if result:
                print(f"  Бектест: {result['direction']} | Conf: {result['conf']:.3f} | Timing: {result['timing']:.2f} | Strength: {result['strength']:.1f}")
                
                if result['passes']:
                    print(f"  ✅ СИГНАЛ НА БЕКТЕСТЕ!")
                else:
                    print(f"  ❌ Сигнал не прошел фильтры на бектесте")
                
                # Сравнить
                if result['direction'] != candle['direction']:
                    print(f"  ⚠️  РАЗНИЦА: Направление отличается!")
                if abs(result['conf'] - candle['conf']) > 0.05:
                    print(f"  ⚠️  РАЗНИЦА: Уверенность отличается на {abs(result['conf'] - candle['conf']):.3f}")
                
                results.append({
                    'pair': pair,
                    'candle': candle,
                    'backtest': result
                })
            else:
                print(f"  ❌ Не удалось проверить свечу")
    
    # Итоги
    print(f"\n\n{'='*70}")
    print("ИТОГИ")
    print(f"{'='*70}")
    
    signals_on_backtest = sum(1 for r in results if r['backtest']['passes'])
    print(f"Сигналов на бектесте: {signals_on_backtest} из {len(results)}")
    
    direction_diff = sum(1 for r in results if r['backtest']['direction'] != r['candle']['direction'])
    print(f"Различий в направлении: {direction_diff} из {len(results)}")
    
    conf_diff = sum(1 for r in results if abs(r['backtest']['conf'] - r['candle']['conf']) > 0.05)
    print(f"Различий в уверенности (>0.05): {conf_diff} из {len(results)}")
    
    if signals_on_backtest > 0:
        print(f"\n⚠️  НА БЕКТЕСТЕ ЕСТЬ СИГНАЛЫ, КОТОРЫХ НЕТ НА ЛАЙВЕ!")
    elif direction_diff > 0 or conf_diff > 0:
        print(f"\n⚠️  ЕСТЬ РАЗНИЦЫ В ПРЕДСКАЗАНИЯХ МЕЖДУ ЛАЙВОМ И БЕКТЕСТОМ!")
    else:
        print(f"\n✅ Предсказания совпадают, но сигналов нет из-за низкой уверенности.")


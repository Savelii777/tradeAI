#!/usr/bin/env python3
"""
Простой скрипт для проверки модели на лайв данных.
Показывает:
1. Какие данные загружаются
2. Какие фичи получаются
3. Какие предсказания делает модель
4. Почему сигналы не проходят фильтры
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import ccxt
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v8_improved")
LOOKBACK = 1500
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# ============================================================
# ФУНКЦИИ (ТОЧНО КАК В ЛАЙВЕ)
# ============================================================
def add_volume_features(df):
    """Точно как в лайве"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # OBV удален (как исправлено)
    
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    
    return df

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe):
    """Точно как в лайве"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Fill NaN
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        ft = ft.ffill().bfill()
        
        if ft.isna().any().any():
            ft = ft.fillna(0)
        
        return ft
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
def fetch_live_data(pair, binance):
    """Загрузить данные через API (как в лайве)"""
    data = {}
    for tf in ['1m', '5m', '15m']:
        try:
            candles = binance.fetch_ohlcv(pair, tf, limit=LOOKBACK)
            if not candles or len(candles) < 50:
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        except Exception as e:
            logger.error(f"Error fetching {pair} {tf}: {e}")
            return None
    
    return data

# ============================================================
# ПРОВЕРКА МОДЕЛИ
# ============================================================
def test_model_on_pair(pair, models, mtf_fe, binance):
    """Проверить модель на одной паре"""
    print(f"\n{'='*70}")
    print(f"Проверка модели на {pair}")
    print(f"{'='*70}")
    
    # 1. Загрузить данные
    print(f"\n1️⃣ Загрузка данных...")
    data = fetch_live_data(pair, binance)
    if data is None:
        print(f"   ❌ Не удалось загрузить данные")
        return None
    
    print(f"   ✅ 1m: {len(data['1m'])} свечей")
    print(f"   ✅ 5m: {len(data['5m'])} свечей")
    print(f"   ✅ 15m: {len(data['15m'])} свечей")
    print(f"   Последняя свеча 5m: {data['5m'].index[-1]}")
    
    # 2. Подготовить фичи
    print(f"\n2️⃣ Подготовка фичей...")
    df = prepare_features(data, mtf_fe)
    if df is None or len(df) < 2:
        print(f"   ❌ Не удалось создать фичи")
        return None
    
    print(f"   ✅ Фичи созданы: {len(df)} строк, {len(df.columns)} колонок")
    
    # 3. Проверить наличие всех фичей модели
    print(f"\n3️⃣ Проверка фичей модели...")
    missing = [f for f in models['features'] if f not in df.columns]
    if missing:
        print(f"   ⚠️ Отсутствуют фичи: {missing[:10]}")
        return None
    print(f"   ✅ Все {len(models['features'])} фичей присутствуют")
    
    # 4. Взять последнюю закрытую свечу (как в лайве)
    print(f"\n4️⃣ Анализ последней свечи...")
    row = df.iloc[[-2]]  # Предпоследняя (последняя закрытая)
    last_candle_time = row.index[0]
    last_candle_close = row['close'].iloc[0]
    print(f"   Свеча: {last_candle_time}")
    print(f"   Close: {last_candle_close:.6f}")
    
    # 5. Подготовить данные для модели
    X = row[models['features']].values
    if pd.isna(X).any():
        print(f"   ⚠️ Есть NaN в фичах, заполняю нулями")
        X = np.nan_to_num(X)
    
    # 6. Предсказания
    print(f"\n5️⃣ Предсказания модели...")
    dir_proba = models['direction'].predict_proba(X)
    dir_conf = float(np.max(dir_proba))
    dir_pred = int(np.argmax(dir_proba))
    timing_pred = float(models['timing'].predict(X)[0])
    strength_pred = float(models['strength'].predict(X)[0])
    
    direction_map = {0: 'SHORT', 1: 'SIDEWAYS', 2: 'LONG'}
    direction_str = direction_map[dir_pred]
    
    print(f"   Direction: {direction_str}")
    print(f"   Confidence: {dir_conf:.3f}")
    print(f"   Timing: {timing_pred:.3f} ATR")
    print(f"   Strength: {strength_pred:.2f}")
    
    # 7. Проверка фильтров
    print(f"\n6️⃣ Проверка фильтров...")
    print(f"   Пороги: Conf>={MIN_CONF}, Timing>={MIN_TIMING}, Strength>={MIN_STRENGTH}")
    
    rejected_reasons = []
    if dir_pred == 1:
        rejected_reasons.append(f"SIDEWAYS")
    if dir_conf < MIN_CONF:
        rejected_reasons.append(f"Conf({dir_conf:.3f}<{MIN_CONF})")
    if timing_pred < MIN_TIMING:
        rejected_reasons.append(f"Timing({timing_pred:.3f}<{MIN_TIMING})")
    if strength_pred < MIN_STRENGTH:
        rejected_reasons.append(f"Strength({strength_pred:.2f}<{MIN_STRENGTH})")
    
    if rejected_reasons:
        print(f"   ❌ Сигнал отклонен: {', '.join(rejected_reasons)}")
    else:
        print(f"   ✅ Сигнал проходит все фильтры!")
    
    # 8. Статистика по последним N свечам
    print(f"\n7️⃣ Статистика по последним 20 свечам...")
    last_20 = df.tail(20)
    X_all = last_20[models['features']].values
    X_all = np.nan_to_num(X_all)
    
    dir_proba_all = models['direction'].predict_proba(X_all)
    dir_preds_all = np.argmax(dir_proba_all, axis=1)
    
    long_count = np.sum(dir_preds_all == 2)
    short_count = np.sum(dir_preds_all == 0)
    sideways_count = np.sum(dir_preds_all == 1)
    
    print(f"   LONG: {long_count} ({long_count/20*100:.0f}%)")
    print(f"   SHORT: {short_count} ({short_count/20*100:.0f}%)")
    print(f"   SIDEWAYS: {sideways_count} ({sideways_count/20*100:.0f}%)")
    
    # 9. Средние значения предсказаний
    timing_all = models['timing'].predict(X_all)
    strength_all = models['strength'].predict(X_all)
    conf_all = np.max(dir_proba_all, axis=1)
    
    print(f"\n8️⃣ Средние значения (последние 20):")
    print(f"   Avg Confidence: {np.mean(conf_all):.3f}")
    print(f"   Avg Timing: {np.mean(timing_all):.3f} ATR")
    print(f"   Avg Strength: {np.mean(strength_all):.2f}")
    
    return {
        'pair': pair,
        'direction': direction_str,
        'conf': dir_conf,
        'timing': timing_pred,
        'strength': strength_pred,
        'passes': len(rejected_reasons) == 0,
        'rejected_reasons': rejected_reasons,
        'last_20_stats': {
            'long': long_count,
            'short': short_count,
            'sideways': sideways_count,
            'avg_conf': float(np.mean(conf_all)),
            'avg_timing': float(np.mean(timing_all)),
            'avg_strength': float(np.mean(strength_all))
        }
    }

# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTC/USDT:USDT", help="Pair to test")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated list of pairs")
    args = parser.parse_args()
    
    print("="*70)
    print("ПРОВЕРКА МОДЕЛИ НА ЛАЙВ ДАННЫХ")
    print("="*70)
    
    # Загрузить модель
    print(f"\n📦 Загрузка модели из {MODEL_DIR}...")
    try:
        models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
            'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
        }
        print(f"   ✅ Модель загружена: {len(models['features'])} фичей")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        return
    
    # Инициализировать
    mtf_fe = MTFFeatureEngine()
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Определить пары для проверки
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(',')]
    else:
        pairs = [args.pair]
    
    # Проверить каждую пару
    results = []
    for pair in pairs:
        try:
            result = test_model_on_pair(pair, models, mtf_fe, binance)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Ошибка при проверке {pair}: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоговый отчет
    if results:
        print(f"\n{'='*70}")
        print("ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'='*70}")
        
        passes = [r for r in results if r['passes']]
        print(f"\n✅ Сигналов проходит фильтры: {len(passes)}/{len(results)}")
        
        if passes:
            print(f"\nПара(ы) с сигналами:")
            for r in passes:
                print(f"  ✅ {r['pair']}: {r['direction']} (Conf={r['conf']:.3f}, Timing={r['timing']:.2f}, Strength={r['strength']:.2f})")
        
        print(f"\n📊 Статистика по всем парам:")
        all_sideways = sum(r['last_20_stats']['sideways'] for r in results)
        all_long = sum(r['last_20_stats']['long'] for r in results)
        all_short = sum(r['last_20_stats']['short'] for r in results)
        total = all_sideways + all_long + all_short
        
        if total > 0:
            print(f"   LONG: {all_long} ({all_long/total*100:.1f}%)")
            print(f"   SHORT: {all_short} ({all_short/total*100:.1f}%)")
            print(f"   SIDEWAYS: {all_sideways} ({all_sideways/total*100:.1f}%)")
        
        avg_conf = np.mean([r['last_20_stats']['avg_conf'] for r in results])
        avg_timing = np.mean([r['last_20_stats']['avg_timing'] for r in results])
        avg_strength = np.mean([r['last_20_stats']['avg_strength'] for r in results])
        
        print(f"\n   Средние значения:")
        print(f"   Avg Confidence: {avg_conf:.3f}")
        print(f"   Avg Timing: {avg_timing:.3f} ATR")
        print(f"   Avg Strength: {avg_strength:.2f}")
        
        print(f"\n💡 Анализ:")
        if all_sideways / total > 0.8:
            print(f"   ⚠️ Проблема: Модель предсказывает SIDEWAYS в {all_sideways/total*100:.0f}% случаев")
            print(f"   → Возможно, модель не актуальна или рынок в боковом тренде")
        if avg_strength < MIN_STRENGTH:
            print(f"   ⚠️ Проблема: Средний Strength ({avg_strength:.2f}) ниже порога ({MIN_STRENGTH})")
            print(f"   → Модель не видит сильных движений")
        if avg_timing < MIN_TIMING:
            print(f"   ⚠️ Проблема: Средний Timing ({avg_timing:.2f}) ниже порога ({MIN_TIMING})")
            print(f"   → Модель не видит хороших точек входа")

if __name__ == '__main__':
    main()


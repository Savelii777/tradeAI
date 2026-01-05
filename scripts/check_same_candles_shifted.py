#!/usr/bin/env python3
"""
Проверка последних свечей - сравнение с live trading
Использует MTFFeatureEngine с shift(1) для M15 (как в live)
"""
import json
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import sys
import argparse
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")

# Парсинг аргументов
parser = argparse.ArgumentParser()
parser.add_argument("--candles", type=int, default=12, help="Сколько последних 5м свечей проверить (по умолчанию 12 = 1 час)")
args = parser.parse_args()

# Загружаем пары
with open(PAIRS_FILE, 'r') as f:
    pairs_data = json.load(f)['pairs'][:20]
    pairs = [p['symbol'] for p in pairs_data]

# Загружаем модели
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
    'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

mtf_fe = MTFFeatureEngine()

def fetch_klines(symbol, interval, limit=500):
    """Fetch klines directly from Binance Futures API"""
    clean_symbol = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
    
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': clean_symbol,
        'interval': interval,
        'limit': limit
    }
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    if isinstance(data, dict) and 'code' in data:
        raise Exception(f"API Error: {data}")
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)
    
    return df

def add_volume_features(df):
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# Используем mtf_fe.align_timeframes() напрямую - тот же код что в live trading
# (MTFFeatureEngine уже содержит shift(1) для M15)


# Генерируем последние N свечей (5м) динамически
now = datetime.now(timezone.utc)
# Округляем до последней закрытой 5м свечи
current_5m = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
# Берем предыдущую закрытую свечу (не текущую формирующуюся)
last_closed = current_5m - timedelta(minutes=5)

# Генерируем список последних N закрытых свечей
num_candles = args.candles
target_times = [last_closed - timedelta(minutes=5*i) for i in range(num_candles)]
target_times = sorted(target_times)  # От старых к новым

first_candle = target_times[0].strftime('%H:%M')
last_candle = target_times[-1].strftime('%H:%M')

print("="*70)
print("🔍 ПРОВЕРКА ПОСЛЕДНИХ СВЕЧЕЙ (как live trading)")
print(f"   Использует MTFFeatureEngine с shift(1) для M15")
print(f"   Период: {first_candle} - {last_candle} UTC ({num_candles} свечей)")
print(f"   Время сейчас: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print("="*70)

all_signals = []
all_predictions = []

for idx, pair in enumerate(pairs):
    print(f"   [{idx+1}/20] {pair}...", end=" ", flush=True)
    try:
        # Загружаем данные
        data = {}
        for tf in ['1m', '5m', '15m']:
            df = fetch_klines(pair, tf, 500)
            data[tf] = df
            time.sleep(0.15)
        
        m1, m5, m15 = data['1m'], data['5m'], data['15m']
        
        # Используем тот же mtf_fe.align_timeframes() что и live trading
        # (уже содержит shift(1) для M15)
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        ft = ft.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)
        
        # Фильтруем только нужные свечи
        mask = ft.index.isin(target_times)
        target_df = ft[mask]
        
        if len(target_df) == 0:
            print(f"no matching candles")
            continue
        
        X = target_df[models['features']].values
        
        dir_proba = models['direction'].predict_proba(X)
        dir_preds = np.argmax(dir_proba, axis=1)
        dir_confs = np.max(dir_proba, axis=1)
        timing_preds = models['timing'].predict(X)
        strength_preds = models['strength'].predict(X)
        
        pair_signals = 0
        for i in range(len(target_df)):
            pred = {
                'time': target_df.index[i],
                'pair': pair.replace('/USDT:USDT', ''),
                'direction': ['SHORT', 'SIDE', 'LONG'][dir_preds[i]],
                'conf': dir_confs[i],
                'timing': timing_preds[i],
                'strength': strength_preds[i],
            }
            all_predictions.append(pred)
            
            # Проверяем фильтры
            if dir_preds[i] == 1:  # SIDEWAYS
                continue
            if dir_confs[i] < 0.50:
                continue
            if timing_preds[i] < 0.80:
                continue
            if strength_preds[i] < 1.40:
                continue
            
            all_signals.append(pred)
            pair_signals += 1
        
        print(f"✓ {len(target_df)} candles, {pair_signals} signals")
        
    except Exception as e:
        print(f"Error: {str(e)[:60]}")

# Статистика
print("\n" + "="*70)
print(f"📊 РЕЗУЛЬТАТЫ С SHIFT (ПРЕДЫДУЩАЯ M15):")
print("-"*70)
print(f"Всего predictions: {len(all_predictions)}")
print(f"✅ Сигналов прошедших ВСЕ фильтры: {len(all_signals)}")

# Считаем predictions по направлениям
df_pred = pd.DataFrame(all_predictions)
if len(df_pred) > 0:
    dir_counts = df_pred['direction'].value_counts()
    print(f"\nПо направлениям:")
    for d, c in dir_counts.items():
        pct = c / len(df_pred) * 100
        print(f"   {d}: {c} ({pct:.1f}%)")

# Показываем сигналы
if len(all_signals) > 0:
    print(f"\n🎯 НАЙДЕННЫЕ СИГНАЛЫ (Conf>0.5, Timing>0.8, Strength>1.4):")
    print("-"*70)
    for s in all_signals:
        print(f"   {s['time'].strftime('%H:%M')} | {s['pair']:12} | {s['direction']} | "
              f"Conf: {s['conf']:.2f} | Timing: {s['timing']:.2f} | Str: {s['strength']:.1f}")
else:
    print(f"\n❌ Сигналов с фильтрами MIN_CONF=0.5 НЕ найдено!")

# Показываем LONG/SHORT
print(f"\n📋 ВСЕ LONG/SHORT сигналы (без фильтров):")
print("-"*70)
rejected = [p for p in all_predictions if p['direction'] != 'SIDE']
rejected_sorted = sorted(rejected, key=lambda x: x['conf'], reverse=True)
for r in rejected_sorted[:20]:
    reasons = []
    if r['conf'] < 0.50:
        reasons.append(f"Conf({r['conf']:.2f}<0.5)")
    if r['timing'] < 0.80:
        reasons.append(f"Tim({r['timing']:.2f}<0.8)")  
    if r['strength'] < 1.40:
        reasons.append(f"Str({r['strength']:.1f}<1.4)")
    reason_str = ", ".join(reasons) if reasons else "✅ PASSED ALL"
    print(f"   {r['time'].strftime('%H:%M')} | {r['pair']:12} | {r['direction']} | "
          f"Conf: {r['conf']:.2f} | {reason_str}")

print("\n" + "="*70)
print("💡 Этот скрипт использует тот же MTFFeatureEngine что и live trading")
print("   Результаты должны совпадать с live_trading.log")
print("="*70)


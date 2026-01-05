#!/usr/bin/env python3
"""
КРИТИЧЕСКАЯ ПРОВЕРКА: Влияние времени запроса данных на предсказания

Гипотеза: Когда мы загружаем 500 свечей, начальная точка датасета
влияет на rolling indicators, что меняет предсказания для одной и той же свечи.
"""
import json
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")

models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

mtf_fe = MTFFeatureEngine()

LOOKBACK = 1500  # Testing with more data

def fetch_klines_with_end(symbol, interval, limit, end_time_ms):
    """Fetch klines ending at specific time"""
    clean_symbol = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': clean_symbol, 
        'interval': interval, 
        'limit': limit,
        'endTime': end_time_ms
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
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

def prepare_features(m1, m5, m15):
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)
    return ft

# Целевая свеча для анализа - берем час назад
now = datetime.now(timezone.utc)
target_candle = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0) - timedelta(hours=1)

print("="*90)
print("🔍 ПРОВЕРКА ВЛИЯНИЯ ВРЕМЕНИ ЗАПРОСА НА ПРЕДСКАЗАНИЯ")
print("="*90)
print(f"Целевая свеча: {target_candle.strftime('%Y-%m-%d %H:%M')} UTC")
print()

pair = 'ADA/USDT:USDT'

# Эмулируем два сценария:
# 1. "Live" - как будто запрашиваем сразу после закрытия свечи
# 2. "Backtest" - как будто запрашиваем сейчас (час позже)

# Сценарий 1: "Live" - endTime = target_candle + 5 минут
live_end = target_candle + timedelta(minutes=10)
live_end_ms = int(live_end.timestamp() * 1000)

# Сценарий 2: "Backtest" - endTime = now  
backtest_end = now
backtest_end_ms = int(backtest_end.timestamp() * 1000)

print(f"📊 {pair}")
print(f"   Live endTime:     {live_end.strftime('%H:%M')} UTC")
print(f"   Backtest endTime: {backtest_end.strftime('%H:%M')} UTC")
print()

# Загружаем данные для обоих сценариев
print("Загрузка данных (Live)...", end=" ", flush=True)
data_live = {}
for tf in ['1m', '5m', '15m']:
    data_live[tf] = fetch_klines_with_end(pair, tf, LOOKBACK, live_end_ms)
print("✓")

print("Загрузка данных (Backtest)...", end=" ", flush=True)
data_backtest = {}
for tf in ['1m', '5m', '15m']:
    data_backtest[tf] = fetch_klines_with_end(pair, tf, LOOKBACK, backtest_end_ms)
print("✓")

# Подготовка фичей
print("Подготовка фичей...", end=" ", flush=True)
ft_live = prepare_features(data_live['1m'], data_live['5m'], data_live['15m'])
ft_backtest = prepare_features(data_backtest['1m'], data_backtest['5m'], data_backtest['15m'])
print("✓")

print()
print(f"Live DataFrame: {len(ft_live)} rows, last={ft_live.index[-1].strftime('%H:%M')}")
print(f"Backtest DataFrame: {len(ft_backtest)} rows, last={ft_backtest.index[-1].strftime('%H:%M')}")

# Проверяем есть ли целевая свеча в обоих
if target_candle not in ft_live.index:
    print(f"❌ Целевая свеча {target_candle} не найдена в Live данных!")
    print(f"   Доступные: {ft_live.index[0]} ... {ft_live.index[-1]}")
else:
    print(f"✅ Целевая свеча найдена в Live")

if target_candle not in ft_backtest.index:
    print(f"❌ Целевая свеча {target_candle} не найдена в Backtest данных!")
    print(f"   Доступные: {ft_backtest.index[0]} ... {ft_backtest.index[-1]}")
else:
    print(f"✅ Целевая свеча найдена в Backtest")

# Если обе найдены - сравниваем
if target_candle in ft_live.index and target_candle in ft_backtest.index:
    print()
    print("="*90)
    print(f"📊 СРАВНЕНИЕ ФИЧ ДЛЯ {target_candle.strftime('%H:%M')} UTC")
    print("="*90)
    
    row_live = ft_live.loc[[target_candle]]
    row_backtest = ft_backtest.loc[[target_candle]]
    
    # Базовые данные
    print(f"\nБазовые данные:")
    print(f"   Live Close:     {row_live['close'].values[0]:.6f}")
    print(f"   Backtest Close: {row_backtest['close'].values[0]:.6f}")
    print(f"   Match: {'✅' if row_live['close'].values[0] == row_backtest['close'].values[0] else '❌'}")
    
    # Предсказания
    X_live = row_live[models['features']].values.astype(float)
    X_backtest = row_backtest[models['features']].values.astype(float)
    
    pred_live = models['direction'].predict_proba(X_live)
    pred_backtest = models['direction'].predict_proba(X_backtest)
    
    dir_live = ['SHORT', 'SIDE', 'LONG'][np.argmax(pred_live)]
    dir_backtest = ['SHORT', 'SIDE', 'LONG'][np.argmax(pred_backtest)]
    
    conf_live = np.max(pred_live)
    conf_backtest = np.max(pred_backtest)
    
    print(f"\n🎯 ПРЕДСКАЗАНИЯ:")
    print(f"   Live:     {dir_live} Conf: {conf_live:.4f}")
    print(f"   Backtest: {dir_backtest} Conf: {conf_backtest:.4f}")
    
    if dir_live == dir_backtest:
        print(f"   ✅ Направления СОВПАДАЮТ!")
    else:
        print(f"   ❌ НАПРАВЛЕНИЯ РАЗНЫЕ! Это BUG!")
    
    # Сравнение фичей
    print(f"\n📊 СРАВНЕНИЕ ФИЧ (топ-10 с наибольшей разницей):")
    diffs = []
    for feat in models['features']:
        v_live = float(row_live[feat].values[0])
        v_back = float(row_backtest[feat].values[0])
        if v_live != 0:
            diff_pct = abs(v_live - v_back) / abs(v_live) * 100
        else:
            diff_pct = abs(v_live - v_back) * 100
        diffs.append({'feature': feat, 'live': v_live, 'backtest': v_back, 'diff': diff_pct})
    
    diffs.sort(key=lambda x: x['diff'], reverse=True)
    
    print(f"\n{'Feature':<40} | {'Live':<15} | {'Backtest':<15} | {'Diff %':<10}")
    print("-"*90)
    for d in diffs[:10]:
        print(f"{d['feature']:<40} | {d['live']:<15.6f} | {d['backtest']:<15.6f} | {d['diff']:.2f}%")
    
    # Итог
    big_diffs = [d for d in diffs if d['diff'] > 1]
    print(f"\n📊 ИТОГ:")
    print(f"   Фичей с разницей >1%: {len(big_diffs)}")
    if len(big_diffs) > 0:
        print(f"   ⚠️ ВНИМАНИЕ: Фичи отличаются! Это объясняет разные предсказания!")
    else:
        print(f"   ✅ Фичи практически идентичны")


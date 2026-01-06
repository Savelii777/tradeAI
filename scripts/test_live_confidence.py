#!/usr/bin/env python3
"""Тест confidence на live данных после переобучения."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import joblib
import ccxt
from train_mtf import MTFFeatureEngine

# Загружаем модели
dir_model = joblib.load('../models/v8_improved/direction_model.joblib')
feature_cols = joblib.load('../models/v8_improved/feature_names.joblib')

print('=== LIVE CONFIDENCE TEST (после переобучения) ===')
print(f'Features: {len(feature_cols)}')

# Получаем live данные BTC
exchange = ccxt.binance()
ohlcv_1m = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=500)
ohlcv_5m = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=300)
ohlcv_15m = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=200)

df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp','open','high','low','close','volume'])
df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp','open','high','low','close','volume'])
df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp','open','high','low','close','volume'])

for df in [df_1m, df_5m, df_15m]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

engine = MTFFeatureEngine()
features = engine.align_timeframes(df_1m, df_5m, df_15m)
features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Добавить отсутствующие фичи со значением 0
missing = [c for c in feature_cols if c not in features.columns]
for m in missing:
    features[m] = 0.0
print(f'Missing features (set to 0): {missing}')
    
# Проверяем доступность фич
print(f'Available after fix: {len([c for c in feature_cols if c in features.columns])}/{len(feature_cols)}')

# Предсказания (в правильном порядке!)
X = features[feature_cols].tail(100)
proba = dir_model.predict_proba(X)
conf = np.max(proba, axis=1)
preds = dir_model.predict(X)

print(f'\n=== CONFIDENCE STATS (last 100 candles) ===')
print(f'Min:  {conf.min():.3f}')
print(f'Max:  {conf.max():.3f}')
print(f'Mean: {conf.mean():.3f}')
print(f'Conf >= 0.5: {(conf >= 0.5).sum()}/{len(conf)} ({(conf >= 0.5).mean()*100:.1f}%)')
print(f'Conf >= 0.6: {(conf >= 0.6).sum()}/{len(conf)} ({(conf >= 0.6).mean()*100:.1f}%)')

# Распределение по направлениям
print(f'\n=== PREDICTIONS ===')
for p in [0, 1, 2]:
    mask = preds == p
    label = ['LONG', 'SHORT', 'SIDEWAYS'][p]
    if mask.sum() > 0:
        print(f'{label}: {mask.sum()} ({conf[mask].mean():.3f} avg conf)')

# СРАВНЕНИЕ: раньше было max 0.43, теперь?
print(f'\n✅ СРАВНЕНИЕ С ПРОШЛЫМ ТЕСТОМ:')
print(f'   Было (до переобучения): max conf = 0.43, mean = 0.37')
print(f'   Сейчас:                 max conf = {conf.max():.3f}, mean = {conf.mean():.3f}')

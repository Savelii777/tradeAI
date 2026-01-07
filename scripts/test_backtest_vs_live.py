#!/usr/bin/env python3
"""
TEST #2: Backtest vs Live - Same Period Comparison
TEST #3: Feature Distribution Check

Проверяем что модель дает одинаковые результаты на бектест данных и live.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import joblib
from train_mtf import MTFFeatureEngine
from datetime import datetime, timedelta

def add_volume_features(df):
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

print('='*70)
print('TEST #2: BACKTEST vs LIVE - SAME PERIOD')
print('='*70)

# Загружаем модель
dir_model = joblib.load('../models/v8_improved/direction_model.joblib')
timing_model = joblib.load('../models/v8_improved/timing_model.joblib')
strength_model = joblib.load('../models/v8_improved/strength_model.joblib')
feature_cols = joblib.load('../models/v8_improved/feature_names.joblib')

# Выбираем пару которая была активна в бектесте
pair = 'ASTER_USDT'
print(f'Pair: {pair}')

# Загружаем данные из CSV (бектест данные)
df_1m = pd.read_csv(f'../data/candles/{pair}_USDT_1m.csv', parse_dates=['timestamp'], index_col='timestamp')
df_5m = pd.read_csv(f'../data/candles/{pair}_USDT_5m.csv', parse_dates=['timestamp'], index_col='timestamp')
df_15m = pd.read_csv(f'../data/candles/{pair}_USDT_15m.csv', parse_dates=['timestamp'], index_col='timestamp')

print(f'Data range: {df_5m.index.min()} to {df_5m.index.max()}')

# Берем последние 48 часов (576 свечей M5)
end_time = df_5m.index.max()
start_time = end_time - timedelta(hours=48)
print(f'Test period: {start_time} to {end_time}')

# Фильтруем данные (с запасом для расчета индикаторов)
warmup_start = start_time - timedelta(hours=24)  # 24 часа на прогрев
m1_period = df_1m[df_1m.index >= warmup_start]
m5_period = df_5m[df_5m.index >= warmup_start]
m15_period = df_15m[df_15m.index >= warmup_start]

print(f'M1: {len(m1_period)} candles, M5: {len(m5_period)} candles, M15: {len(m15_period)} candles')

# Генерируем фичи (как в бектесте)
engine = MTFFeatureEngine()
features = engine.align_timeframes(m1_period, m5_period, m15_period)
features = features.join(m5_period[['open', 'high', 'low', 'close', 'volume']])
features = add_volume_features(features)
features['atr'] = calculate_atr(features)
features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Фильтруем только тестовый период
features = features[features.index >= start_time]
print(f'Features for test period: {len(features)} rows')

# Предсказания
X = features[feature_cols]
proba = dir_model.predict_proba(X)
conf = np.max(proba, axis=1)
preds = dir_model.predict(X)

# Timing и Strength
timing_pred = timing_model.predict(X)
strength_pred = strength_model.predict(X)

# Собираем результаты
results = pd.DataFrame({
    'timestamp': features.index,
    'close': features['close'].values,
    'direction': preds,
    'confidence': conf,
    'timing': timing_pred,
    'strength': strength_pred
})

# Статистика
print(f'\n=== SIGNAL STATS (48h) ===')
labels = {0: 'LONG', 1: 'SHORT', 2: 'SIDEWAYS'}
for p in [0, 1, 2]:
    mask = results['direction'] == p
    cnt = mask.sum()
    avg_conf = results.loc[mask, 'confidence'].mean() if cnt > 0 else 0
    print(f'{labels[p]:8}: {cnt:3} signals ({cnt/len(results)*100:.1f}%) | Avg conf: {avg_conf:.3f}')

# Фильтруем сигналы по порогам (как в live)
MIN_CONF = 0.5
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

valid_signals = results[
    (results['direction'] != 2) &  # Not SIDEWAYS
    (results['confidence'] >= MIN_CONF) &
    (results['timing'] >= MIN_TIMING) &
    (results['strength'] >= MIN_STRENGTH)
]

print(f'\n=== VALID SIGNALS (Conf>={MIN_CONF}, Timing>={MIN_TIMING}, Strength>={MIN_STRENGTH}) ===')
print(f'Total valid: {len(valid_signals)} out of {len(results)} ({len(valid_signals)/len(results)*100:.1f}%)')

if len(valid_signals) > 0:
    print(f'\nLast 15 valid signals:')
    for _, row in valid_signals.tail(15).iterrows():
        direction = labels[row['direction']]
        print(f"  {row['timestamp']} | {direction:5} | Conf: {row['confidence']:.2f} | Tim: {row['timing']:.2f} | Str: {row['strength']:.2f} | Close: {row['close']:.4f}")

# Сохраняем для сравнения
results.to_csv('../results/backtest_signals_48h.csv', index=False)
print(f'\nSignals saved to results/backtest_signals_48h.csv')

# ============================================================
# TEST #3: FEATURE DISTRIBUTION CHECK
# ============================================================
print('\n' + '='*70)
print('TEST #3: FEATURE DISTRIBUTION CHECK')
print('='*70)

# Сравниваем распределения фич на бектесте vs последние 48 часов
# Загружаем ВСЕ данные для бектеста (как при обучении)
print('\nLoading full backtest data...')
full_1m = pd.read_csv(f'../data/candles/{pair}_USDT_1m.csv', parse_dates=['timestamp'], index_col='timestamp')
full_5m = pd.read_csv(f'../data/candles/{pair}_USDT_5m.csv', parse_dates=['timestamp'], index_col='timestamp')
full_15m = pd.read_csv(f'../data/candles/{pair}_USDT_15m.csv', parse_dates=['timestamp'], index_col='timestamp')

# Генерируем фичи для полного периода (последние 30 дней)
full_start = full_5m.index.max() - timedelta(days=30)
full_1m = full_1m[full_1m.index >= full_start]
full_5m = full_5m[full_5m.index >= full_start]
full_15m = full_15m[full_15m.index >= full_start]

full_features = engine.align_timeframes(full_1m, full_5m, full_15m)
full_features = full_features.join(full_5m[['open', 'high', 'low', 'close', 'volume']])
full_features = add_volume_features(full_features)
full_features['atr'] = calculate_atr(full_features)
full_features = full_features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

print(f'Full backtest features: {len(full_features)} rows')
print(f'Last 48h features: {len(features)} rows')

# Сравниваем статистики ключевых фич
key_features = ['m5_rsi_14', 'm5_momentum_10', 'm15_trend_strength', 'm1_rsi_5_last', 'vol_ratio', 'atr']
available_key = [f for f in key_features if f in feature_cols]

print(f'\n=== KEY FEATURE COMPARISON (Backtest 30d vs Last 48h) ===')
print(f'{"Feature":<25} | {"BT Mean":>10} | {"48h Mean":>10} | {"Diff %":>8} | {"Status":<8}')
print('-'*75)

all_ok = True
for feat in available_key:
    bt_mean = full_features[feat].mean()
    live_mean = features[feat].mean()
    diff_pct = abs(bt_mean - live_mean) / (abs(bt_mean) + 1e-10) * 100
    
    status = '✅ OK' if diff_pct < 30 else '⚠️ DIFF' if diff_pct < 50 else '❌ BAD'
    if diff_pct >= 30:
        all_ok = False
    
    print(f'{feat:<25} | {bt_mean:>10.4f} | {live_mean:>10.4f} | {diff_pct:>7.1f}% | {status}')

# Проверяем распределение confidence
print(f'\n=== CONFIDENCE DISTRIBUTION ===')
bt_X = full_features[feature_cols]
bt_proba = dir_model.predict_proba(bt_X)
bt_conf = np.max(bt_proba, axis=1)

print(f'Backtest 30d: Mean={bt_conf.mean():.3f}, Std={bt_conf.std():.3f}, >=0.5: {(bt_conf >= 0.5).mean()*100:.1f}%')
print(f'Last 48h:     Mean={conf.mean():.3f}, Std={conf.std():.3f}, >=0.5: {(conf >= 0.5).mean()*100:.1f}%')

conf_diff = abs(bt_conf.mean() - conf.mean())
if conf_diff < 0.1:
    print(f'\n✅ Confidence distributions are SIMILAR (diff={conf_diff:.3f})')
else:
    print(f'\n⚠️ Confidence distributions DIFFER (diff={conf_diff:.3f})')

# Финальный вердикт
print('\n' + '='*70)
print('FINAL VERDICT')
print('='*70)
if all_ok and conf_diff < 0.1:
    print('✅ MODEL IS CONSISTENT - Ready for live trading!')
else:
    print('⚠️ Some differences detected - Review before live trading')

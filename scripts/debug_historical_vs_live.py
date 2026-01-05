#!/usr/bin/env python3
"""
КРИТИЧЕСКАЯ ДИАГНОСТИКА: Почему исторические и live предсказания отличаются?

Гипотеза: shift(1) для M15 работает по-разному для:
1. "Текущей" свечи (только что закрылась)
2. "Исторической" свечи (закрылась давно)
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
PAIRS_FILE = Path("config/pairs_list.json")

# Load models
models = {
    'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
    'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
    'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
}

mtf_fe = MTFFeatureEngine()

def fetch_klines(symbol, interval, limit=500):
    clean_symbol = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {'symbol': clean_symbol, 'interval': interval, 'limit': limit}
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

# Load pairs
with open(PAIRS_FILE, 'r') as f:
    pairs = [p['symbol'] for p in json.load(f)['pairs'][:20]]

print("="*90)
print("🔍 АНАЛИЗ: Как M15 shift влияет на предсказания")
print("="*90)

now = datetime.now(timezone.utc)
current_5m = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
last_closed = current_5m - timedelta(minutes=5)

print(f"\nТекущее время UTC: {now.strftime('%H:%M:%S')}")
print(f"Последняя закрытая M5 свеча: {last_closed.strftime('%H:%M')}")

# Определяем какая M15 свеча используется
current_m15_start = last_closed.replace(minute=(last_closed.minute // 15) * 15)
prev_m15_start = current_m15_start - timedelta(minutes=15)

print(f"\nДля M5 свечи {last_closed.strftime('%H:%M')}:")
print(f"   Текущая M15 свеча: {current_m15_start.strftime('%H:%M')} (ещё формируется или только закрылась)")
print(f"   После shift(1): используется M15 {prev_m15_start.strftime('%H:%M')}")

# Тестируем несколько пар
test_pairs = ['PIPPIN/USDT:USDT', '1000PEPE/USDT:USDT', 'HYPE/USDT:USDT', 'ADA/USDT:USDT', 'TAO/USDT:USDT']

print("\n" + "="*90)
print("📊 СРАВНЕНИЕ: Последняя свеча (iloc[-2]) vs Исторические свечи")  
print("="*90)

for pair in test_pairs:
    print(f"\n{'='*90}")
    print(f"📊 {pair}")
    print("="*90)
    
    try:
        # Fetch data
        data = {}
        for tf in ['1m', '5m', '15m']:
            data[tf] = fetch_klines(pair, tf, 500)
        
        ft = prepare_features(data['1m'], data['5m'], data['15m'])
        
        print(f"\n   Всего строк после prepare_features: {len(ft)}")
        print(f"   Последние 5 индексов: {[t.strftime('%H:%M') for t in ft.index[-5:]]}")
        
        # МЕТОД 1: Live style - iloc[-2]
        row_live = ft.iloc[[-2]]
        live_time = row_live.index[0]
        X_live = row_live[models['features']].values
        
        dir_proba_live = models['direction'].predict_proba(X_live)
        dir_pred_live = int(np.argmax(dir_proba_live))
        dir_conf_live = float(np.max(dir_proba_live))
        timing_live = float(models['timing'].predict(X_live)[0])
        strength_live = float(models['strength'].predict(X_live)[0])
        dir_str_live = ['SHORT', 'SIDE', 'LONG'][dir_pred_live]
        
        # МЕТОД 2: Backtest style - несколько исторических свечей
        historical_times = [ft.index[-2] - timedelta(minutes=5*i) for i in range(1, 13)]
        historical_times = [t for t in historical_times if t in ft.index]
        
        print(f"\n   {'Свеча':<12} | {'Направление':<10} | {'Conf':<8} | {'Timing':<8} | {'Strength':<8} | Close")
        print(f"   {'-'*80}")
        
        # Показываем последнюю (live style)
        close_live = row_live['close'].values[0]
        print(f"   {live_time.strftime('%H:%M'):<12} | {dir_str_live:<10} | {dir_conf_live:<8.3f} | {timing_live:<8.2f} | {strength_live:<8.2f} | {close_live:.6f} ← iloc[-2]")
        
        # Показываем исторические
        for hist_time in historical_times[:8]:
            row_hist = ft.loc[[hist_time]]
            X_hist = row_hist[models['features']].values
            
            dir_proba_hist = models['direction'].predict_proba(X_hist)
            dir_pred_hist = int(np.argmax(dir_proba_hist))
            dir_conf_hist = float(np.max(dir_proba_hist))
            timing_hist = float(models['timing'].predict(X_hist)[0])
            strength_hist = float(models['strength'].predict(X_hist)[0])
            dir_str_hist = ['SHORT', 'SIDE', 'LONG'][dir_pred_hist]
            close_hist = row_hist['close'].values[0]
            
            # Отмечаем сигналы которые проходят фильтры
            passes_filters = dir_pred_hist != 1 and dir_conf_hist >= 0.50 and timing_hist >= 0.8 and strength_hist >= 1.4
            marker = "✅ SIGNAL" if passes_filters else ""
            
            print(f"   {hist_time.strftime('%H:%M'):<12} | {dir_str_hist:<10} | {dir_conf_hist:<8.3f} | {timing_hist:<8.2f} | {strength_hist:<8.2f} | {close_hist:.6f} {marker}")
        
    except Exception as e:
        print(f"   ERROR: {e}")

print("\n" + "="*90)
print("💡 ВЫВОД:")
print("   Если направления меняются от свечи к свече - это НОРМАЛЬНО (рынок меняется)")
print("   Проблема если iloc[-2] и ft.loc[same_time] дают РАЗНЫЕ результаты")
print("="*90)

# Финальная проверка: iloc[-2] vs loc[same_timestamp]
print("\n" + "="*90)
print("🔍 КРИТИЧЕСКАЯ ПРОВЕРКА: iloc[-2] vs loc[same_time]")
print("="*90)

for pair in test_pairs[:3]:
    print(f"\n{pair}:")
    data = {}
    for tf in ['1m', '5m', '15m']:
        data[tf] = fetch_klines(pair, tf, 500)
    ft = prepare_features(data['1m'], data['5m'], data['15m'])
    
    # iloc[-2]
    row_iloc = ft.iloc[[-2]]
    time_iloc = row_iloc.index[0]
    
    # loc[same_time]
    row_loc = ft.loc[[time_iloc]]
    
    # Compare
    X_iloc = row_iloc[models['features']].values
    X_loc = row_loc[models['features']].values
    
    features_match = np.allclose(X_iloc, X_loc, equal_nan=True)
    
    pred_iloc = models['direction'].predict_proba(X_iloc)
    pred_loc = models['direction'].predict_proba(X_loc)
    
    preds_match = np.allclose(pred_iloc, pred_loc)
    
    print(f"   Time: {time_iloc.strftime('%H:%M')}")
    print(f"   Features match: {'✅' if features_match else '❌'}")
    print(f"   Predictions match: {'✅' if preds_match else '❌'}")
    
    if not features_match:
        diff_count = np.sum(~np.isclose(X_iloc, X_loc, equal_nan=True))
        print(f"   ⚠️ {diff_count} features differ!")


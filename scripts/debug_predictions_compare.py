#!/usr/bin/env python3
"""
ГЛУБОКАЯ ДИАГНОСТИКА: Сравнение предсказаний live vs backtest для ОДНОЙ свечи
Находит где именно расходятся данные/предсказания
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

# Load pairs
with open(PAIRS_FILE, 'r') as f:
    pairs_data = json.load(f)['pairs'][:20]
    pairs = [p['symbol'] for p in pairs_data]

mtf_fe = MTFFeatureEngine()

def fetch_klines(symbol, interval, limit=500):
    """Fetch klines directly from Binance Futures API"""
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

def prepare_features_backtest_style(m1, m5, m15):
    """Backtest style: mask = ft.index.isin(target_times)"""
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)
    return ft

print("="*90)
print("🔍 ГЛУБОКАЯ ДИАГНОСТИКА: Сравнение предсказаний")
print("="*90)

# Определяем целевую свечу (последняя закрытая)
now = datetime.now(timezone.utc)
current_5m = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
target_candle = current_5m - timedelta(minutes=5)

print(f"Текущее время: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"Анализируемая свеча: {target_candle.strftime('%H:%M:%S')} UTC")
print("="*90)

# Сравним несколько последних свечей
target_candles = [target_candle - timedelta(minutes=5*i) for i in range(6)]
target_candles = sorted(target_candles)

print(f"\nАнализируем свечи: {[t.strftime('%H:%M') for t in target_candles]}")
print("-"*90)

# Тестируем на первых 5 парах
test_pairs = pairs[:5]

for pair in test_pairs:
    print(f"\n{'='*90}")
    print(f"📊 {pair}")
    print("="*90)
    
    try:
        # Fetch data
        data = {}
        for tf in ['1m', '5m', '15m']:
            df = fetch_klines(pair, tf, 500)
            data[tf] = df
        
        m1, m5, m15 = data['1m'], data['5m'], data['15m']
        
        # Prepare features (backtest style)
        ft = prepare_features_backtest_style(m1, m5, m15)
        
        print(f"\n   DataFrame размер: {len(ft)} строк")
        print(f"   Последние 3 индекса: {list(ft.index[-3:])}")
        
        # Метод 1: BACKTEST style - по индексу
        print(f"\n   {'Свеча':<12} | {'BACKTEST (по индексу)':<35} | {'LIVE (iloc[-X])':<35}")
        print(f"   {'-'*12}-+-{'-'*35}-+-{'-'*35}")
        
        for i, target in enumerate(target_candles):
            # Backtest style: select by timestamp
            if target in ft.index:
                row_bt = ft.loc[[target]]
                X_bt = row_bt[models['features']].values
                dir_proba_bt = models['direction'].predict_proba(X_bt)
                dir_pred_bt = int(np.argmax(dir_proba_bt))
                dir_conf_bt = float(np.max(dir_proba_bt))
                dir_str_bt = ['SHORT', 'SIDE', 'LONG'][dir_pred_bt]
                bt_result = f"{dir_str_bt} Conf:{dir_conf_bt:.3f}"
            else:
                bt_result = "NOT FOUND"
            
            # Live style: select by position
            # iloc[-1] = current (forming), iloc[-2] = last closed, etc.
            position = -(len(target_candles) - i)  # -6, -5, -4, -3, -2, -1
            if abs(position) <= len(ft):
                row_live = ft.iloc[[position]]
                actual_time = row_live.index[0]
                X_live = row_live[models['features']].values
                dir_proba_live = models['direction'].predict_proba(X_live)
                dir_pred_live = int(np.argmax(dir_proba_live))
                dir_conf_live = float(np.max(dir_proba_live))
                dir_str_live = ['SHORT', 'SIDE', 'LONG'][dir_pred_live]
                live_result = f"{dir_str_live} Conf:{dir_conf_live:.3f} @{actual_time.strftime('%H:%M')}"
            else:
                live_result = "OUT OF RANGE"
            
            # Check match
            match = "✅" if bt_result.split()[0] == live_result.split()[0] else "❌ MISMATCH!"
            print(f"   {target.strftime('%H:%M'):<12} | {bt_result:<35} | {live_result:<35} {match}")
        
        # Дополнительная проверка: iloc[-2] vs ожидаемое время
        print(f"\n   🔍 Детальная проверка iloc[-2]:")
        row_iloc2 = ft.iloc[[-2]]
        expected_time = target_candles[-2]  # Предпоследняя из списка = последняя закрытая
        actual_time = row_iloc2.index[0]
        print(f"      Ожидаемое время: {expected_time.strftime('%H:%M')}")
        print(f"      Фактическое время iloc[-2]: {actual_time.strftime('%H:%M')}")
        print(f"      Совпадает: {'✅' if expected_time == actual_time else '❌ ПРОБЛЕМА!'}")
        
    except Exception as e:
        print(f"   ERROR: {e}")

print("\n" + "="*90)
print("💡 ИТОГ:")
print("   Если все строки показывают ✅ - проблема НЕ в выборе свечей")
print("   Если есть ❌ MISMATCH - найден источник расхождения")
print("="*90)


#!/usr/bin/env python3
"""
Диагностика: сравнение выбора свечей между backtest и live trading
"""
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

PAIRS_FILE = Path("config/pairs_list.json")

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

# Load pairs
with open(PAIRS_FILE, 'r') as f:
    pairs_data = json.load(f)['pairs'][:20]
    pairs = [p['symbol'] for p in pairs_data]

mtf_fe = MTFFeatureEngine()

print("="*80)
print("🔍 ДИАГНОСТИКА: Сравнение выбора свечей")
print("="*80)

now = datetime.now(timezone.utc)
current_5m = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
expected_closed_candle = current_5m - timedelta(minutes=5)

print(f"Текущее время: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"Ожидаемая последняя закрытая свеча: {expected_closed_candle.strftime('%H:%M:%S')} UTC")
print("-"*80)
print(f"{'Пара':<20} {'Всего строк':<12} {'iloc[-1]':<20} {'iloc[-2]':<20} {'Отклонение'}")
print("-"*80)

misaligned = []

for pair in pairs:
    try:
        # Fetch data
        data = {}
        for tf in ['1m', '5m', '15m']:
            df = fetch_klines(pair, tf, 500)
            data[tf] = df
        
        m1, m5, m15 = data['1m'], data['5m'], data['15m']
        
        # Align timeframes (как в live trading)
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Critical: dropna как в live
        before_drop = len(ft)
        ft = ft.dropna(subset=['close', 'atr']).ffill().bfill().fillna(0)
        after_drop = len(ft)
        
        # Live trading использует iloc[-2]
        live_last = ft.index[-1] if len(ft) > 0 else None
        live_second_last = ft.index[-2] if len(ft) > 1 else None
        
        # Проверяем отклонение
        status = "✅"
        if live_second_last:
            diff_minutes = abs((expected_closed_candle - live_second_last).total_seconds() / 60)
            if diff_minutes > 0:
                status = f"⚠️ {diff_minutes:.0f}m"
                misaligned.append({
                    'pair': pair,
                    'expected': expected_closed_candle.strftime('%H:%M'),
                    'actual': live_second_last.strftime('%H:%M'),
                    'diff': diff_minutes
                })
        
        print(f"{pair:<20} {after_drop:<12} {live_last.strftime('%H:%M') if live_last else 'N/A':<20} "
              f"{live_second_last.strftime('%H:%M') if live_second_last else 'N/A':<20} {status}")
        
    except Exception as e:
        print(f"{pair:<20} ERROR: {str(e)[:50]}")

print("="*80)
if misaligned:
    print(f"🔴 ПРОБЛЕМА: {len(misaligned)} пар имеют неправильное выравнивание!")
    print("\nДетали:")
    for m in misaligned:
        print(f"   {m['pair']}: ожидалось {m['expected']}, получено {m['actual']} (разница {m['diff']:.0f}m)")
    
    print("\n💡 РЕШЕНИЕ: Live trading должен использовать timestamp selection вместо iloc[-2]:")
    print("   expected_time = current_5m - timedelta(minutes=5)")
    print("   row = df.loc[[expected_time]] if expected_time in df.index else df.iloc[[-2]]")
else:
    print("✅ Все пары имеют правильное выравнивание!")


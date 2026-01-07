#!/usr/bin/env python3
"""
КРИТИЧЕСКИЙ ТЕСТ: Проверяем совпадение свечей между CSV и API Binance

Гипотеза: Данные из Binance API могут отличаться от CSV (разные индексы, сдвиги времени и т.д.)
"""

import sys
import pandas as pd
import numpy as np
import ccxt
import time
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"


def compare_candles():
    print("=" * 70)
    print("СРАВНЕНИЕ CSV vs BINANCE API")
    print("=" * 70)
    
    # Init Binance
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    
    pair = "BTC/USDT:USDT"
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    # Load CSV
    csv_m5 = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    # Ensure UTC timezone
    if csv_m5.index.tz is None:
        csv_m5.index = csv_m5.index.tz_localize('UTC')
    
    print(f"\n📁 CSV M5: {len(csv_m5)} свечей")
    print(f"   Диапазон: {csv_m5.index[0]} → {csv_m5.index[-1]}")
    
    # Fetch latest from Binance
    print(f"\n🌐 Загрузка с Binance API...")
    candles = binance.fetch_ohlcv(pair, '5m', limit=100)
    
    api_m5 = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    api_m5['timestamp'] = pd.to_datetime(api_m5['timestamp'], unit='ms', utc=True)
    api_m5.set_index('timestamp', inplace=True)
    
    print(f"📡 API M5: {len(api_m5)} свечей")
    print(f"   Диапазон: {api_m5.index[0]} → {api_m5.index[-1]}")
    
    # Find overlapping timestamps
    common_times = csv_m5.index.intersection(api_m5.index)
    print(f"\n🔗 Общих временных меток: {len(common_times)}")
    
    if len(common_times) == 0:
        print("⚠️  НЕТ ПЕРЕСЕЧЕНИЙ! CSV устарел.")
        print(f"   Последняя свеча CSV: {csv_m5.index[-1]}")
        print(f"   Первая свеча API: {api_m5.index[0]}")
        return
    
    # Compare OHLCV for common timestamps
    print(f"\n📊 СРАВНЕНИЕ OHLCV:")
    print("-" * 90)
    print(f"{'Timestamp':<25} {'CSV Close':>12} {'API Close':>12} {'Diff':>10} {'Match':>8}")
    print("-" * 90)
    
    mismatches = 0
    for ts in common_times[-20:]:  # Last 20 common candles
        csv_close = csv_m5.loc[ts, 'close']
        api_close = api_m5.loc[ts, 'close']
        diff = abs(csv_close - api_close)
        match = "✅" if diff < 0.01 else "❌"
        
        if diff >= 0.01:
            mismatches += 1
        
        print(f"{str(ts):<25} {csv_close:>12.2f} {api_close:>12.2f} {diff:>10.2f} {match:>8}")
    
    print("-" * 90)
    
    if mismatches > 0:
        print(f"\n🔥 НАЙДЕНО {mismatches} НЕСОВПАДЕНИЙ!")
        print("   Это может быть причиной разницы в предсказаниях!")
    else:
        print(f"\n✅ Все свечи совпадают идеально")
    
    # Check timezone handling
    print(f"\n⏰ ПРОВЕРКА ЧАСОВОГО ПОЯСА:")
    print(f"   CSV index timezone: {csv_m5.index.tz}")
    print(f"   API index timezone: {api_m5.index.tz}")
    
    # Check if last API candle is the current (incomplete) one
    now = datetime.now(timezone.utc)
    last_api_ts = api_m5.index[-1]
    print(f"\n⏰ ПРОВЕРКА ЗАКРЫТИЯ СВЕЧИ:")
    print(f"   Сейчас UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Последняя API свеча: {last_api_ts}")
    
    # 5-min candle closes at :00, :05, :10 etc.
    current_5m_start = now.replace(second=0, microsecond=0)
    current_5m_start = current_5m_start.replace(minute=(now.minute // 5) * 5)
    
    if last_api_ts.tz_localize(None) >= current_5m_start.replace(tzinfo=None):
        print(f"   ⚠️  Последняя свеча НЕ ЗАКРЫТА!")
        print(f"   В live scanner мы берём [-2], т.е. предпоследнюю = {api_m5.index[-2]}")
    else:
        print(f"   ✅ Последняя свеча закрыта")


if __name__ == "__main__":
    compare_candles()

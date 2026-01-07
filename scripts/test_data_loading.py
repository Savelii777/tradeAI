#!/usr/bin/env python3
"""
Быстрый тест загрузки данных - проверяет, сколько данных загружается
"""

import ccxt
from datetime import datetime, timedelta, timezone

LOOKBACK = 1500
TIMEFRAMES = ['1m', '5m', '15m']
TEST_PAIR = 'BTC/USDT:USDT'

binance = ccxt.binance({
    'timeout': 10000,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

print("=" * 70)
print("🔍 Тест загрузки данных с Binance API")
print("=" * 70)
print(f"Пара: {TEST_PAIR}")
print(f"LOOKBACK: {LOOKBACK} свечей")
print()

for tf in TIMEFRAMES:
    print(f"📊 Загрузка {tf}...")
    
    # Calculate hours needed
    hours_needed = {
        '1m': LOOKBACK / 60,
        '5m': LOOKBACK * 5 / 60,
        '15m': LOOKBACK * 15 / 60
    }
    
    since_time = datetime.now(timezone.utc) - timedelta(hours=hours_needed[tf] + 1)
    since_ms = int(since_time.timestamp() * 1000)
    
    try:
        # Try with 'since' parameter
        candles = binance.fetch_ohlcv(TEST_PAIR, tf, since=since_ms, limit=LOOKBACK)
        method = "with 'since'"
    except Exception as e:
        # Fallback: try without since
        print(f"   ⚠️  Failed with 'since': {e}")
        candles = binance.fetch_ohlcv(TEST_PAIR, tf, limit=LOOKBACK)
        method = "limit only"
    
    candles_count = len(candles)
    
    min_required = {
        '1m': 500,
        '5m': 200,
        '15m': 100
    }
    
    # Check data quality
    if candles_count < min_required[tf]:
        status = "❌ КРИТИЧНО"
        print(f"   {status}: {candles_count} свечей (нужно {min_required[tf]}+)")
        print(f"   ⚠️  Фичи будут неверными!")
    elif candles_count < LOOKBACK * 0.8:
        status = "⚠️  ПРЕДУПРЕЖДЕНИЕ"
        print(f"   {status}: {candles_count}/{LOOKBACK} свечей ({candles_count/LOOKBACK*100:.1f}%)")
        print(f"   ⚠️  Может повлиять на качество фичей")
    else:
        status = "✅ OK"
        print(f"   {status}: {candles_count}/{LOOKBACK} свечей ({candles_count/LOOKBACK*100:.1f}%)")
    
    if candles_count > 0:
        first_ts = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
        last_ts = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
        duration = last_ts - first_ts
        print(f"   Период: {first_ts.strftime('%Y-%m-%d %H:%M')} - {last_ts.strftime('%Y-%m-%d %H:%M')} ({duration})")
        print(f"   Метод: {method}")
    
    print()

print("=" * 70)
print("📝 Вывод:")
print("  ✅ OK - данных достаточно")
print("  ⚠️  ПРЕДУПРЕЖДЕНИЕ - данных меньше, чем нужно, но можно работать")
print("  ❌ КРИТИЧНО - данных слишком мало, фичи будут неверными")
print("=" * 70)


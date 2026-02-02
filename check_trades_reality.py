#!/usr/bin/env python3
import ccxt
import pandas as pd
from datetime import datetime, timezone
import time

# Последние 10 сделок из бэктеста
trades = [
    ("2026-01-23 06:25:00", "DOT/USDT", "LONG", 1.944, 0.006035),
    ("2026-01-23 07:45:00", "DOT/USDT", "LONG", 1.935, 0.005595),
    ("2026-01-23 08:45:00", "SOL/USDT", "LONG", 127.03, 0.2788),
    ("2026-01-23 12:30:00", "SUI/USDT", "LONG", 1.4864, 0.002884),
    ("2026-01-23 14:30:00", "BCH/USDT", "SHORT", 596.75, 1.0644),
    ("2026-01-23 15:05:00", "TRX/USDT", "LONG", 0.29831, 0.000845),
    ("2026-01-23 19:30:00", "DOGE/USDT", "LONG", 0.12438, 0.000526),
    ("2026-01-23 20:30:00", "APT/USDT", "LONG", 1.5345, 0.007543),
    ("2026-01-24 07:10:00", "POL/USDT", "LONG", 0.12669, 0.000263),
    ("2026-01-24 13:55:00", "UNI/USDT", "LONG", 4.851, 0.009043),
]

print("Проверяю последние 10 сделок на реальность...")
print("=" * 100)

exchange = ccxt.mexc()

valid_count = 0
invalid_count = 0

for timestamp_str, pair, direction, entry_price, atr in trades:
    dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
    ts = int(dt.timestamp() * 1000)
    
    print(f"\n{timestamp_str} | {pair} {direction} @ ${entry_price:.6f}")
    
    try:
        # Получаем свечу на момент входа
        candles = exchange.fetch_ohlcv(pair, '5m', since=ts - 60000, limit=2)
        time.sleep(0.3)
        
        if len(candles) > 0:
            candle = candles[0]
            real_time = datetime.fromtimestamp(candle[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            real_open = candle[1]
            real_high = candle[2]
            real_low = candle[3]
            real_close = candle[4]
            
            # Проверяем попадает ли entry в диапазон
            in_range = real_low <= entry_price <= real_high
            diff_pct = abs(entry_price - real_close) / real_close * 100
            atr_pct = atr / entry_price * 100
            
            print(f"  ⏰ Время свечи: {real_time}")
            print(f"  📊 OHLC: O={real_open:.6f} H={real_high:.6f} L={real_low:.6f} C={real_close:.6f}")
            print(f"  {'✅' if in_range else '❌'} Entry {entry_price:.6f} {'В ДИАПАЗОНЕ' if in_range else '⚠️  ВНЕ ДИАПАЗОНА!'}")
            print(f"  📈 Отклонение от Close: {diff_pct:.2f}%")
            print(f"  📏 ATR: {atr:.6f} ({atr_pct:.2f}% от цены)")
            
            if in_range:
                valid_count += 1
            else:
                invalid_count += 1
                
        else:
            print(f"  ❌ НЕТ ДАННЫХ на бирже!")
            invalid_count += 1
            
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        invalid_count += 1

print("\n" + "=" * 100)
print(f"\n📊 ИТОГО:")
print(f"   ✅ Валидных сделок: {valid_count}/{len(trades)} ({valid_count/len(trades)*100:.1f}%)")
print(f"   ❌ Невалидных: {invalid_count}/{len(trades)}")
print("\n" + "=" * 100)

#!/usr/bin/env python3
import math

# Исходные данные бэктеста
start = 61
end = 10000
days = 14

# Расчёт
growth = end / start
daily_mult = growth ** (1/days)
daily_pct = (daily_mult - 1) * 100

print('='*50)
print('РАСЧЁТ ВРЕМЕНИ $100K -> $200K')
print('='*50)
print(f'Бэктест: ${start} -> ${end:,} за {days} дней')
print(f'Рост: {growth:.1f}x')
print(f'Дневной множитель: {daily_mult:.4f} (+{daily_pct:.1f}%/день)')
print()

# Время для удвоения
target_growth = 2  # 100K → 200K
days_to_double = math.log(target_growth) / math.log(daily_mult)

print(f'Для удвоения (2x):')
print(f'  При 100% эффективности: {days_to_double:.1f} дней')
print()

# С учётом снижения эффективности на большом депо
print('С учётом размера депозита:')
for efficiency in [1.0, 0.7, 0.5, 0.3, 0.2]:
    adj_daily_mult = 1 + (daily_mult - 1) * efficiency
    adj_days = math.log(2) / math.log(adj_daily_mult)
    adj_pct = (adj_daily_mult - 1) * 100
    print(f'  При {int(efficiency*100)}% эффективности (+{adj_pct:.1f}%/день): {adj_days:.1f} дней')

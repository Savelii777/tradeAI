#!/usr/bin/env python3
"""Debug risk-based sizing"""

# Симулируем APT trade
capital = 61.0  # Начальный депозит был $61
risk_pct = 0.05
risk_amount = capital * risk_pct

# APT примерные значения
entry_price = 3.50
atr = 0.05  # примерно 1.4% от цены
sl_mult = 1.5
stop_distance = atr * sl_mult

sl_pct = stop_distance / entry_price
print(f'Entry: {entry_price}')
print(f'ATR: {atr} ({atr/entry_price*100:.2f}%)')
print(f'Stop distance: {stop_distance} ({sl_pct*100:.2f}%)')
print()

# Position sizing
position_value = risk_amount / sl_pct
print(f'Risk amount: ${risk_amount:.2f}')
print(f'Position (calculated): ${position_value:.2f}')

# Limits
max_leverage = 50.0
max_pos_by_lev = capital * max_leverage
max_pos = 4_000_000
print(f'Max by leverage (50x): ${max_pos_by_lev:.2f}')
position_value_limited = min(position_value, max_pos_by_lev, max_pos)
print(f'Position (after limits): ${position_value_limited:.2f}')

leverage = position_value_limited / capital
print(f'Leverage: {leverage:.1f}x')
print()

# Loss at stop
actual_loss = position_value_limited * sl_pct
print(f'Loss at stop: ${actual_loss:.2f}')
print(f'Loss as pct of capital: {actual_loss/capital*100:.1f}%')
print()

# ===============================================
# Теперь с реальными данными из лога
# AVG LOSS: -1.70% price move -> -17.2% ROE
# ===============================================
print('='*50)
print('АНАЛИЗ ИЗ ЛОГА:')
print('='*50)
print('AVG leverage: 10x')
print('AVG LOSS price move: 1.70%')
print('AVG LOSS ROE: 17.2%')
print()

# Если leverage 10x и loss 1.7% price:
avg_leverage = 10
price_move = 0.017
roe_loss = price_move * avg_leverage
print(f'Расчёт: {price_move*100:.1f}% * {avg_leverage}x = {roe_loss*100:.1f}% ROE')
print()

# Для 5% loss при 10x leverage, price move должен быть:
target_loss_pct = 0.05
required_price_move = target_loss_pct / avg_leverage
print(f'Для 5% loss при {avg_leverage}x, нужен стоп: {required_price_move*100:.2f}% price move')
print()

# А у нас стоп 1.7% - это ATR-based
# Значит position sizing НЕ РАБОТАЕТ!
print('ПРОБЛЕМА:')
print(f'  Нужен стоп: {required_price_move*100:.2f}%')
print(f'  Реальный стоп: 1.70%')
print(f'  Разница: {(0.017 / required_price_move):.1f}x больше чем нужно!')

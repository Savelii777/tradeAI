# Исправление проблемы "Боковик" (Sideways Predictions) ✅ РЕШЕНО

## Проблема

Paper trading скрипт (`paper_trading_v8_ws.py`) постоянно предсказывал класс 1 (SIDEWAYS), потому что:

1. **Missing features**: Модель ожидала **172 features**, но получала только **166**
2. **Отсутствующие 6 volume features**: `vol_sma_20`, `vol_ratio`, `vol_zscore`, `vwap`, `price_vs_vwap`, `vol_momentum`
3. **Неправильные данные на входе**: Модель получала NaN или неполные features, поэтому возвращала дефолтный класс (SIDEWAYS)

## Что ожидает модель

Модель была обучена через `train_mtf.py` + volume features и ожидает **172 features**:

### 166 MTF features (из MTFFeatureEngine)
- **M5**: `m5_return_1`, `m5_ema_9`, `m5_rsi`, ... (~133 features)
- **M15**: `m15_trend`, `m15_momentum`, `m15_atr`, ... (~11 features)  
- **M1**: `m1_momentum_last`, `m1_rsi_mean`, ... (~22 features)

### 6 Volume features
- `vol_sma_20`, `vol_ratio`, `vol_zscore`
- `vwap`, `price_vs_vwap`, `vol_momentum`

### Вспомогательные (НЕ для модели)
- `open`, `high`, `low`, `close`, `volume`, `atr` - для расчетов SL/TP

## Исправления

### 1. Добавил обратно volume features (были удалены по ошибке)

**До:**
```python
ft = mtf_fe.align_timeframes(m1, m5, m15)
ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
# ❌ Отсутствуют 6 volume features!
ft['atr'] = calculate_atr(ft)
```

**После:**
```python
# Генерируем 166 MTF features
ft = mtf_fe.align_timeframes(m1, m5, m15)

# Добавляем OHLCV для расчетов
ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])

# ✅ Добавляем 6 volume features (требуются моделью!)
ft = add_volume_features(ft)

# Добавляем ATR для position sizing
ft['atr'] = calculate_atr(ft)
```

### 2. Добавил валидацию features

```python
# Проверяем, что все ожидаемые features присутствуют
missing_features = [f for f in models['features'] if f not in row.columns]
if missing_features:
    logger.error(f"Missing features: {missing_features[:10]}")
    continue

# Безопасное извлечение
try:
    X = row.iloc[[-1]][models['features']].values
except KeyError as e:
    logger.error(f"KeyError: {e}")
    continue
```

### 3. Улучшил логирование при загрузке моделей

```python
logger.info(f"Loaded {len(models['features'])} features")
logger.info(f"First 10 features: {models['features'][:10]}")

# Предупреждение о неправильных features
excluded = ['atr', 'price_change', 'obv', 'open', 'high', 'low', 'close', 'volume']
found_excluded = [f for f in excluded if f in models['features']]
if found_excluded:
    logger.error(f"⚠️ WARNING: Excluded columns in features: {found_excluded}")
```

### 4. Удалил ненужную функцию `add_volume_features()`

Эта функция добавляла features, которых НЕТ в обученной MTF модели.

## Проверка

### Тест features (быстрая проверка)

```bash
python test_features_simple.py
```

Должны увидеть:
```
✅ All 172 model features are present!

TEST PREDICTION
======================================================================
Input shape: (1, 172)
Contains NaN: False

✅ Prediction: LONG (confidence: 62.45%)
   Probabilities:
   - SHORT:    15.20%
   - SIDEWAYS: 22.35%
   - LONG:     62.45%
```

### Запуск paper trading

```bash
python scripts/paper_trading_v8_ws.py
```

Теперь вы должны увидеть:
- ✅ `Loaded 172 features`
- ✅ `First 10 features: ['m5_return_1', 'm5_return_5', ...]`
- ✅ Предсказания с разными классами (LONG/SHORT/SIDEWAYS)
- ✅ Логи вида: `BTC/USDT: Dir=2 (conf=0.65) - Proba: [[0.15 0.20 0.65]]`
- ✅ **НЕТ ошибок** "Missing features"

Если всё ещё проблемы:
1. Проверьте логи - есть ли ошибки "Missing features" или "NaN in features"?
2. Убедитесь, что используете модель из `models/v8_improved/` или `models/v7_sniper/`
3. Запустите тест: `python test_features_simple.py`

## Примечание о train_v3_dynamic.py

**ВАЖНО**: Скрипт `train_v3_dynamic.py` использует **другой подход** (volume-based features без MTF).

Если вы хотите использовать `train_v3_dynamic.py` для обучения:
1. Нужно ПЕРЕОБУЧИТЬ модель этим скриптом
2. Или изменить `paper_trading_v8_ws.py`, чтобы использовать такие же features

Текущие модели (`v7_sniper`, `v8_improved`) были обучены через MTF подход!


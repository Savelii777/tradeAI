# Анализ проблемы: Бэктест vs Live Trading

## Краткое описание проблемы

**Симптом**: На бэктесте модель показывает отличные результаты (Win Rate 80%+, Walk-Forward 70%+), но в live trading сигналы LONG и SHORT имеют confidence только ~40% (вместо требуемых 50%+), из-за чего сигналы не срабатывают.

**Причина**: Feature Distribution Shift (сдвиг распределения признаков) между условиями обучения и live trading.

---

## ✅ Исправления в этом PR

В файле `src/utils/constants.py` были добавлены недостающие абсолютные признаки:

```python
ABSOLUTE_PRICE_FEATURES = [
    # ... существующие признаки ...
    # ✅ ДОБАВЛЕНО: MACD признаки (абсолютная разница цен)
    'm5_macd', 'm5_macd_signal', 'm5_macd_histogram', 'm5_macd_histogram_change',
]
```

**Почему это важно**: MACD = EMA(12) - EMA(26) — это абсолютная разница в ценах. При разных ценах актива (например, BTC $25,000 vs $95,000) значения MACD будут кардинально отличаться, вызывая Feature Distribution Shift.

---

## Детальный анализ

### 1. Что такое Feature Distribution Shift?

Это ситуация, когда распределение входных признаков (features) модели отличается между:
- **Обучением/бэктестом**: модель обучена на данных определённого периода
- **Live trading**: модель применяется на текущих данных, которые могут иметь другую структуру

### 2. Выявленные проблемы в коде

#### 2.1. Накопительные (cumsum) признаки

В файле `src/features/feature_engine.py` (строки 100-102) используются признаки с `cumsum()`:

```python
'consecutive_up': (direction == 1).groupby((direction != 1).cumsum()).cumsum(),
'consecutive_down': (direction == -1).groupby((direction != -1).cumsum()).cumsum()
```

**Проблема**: 
- В бэктесте: данные начинаются с 2017 года → cumsum накапливается за 8 лет
- В live: данные начинаются с последних 1000-2000 свечей → cumsum накапливается только за несколько дней
- **Результат**: Кардинально разные значения признаков!

**Пример**:
- Бэктест: `consecutive_up = 1547` (накоплено за годы)
- Live: `consecutive_up = 23` (накоплено за дни)

#### 2.2. Абсолютные ценовые признаки

В `src/utils/constants.py` уже определены проблемные признаки:

```python
ABSOLUTE_PRICE_FEATURES = [
    'm5_ema_9', 'm5_ema_21', 'm5_ema_50', 'm5_ema_200',  # Абсолютные EMA
    'm5_bb_upper', 'm5_bb_middle', 'm5_bb_lower',        # Абсолютные уровни BB  
    'm5_volume_ma_5', 'm5_volume_ma_10', 'm5_volume_ma_20',
    'm5_atr_7', 'm5_atr_14', 'm5_atr_21', 'm5_atr_14_ma',
    'm5_volume_delta', 'm5_volume_trend',
    # ✅ ИСПРАВЛЕНО: MACD тоже абсолютная разница цен!
    'm5_macd', 'm5_macd_signal', 'm5_macd_histogram',
]
```

**Проблема**:
- В обучении: BTC стоил $25,000 → `m5_ema_200 = 25000`
- В live: BTC стоит $95,000 → `m5_ema_200 = 95000`
- **Результат**: Модель видит совершенно другие значения и теряет уверенность!

**MACD** - особенно важно:
- MACD = EMA(12) - EMA(26) - это **абсолютная разница** в ценах
- При BTC $25,000: MACD может быть $500
- При BTC $95,000: MACD может быть $2,000
- **Этот признак не был исключён ранее!** ← Исправлено в этом PR

#### 2.3. Rolling Normalization с разной длиной окна

В `train_mtf.py` (строка 181):

```python
features = self.feature_engine.generate_all_features(df, normalize=False)
```

**Статус**: ✅ Уже исправлено - normalization отключена.

Но в `feature_engine.py` есть метод `normalize_features()` с `window=500`:
- При обучении: окно из 500 точек на 60 днях данных
- В live: окно из 500 точек на последних 1000 свечах
- **Результат**: Разные z-score из-за разных базовых статистик

### 3. Почему Confidence низкий?

Модель LightGBM для классификации направления (`direction_model`) выдаёт вероятности через `predict_proba()`.

Когда модель видит:
- Признаки, значения которых сильно отличаются от обучающих данных
- Признаки с неожиданным распределением

Она становится "неуверенной" и выдаёт вероятности близкие к 33% для каждого класса (LONG, SHORT, SIDEWAYS), вместо уверенных 60%+ для одного направления.

**Ваш случай**: max confidence = 40% означает модель "путается" между классами.

---

## Рекомендации по исправлению

### Приоритет 1: Немедленные исправления

#### 1.1. Исключить cumsum-зависимые признаки из обучения

В `train_v3_dynamic.py` уже есть исключение:
```python
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES

features = [c for c in train_df.columns if c not in all_exclude 
            and not any(p in c.lower() for p in CUMSUM_PATTERNS)]
```

**Проверьте**: что `CUMSUM_PATTERNS` включает все проблемные паттерны:
- `'consecutive_up'`, `'consecutive_down'` - добавить если отсутствуют!

#### 1.2. Переобучить модель на актуальных данных

```bash
cd /Users/savelii/tradeAI && python scripts/train_v3_dynamic.py \
    --days 60 \
    --test_days 14 \
    --pairs 20 \
    --initial_balance 20 \
    --output ./models/v8_improved \
    --reverse \
    --walk-forward
```

**Важно**: Использовать `--reverse` чтобы обучение было на последних 60 днях, а тест на предыдущих 30.

### Приоритет 2: Улучшения признаков

#### 2.1. Использовать только относительные признаки

Вместо абсолютных значений EMA, использовать:
- `ema_distance = (close - ema) / ema * 100` — расстояние от цены до EMA в %
- `ema_slope = (ema - ema.shift(1)) / ema.shift(1) * 100` — наклон EMA в %

Это уже частично сделано в `train_mtf.py` строки 218-220:
```python
# ✅ FIX: Store as % distance from price instead of absolute price
features['m1_ema_3'] = (ema_3 - df['close']) / df['close'] * 100
features['m1_ema_8'] = (ema_8 - df['close']) / df['close'] * 100
```

#### 2.2. Нормализовать признаки к ATR

Вместо абсолютных значений волатильности:
```python
# Было:
features['m5_atr_14'] = atr_14  # $500 для BTC

# Надо:
features['m5_atr_14_pct'] = atr_14 / close * 100  # 2.5%
```

### Приоритет 3: Валидация перед запуском

#### 3.1. Запустить диагностику

```bash
python scripts/diagnose_live_vs_backtest.py --all-pairs
```

Это покажет:
- Какие признаки имеют drift >30%
- Почему live не генерирует сигналы

#### 3.2. Сравнить распределения признаков

```bash
python scripts/compare_feature_distributions.py
```

---

## Рекомендуемый порядок действий

1. **Сейчас**: Запустить диагностику чтобы увидеть конкретные проблемные признаки
2. **Исправить**: Добавить недостающие паттерны в `CUMSUM_PATTERNS`
3. **Переобучить**: Модель на свежих данных с правильными признаками
4. **Протестировать**: Walk-forward validation должен показывать >50% Win Rate
5. **Paper Trading**: Запустить live_trading без реального исполнения
6. **Live**: Только после успешного paper trading

---

## Ссылки и источники

### Статьи по теме (рекомендованы пользователем):
- https://habr.com/ru/articles/561638/ - Переобучение в ML для трейдинга
- https://habr.com/ru/companies/ods/articles/560312/ - Feature Engineering для финансов

### Дополнительные ресурсы:
- [Feature Distribution Shift in ML](https://www.seas.upenn.edu/~obastani/cis7000/spring2024/docs/lecture3.pdf)
- [Backtesting vs Live Trading - MultiCharts](https://www.multicharts.com/trading-software/index.php?title=Backtesting_vs_Live_Trading)
- [Debugging Backtest vs Live Discrepancies](https://deepwiki.com/nkaz001/hftbacktest/7.4-debugging-backtest-vs-live-discrepancies)

---

## Заключение

Проблема "нет сигналов в live" — это **классическая проблема ML в трейдинге**, связанная с Feature Distribution Shift. 

Модель обучена на признаках с одними значениями, но в live видит совершенно другие значения тех же признаков. Это приводит к низкой уверенности модели (confidence ~40% вместо 50%+).

**Решение**: Использовать только **относительные, нормализованные признаки**, которые не зависят от:
1. Начальной точки окна данных (cumsum)
2. Абсолютного уровня цен (EMA, BB levels)
3. Длины исторического окна (rolling stats)

После исправлений и переобучения модели, сигналы в live должны появиться с нормальной уверенностью.

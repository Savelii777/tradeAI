# PRD: AI Trading Bot MTF V2 - Multi-Timeframe Ensemble Model

## 1. Executive Summary

### Проект
AI Trading Bot с машинным обучением для автоматической торговли криптовалютными фьючерсами на Binance/Bybit.

### Текущий статус
- **Версия:** MTF V2 (Multi-Timeframe Version 2)
- **Стадия:** Бэктестинг, оптимизация
- **Результат последнего бэктеста:** -11.79% за 14 дней
- **Win Rate:** 28.6% (breakeven для RR 1:3 = 25%)
- **Theoretical Expectancy:** +0.14R per trade (положительная!)

### Ключевая проблема
Модель теоретически прибыльная (положительное мат. ожидание), но на практике убыточная из-за:
1. Высокого drawdown (84%)
2. Слабой модели timing (correlation ~0.05)
3. Недостаточного Win Rate для компенсации комиссий

---

## 2. System Architecture

### 2.1 Общая архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRADING BOT                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Scanner    │───▶│   Feature    │───▶│   Ensemble   │       │
│  │  (Pairs)     │    │   Engine     │    │   Model V2   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Exchange   │◀───│   Order      │◀───│   Signal     │       │
│  │   API        │    │   Manager    │    │   Generator  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Ensemble Model V2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENSEMBLE MODEL V2                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 133 Features (OHLCV, Indicators, Patterns, Structure)   │
│                           │                                      │
│           ┌───────────────┼───────────────┐                      │
│           ▼               ▼               ▼                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Direction  │  │   Timing    │  │  Volatility │              │
│  │  Model V2   │  │  Model V2   │  │   Model     │              │
│  │ (3-class)   │  │ (regression)│  │ (regression)│              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────┐                │
│  │           Signal Generator                   │                │
│  │  - Direction prob > threshold (0.32)         │                │
│  │  - Timing score > threshold (0.15)           │                │
│  │  - Generate BUY/SELL/HOLD                    │                │
│  └─────────────────────────────────────────────┘                │
│                           │                                      │
│                           ▼                                      │
│              TradingSignal (buy/sell/hold)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Component Models

| Model | Type | Output | Status | Performance |
|-------|------|--------|--------|-------------|
| **DirectionModelV2** | LightGBM Classifier | 3 classes (down/sideways/up) | ✅ Trained | Accuracy: 44.2% |
| **TimingModelV2** | LightGBM Regressor | Score 0-1 | ✅ Trained | Correlation: 0.053 |
| **StrengthModel** | LightGBM Regressor | Expected move | ❌ Not trained | N/A |
| **VolatilityModel** | LightGBM Regressor | Volatility forecast | ❌ Not trained | N/A |

---

## 3. Data Pipeline

### 3.1 Data Sources
- **Exchange:** Binance Futures (USDT-M)
- **Timeframe:** 5-minute candles
- **Pairs:** BTC/USDT, ETH/USDT, SOL/USDT (+ 17 других)
- **History:** 30 days для обучения

### 3.2 Feature Engineering

**Всего: 133 фичи**

| Category | Count | Examples |
|----------|-------|----------|
| OHLCV | 12 | open, high, low, close, volume, returns |
| Technical Indicators | 65 | RSI, MACD, Bollinger, ATR, ADX, Stochastic |
| Candlestick Patterns | 11 | doji, hammer, engulfing, morning_star |
| Market Structure | 36 | support_distance, resistance_distance, trend_strength |
| Time Features | 9 | hour, day_of_week, is_session_overlap |

### 3.3 Target Variables

**Direction Target (3-class classification):**
```python
# RR 1:3 optimized targets
ATR_MULT_TP = 4.5  # 4.5 ATR for Take Profit
ATR_MULT_SL = 1.5  # 1.5 ATR for Stop Loss

# Logic:
if price hits TP first → UP (2)
if price hits SL first → DOWN (0)
else → SIDEWAYS (1)
```

**Timing Target (regression 0-1):**
```python
# Measures how quickly price moves in predicted direction
timing_score = time_to_target / max_time
# Normalized and clipped to [0, 1]
```

---

## 4. Current Model Performance

### 4.1 Training Metrics (30 days, 3 pairs)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Direction Accuracy** | 48.2% | 44.2% | ~42% |
| **Direction - Sideways %** | 56.9% | ~55% | ~55% |
| **Timing Correlation** | 0.15 | 0.053 | ~0.05 |
| **Timing MSE** | 0.018 | 0.019 | ~0.02 |

### 4.2 Backtest Results (14 days)

```
======================================================================
MTF V2 BACKTEST RESULTS
======================================================================

Capital: $10000.00 → $8821.29
Total Return: -11.79%
Total Trades: 126
Win Rate: 28.6%
Profit Factor: 0.87
Sharpe Ratio: 26.95
Max Drawdown: 84.34%

----------------------------------------
RR 1:3 ANALYSIS
----------------------------------------
Average Win: 0.03R
Average Loss: 0.01R
Expectancy: 0.001R per trade
Theoretical EV (WR=28.6%, RR=3): 0.14R
✓ Positive expectancy achieved

----------------------------------------
PER-PAIR RESULTS
----------------------------------------
ETH/USDT: 42 trades, WR=31.0%, PF=0.98, Return=-1.47%
SOL/USDT: 42 trades, WR=28.6%, PF=0.90, Return=-8.27%
BTC/USDT: 42 trades, WR=21.4%, PF=0.53, Return=-25.62%

----------------------------------------
EXIT REASONS
----------------------------------------
take_profit: 36 trades, PnL=$8149.33
stop_loss: 90 trades, PnL=$-9328.05
```

### 4.3 Signal Distribution

```
Direction Model Output Distribution:
- Down:     21.1%
- Sideways: 59.2%
- Up:       19.7%

Trading Signals Generated:
- Buy:  31 signals (24.6%)
- Sell: 95 signals (75.4%)
- Hold: ~3800 (filtered out)
```

---

## 5. Identified Problems

### 5.1 Critical Issues

| # | Problem | Impact | Root Cause |
|---|---------|--------|------------|
| 1 | **Max Drawdown 84%** | Capital destruction | No position sizing limits, no daily loss limits |
| 2 | **Timing Model Useless** | Correlation 0.05 | Target definition unclear, regression overfits |
| 3 | **BTC Underperforms** | WR 21.4% | Model not capturing BTC-specific patterns |
| 4 | **Strength/Volatility Models Not Trained** | Missing filters | Training script doesn't train them |

### 5.2 Model Issues

**Direction Model:**
- ✅ Sideways bias fixed (was 80%, now ~57%)
- ✅ Class weights balanced with power=0.85 formula
- ⚠️ Overfitting: Train 48%, Val 44%, gap still exists
- ⚠️ 3-class accuracy 44% is barely above random (33%)

**Timing Model:**
- ❌ Correlation 0.05 — практически случайный
- ❌ Mean output 0.20, Max 0.35 — узкий диапазон
- ❌ Не помогает фильтровать плохие сигналы

**Signal Generation:**
- ⚠️ Порог direction 0.32 слишком низкий?
- ⚠️ Порог timing 0.15 слишком низкий (практически не фильтрует)
- ⚠️ Strength model не обучена — всегда возвращает 1.0

### 5.3 Backtest Issues

**Fixed:**
- ✅ Features indexing (iloc → loc by timestamp)
- ✅ Missing dropna() for features
- ✅ Config thresholds overwriting defaults
- ✅ use_dynamic_thresholds not passed

**Remaining:**
- ⚠️ No slippage simulation beyond fixed %
- ⚠️ No funding rate costs
- ⚠️ Commission calculation may be off

---

## 6. Model Configuration

### 6.1 LightGBM Hyperparameters

**Direction Model V2:**
```python
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'boosting_type': 'gbdt',
    'num_leaves': 12,        # Reduced for regularization
    'max_depth': 4,          # Shallow trees
    'learning_rate': 0.02,   # Slow learning
    'n_estimators': 300,
    'min_child_samples': 200, # High minimum samples
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbosity': -1
}
```

**Class Weights Formula:**
```python
# Power-based inverse frequency weights
power = 0.85  # Between sqrt (0.5) and linear (1.0)
freq_down, freq_side, freq_up = class_frequencies

w_down = (1.0 / freq_down) ** power
w_side = (1.0 / freq_side) ** power  
w_up = (1.0 / freq_up) ** power

# Normalize
total = w_down + w_side + w_up
weights = normalize(weights)

# Result: Down=1.17, Sideways=0.50, Up=1.33
```

**Timing Model V2:**
```python
params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'num_leaves': 12,
    'max_depth': 4,
    'learning_rate': 0.02,
    'n_estimators': 200,
    'min_child_samples': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

### 6.2 Signal Thresholds

```yaml
# config/trading_params.yaml
entry:
  min_direction_probability: 0.32  # Just above 1/3 for 3-class
  min_timing_score: 0.15           # Model outputs 0.08-0.35
  min_strength_score: 0.20         # Not used (model not trained)
  use_dynamic_thresholds: false    # Disabled - causes over-filtering
```

### 6.3 Risk Management

```yaml
risk:
  max_risk_per_trade: 0.05    # 5% of capital per trade
  max_daily_loss: 0.15        # 15% daily limit
  max_consecutive_losses: 3
  max_drawdown: 0.50          # 50% max drawdown
  
exit:
  stop_loss_atr_multiplier: 1.5   # SL at 1.5 ATR
  take_profit_atr_multiplier: 4.5 # TP at 4.5 ATR (RR 1:3)
  rr_ratio: 3.0
```

---

## 7. File Structure

```
/Users/savelii/tradeAI/
├── main.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── config/
│   ├── settings.yaml               # General settings
│   └── trading_params.yaml         # Trading parameters ⭐
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   └── mtf_v2/                     # Saved V2 models
│       ├── direction_v2.pkl
│       ├── timing_v2.pkl
│       └── ensemble_v2.pkl
├── scripts/
│   ├── train_mtf_v2.py            # V2 training script ⭐
│   ├── backtest_mtf_v2.py         # V2 backtest script ⭐
│   └── debug_v2_predictions.py    # Debug script
├── src/
│   ├── features/
│   │   ├── feature_engine.py      # Feature generation ⭐
│   │   ├── indicators.py          # Technical indicators
│   │   ├── patterns.py            # Candlestick patterns
│   │   └── market_structure.py    # Support/resistance
│   ├── models/
│   │   ├── direction_v2.py        # Direction model V2 ⭐
│   │   ├── timing_v2.py           # Timing model V2
│   │   ├── ensemble_v2.py         # Ensemble V2 ⭐
│   │   ├── strength.py            # Strength model (not trained)
│   │   └── volatility.py          # Volatility model (not trained)
│   ├── execution/
│   │   ├── exchange_api.py
│   │   ├── order_manager.py
│   │   └── position_manager.py
│   ├── risk/
│   │   ├── limits.py
│   │   └── drawdown.py
│   └── strategy/
│       ├── decision_engine.py
│       ├── signals.py
│       └── position_sizing.py
└── results/
    └── backtest_v2/
        ├── backtest_summary.json
        └── trades.json
```

---

## 8. Key Code Snippets

### 8.1 Direction Target Creation (RR 1:3)

```python
def create_direction_target_rr3(
    df: pd.DataFrame,
    atr_mult_tp: float = 4.5,  # Take profit at 4.5 ATR
    atr_mult_sl: float = 1.5,  # Stop loss at 1.5 ATR
    max_bars: int = 48         # Max 4 hours
) -> pd.Series:
    """
    Creates direction labels based on RR 1:3 logic.
    UP(2): price hits TP first
    DOWN(0): price hits SL first
    SIDEWAYS(1): neither within max_bars
    """
    atr = calculate_atr(df, period=14)
    targets = []
    
    for i in range(len(df) - max_bars):
        tp_dist = atr.iloc[i] * atr_mult_tp
        sl_dist = atr.iloc[i] * atr_mult_sl
        entry = df['close'].iloc[i]
        
        # Check future bars
        for j in range(1, max_bars + 1):
            high = df['high'].iloc[i + j]
            low = df['low'].iloc[i + j]
            
            # TP hit (long direction)
            if high >= entry + tp_dist:
                targets.append(2)  # UP
                break
            # SL hit (long direction)  
            elif low <= entry - sl_dist:
                targets.append(0)  # DOWN
                break
        else:
            targets.append(1)  # SIDEWAYS
    
    return pd.Series(targets)
```

### 8.2 Class Weight Calculation

```python
def _compute_class_weights(self, y: pd.Series, sideways_penalty: float = 1.0):
    """
    Compute class weights using power-based inverse frequency.
    """
    counts = y.value_counts()
    total = len(y)
    
    freq_down = counts.get(0, 1) / total
    freq_side = counts.get(1, 1) / total
    freq_up = counts.get(2, 1) / total
    
    # Power-based weights: power=0.85 best for ~50% sideways
    power = 0.85
    w_down = (1.0 / max(freq_down, 0.1)) ** power
    w_side = (1.0 / max(freq_side, 0.1)) ** power
    w_up = (1.0 / max(freq_up, 0.1)) ** power
    
    # Normalize
    total_weight = w_down + w_side + w_up
    avg_weight = total_weight / 3
    
    return {
        0: w_down / avg_weight,      # ~1.17
        1: w_side / avg_weight,      # ~0.50
        2: w_up / avg_weight         # ~1.33
    }
```

### 8.3 Signal Generation Logic

```python
def get_trading_signal(self, X, min_direction_prob=0.32, min_timing=0.15, ...):
    predictions = self.predict(X)
    
    for i in range(len(X)):
        p_down, p_sideways, p_up = predictions['direction_proba'][i]
        timing = predictions['timing'][i]
        strength = predictions.get('strength', 1.0)  # Default if not trained
        
        # Long signal conditions
        if p_up > min_direction_prob and p_up > p_down:
            if timing >= min_timing and strength >= min_strength:
                signal = 'buy'
                confidence = p_up
        
        # Short signal conditions
        elif p_down > min_direction_prob and p_down > p_up:
            if timing >= min_timing and strength >= min_strength:
                signal = 'sell'
                confidence = p_down
        
        else:
            signal = 'hold'
```

---

## 9. Requirements for Improvement

### 9.1 Must Have (P0)

1. **Reduce Max Drawdown to < 30%**
   - Implement proper position sizing based on Kelly criterion
   - Add daily loss limits that actually stop trading
   - Reduce max risk per trade from 5% to 2%

2. **Improve Win Rate to > 35%**
   - Need WR 35%+ for profitable RR 1:3 strategy after costs
   - Current 28.6% is too close to breakeven

3. **Fix Timing Model**
   - Current correlation 0.05 is useless
   - Either improve or remove from signal generation

4. **Train Strength/Volatility Models**
   - Currently return default values
   - Need proper targets and training

### 9.2 Should Have (P1)

1. **Better Feature Selection**
   - 133 features may cause overfitting
   - Use feature importance to reduce to top 50

2. **Per-Pair Model Training**
   - BTC behaves differently than ETH/SOL
   - Consider pair-specific models or features

3. **Add Market Regime Detection**
   - Don't trade in extreme conditions
   - Filter by volatility regime

4. **Improve Backtesting Realism**
   - Add order book slippage simulation
   - Include funding rate costs
   - Add latency simulation

### 9.3 Nice to Have (P2)

1. **Multi-Timeframe Features**
   - Currently only 5m timeframe
   - Add 15m, 1h, 4h trend features

2. **Alternative Models**
   - Try XGBoost, CatBoost
   - Try neural networks (LSTM, Transformer)

3. **Online Learning**
   - Update model weights based on recent performance
   - Adaptive thresholds

---

## 10. Metrics to Track

### 10.1 Model Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Direction Accuracy | 44.2% | > 50% | 3-class, random = 33% |
| Direction - Sideways % | 57% | 40-50% | Balanced distribution |
| Timing Correlation | 0.05 | > 0.30 | Need significant signal |
| Train-Val Gap | 4% | < 3% | Reduce overfitting |

### 10.2 Trading Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Win Rate | 28.6% | > 35% | Breakeven for RR 1:3 = 25% |
| Profit Factor | 0.87 | > 1.20 | Gross profit / Gross loss |
| Max Drawdown | 84% | < 30% | Critical! |
| Sharpe Ratio | 26.9 | > 1.5 | Risk-adjusted return |
| Return (14d) | -11.79% | > 0% | Profitable |

### 10.3 Operational Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Trades per day | ~9 | 5-10 |
| Avg trade duration | ~4h | 2-8h |
| Signal to trade ratio | 3% | 5-10% |

---

## 11. Questions for Review

1. **Model Architecture:**
   - Is 3-class direction (down/sideways/up) the best approach?
   - Should we use binary classification (long/short) with confidence filtering?
   - Would regression for expected return work better?

2. **Feature Engineering:**
   - Which of 133 features are actually predictive?
   - Should we add cross-pair correlation features?
   - Are time features (hour, day) useful or noise?

3. **Training:**
   - 30 days enough? Should use more historical data?
   - How to handle non-stationarity of financial data?
   - Should we train separate models per market regime?

4. **Signal Generation:**
   - Current thresholds (direction 0.32, timing 0.15) optimal?
   - Should we use Kelly criterion for position sizing?
   - How to incorporate volatility into signal strength?

5. **Risk Management:**
   - How to prevent 84% drawdown in live trading?
   - What's the optimal risk per trade (currently 5%)?
   - Should we have correlation-based position limits?

---

## 12. Next Steps

### Immediate (This Week)

1. [ ] Analyze feature importance, reduce to top 50 features
2. [ ] Fix timing model or remove from signal generation
3. [ ] Train strength and volatility models
4. [ ] Implement proper position sizing limits

### Short-term (2 Weeks)

1. [ ] Improve direction model accuracy to > 50%
2. [ ] Reduce max drawdown to < 30%
3. [ ] Achieve positive return on 14-day backtest
4. [ ] Add market regime filter

### Medium-term (1 Month)

1. [ ] Paper trading on live data
2. [ ] Validate out-of-sample performance
3. [ ] Implement online learning
4. [ ] Prepare for live trading

---

## 13. Appendix

### A. Training Command

```bash
docker-compose -f docker/docker-compose.yml run --rm trading-bot \
  python scripts/train_mtf_v2.py \
  --pairs "BTC,ETH,SOL" \
  --days 30 \
  --output models/mtf_v2
```

### B. Backtest Command

```bash
docker-compose -f docker/docker-compose.yml run --rm trading-bot \
  python scripts/backtest_mtf_v2.py \
  --pairs "BTC,ETH,SOL" \
  --days 14 \
  --config config/trading_params.yaml
```

### C. Debug Predictions

```bash
docker-compose -f docker/docker-compose.yml run --rm trading-bot \
  python scripts/debug_v2_predictions.py
```

### D. Key Files to Review

1. `src/models/direction_v2.py` - Direction model with class weights
2. `src/models/ensemble_v2.py` - Signal generation logic
3. `scripts/backtest_mtf_v2.py` - Backtest implementation
4. `config/trading_params.yaml` - Trading configuration

---

*Document created: 2025-12-24*
*Last updated: 2025-12-24*
*Version: 2.0*

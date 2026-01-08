# AI Trading Bot v2.1

🤖 **Autonomous ML-powered cryptocurrency trading bot for scalping**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A fully autonomous trading bot with its own machine learning system, capable of analyzing real-time market data and executing profitable trades on cryptocurrency exchanges without relying on external AI APIs.

### 🚀 What's New in v2.1

**Multi-Pair Scanner + M1 Sniper + Aggressive Trading**

- **Multi-Pair Scanner**: Continuously scans 25+ cryptocurrency pairs for high-potential setups
- **M1 Sniper Entry**: Precise entry execution on 1-minute timeframe for optimal fills
- **Aggressive Position Sizing**: 100% deposit usage with auto-calculated leverage (5x-20x)
- **Single Position Mode**: "One shot, one target" strategy for maximum focus
- **Fixed Risk Management**: 5% risk per trade with automatic leverage calculation

### Key Features

- **Autonomous ML Models**: Own ensemble of models trained on historical data
- **Real-time Data Processing**: Continuous market data collection and analysis
- **Advanced Feature Engineering**: Technical indicators, candlestick patterns, market structure
- **Risk Management**: Multi-level limits, drawdown control, position sizing
- **Paper & Live Trading**: Test strategies before deploying real capital
- **Monitoring Dashboard**: Real-time performance tracking and alerts

### Target Performance (Aggressive Mode)

| Metric | Target |
|--------|--------|
| Win Rate | ≥ 55% |
| Risk:Reward | 1:3 |
| Risk per Trade | 5% |
| Leverage | 5x-20x (auto) |
| Trades/Day | 3-10 |

## Architecture

### v2.1 Architecture with Scanner

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Multi-Pair     │────▶│ Feature Engine  │────▶│   ML Models     │
│  Scanner        │     │  (Indicators)   │     │  (Ensemble)     │
│  (25+ pairs)    │     └─────────────────┘     └────────┬────────┘
└─────────────────┘                                      │
        │                                                │
        ▼                                       ┌────────▼────────┐
┌─────────────────┐     ┌─────────────────┐     │ Decision Engine │
│   M1 Sniper     │◀────│  Best Setup     │◀────│  (Score: 0-100) │
│ (Precise Entry) │     │  Selection      │     └─────────────────┘
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Execution    │◀────│  Risk Manager   │◀────│ Aggressive Sizer│
│ (Futures/Lever) │     │  (5% Fixed)     │     │ (100% Deposit)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### ML Model Ensemble

- **Direction Model**: Predicts price direction (up/down/sideways)
- **Strength Model**: Predicts movement magnitude in ATR
- **Volatility Model**: Predicts expected future volatility
- **Timing Model**: Determines optimal entry timing

## v2.1 Trading Flow

```
1. SCAN → Scan 25+ pairs for opportunities (every 60s)
2. SCORE → ML models score each pair (0-100)
3. SELECT → Choose best setup (score ≥ 70)
4. SNIPE → Wait for M1 entry trigger (max 15 candles)
5. CALCULATE → Auto-calculate leverage (Risk 5% / Stop Distance)
6. EXECUTE → Open position with 100% deposit
7. MANAGE → Trail stop, move to breakeven
8. CLOSE → Take profit at 1:3 RR or stop loss
9. REPEAT → Immediately scan for next opportunity
```

## Leverage Calculation

Leverage is auto-calculated to maintain 5% fixed risk:

| Stop Distance | Leverage | Risk |
|--------------|----------|------|
| 0.25% | 20x (max) | 5% |
| 0.50% | 10x | 5% |
| 1.00% | 5x (min) | 5% |

**Formula**: `Leverage = Fixed_Risk / Stop_Distance`

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/tradeAI.git
cd tradeAI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure the bot**
```bash
cp config/secrets.yaml.example config/secrets.yaml
# Edit config/secrets.yaml with your API keys
# Edit config/settings.yaml for general settings
# Edit config/trading_params.yaml for trading parameters
```

5. **Initialize database**
```bash
# Start PostgreSQL and Redis (or use Docker)
docker-compose -f docker/docker-compose.yml up -d postgres redis
```

6. **Train models (V8 Improved - Anti-Overfitting)**
```bash
# ⚠️ IMPORTANT: Use train_v3_dynamic.py for V8 models with proper feature exclusion
python scripts/train_v3_dynamic.py --days 60 --test_days 14 --walk-forward
```

7. **Validate model is ready for live trading**
```bash
python scripts/preflight_check.py --model-dir models/v8_improved --verbose
```

8. **Run backtest**
```bash
python scripts/backtest.py --symbol BTCUSDT --days 90
```

9. **Start the bot (paper trading)**
```bash
python main.py
```

> ⚠️ **CRITICAL:** Before live trading, ensure your model was trained with the latest `train_v3_dynamic.py` 
> which excludes absolute price features (EMA values, BB levels, ATR values) that cause backtest vs live discrepancy.

### Docker Deployment

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Configuration

### Main Settings (`config/settings.yaml`)

```yaml
app:
  version: "2.1.0"

exchange:
  name: "binance"
  testnet: true

trading:
  symbol: "BTCUSDT"
  mode: "paper"
  # v2.1: New settings
  aggressive_mode: true    # 100% deposit with leverage
  single_position: true    # One position at a time
  use_scanner: true        # Multi-pair scanning
```

### Trading Parameters (`config/trading_params.yaml`)

```yaml
# v2.1: Aggressive Mode Settings
risk:
  max_risk_per_trade: 0.05  # 5% fixed risk
  max_position_size: 1.0    # 100% of deposit

scanner:
  enabled: true
  min_score: 70
  pairs:
    - "BTCUSDT"
    - "ETHUSDT"
    - "SOLUSDT"
    # ... 25+ pairs

sniper:
  max_wait_candles: 15
  min_stop_pct: 0.002    # 0.2%
  max_stop_pct: 0.005    # 0.5%

aggressive_sizing:
  fixed_risk_pct: 0.05   # 5%
  min_leverage: 5
  max_leverage: 20
  take_profit_rr: 3.0    # 1:3 risk-reward
```

## Project Structure

```
ai-trading-bot/
├── config/                     # Configuration files
│   ├── settings.yaml           # Main settings
│   ├── trading_params.yaml     # Trading parameters
│   └── secrets.yaml            # API keys (not committed)
├── src/
│   ├── data/                   # Data collection & storage
│   ├── features/               # Feature engineering
│   ├── models/                 # ML models
│   ├── strategy/               # Trading strategy
│   ├── execution/              # Order execution
│   ├── risk/                   # Risk management
│   ├── monitoring/             # Monitoring & alerts
│   ├── scanner/                # v2.1: Multi-pair scanner
│   │   ├── pair_scanner.py     # Scans all pairs
│   │   ├── m1_sniper.py        # M1 precise entry
│   │   └── aggressive_sizing.py # 100% deposit sizing
│   └── utils/                  # Utilities
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
├── docker/                     # Docker configuration
├── main.py                     # Entry point
└── requirements.txt            # Dependencies
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## Risk Disclaimer

⚠️ **WARNING**: Cryptocurrency trading involves substantial risk of loss. This software is provided for educational purposes only.

- **Aggressive mode uses high leverage (up to 20x) - extreme caution required**
- **Never trade with money you cannot afford to lose**
- **Past performance does not guarantee future results**
- **Always start with paper trading**
- **Use small positions when going live**
- **The authors are not responsible for any trading losses**

## Changelog

### v2.1.0 (Current)
- ✅ Multi-pair scanner (25+ pairs)
- ✅ M1 sniper entry system
- ✅ Aggressive 100% deposit sizing
- ✅ Auto leverage calculation (5x-20x)
- ✅ Fixed 5% risk per trade
- ✅ Single position mode

### v1.0.0
- ✅ Basic trading bot
- ✅ ML ensemble models
- ✅ Risk management
- ✅ Paper trading mode

## Pre-Live Validation

Before going live with V8 Improved model, run these validation scripts:

### 1. Pre-Flight Check
```bash
# Check model files, features, and configuration
python scripts/preflight_check.py --model-dir models/v8_improved

# With verbose output
python scripts/preflight_check.py --model-dir models/v8_improved --verbose
```

### 2. Feature Distribution Check
```bash
# Check for feature drift
python scripts/compare_feature_distributions.py --pair BTC_USDT_USDT --hours 48

# Check all pairs
python scripts/compare_feature_distributions.py --all-pairs
```

### 3. Live Simulation Test
```bash
# Compare live-like feature generation with backtest
python scripts/simulate_live_trading.py --pair BTC_USDT_USDT --hours 48
```

### 4. Walk-Forward Validation
```bash
# Train and validate with walk-forward
python scripts/train_v3_dynamic.py --walk-forward --days 60 --test_days 14
```

### Pre-Live Checklist

```
[ ] All preflight_check.py checks pass
[ ] No cumsum-dependent features in model
[ ] Walk-forward win rate >= 60%
[ ] Paper trading for 7+ days
[ ] Paper WR matches walk-forward (+/- 10%)
```

See `docs/LIVE_TRADING_ANALYSIS.md` for detailed analysis.

## Roadmap

### Version 2.2
- [ ] Portfolio mode (multiple positions)
- [ ] Advanced trailing stops
- [ ] Custom pair filtering

### Version 3.0
- [ ] Reinforcement learning models
- [ ] Automatic hyperparameter optimization
- [ ] Web-based dashboard

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ccxt](https://github.com/ccxt/ccxt) - Cryptocurrency exchange library
- [LightGBM](https://github.com/microsoft/LightGBM) - Gradient boosting framework
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical analysis library

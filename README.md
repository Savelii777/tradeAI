# AI Trading Bot v1.0

🤖 **Autonomous ML-powered cryptocurrency trading bot for scalping**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A fully autonomous trading bot with its own machine learning system, capable of analyzing real-time market data and executing profitable trades on cryptocurrency exchanges without relying on external AI APIs.

### Key Features

- **Autonomous ML Models**: Own ensemble of models trained on historical data
- **Real-time Data Processing**: Continuous market data collection and analysis
- **Advanced Feature Engineering**: Technical indicators, candlestick patterns, market structure
- **Risk Management**: Multi-level limits, drawdown control, position sizing
- **Paper & Live Trading**: Test strategies before deploying real capital
- **Monitoring Dashboard**: Real-time performance tracking and alerts

### Target Performance

| Metric | Target |
|--------|--------|
| Win Rate | ≥ 55% |
| Profit Factor | ≥ 1.5 |
| Max Drawdown | ≤ 10% |
| Trades/Day | 5-15 |
| Monthly Return | 3-10% |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Collector │────▶│ Feature Engine  │────▶│   ML Models     │
│  (Exchange API) │     │  (Indicators)   │     │  (Ensemble)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│    Execution    │◀────│  Risk Manager   │◀────│ Decision Engine │
│  (Orders/Pos)   │     │  (Limits)       │     │  (Signals)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### ML Model Ensemble

- **Direction Model**: Predicts price direction (up/down/sideways)
- **Strength Model**: Predicts movement magnitude in ATR
- **Volatility Model**: Predicts expected future volatility
- **Timing Model**: Determines optimal entry timing

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

6. **Train models**
```bash
python scripts/train_models.py --symbol BTCUSDT --days 180
```

7. **Run backtest**
```bash
python scripts/backtest.py --symbol BTCUSDT --days 90
```

8. **Start the bot (paper trading)**
```bash
python main.py
```

### Docker Deployment

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Configuration

### Main Settings (`config/settings.yaml`)

```yaml
exchange:
  name: "binance"
  testnet: true  # Use testnet for paper trading

trading:
  symbol: "BTCUSDT"
  mode: "paper"  # "paper" or "live"

data:
  primary_timeframe: "5m"
  history_days: 30
```

### Trading Parameters (`config/trading_params.yaml`)

```yaml
risk:
  max_risk_per_trade: 0.02  # 2%
  max_daily_loss: 0.03      # 3%
  max_drawdown: 0.15        # 15%

entry:
  min_direction_probability: 0.60
  min_expected_move_atr: 1.5

exit:
  stop_loss_atr_multiplier: 1.5
  take_profit_min_rr: 2.0
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

- **Never trade with money you cannot afford to lose**
- **Past performance does not guarantee future results**
- **Always start with paper trading**
- **Use small positions when going live**
- **The authors are not responsible for any trading losses**

## Roadmap

### Version 1.1
- [ ] Multiple trading pairs support
- [ ] Improved execution speed
- [ ] Enhanced notifications

### Version 1.2
- [ ] Portfolio management
- [ ] Correlation analysis
- [ ] Multi-exchange support

### Version 2.0
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

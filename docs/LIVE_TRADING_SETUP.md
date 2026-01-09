# Live Trading Setup Guide

This guide explains how to set up and run the live trading bot on MEXC using market data from Binance.

## Prerequisites

1. **Trained Model**: The bot requires a trained V8 model. Train it with:
   ```bash
   python scripts/train_v3_dynamic.py --days 90 --test_days 30 --pairs 20 --initial_balance 20 --output ./models/v8_improved --reverse --walk-forward
   ```

2. **MEXC Account**: 
   - Create an account at [MEXC](https://www.mexc.com)
   - Enable Futures trading
   - Generate API keys with Futures trading permission (do NOT enable withdrawal)

3. **Telegram Bot** (optional but recommended):
   - Create a bot via [@BotFather](https://t.me/BotFather)
   - Get your chat ID via [@userinfobot](https://t.me/userinfobot)

## Configuration

### Step 1: Create secrets.yaml

Copy the example file and fill in your credentials:

```bash
cp config/secrets.yaml.example config/secrets.yaml
```

Edit `config/secrets.yaml`:

```yaml
# MEXC Futures API
mexc:
  api_key: "your_mexc_api_key"
  api_secret: "your_mexc_api_secret"

# Telegram notifications  
notifications:
  telegram:
    bot_token: "your_telegram_bot_token"
    chat_id: "your_telegram_chat_id"
```

**IMPORTANT**: Never commit `secrets.yaml` to version control! It's already in `.gitignore`.

### Step 2: Verify Model Files

Ensure the model files exist:

```bash
ls models/v8_improved/
# Should show:
# - direction_model.joblib
# - timing_model.joblib  
# - strength_model.joblib
# - feature_names.joblib
```

## Running the Bot

### Option 1: V9 Script (Recommended)

The new V9 script has cleaner code and better error handling:

```bash
python scripts/live_trading_v9.py
```

### Option 2: V8 Script (Legacy)

```bash
python scripts/live_trading_mexc_v8.py
```

## How It Works

1. **Data Source**: Market data (1m, 5m, 15m candles) is fetched from Binance (free, no auth required)
2. **Signal Generation**: The V8 model generates signals based on the trained features
3. **Trade Execution**: Trades are executed on MEXC Futures via their API
4. **Risk Management**: Uses the exact same backtest logic:
   - 5% risk per trade
   - Adaptive stop loss (1.2-1.6 ATR based on signal strength)
   - Breakeven trigger (1.2-1.8 ATR based on strength)
   - Progressive trailing stop

## Signal Thresholds

The bot uses these thresholds (matching backtest):

| Parameter | Value | Description |
|-----------|-------|-------------|
| MIN_CONF | 0.50 | Direction confidence |
| MIN_TIMING | 0.8 | ATR gain potential |
| MIN_STRENGTH | 1.4 | Predicted move strength |

## Monitoring

### Logs

Logs are written to `logs/live_trading_v9.log` (or `logs/live_trading.log` for V8).

View logs in real-time:
```bash
tail -f logs/live_trading_v9.log
```

### Telegram Notifications

The bot sends notifications for:
- 🟢 Trade opened (with entry, SL, leverage details)
- ✅/❌ Trade closed (with PnL, ROE)
- 🚀 Bot started/stopped

### State File

Active trades are saved to `active_trades_v9.json` (or `active_trades_mexc.json` for V8).

## Troubleshooting

### "Secrets file not found"

Copy the example file:
```bash
cp config/secrets.yaml.example config/secrets.yaml
```

### "Model directory not found"

Train the model first:
```bash
python scripts/train_v3_dynamic.py --days 90 --test_days 30
```

### "MEXC API request failed"

1. Check your API keys are correct
2. Ensure Futures trading is enabled on your MEXC account
3. Check your IP is whitelisted (if you set IP restrictions)

### No signals generated

1. Check the log for detailed rejection reasons
2. The model is selective - it may take time to find high-quality signals
3. Verify the model is properly trained with walk-forward validation

## Safety Notes

1. **Start Small**: Test with a small amount first
2. **Monitor**: Watch the first few trades closely
3. **No Guarantees**: Past performance doesn't guarantee future results
4. **Risk Only What You Can Lose**: Crypto trading involves significant risk

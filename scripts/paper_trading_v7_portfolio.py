#!/usr/bin/env python3
"""
Paper Trading Script (V7 Sniper) - PORTFOLIO MODE
- Loads trained models from models/v7_sniper_final/
- Connects to Binance via CCXT
- Fetches live data every 1 minute
- Manages a VIRTUAL PORTFOLIO with real backtest logic (Risk, Leverage, Fees)
- Sends alerts to Telegram
"""

import sys
import time
import json
import joblib
import ccxt
import requests
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v7_sniper_final")
PAIRS_FILE = Path("config/pairs_list.json")
TRADES_FILE = Path("active_trades.json")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 1000 # Candles to fetch

# Thresholds (Must match training)
# ORIGINAL (SAFE) VALUES:
MIN_CONF = 0.55
MIN_TIMING = 0.60
MIN_STRENGTH = 2.0

# TEST VALUES (AGGRESSIVE):
# MIN_CONF = 0.40
# MIN_TIMING = 0.40
# MIN_STRENGTH = 1.0

# Risk Management (From Backtest)
RISK_PCT = 0.05          # 5% risk per trade
MAX_LEVERAGE = 20.0      # Max leverage
SL_ATR = 1.5             # Stop Loss = 1.5 * ATR
TP_RR = 2.0              # Take Profit = 2.0 * Risk (Reward/Risk)
MAX_HOLDING_BARS = 50    # Max holding time (5m bars)
ENTRY_FEE = 0.0002       # 0.02%
EXIT_FEE = 0.0002        # 0.02%
INITIAL_CAPITAL = 20.0 # Virtual Capital

# Telegram Config
TELEGRAM_TOKEN = "8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc"
TELEGRAM_CHAT_ID = "677822370"

# ============================================================
# PORTFOLIO MANAGER
# ============================================================
class PortfolioManager:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.position = None # Only 1 position allowed (Global Limit)
        self.trades_history = []
        self.load_state()
        
    def load_state(self):
        if TRADES_FILE.exists():
            try:
                with open(TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    self.capital = data.get('capital', INITIAL_CAPITAL)
                    self.position = data.get('position', None)
                    self.trades_history = data.get('history', [])
                    logger.info(f"Loaded state. Capital: ${self.capital:.2f}, Position: {self.position['pair'] if self.position else 'None'}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        else:
            logger.info("No previous state found. Starting fresh.")

    def save_state(self):
        data = {
            'capital': self.capital,
            'position': self.position,
            'history': self.trades_history
        }
        with open(TRADES_FILE, 'w') as f:
            json.dump(data, f, indent=4, default=str)

    def open_position(self, signal):
        if self.position is not None:
            logger.info(f"Ignored signal for {signal['pair']} - Already in position {self.position['pair']}")
            return

        entry_price = signal['price']
        atr = signal['atr']
        
        # Calculate SL/TP
        stop_distance = atr * SL_ATR
        stop_loss_pct = stop_distance / entry_price
        
        if signal['direction'] == 'LONG':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * TP_RR)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * TP_RR)
            
        # Calculate Size & Leverage
        # Leverage = Risk% / SL% (capped at Max Leverage)
        leverage = min(RISK_PCT / stop_loss_pct, MAX_LEVERAGE)
        position_value = self.capital * leverage
        size = position_value / entry_price
        
        # Entry Fee
        fee = position_value * ENTRY_FEE
        self.capital -= fee # Deduct fee immediately
        
        self.position = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': datetime.now().isoformat(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
            'leverage': leverage,
            'position_value': position_value,
            'bars_held': 0,
            'highest_pnl': 0
        }
        
        self.save_state()
        self.send_alert(f"🟢 <b>OPEN {signal['direction']}</b> {signal['pair']}", 
                        f"Price: {entry_price:.4f}\nLev: {leverage:.1f}x\nSize: ${position_value:.0f}\nSL: {stop_loss:.4f}\nTP: {take_profit:.4f}")

    def update(self, current_prices):
        if self.position is None:
            return

        pair = self.position['pair']
        if pair not in current_prices:
            return
            
        current_price = current_prices[pair]
        self.position['bars_held'] += 1
        
        # Check Exit Conditions
        should_exit = False
        reason = ""
        
        # 1. Time Limit
        if self.position['bars_held'] >= MAX_HOLDING_BARS:
            should_exit = True
            reason = "Time Limit"
            
        # 2. Stop Loss / Take Profit
        if self.position['direction'] == 'LONG':
            if current_price <= self.position['stop_loss']:
                should_exit = True
                reason = "Stop Loss"
            elif current_price >= self.position['take_profit']:
                should_exit = True
                reason = "Take Profit"
        else: # SHORT
            if current_price >= self.position['stop_loss']:
                should_exit = True
                reason = "Stop Loss"
            elif current_price <= self.position['take_profit']:
                should_exit = True
                reason = "Take Profit"
                
        if should_exit:
            self.close_position(current_price, reason)

    def close_position(self, exit_price, reason):
        pos = self.position
        
        # Calculate PnL
        if pos['direction'] == 'LONG':
            price_change = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            price_change = (pos['entry_price'] - exit_price) / pos['entry_price']
            
        raw_pnl = price_change * pos['position_value']
        exit_fee = pos['position_value'] * EXIT_FEE
        net_pnl = raw_pnl - exit_fee
        
        self.capital += net_pnl
        pnl_pct = (net_pnl / (pos['position_value'] / pos['leverage'])) * 100 # ROE
        
        # Log
        trade_record = {
            'pair': pos['pair'],
            'direction': pos['direction'],
            'entry': pos['entry_price'],
            'exit': exit_price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'time': datetime.now().isoformat()
        }
        self.trades_history.append(trade_record)
        
        # Alert
        emoji = "✅" if net_pnl > 0 else "❌"
        self.send_alert(f"{emoji} <b>CLOSE {pos['direction']}</b> {pos['pair']}",
                        f"Reason: {reason}\nPrice: {exit_price:.4f}\nPnL: ${net_pnl:.2f} ({pnl_pct:.1f}%)\nCapital: ${self.capital:.2f}")
        
        self.position = None
        self.save_state()

    def send_alert(self, title, body):
        msg = f"{title}\n\n{body}"
        logger.info(f"TELEGRAM: {title}")
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'}
            requests.post(url, data=data)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# ============================================================
# UTILS
# ============================================================
def load_models():
    logger.info(f"Loading models from {MODEL_DIR}...")
    try:
        models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.pkl'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.pkl'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.pkl'),
            'features': joblib.load(MODEL_DIR / 'features.pkl')
        }
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

def get_pairs():
    with open(PAIRS_FILE, "r") as f:
        data = json.load(f)
        return [p['symbol'] for p in data['pairs']][:20]

def fetch_live_data(exchange, symbol):
    data = {}
    clean_symbol = symbol.replace('_', '/')
    if '/' not in clean_symbol:
        clean_symbol = f"{clean_symbol[:-4]}/{clean_symbol[-4:]}"
    try:
        for tf in TIMEFRAMES:
            candles = exchange.fetch_ohlcv(clean_symbol, tf, limit=LOOKBACK)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        return data
    except Exception as e:
        logger.warning(f"Error fetching {symbol}: {e}")
        return None

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe):
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    return ft.dropna()

# ============================================================
# SIGNAL GENERATION
# ============================================================
def check_signal(df, models, pair):
    if len(df) < 2: return None, None
    latest = df.iloc[[-2]] 
    feature_cols = models['features']
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing: return None, None
    X = latest[feature_cols].values
    
    dir_proba = models['direction'].predict_proba(X)[0]
    dir_pred = np.argmax(dir_proba)
    dir_conf = np.max(dir_proba)
    timing_prob = models['timing'].predict_proba(X)[0][1]
    strength_pred = models['strength'].predict(X)[0]
    
    stats = {'pair': pair, 'dir': dir_pred, 'conf': dir_conf, 'timing': timing_prob, 'strength': strength_pred}
    
    if dir_pred == 1: return None, stats
    if dir_conf < MIN_CONF: return None, stats
    if timing_prob < MIN_TIMING: return None, stats
    if strength_pred < MIN_STRENGTH: return None, stats
    
    direction = 'LONG' if dir_pred == 2 else 'SHORT'
    return {
        'timestamp': latest.index[0],
        'pair': pair,
        'direction': direction,
        'price': latest['close'].iloc[0],
        'atr': latest['atr'].iloc[0],
        'conf': dir_conf,
        'timing': timing_prob,
        'strength': strength_pred
    }, stats

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    logger.info("Starting Paper Trading Bot (V7 Sniper) - PORTFOLIO MODE...")
    
    models = load_models()
    pairs = get_pairs()
    exchange = ccxt.binance()
    mtf_fe = MTFFeatureEngine()
    portfolio = PortfolioManager()
    
    logger.info(f"Monitoring {len(pairs)} pairs. Capital: ${portfolio.capital:.2f}")
    
    last_processed = {}
    
    while True:
        logger.info(f"Scanning... {datetime.now().strftime('%H:%M:%S')}")
        current_prices = {}
        best_candidate = None
        
        for pair in pairs:
            try:
                data = fetch_live_data(exchange, pair)
                if not data: continue
                
                # Store current price for portfolio update
                current_price = data['5m']['close'].iloc[-1]
                current_prices[pair] = current_price
                
                df = prepare_features(data, mtf_fe)
                signal, stats = check_signal(df, models, pair)
                
                # Logging logic
                if stats:
                    is_directional = stats['dir'] != 1
                    if best_candidate is None:
                        best_candidate = stats
                    else:
                        best_is_directional = best_candidate['dir'] != 1
                        if is_directional and not best_is_directional:
                            best_candidate = stats
                        elif is_directional == best_is_directional:
                            if stats['conf'] > best_candidate['conf']:
                                best_candidate = stats
                
                # Trading logic
                if signal:
                    last_time = last_processed.get(pair)
                    current_time = signal['timestamp']
                    if last_time != current_time:
                        portfolio.open_position(signal)
                        last_processed[pair] = current_time
                    
            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")
        
        # Update active positions
        portfolio.update(current_prices)
        
        if best_candidate:
            dir_map = {0: 'SHORT', 1: 'SIDEWAYS', 2: 'LONG'}
            d_str = dir_map.get(best_candidate['dir'], 'UNKNOWN')
            logger.info(f"Best: {best_candidate['pair']} [{d_str}] (Conf: {best_candidate['conf']:.1%}, Timing: {best_candidate['timing']:.1%}, Str: {best_candidate['strength']:.2f})")
        else:
            logger.info("No data fetched.")
            
        time.sleep(60)

if __name__ == '__main__':
    main()

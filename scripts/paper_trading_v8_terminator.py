#!/usr/bin/env python3
"""
Paper Trading Script (V8 TERMINATOR) - PORTFOLIO MODE
- Loads trained models from models/v8_terminator/
- Connects to Binance via CCXT
- Fetches live data every 1 minute
- Manages a VIRTUAL PORTFOLIO with V8 Logic (Smart Exit, Wick Veto, BTC Veto)
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
MODEL_DIR = Path("models/v8_terminator")
PAIRS_FILE = Path("config/pairs_list.json")
TRADES_FILE = Path("active_trades_v8.json")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 1000 # Candles to fetch

# Thresholds (V8 Terminator)
MIN_CONF = 0.55
MIN_TIMING = 0.60
MIN_STRENGTH = 2.0

# Smart Logic
RSI_EXIT_LONG = 75
RSI_EXIT_SHORT = 25
WICK_REJECTION_RATIO = 2.0
BTC_VETO_RSI_LOW = 30
BTC_VETO_RSI_HIGH = 70

# Risk Management
RISK_PCT = 0.05          # 5% risk per trade
MAX_LEVERAGE = 20.0      # Max leverage
SL_ATR = 1.5             # Stop Loss = 1.5 * ATR
TP_RR = 2.0              # Take Profit = 2.0 * Risk (Reward/Risk) - Dynamic in V8 but base is here
MAX_HOLDING_BARS = 150   # Max holding time (5m bars) - Increased for V8
ENTRY_FEE = 0.0002       # 0.02%
EXIT_FEE = 0.0002        # 0.02%
INITIAL_CAPITAL = 10000.0 # Virtual Capital

# Telegram Config
TELEGRAM_TOKEN = "8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc"
TELEGRAM_CHAT_ID = "677822370"

# ============================================================
# FEATURE ENGINEERING (V8)
# ============================================================
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume + Wick + Microstructure features (V8)."""
    df = df.copy()
    
    # 1. Volume Features
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # OBV
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # Volume momentum
    df['vol_momentum'] = df['volume'].pct_change(5)

    # 2. Wick Features
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    
    # Ratios
    df['upper_wick_pct'] = df['upper_wick'] / df['close']
    df['lower_wick_pct'] = df['lower_wick'] / df['close']
    df['body_pct'] = df['body'] / df['close']
    
    # Wick-to-Body Ratio
    df['wick_ratio_up'] = df['upper_wick'] / (df['body'] + 1e-9)
    df['wick_ratio_down'] = df['lower_wick'] / (df['body'] + 1e-9)
    
    # 3. RSI (For Smart Exit)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
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
    ft = add_advanced_features(ft)
    ft['atr'] = calculate_atr(ft)
    return ft.dropna()

# ============================================================
# PORTFOLIO MANAGER
# ============================================================
class PortfolioManager:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.position = None # Only 1 position allowed
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
            return

        # Wick Veto Check
        if signal['direction'] == 'LONG':
            if signal['wick_ratio_up'] > WICK_REJECTION_RATIO:
                logger.info(f"🚫 VETO: Long rejected due to upper wick (Ratio: {signal['wick_ratio_up']:.2f})")
                return
        else:
            if signal['wick_ratio_down'] > WICK_REJECTION_RATIO:
                logger.info(f"🚫 VETO: Short rejected due to lower wick (Ratio: {signal['wick_ratio_down']:.2f})")
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
        leverage = min(RISK_PCT / stop_loss_pct, MAX_LEVERAGE)
        position_value = self.capital * leverage
        size = position_value / entry_price
        
        # Entry Fee
        fee = position_value * ENTRY_FEE
        self.capital -= fee
        
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
            'breakeven_active': False
        }
        
        self.save_state()
        self.send_alert(f"🤖 <b>TERMINATOR ENTRY {signal['direction']}</b> {signal['pair']}", 
                        f"Price: {entry_price:.4f}\nLev: {leverage:.1f}x\nSize: ${position_value:.0f}\nSL: {stop_loss:.4f}\nRSI: {signal['rsi']:.1f}")

    def update(self, data_map):
        if self.position is None:
            return

        pair = self.position['pair']
        if pair not in data_map:
            return
            
        # Get latest 5m candle
        df = data_map[pair]['5m']
        if df.empty:
            return
            
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        
        # Calculate RSI on the fly (using the dataframe which has history)
        # We need to ensure 'rsi' is in the dataframe. 
        # It is added in prepare_features, but data_map contains raw OHLCV.
        # So we need to calc RSI here quickly.
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        self.position['bars_held'] += 1
        
        should_exit = False
        reason = ""
        
        # 1. Time Limit
        if self.position['bars_held'] >= MAX_HOLDING_BARS:
            should_exit = True
            reason = "Time Limit"
            
        # 2. Stop Loss / Take Profit (Hard)
        if self.position['direction'] == 'LONG':
            if current_price <= self.position['stop_loss']:
                should_exit = True
                reason = "Stop Loss"
            # Smart Exit (RSI)
            elif current_rsi > RSI_EXIT_LONG:
                should_exit = True
                reason = "Smart Exit (RSI)"
        else: # SHORT
            if current_price >= self.position['stop_loss']:
                should_exit = True
                reason = "Stop Loss"
            # Smart Exit (RSI)
            elif current_rsi < RSI_EXIT_SHORT:
                should_exit = True
                reason = "Smart Exit (RSI)"
                
        # 3. Breakeven Logic (Simplified for Paper)
        # If we are > 1R in profit, move SL to Breakeven
        entry = self.position['entry_price']
        sl = self.position['stop_loss']
        risk = abs(entry - sl)
        
        if not self.position['breakeven_active']:
            if self.position['direction'] == 'LONG':
                if current_price > entry + risk:
                    self.position['stop_loss'] = entry + (risk * 0.1)
                    self.position['breakeven_active'] = True
                    logger.info(f"🛡️ Moved {pair} SL to Breakeven")
            else:
                if current_price < entry - risk:
                    self.position['stop_loss'] = entry - (risk * 0.1)
                    self.position['breakeven_active'] = True
                    logger.info(f"🛡️ Moved {pair} SL to Breakeven")

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
        pnl_pct = (net_pnl / (pos['position_value'] / pos['leverage'])) * 100 
        
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
            'direction': joblib.load(MODEL_DIR / 'model_direction.pkl'),
            'timing': joblib.load(MODEL_DIR / 'model_timing.pkl'),
            'strength': joblib.load(MODEL_DIR / 'model_strength.pkl'),
            # 'features': joblib.load(MODEL_DIR / 'features.pkl') # Not used directly, we reconstruct
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

def get_btc_veto(exchange):
    """Check BTC Trend for Veto."""
    try:
        candles = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calc RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        veto = {'long': False, 'short': False}
        if current_rsi < BTC_VETO_RSI_LOW:
            veto['long'] = True # Don't buy the dip if it's crashing
        if current_rsi > BTC_VETO_RSI_HIGH:
            veto['short'] = True # Don't short the pump
            
        return veto
    except Exception as e:
        logger.warning(f"BTC Veto Error: {e}")
        return {'long': False, 'short': False}

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    logger.info("Starting V8 TERMINATOR Paper Trading...")
    
    # Init
    exchange = ccxt.binance()
    models = load_models()
    pairs = get_pairs()
    mtf_fe = MTFFeatureEngine()
    portfolio = PortfolioManager()
    
    # Feature Columns (Must match training)
    # We need to generate a dummy df to get columns or hardcode them
    # For now, we assume the feature engine produces consistent columns
    
    while True:
        try:
            logger.info(f"Scanning {len(pairs)} pairs...")
            
            # 1. Check BTC Veto
            veto = get_btc_veto(exchange)
            if veto['long']: logger.warning("BTC DUMPING (RSI < 30) - Longs Vetoed")
            if veto['short']: logger.warning("BTC PUMPING (RSI > 70) - Shorts Vetoed")
            
            # 2. Fetch Data & Update Portfolio
            data_map = {}
            for pair in pairs:
                data = fetch_live_data(exchange, pair)
                if data:
                    data_map[pair] = data
            
            # Update existing position
            portfolio.update(data_map)
            
            # If we have a position, we don't scan for new ones (Single Slot)
            if portfolio.position is not None:
                logger.info(f"Position active: {portfolio.position['pair']}. Waiting...")
                time.sleep(60)
                continue
                
            # 3. Scan for Signals
            best_signal = None
            
            for pair, data in data_map.items():
                try:
                    df = prepare_features(data, mtf_fe)
                    if df.empty: continue
                    
                    # Predict
                    # Ensure columns match model expectation
                    # We use the model's feature_name_ property if available, or just pass the df
                    # LightGBM usually handles pandas df if columns match
                    
                    # Get latest row
                    row = df.iloc[[-1]]
                    X = row.values # Or row depending on how model was trained. 
                    # In training: X_train = df[feature_cols].values
                    # We need to ensure we drop non-feature columns like 'open', 'close', etc if they weren't in X
                    # Actually, in training we passed the whole DF to generate_signals but extracted feature_cols
                    # We need to know which columns are features.
                    
                    # HACK: We assume all numeric columns generated by prepare_features ARE features
                    # EXCEPT: open, high, low, close, volume, timestamp, target_*, etc.
                    # Let's filter standard OHLCV out if they are present
                    # But wait, prepare_features returns a DF with features AND OHLCV.
                    # We need the exact list of features.
                    # Let's load it from the model object if possible or infer.
                    
                    # In train_v4_smart.py, we didn't save feature_cols list explicitly in the pickle dict.
                    # But LightGBM models store feature names.
                    feature_names = models['direction'].feature_name_
                    X = row[feature_names].values
                    
                    # 1. Direction
                    dir_proba = models['direction'].predict_proba(X)
                    dir_pred = np.argmax(dir_proba, axis=1)[0]
                    dir_conf = np.max(dir_proba, axis=1)[0]
                    
                    if dir_pred == 1: continue # Sideways
                    if dir_conf < MIN_CONF: continue
                    
                    # BTC Veto Check
                    direction = 'LONG' if dir_pred == 2 else 'SHORT'
                    if direction == 'LONG' and veto['long']: continue
                    if direction == 'SHORT' and veto['short']: continue
                    
                    # 2. Timing
                    timing_prob = models['timing'].predict_proba(X)[:, 1][0]
                    if timing_prob < MIN_TIMING: continue
                    
                    # 3. Strength
                    strength_pred = models['strength'].predict(X)[0]
                    if strength_pred < MIN_STRENGTH: continue
                    
                    # Signal Found
                    signal = {
                        'pair': pair,
                        'direction': direction,
                        'price': row['close'].values[0],
                        'atr': row['atr'].values[0],
                        'score': dir_conf * timing_prob,
                        'wick_ratio_up': row['wick_ratio_up'].values[0],
                        'wick_ratio_down': row['wick_ratio_down'].values[0],
                        'rsi': row['rsi'].values[0]
                    }
                    
                    # Compare with best signal
                    if best_signal is None or signal['score'] > best_signal['score']:
                        best_signal = signal
                        
                except Exception as e:
                    logger.error(f"Error processing {pair}: {e}")
                    continue
            
            # Execute Best Signal
            if best_signal:
                logger.info(f"Found Signal: {best_signal['pair']} {best_signal['direction']} (Score: {best_signal['score']:.2f})")
                portfolio.open_position(best_signal)
            else:
                logger.info("No valid signals found.")
                
            time.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Stopping...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

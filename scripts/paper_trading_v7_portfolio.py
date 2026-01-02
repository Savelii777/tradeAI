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
from datetime import datetime, timezone, timedelta
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
TRADES_FILE = Path("active_trades.json")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 500 # Candles to fetch (enough for EMA200 + buffer)

# Thresholds (Must match training)
# ORIGINAL (SAFE) VALUES:
# MIN_CONF = 0.55
# MIN_TIMING = 0.60
# MIN_STRENGTH = 2.0

# TEST VALUES (AGGRESSIVE):
MIN_CONF = 0.50      
MIN_TIMING = 0.55    
MIN_STRENGTH = 1.4  

# Risk Management (From Backtest)
RISK_PCT = 0.05          # 5% risk per trade
MAX_LEVERAGE = 20.0      # Max leverage
SL_ATR_BASE = 1.5        # Base Stop Loss = 1.5 * ATR
MAX_HOLDING_BARS = 150   # Max holding time (12.5 hours)
ENTRY_FEE = 0.0002       # 0.02%
EXIT_FEE = 0.0002        # 0.02%
INITIAL_CAPITAL = 20.0   # Virtual Capital

# V8 IMPROVEMENTS
USE_ADAPTIVE_SL = True       # Adjust SL based on predicted strength
USE_DYNAMIC_LEVERAGE = True  # Boost leverage for high-confidence signals
USE_AGGRESSIVE_TRAIL = True  # Tighter trailing at medium R-multiples

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
        pred_strength = signal.get('strength', 2.0)
        conf = signal.get('conf', 0.5)
        timing = signal.get('timing', 0.5)
        
        # === V8: ADAPTIVE STOP LOSS ===
        # High strength prediction → tighter SL (more leverage, bigger wins)
        # Low strength prediction → wider SL (safer, but smaller position)
        if USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:      # Strong signal: tight SL
                sl_mult = 1.2
            elif pred_strength >= 2.0:    # Medium signal: normal SL
                sl_mult = 1.5
            else:                          # Weak signal: wider SL
                sl_mult = 1.8
        else:
            sl_mult = SL_ATR_BASE
        
        stop_distance = atr * sl_mult
        stop_loss_pct = stop_distance / entry_price
        
        if signal['direction'] == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # === V8: DYNAMIC BREAKEVEN TRIGGER ===
        if pred_strength >= 3.0:
            be_trigger_mult = 1.8   # Wait longer for strong signals
        elif pred_strength >= 2.0:
            be_trigger_mult = 1.5   # Standard
        else:
            be_trigger_mult = 1.2   # Fast BE for weak signals
            
        # === V8: DYNAMIC RISK/LEVERAGE ===
        if USE_DYNAMIC_LEVERAGE:
            # Quality multiplier: 0.8x to 1.5x based on signal quality
            score = conf * timing
            quality = (score / 0.3) * (timing / 0.6) * (pred_strength / 2.0)
            quality_mult = np.clip(quality, 0.8, 1.5)
            risk_pct = RISK_PCT * quality_mult
        else:
            risk_pct = RISK_PCT
            
        # Calculate Size & Leverage
        # Leverage = Risk% / SL% (capped at Max Leverage)
        leverage = min(risk_pct / stop_loss_pct, MAX_LEVERAGE)
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
            'stop_distance': stop_distance, # Saved for Trailing Stop
            'size': size,
            'leverage': leverage,
            'position_value': position_value,
            'bars_held': 0,
            'breakeven_active': False,  # Track if breakeven is active
            'last_candle_time': None,   # Track last processed candle to avoid duplicates
            'pred_strength': pred_strength,  # V8: Save for adaptive trailing
            'be_trigger_mult': be_trigger_mult  # V8: Dynamic BE trigger
        }
        
        self.save_state()
        self.send_alert(f"🟢 <b>OPEN {signal['direction']}</b> {signal['pair']}", 
                        f"Price: {entry_price:.4f}\nLev: {leverage:.1f}x\nSize: ${position_value:.0f}\nSL: {stop_loss:.4f} ({sl_mult:.1f}ATR)\nStrength: {pred_strength:.1f}")

    def update_trailing_stop(self, candle_data):
        """
        Update trailing stop using last completed candle (matches backtest logic).
        V8 IMPROVEMENTS: Adaptive BE trigger, Aggressive trailing
        """
        if self.position is None:
            return

        pair = self.position['pair']
        if pair not in candle_data:
            return
            
        candle = candle_data[pair]
        bar_high = candle['high']
        bar_low = candle['low']
        bar_close = candle['close']
        
        # Only update on new candle (track last processed timestamp)
        last_candle_time = self.position.get('last_candle_time')
        current_candle_time = candle.get('timestamp')
        
        # Skip if same candle (avoid duplicate processing)
        if last_candle_time == current_candle_time:
            return
            
        self.position['last_candle_time'] = current_candle_time
        self.position['bars_held'] += 1
        
        # --- V8 TRAILING STOP LOGIC ---
        pos = self.position
        entry_price = pos['entry_price']
        sl_dist = pos['stop_distance']
        # Calculate ATR from stop distance (account for adaptive SL)
        pred_strength = pos.get('pred_strength', 2.0)
        if USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:
                sl_mult = 1.2
            elif pred_strength >= 2.0:
                sl_mult = 1.5
            else:
                sl_mult = 1.8
        else:
            sl_mult = SL_ATR_BASE
        atr = sl_dist / sl_mult
        
        sl_price = pos['stop_loss']
        breakeven_active = pos.get('breakeven_active', False)
        
        # V8: Dynamic BE trigger from position data
        be_trigger_mult = pos.get('be_trigger_mult', 1.5)
        be_trigger_dist = atr * be_trigger_mult
        
        if pos['direction'] == 'LONG':
            # 1. Check Stop Loss FIRST
            if bar_low <= sl_price:
                exit_price = sl_price
                reason = "Stop Loss" if not breakeven_active else "Trailing Stop"
                self.close_position(exit_price, reason)
                return
            
            # 2. Check Breakeven Trigger
            be_trigger_price = entry_price + be_trigger_dist
            if not breakeven_active and bar_high >= be_trigger_price:
                breakeven_active = True
                pos['breakeven_active'] = True
                sl_price = entry_price + (atr * 0.3)  # V8: Higher BE margin
                pos['stop_loss'] = sl_price
                logger.info(f"Moved to Breakeven+: {sl_price:.4f}")
            
            # 3. V8: AGGRESSIVE Progressive Trailing Stop
            if breakeven_active:
                current_profit = bar_high - entry_price
                r_multiple = current_profit / sl_dist
                
                if USE_AGGRESSIVE_TRAIL:
                    if r_multiple > 5.0:      # Super Pump: Lock it in
                        trail_mult = 0.4
                    elif r_multiple > 3.0:    # Good Trend: Tight trail
                        trail_mult = 0.8
                    elif r_multiple > 2.0:    # Medium: Medium trail
                        trail_mult = 1.2
                    else:                      # Early: Loose trail
                        trail_mult = 1.8
                else:
                    trail_mult = 2.0
                    if r_multiple > 5.0:
                        trail_mult = 0.5
                    elif r_multiple > 3.0:
                        trail_mult = 1.5
                
                new_sl = bar_high - (atr * trail_mult)
                if new_sl > sl_price:
                    sl_price = new_sl
                    pos['stop_loss'] = sl_price
                    logger.info(f"Trailing Stop Updated (R={r_multiple:.1f}, trail={trail_mult}ATR): {sl_price:.4f}")
                    
        else: # SHORT
            # 1. Check Stop Loss FIRST
            if bar_high >= sl_price:
                exit_price = sl_price
                reason = "Stop Loss" if not breakeven_active else "Trailing Stop"
                self.close_position(exit_price, reason)
                return
            
            # 2. Check Breakeven Trigger
            be_trigger_price = entry_price - be_trigger_dist
            if not breakeven_active and bar_low <= be_trigger_price:
                breakeven_active = True
                pos['breakeven_active'] = True
                sl_price = entry_price - (atr * 0.3)  # V8: Higher BE margin
                pos['stop_loss'] = sl_price
                logger.info(f"Moved to Breakeven+: {sl_price:.4f}")
            
            # 3. V8: AGGRESSIVE Progressive Trailing Stop
            if breakeven_active:
                current_profit = entry_price - bar_low
                r_multiple = current_profit / sl_dist
                
                if USE_AGGRESSIVE_TRAIL:
                    if r_multiple > 5.0:      # Super Pump: Lock it in
                        trail_mult = 0.4
                    elif r_multiple > 3.0:    # Good Trend: Tight trail
                        trail_mult = 0.8
                    elif r_multiple > 2.0:    # Medium: Medium trail
                        trail_mult = 1.2
                    else:                      # Early: Loose trail
                        trail_mult = 1.8
                else:
                    trail_mult = 2.0
                    if r_multiple > 5.0:
                        trail_mult = 0.5
                    elif r_multiple > 3.0:
                        trail_mult = 1.5
                
                new_sl = bar_low + (atr * trail_mult)
                if new_sl < sl_price:
                    sl_price = new_sl
                    pos['stop_loss'] = sl_price
                    logger.info(f"Trailing Stop Updated (R={r_multiple:.1f}, trail={trail_mult}ATR): {sl_price:.4f}")

        # Save state after updating stop loss
        self.save_state()

    def check_stop_loss(self, current_price):
        """
        Check stop loss using current price (called frequently for better protection).
        This allows faster exit than waiting for candle close.
        """
        if self.position is None:
            return False
            
        pos = self.position
        sl_price = pos['stop_loss']
        entry_price = pos['entry_price']
        breakeven_active = pos.get('breakeven_active', False)
        
        should_exit = False
        reason = ""
        
        # Time Limit Check
        if pos['bars_held'] >= MAX_HOLDING_BARS:
            should_exit = True
            reason = "Time Limit"
        
        # Stop Loss Check (using current price for faster execution)
        elif pos['direction'] == 'LONG':
            if current_price <= sl_price:
                should_exit = True
                reason = "Stop Loss" if sl_price < entry_price else "Trailing Stop"
        else: # SHORT
            if current_price >= sl_price:
                should_exit = True
                reason = "Stop Loss" if sl_price > entry_price else "Trailing Stop"
        
        if should_exit:
            self.close_position(current_price, reason)
            return True
        
        return False

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
            'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
            'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
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
        print("\n" + "="*60)
        print(f"SCANNING MARKET - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        print(f"{'PAIR':<15} | {'DIR':<8} | {'CONF':<6} | {'TIMING':<6} | {'STR':<5} | {'STATUS'}")
        print("-" * 75)

        candle_data = {}  # Store last completed candle data for position updates
        
        for pair in pairs:
            try:
                data = fetch_live_data(exchange, pair)
                if not data: continue
                
                # Use last completed candle ([-2]) for signal generation and position updates
                # This matches backtest logic which uses closed candles
                if len(data['5m']) < 2: continue
                last_completed_candle = data['5m'].iloc[-2]
                
                # Store candle data for position updates (matches backtest bar-by-bar logic)
                candle_data[pair] = {
                    'high': last_completed_candle['high'],
                    'low': last_completed_candle['low'],
                    'close': last_completed_candle['close'],
                    'timestamp': last_completed_candle.name
                }
                current_price = last_completed_candle['close']  # For signal entry price
                
                df = prepare_features(data, mtf_fe)
                
                # --- INLINE SIGNAL CHECK FOR PRINTING ---
                # Use [-2] (Last Completed Candle) to avoid repainting
                if len(df) < 2: continue
                row = df.iloc[[-2]]
                X = row[models['features']].values
                
                # 1. Direction
                dir_proba = models['direction'].predict_proba(X)
                dir_pred = np.argmax(dir_proba, axis=1)[0]
                dir_conf = np.max(dir_proba, axis=1)[0]
                
                dir_map = {0: 'SHORT', 1: 'SIDE', 2: 'LONG'}
                direction = dir_map.get(dir_pred, 'UNK')
                
                # 2. Timing
                timing_prob = models['timing'].predict_proba(X)[:, 1][0]
                
                # 3. Strength
                strength_pred = models['strength'].predict(X)[0]
                
                # Status
                status = "WAIT"
                if direction != "SIDE":
                    if dir_conf >= MIN_CONF and timing_prob >= MIN_TIMING and strength_pred >= MIN_STRENGTH:
                        status = "SIGNAL 🚀"
                        
                        # Execute Signal
                        signal = {
                            'timestamp': row.index[0],
                            'pair': pair,
                            'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                            'price': current_price,
                            'atr': row['atr'].values[0]
                        }
                        
                        last_time = last_processed.get(pair)
                        current_time = signal['timestamp']
                        
                        # Check for Stale Signal (Must be within 2 minutes of candle close)
                        # Candle Timestamp is Open Time. Close Time is +5m.
                        candle_close_time = current_time + timedelta(minutes=5)
                        # Use UTC to match CCXT/Pandas timestamps
                        now = datetime.now(timezone.utc).replace(tzinfo=None)
                        delay = (now - candle_close_time).total_seconds()
                        
                        if delay > 120: # If more than 2 minutes late
                             status = f"STALE ({int(delay)}s)"
                        elif last_time != current_time:
                            portfolio.open_position(signal)
                            last_processed[pair] = current_time
                    else:
                        reasons = []
                        if dir_conf < MIN_CONF: reasons.append("LowConf")
                        if timing_prob < MIN_TIMING: reasons.append("BadTime")
                        if strength_pred < MIN_STRENGTH: reasons.append("Weak")
                        status = f"SKIP ({','.join(reasons)})"
                
                # Print Table Row
                print(f"{pair:<15} | {direction:<8} | {dir_conf:.2f} | {timing_prob:.2f}   | {strength_pred:.2f}  | {status}")
                    
            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")
        
        print("="*60 + "\n")
        
        # 1. Update trailing stops using last completed candles (matches backtest logic)
        # This updates stop loss based on candle high/low (like backtest) - called every 10s but only processes new candles
        portfolio.update_trailing_stop(candle_data)
        
        # 2. Check stop loss with current prices (frequent check every 10s for better protection)
        # This allows faster exit than waiting for candle close
        if portfolio.position:
            pair = portfolio.position['pair']
            try:
                # Get current ticker price for stop loss check
                clean_symbol = pair.replace('_', '/')
                if '/' not in clean_symbol:
                    clean_symbol = f"{clean_symbol[:-4]}/{clean_symbol[-4:]}"
                ticker = exchange.fetch_ticker(clean_symbol)
                current_price = ticker['last']  # Last traded price
                
                # Check if stop loss was hit (uses current price for faster execution)
                portfolio.check_stop_loss(current_price)
            except Exception as e:
                logger.error(f"Error checking stop loss for {pair}: {e}")
            
        time.sleep(10)

if __name__ == '__main__':
    main()

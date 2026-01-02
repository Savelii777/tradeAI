#!/usr/bin/env python3
"""
Paper Trading Script (V8 Sniper) - WEBSOCKET VERSION (FULL LOGIC PORT)
- Uses WebSocketManager for real-time data (No more lag!)
- Thread-safe data sharing
- Instant Stop-Loss & Trailing Execution
- FULL Feature Parity with v7 (Time Exit, Dynamic BE, Exact Trailing)
"""

import sys
import time
import json
import joblib
import ccxt
import requests
import pandas as pd
import numpy as np
import threading
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine
from src.data.websocket_manager import WebSocketManager

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
TRADES_FILE = Path("active_trades_ws.json") 
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 500

# Thresholds
MIN_CONF = 0.50      
MIN_TIMING = 0.55    
MIN_STRENGTH = 1.4  

# Risk Management
RISK_PCT = 0.05          # 5% risk (Base)
MAX_LEVERAGE = 20.0      # Max leverage (Matches V7)
MAX_HOLDING_BARS = 150   # 12.5 Hours (on 5m)
ENTRY_FEE = 0.0002       
EXIT_FEE = 0.0002        
INITIAL_CAPITAL = 20.0   
SL_ATR_BASE = 1.5

# V8 Features
USE_ADAPTIVE_SL = True       
USE_DYNAMIC_LEVERAGE = True  
USE_AGGRESSIVE_TRAIL = True  

# Telegram
TELEGRAM_TOKEN = "8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc"
TELEGRAM_CHAT_ID = "677822370"

# ============================================================
# DATA STREAMER
# ============================================================
class DataStreamer:
    def __init__(self, pairs):
        self.pairs = pairs
        self.ws_manager = WebSocketManager('binance')
        self.lock = threading.Lock()
        self.ready = False
        self.current_prices = {} 

    async def _run_ws(self):
        await self.ws_manager.connect()
        logger.info("WebSocket Connected. Subscribing...")
        
        for pair in self.pairs:
            # Subscribe to TRADES for instant updates
            await self.ws_manager.subscribe_trades(pair, self._on_trade)
            await asyncio.sleep(0.2) # Rate limit protection (Binance allows 5 msg/sec)
            
        while True:
            await asyncio.sleep(1)

    def _on_trade(self, trade):
        with self.lock:
            self.current_prices[trade.symbol] = trade.price
            
        # INSTANT EXIT CHECK
        if hasattr(self, 'on_price_update'):
            self.on_price_update(trade.symbol, trade.price)

    def start(self):
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_ws())
        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        logger.info("WS Thread started.")

# ============================================================
# PORTFOLIO MANAGER
# ============================================================
class PortfolioManager:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.position = None 
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
                    
                    # Convert timestamps back to datetime if needed
                    if self.position and 'entry_time' in self.position:
                        if isinstance(self.position['entry_time'], str):
                            self.position['entry_time'] = datetime.fromisoformat(self.position['entry_time'])
            except Exception:
                pass

    def save_state(self):
        # Serialize datetime
        pos_copy = None
        if self.position:
            pos_copy = self.position.copy()
            if isinstance(pos_copy['entry_time'], datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
                
        data = {'capital': self.capital, 'position': pos_copy, 'history': self.trades_history}
        with open(TRADES_FILE, 'w') as f:
            json.dump(data, f, indent=4, default=str)

    def open_position(self, signal):
        if self.position is not None: return

        entry_price = signal['price']
        atr = signal['atr']
        pred_strength = signal.get('strength', 2.0)
        conf = signal.get('conf', 0.5)
        timing = signal.get('timing', 0.5)
        
        # === V8: ADAPTIVE STOP LOSS (FIXED LOGIC) ===
        # Weak -> Tight (1.2), Strong -> Wide (1.6)
        if USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:      # Strong
                sl_mult = 1.6
            elif pred_strength >= 2.0:    # Medium
                sl_mult = 1.5
            else:                         # Weak
                sl_mult = 1.2
        else:
            sl_mult = SL_ATR_BASE
        
        stop_distance = atr * sl_mult
        
        if signal['direction'] == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
            
        # === V8: DYNAMIC BREAKEVEN TRIGGER (MATCHING V7) ===
        if pred_strength >= 3.0:
            be_trigger_mult = 1.8   # Wait longer for strong signals
        elif pred_strength >= 2.0:
            be_trigger_mult = 1.5   # Standard
        else:
            be_trigger_mult = 1.2   # Fast BE for weak signals

        # === V8: DYNAMIC RISK ===
        if USE_DYNAMIC_LEVERAGE:
            # Quality multiplier: 0.8x to 1.5x
            score = conf * timing
            quality = (score / 0.3) * (timing / 0.6) * (pred_strength / 2.0)
            quality_mult = np.clip(quality, 0.8, 1.5)
            risk_pct = RISK_PCT * quality_mult
        else:
            risk_pct = RISK_PCT
            
        stop_loss_pct = stop_distance / entry_price
        leverage = min(risk_pct / stop_loss_pct, MAX_LEVERAGE)
        position_value = self.capital * leverage
        
        # Deduct Fee
        self.capital -= position_value * ENTRY_FEE
        
        self.position = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': datetime.now(), # Store as object
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'position_value': position_value,
            'leverage': leverage,
            'breakeven_active': False,
            'pred_strength': pred_strength,
            'be_trigger_mult': be_trigger_mult 
        }
        self.save_state()
        self.send_alert(f"🟢 <b>OPEN {signal['direction']}</b> {signal['pair']}", 
                        f"Price: {entry_price}\nSL: {stop_loss:.4f} ({sl_mult} ATR)\nLev: {leverage:.1f}x\nStr: {pred_strength:.1f}")

    def check_instant_exit(self, pair, current_price):
        """Called INSTANTLY by WebSocket thread."""
        if self.position is None or self.position['pair'] != pair:
            return

        pos = self.position
        
        # 1. TIME LIMIT CHECK
        # 150 bars * 5 mins = 750 mins
        duration = datetime.now() - pos['entry_time']
        if duration > timedelta(minutes=MAX_HOLDING_BARS * 5):
            self.close_position(current_price, "Time Limit")
            return

        sl = pos['stop_loss']
        should_exit = False
        reason = ""
        
        # 2. DIRECTIONAL CHECKS
        if pos['direction'] == 'LONG':
            # Stop Loss
            if current_price <= sl:
                should_exit = True
                reason = "Stop Loss" if not pos['breakeven_active'] else "Trailing Stop"
            
            # Activate Breakeven?
            if not pos['breakeven_active']:
                trigger_dist = pos['stop_distance'] * pos['be_trigger_mult']
                trigger_price = pos['entry_price'] + trigger_dist
                if current_price >= trigger_price:
                    self.activate_breakeven()
                    
            # Update Trailing?
            if pos['breakeven_active']:
                self.update_trailing_fast(current_price)

        else: # SHORT
            # Stop Loss
            if current_price >= sl:
                should_exit = True
                reason = "Stop Loss" if not pos['breakeven_active'] else "Trailing Stop"
                
            # Activate Breakeven?
            if not pos['breakeven_active']:
                trigger_dist = pos['stop_distance'] * pos['be_trigger_mult']
                trigger_price = pos['entry_price'] - trigger_dist
                if current_price <= trigger_price:
                    self.activate_breakeven()
                    
            # Update Trailing?
            if pos['breakeven_active']:
                self.update_trailing_fast(current_price)

        if should_exit:
            self.close_position(current_price, reason)

    def activate_breakeven(self):
        """Move SL to Breakeven + buffer"""
        pos = self.position
        # V8: Higher BE margin (0.3 ATR) to cover slippage
        margin = pos['stop_distance'] * 0.2 
        
        if pos['direction'] == 'LONG':
            new_sl = pos['entry_price'] + margin
            pos['stop_loss'] = new_sl
        else:
            new_sl = pos['entry_price'] - margin
            pos['stop_loss'] = new_sl
            
        pos['breakeven_active'] = True
        logger.info(f"Moved to Breakeven+: {new_sl:.4f}")
        self.save_state()

    def update_trailing_fast(self, current_price):
        """
        Exact copy of V7 Trailing Logic.
        Calculates R-multiple and adjusts trail distance dynamically.
        """
        pos = self.position
        entry = pos['entry_price']
        sl_dist = pos['stop_distance']
        atr = sl_dist / 1.5 # Approx ATR
        
        # Calculate R-Multiple
        if pos['direction'] == 'LONG':
            current_profit = current_price - entry
        else:
            current_profit = entry - current_price
            
        r_multiple = current_profit / sl_dist
        
        # === V8: AGGRESSIVE TRAILING ===
        if USE_AGGRESSIVE_TRAIL:
            if r_multiple > 5.0:      # Super Pump
                trail_mult = 0.4
            elif r_multiple > 3.0:    # Good Trend
                trail_mult = 0.8
            elif r_multiple > 2.0:    # Medium
                trail_mult = 1.2
            else:                      # Early
                trail_mult = 1.8
        else:
            trail_mult = 2.0
            if r_multiple > 5.0: trail_mult = 0.5
            elif r_multiple > 3.0: trail_mult = 1.5

        # Update SL
        if pos['direction'] == 'LONG':
            new_sl = current_price - (atr * trail_mult)
            if new_sl > pos['stop_loss']:
                pos['stop_loss'] = new_sl
                # logger.info(f"Trailing Update: {new_sl:.4f} (R={r_multiple:.1f})")
        else:
            new_sl = current_price + (atr * trail_mult)
            if new_sl < pos['stop_loss']:
                pos['stop_loss'] = new_sl
                # logger.info(f"Trailing Update: {new_sl:.4f} (R={r_multiple:.1f})")

    def close_position(self, price, reason):
        pos = self.position
        
        if pos['direction'] == 'LONG':
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
            
        gross = pos['position_value'] * pnl_pct
        net = gross - (pos['position_value'] * EXIT_FEE)
        self.capital += net
        roe = (net / (pos['position_value'] / pos['leverage'])) * 100
        
        self.trades_history.append({
            'pair': pos['pair'],
            'direction': pos['direction'],
            'entry': pos['entry_price'],
            'exit': price,
            'pnl': net,
            'roe': roe,
            'reason': reason,
            'time': datetime.now().isoformat()
        })
        
        emoji = "✅" if net > 0 else "❌"
        self.send_alert(f"{emoji} <b>CLOSE {pos['direction']}</b> {pos['pair']}",
                        f"Reason: {reason}\nPrice: {price}\nPnL: ${net:.2f} ({roe:.1f}%)\nCap: ${self.capital:.2f}")
        
        self.position = None
        self.save_state()

    def send_alert(self, title, body):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {'chat_id': TELEGRAM_CHAT_ID, 'text': f"{title}\n{body}", 'parse_mode': 'HTML'}
            requests.post(url, data=data, timeout=5)
        except Exception:
            pass

# ============================================================
# UTILS & MAIN
# ============================================================
def load_models():
    logger.info("Loading V8 Models...")
    return {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }

def get_pairs():
    with open(PAIRS_FILE, 'r') as f:
        return [p['symbol'] for p in json.load(f)['pairs']][:20]

def add_volume_features(df):
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

def calculate_atr(df, period=14):
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

def main():
    logger.info("Starting V8 Sniper (Hybrid WS Mode)...")
    
    models = load_models()
    pairs = get_pairs()
    portfolio = PortfolioManager()
    mtf_fe = MTFFeatureEngine()
    
    # 1. Start WebSocket Streamer (Exits)
    streamer = DataStreamer(pairs)
    streamer.on_price_update = portfolio.check_instant_exit
    streamer.start() 
    
    logger.info("Monitoring 20 pairs. Waiting for signals...")
    
    # 2. Main Loop (Entries)
    while True:
        # Check WS readiness
        if not streamer.current_prices:
            time.sleep(2)
            continue
            
        logger.info(f"Scanning... (Position: {portfolio.position['pair'] if portfolio.position else 'None'})")
        
        exchange = ccxt.binance()
        
        for pair in pairs:
            # Single Slot Logic: Only scan if empty
            if portfolio.position is not None:
                continue
                
            try:
                # Fetch fresh history (Safe Entry)
                # Matches V7 logic: 1m delay is fine for entries
                data = {}
                clean_symbol = pair.replace('_', '/')
                if '/' not in clean_symbol: clean_symbol = f"{clean_symbol[:-4]}/{clean_symbol[-4:]}"
                
                valid = True
                for tf in TIMEFRAMES:
                    candles = exchange.fetch_ohlcv(clean_symbol, tf, limit=LOOKBACK)
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    data[tf] = df
                    if len(df) < 50: valid = False
                if not valid: continue
                
                df = prepare_features(data, mtf_fe)
                if len(df) < 2: continue
                
                # Check Signal on last closed candle
                row = df.iloc[[-2]] 
                X = row[models['features']].values
                
                # STALE CHECK (V7 Feature)
                # Candle time is Open Time. Close is +5m.
                # If current time is > Close Time + 2 mins, it's stale.
                last_candle_time = row.index[0]
                candle_close_time = last_candle_time + timedelta(minutes=5)
                now = datetime.now()
                if (now - candle_close_time).total_seconds() > 120:
                    # logger.warning(f"Skipping stale signal {pair}")
                    continue

                dir_proba = models['direction'].predict_proba(X)
                dir_conf = np.max(dir_proba)
                dir_pred = np.argmax(dir_proba)
                
                if dir_pred == 1 or dir_conf < MIN_CONF: continue
                
                timing_prob = models['timing'].predict_proba(X)[0][1]
                if timing_prob < MIN_TIMING: continue
                
                strength_pred = models['strength'].predict(X)[0]
                if strength_pred < MIN_STRENGTH: continue
                
                # Signal Found!
                signal = {
                    'pair': pair,
                    'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                    'price': row['close'].iloc[0],
                    'atr': row['atr'].iloc[0],
                    'conf': dir_conf,
                    'timing': timing_prob,
                    'strength': strength_pred
                }
                portfolio.open_position(signal)
                break # Taken a slot
                
            except Exception as e:
                logger.error(f"Scan error {pair}: {e}")
                
        time.sleep(10)

if __name__ == '__main__':
    main()

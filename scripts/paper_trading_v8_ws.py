#!/usr/bin/env python3
"""
Paper Trading Script (V8 Sniper) - WEBSOCKET VERSION (BACKTEST LOGIC)
- Uses WebSocketManager for real-time data (No lag!)
- Thread-safe data sharing
- Instant Stop-Loss checks (every tick)
- Trailing Stop UPDATES only on candle close (MATCHES BACKTEST)
- Slippage & Position Size limits (MATCHES BACKTEST)
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

# Risk Management (MATCHES BACKTEST)
RISK_PCT = 0.05          # 5% risk (Base)
MAX_LEVERAGE = 20.0      # Max leverage
MAX_HOLDING_BARS = 150   # 12.5 Hours (on 5m)
ENTRY_FEE = 0.0002       
EXIT_FEE = 0.0002        
INITIAL_CAPITAL = 20.0   
SL_ATR_BASE = 1.5
MAX_POSITION_SIZE = 50000.0  # Max $50K position (BACKTEST LIMIT)
SLIPPAGE_PCT = 0.0001        # 0.01% slippage (BACKTEST REALISM)

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
        
        # === V8: ADAPTIVE STOP LOSS (BACKTEST LOGIC) ===
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
            
        # === V8: DYNAMIC BREAKEVEN TRIGGER (BACKTEST LOGIC) ===
        if pred_strength >= 3.0:
            be_trigger_mult = 1.8
        elif pred_strength >= 2.0:
            be_trigger_mult = 1.5
        else:
            be_trigger_mult = 1.2

        # === V8: DYNAMIC RISK (BACKTEST LOGIC) ===
        if USE_DYNAMIC_LEVERAGE:
            score = conf * timing
            quality = (score / 0.5) * (timing / 0.6) * (pred_strength / 2.0)
            quality_mult = np.clip(quality, 0.8, 1.5)
            risk_pct = RISK_PCT * quality_mult
        else:
            risk_pct = RISK_PCT
        
        # Calculate Position Size (BACKTEST LOGIC)
        stop_loss_pct = stop_distance / entry_price
        leverage = min(risk_pct / stop_loss_pct, MAX_LEVERAGE)
        position_value = self.capital * leverage
        
        # === BACKTEST LIMIT: Cap position size ===
        if position_value > MAX_POSITION_SIZE:
            position_value = MAX_POSITION_SIZE
            leverage = position_value / self.capital
        
        # Deduct Fee
        self.capital -= position_value * ENTRY_FEE
        
        # CRITICAL: Store ORIGINAL entry price (without slippage) for SL/BE/Trailing calculations
        # Slippage will be applied ONLY in PnL calculation (like backtest)
        self.position = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'entry_price': entry_price,  # ORIGINAL price (NO slippage) - for SL/BE/Trailing
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'position_value': position_value,
            'leverage': leverage,
            'breakeven_active': False,
            'pred_strength': pred_strength,
            'be_trigger_mult': be_trigger_mult,
            'last_candle_time': None,
            'bars_held': 0
        }
        self.save_state()
        self.send_alert(f"🟢 <b>OPEN {signal['direction']}</b> {signal['pair']}", 
                        f"Price: {entry_price:.4f}\nSL: {stop_loss:.4f} ({sl_mult}ATR)\nLev: {leverage:.1f}x\nSize: ${position_value:.0f}\nStr: {pred_strength:.1f}")

    def check_instant_exit(self, pair, current_price):
        """
        Called INSTANTLY by WebSocket on every tick.
        ONLY checks SL hits - does NOT update trailing stop.
        Trailing stop updates happen ONLY on candle close (like backtest).
        """
        if self.position is None or self.position['pair'] != pair:
            return

        pos = self.position
        
        # 1. TIME LIMIT CHECK
        duration = datetime.now() - pos['entry_time']
        if duration > timedelta(minutes=MAX_HOLDING_BARS * 5):
            self.close_position(current_price, "Time Limit")
            return

        sl = pos['stop_loss']
        should_exit = False
        reason = ""
        
        # 2. CHECK STOP LOSS ONLY (no trailing update here)
        if pos['direction'] == 'LONG':
            if current_price <= sl:
                should_exit = True
                reason = "Stop Loss" if not pos['breakeven_active'] else "Trailing Stop"
        else: # SHORT
            if current_price >= sl:
                should_exit = True
                reason = "Stop Loss" if not pos['breakeven_active'] else "Trailing Stop"

        if should_exit:
            # Pass raw price - slippage will be applied in close_position()
            self.close_position(current_price, reason)

    def update_trailing_on_candle(self, candle_high, candle_low, candle_close, candle_time):
        """
        BACKTEST LOGIC: Update trailing stop ONLY on candle close.
        Called once per 5m candle (not on every tick).
        This matches the backtest bar-by-bar simulation.
        """
        if self.position is None:
            return
        
        pos = self.position
        
        # Only process new candles (avoid duplicates)
        if pos.get('last_candle_time') == candle_time:
            return
        
        pos['last_candle_time'] = candle_time
        pos['bars_held'] += 1
        
        # Restore ATR (same logic as backtest)
        pred_strength = pos.get('pred_strength', 2.0)
        if USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:
                sl_mult = 1.6
            elif pred_strength >= 2.0:
                sl_mult = 1.5
            else:
                sl_mult = 1.2
        else:
            sl_mult = SL_ATR_BASE
        
        atr = pos['stop_distance'] / sl_mult
        be_trigger_dist = atr * pos['be_trigger_mult']
        
        # LONG LOGIC
        if pos['direction'] == 'LONG':
            # 1. Check Breakeven Trigger
            be_trigger_price = pos['entry_price'] + be_trigger_dist
            if not pos['breakeven_active'] and candle_high >= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] + (atr * 0.3)
                logger.info(f"Breakeven activated at {pos['stop_loss']:.4f}")
            
            # 2. Update Trailing Stop (if breakeven active)
            if pos['breakeven_active']:
                current_profit = candle_high - pos['entry_price']
                r_multiple = current_profit / pos['stop_distance']
                
                # Trailing multiplier (BACKTEST LOGIC)
                if USE_AGGRESSIVE_TRAIL:
                    if r_multiple > 5.0:
                        trail_mult = 0.4
                    elif r_multiple > 3.0:
                        trail_mult = 0.8
                    elif r_multiple > 2.0:
                        trail_mult = 1.2
                    else:
                        trail_mult = 1.8
                else:
                    trail_mult = 2.0
                    if r_multiple > 5.0: trail_mult = 0.5
                    elif r_multiple > 3.0: trail_mult = 1.5
                
                new_sl = candle_high - (atr * trail_mult)
                if new_sl > pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    logger.info(f"Trailing SL: {new_sl:.4f} (R={r_multiple:.1f})")
        
        # SHORT LOGIC
        else:
            # 1. Check Breakeven Trigger
            be_trigger_price = pos['entry_price'] - be_trigger_dist
            if not pos['breakeven_active'] and candle_low <= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] - (atr * 0.3)
                logger.info(f"Breakeven activated at {pos['stop_loss']:.4f}")
            
            # 2. Update Trailing Stop (if breakeven active)
            if pos['breakeven_active']:
                current_profit = pos['entry_price'] - candle_low
                r_multiple = current_profit / pos['stop_distance']
                
                # Trailing multiplier (BACKTEST LOGIC)
                if USE_AGGRESSIVE_TRAIL:
                    if r_multiple > 5.0:
                        trail_mult = 0.4
                    elif r_multiple > 3.0:
                        trail_mult = 0.8
                    elif r_multiple > 2.0:
                        trail_mult = 1.2
                    else:
                        trail_mult = 1.8
                else:
                    trail_mult = 2.0
                    if r_multiple > 5.0: trail_mult = 0.5
                    elif r_multiple > 3.0: trail_mult = 1.5
                
                new_sl = candle_low + (atr * trail_mult)
                if new_sl < pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    logger.info(f"Trailing SL: {new_sl:.4f} (R={r_multiple:.1f})")
        
        self.save_state()

    def close_position(self, price, reason):
        """
        Close position with BACKTEST LOGIC.
        Slippage applied HERE in PnL calculation (not at entry).
        """
        pos = self.position
        
        # Apply slippage to BOTH entry and exit (like backtest)
        if pos['direction'] == 'LONG':
            effective_entry = pos['entry_price'] * (1 + SLIPPAGE_PCT)
            effective_exit = price * (1 - SLIPPAGE_PCT)
            pnl_pct = (effective_exit - effective_entry) / effective_entry
        else:
            effective_entry = pos['entry_price'] * (1 - SLIPPAGE_PCT)
            effective_exit = price * (1 + SLIPPAGE_PCT)
            pnl_pct = (effective_entry - effective_exit) / effective_entry
            
        gross = pos['position_value'] * pnl_pct
        fees = pos['position_value'] * EXIT_FEE
        net = gross - fees
        
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
            'bars_held': pos.get('bars_held', 0),
            'time': datetime.now().isoformat()
        })
        
        emoji = "✅" if net > 0 else "❌"
        self.send_alert(f"{emoji} <b>CLOSE {pos['direction']}</b> {pos['pair']}",
                        f"Reason: {reason}\nPrice: {price:.4f}\nPnL: ${net:.2f} ({roe:.1f}%)\nBars: {pos.get('bars_held', 0)}\nCap: ${self.capital:.2f}")
        
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
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    logger.info(f"Loaded {len(models['features'])} features")
    logger.info(f"First 10 features: {models['features'][:10]}")
    
    # Check excluded columns (should NOT be in features)
    excluded = ['atr', 'price_change', 'obv', 'obv_sma', 'open', 'high', 'low', 'close', 'volume']
    found_excluded = [f for f in excluded if f in models['features']]
    if found_excluded:
        logger.error(f"⚠️ WARNING: Excluded columns in features: {found_excluded}")
    
    return models

def get_pairs():
    with open(PAIRS_FILE, 'r') as f:
        return [p['symbol'] for p in json.load(f)['pairs']][:20]

def add_volume_features(df):
    """Add volume features required by the model (6 features)."""
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
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
    
    # Check data quality
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        logger.debug(f"Insufficient data: m1={len(m1)}, m5={len(m5)}, m15={len(m15)}")
        return pd.DataFrame()
    
    # Generate MTF features (166 features aligned to M5)
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    
    # === ADD OHLCV for volume features & ATR calculation ===
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    
    # Add volume features (6 features required by model)
    ft = add_volume_features(ft)
    
    # Add ATR for position sizing (not used by model)
    ft['atr'] = calculate_atr(ft)
    
    # Remove rows with NaN (important!)
    before_dropna = len(ft)
    ft = ft.dropna()
    after_dropna = len(ft)
    
    if after_dropna < before_dropna * 0.5:
        logger.warning(f"Heavy data loss after dropna: {before_dropna} -> {after_dropna}")
    
    return ft

def main():
    logger.info("Starting V8 Sniper (BACKTEST LOGIC + Real-time Exits)...")
    
    models = load_models()
    pairs = get_pairs()
    portfolio = PortfolioManager()
    mtf_fe = MTFFeatureEngine()
    exchange = ccxt.binance()
    
    # 1. Start WebSocket Streamer (Instant SL checks)
    streamer = DataStreamer(pairs)
    streamer.on_price_update = portfolio.check_instant_exit
    streamer.start() 
    
    logger.info("Monitoring 20 pairs. Waiting for signals...")
    
    last_trailing_update = {}  # Track last trailing update per pair
    
    # 2. Main Loop (Entries + Trailing Updates)
    while True:
        # Check WS readiness
        if not streamer.current_prices:
            time.sleep(2)
            continue
        
        # === UPDATE TRAILING STOP ON CANDLE CLOSE ===
        # Fetch last completed 5m candle for active position
        if portfolio.position:
            pair = portfolio.position['pair']
            try:
                clean_symbol = pair.replace('_', '/')
                if '/' not in clean_symbol:
                    clean_symbol = f"{pair[:-4]}/{pair[-4:]}"
                
                # Fetch last 2 candles (to get completed one)
                candles_5m = exchange.fetch_ohlcv(clean_symbol, '5m', limit=2)
                if len(candles_5m) >= 2:
                    # Use second-to-last (fully closed candle)
                    last_candle = candles_5m[-2]
                    candle_time = pd.to_datetime(last_candle[0], unit='ms')
                    
                    # Only update if it's a new candle
                    if last_trailing_update.get(pair) != candle_time:
                        last_trailing_update[pair] = candle_time
                        
                        # Update trailing stop (BACKTEST LOGIC)
                        portfolio.update_trailing_on_candle(
                            candle_high=last_candle[2],
                            candle_low=last_candle[3],
                            candle_close=last_candle[4],
                            candle_time=candle_time
                        )
            except Exception as e:
                logger.error(f"Error updating trailing for {pair}: {e}")
        
        # === SCAN FOR NEW SIGNALS ===
        # Only scan if no position (single slot)
        if portfolio.position is None:
            for pair in pairs:
                try:
                    # Fetch fresh history for signal generation
                    data = {}
                    clean_symbol = pair.replace('_', '/')
                    if '/' not in clean_symbol:
                        clean_symbol = f"{pair[:-4]}/{pair[-4:]}"
                    
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
                    
                    # Check Signal on last closed candle (BACKTEST LOGIC)
                    row = df.iloc[-2:]  # Get last 2 rows for safety
                    
                    # === VALIDATE FEATURES ===
                    missing_features = [f for f in models['features'] if f not in row.columns]
                    if missing_features:
                        if len(missing_features) < 10:
                            logger.error(f"{pair}: Missing {len(missing_features)} features: {missing_features}")
                        else:
                            logger.error(f"{pair}: Missing {len(missing_features)} features (first 10): {missing_features[:10]}")
                        # Don't skip - this is a critical error that needs fixing
                        continue
                    
                    # Extract features (use iloc[-1] for last closed candle)
                    try:
                        X = row.iloc[[-1]][models['features']].values
                    except KeyError as e:
                        logger.error(f"{pair}: KeyError extracting features: {e}")
                        continue
                    
                    # Check for NaN
                    if np.isnan(X).any():
                        nan_count = np.isnan(X).sum()
                        logger.warning(f"{pair}: {nan_count} NaN values in features, skipping")
                        continue
                    
                    # NO STALE CHECK - Enter immediately on signal (like backtest)
                    
                    dir_proba = models['direction'].predict_proba(X)
                    dir_conf = np.max(dir_proba)
                    dir_pred = np.argmax(dir_proba)
                    
                    # === DEBUG: Log predictions ===
                    if dir_conf > 0.4:  # Log interesting signals
                        logger.info(f"{pair}: Dir={dir_pred} (conf={dir_conf:.2f}) - Proba: {dir_proba[0]}")
                    
                    if dir_pred == 1 or dir_conf < MIN_CONF: continue
                    
                    timing_prob = models['timing'].predict_proba(X)[0][1]
                    if timing_prob < MIN_TIMING: continue
                    
                    strength_pred = models['strength'].predict(X)[0]
                    if strength_pred < MIN_STRENGTH: continue
                    
                    # === SIGNAL FOUND - ENTER AT CURRENT PRICE ===
                    # Use current live price from WebSocket (faster entry, like backtest)
                    current_price = streamer.current_prices.get(pair, row['close'].iloc[-1])
                    
                    signal = {
                        'pair': pair,
                        'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                        'price': current_price,  # Live price for faster entry
                        'atr': row['atr'].iloc[-1],
                        'conf': dir_conf,
                        'timing': timing_prob,
                        'strength': strength_pred
                    }
                    portfolio.open_position(signal)
                    logger.info(f"🚀 Signal taken: {pair} {signal['direction']} @ {current_price:.4f}")
                    break  # Taken a slot
                    
                except Exception as e:
                    logger.error(f"Scan error {pair}: {e}")
        
        # Sleep (shorter interval for faster trailing updates)
        time.sleep(10)

if __name__ == '__main__':
    main()

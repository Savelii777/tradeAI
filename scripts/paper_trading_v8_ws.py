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
# CANDLE BUILDER - Build candles from WebSocket in real-time
# ============================================================
class CandleBuilder:
    """Builds and maintains candles from WebSocket data."""
    
    def __init__(self):
        self.candles = {}  # {pair: {timeframe: deque of candles}}
        self.lock = threading.Lock()
        self.max_candles = LOOKBACK  # Keep last N candles
    
    def preload_history(self, pair, timeframe, candles_data):
        """
        Preload historical candles from CCXT.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe (1m, 5m, 15m)
            candles_data: List of [timestamp, open, high, low, close, volume]
        """
        with self.lock:
            if pair not in self.candles:
                self.candles[pair] = {}
            if timeframe not in self.candles[pair]:
                from collections import deque
                self.candles[pair][timeframe] = deque(maxlen=self.max_candles)
            
            # Convert CCXT format to our format
            for candle in candles_data:
                timestamp = pd.to_datetime(candle[0], unit='ms', utc=True)
                self.candles[pair][timeframe].append({
                    'timestamp': timestamp,
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                })
        
    def on_candle_update(self, candle):
        """Called when new candle data arrives from WebSocket."""
        with self.lock:
            pair = candle.symbol
            tf = candle.timeframe
            
            # Initialize storage
            if pair not in self.candles:
                self.candles[pair] = {}
            if tf not in self.candles[pair]:
                from collections import deque
                self.candles[pair][tf] = deque(maxlen=self.max_candles)
            
            # Add or update candle
            candles_deque = self.candles[pair][tf]
            
            # Check if this is an update to existing candle or new one
            if candles_deque and candles_deque[-1]['timestamp'] == candle.timestamp:
                # Update existing (incomplete) candle
                candles_deque[-1].update({
                    'high': max(candles_deque[-1]['high'], candle.high),
                    'low': min(candles_deque[-1]['low'], candle.low),
                    'close': candle.close,
                    'volume': candle.volume,
                })
            else:
                # New candle
                candles_deque.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                })
    
    def get_candles(self, pair, timeframe):
        """Get candles as DataFrame."""
        with self.lock:
            if pair not in self.candles or timeframe not in self.candles[pair]:
                return None
            
            candles_list = list(self.candles[pair][timeframe])
            if not candles_list:
                return None
                
            df = pd.DataFrame(candles_list)
            df.set_index('timestamp', inplace=True)
            return df

# ============================================================
# DATA STREAMER - WebSocket data manager
# ============================================================
class DataStreamer:
    def __init__(self, pairs):
        self.pairs = pairs
        self.ws_manager = WebSocketManager('binance')
        self.lock = threading.Lock()
        self.ready = False
        self.current_prices = {}
        self.candle_builder = CandleBuilder()
        self.subscribed_candles = False

    async def _run_ws(self):
        await self.ws_manager.connect()
        logger.info("WebSocket Connected. Subscribing...")
        logger.info(f"⏳ Subscribing to {len(self.pairs) * (len(TIMEFRAMES) + 1)} streams (will take ~{len(self.pairs) * (len(TIMEFRAMES) + 1) / 4:.0f}s)...")
        
        subscription_count = 0
        for i, pair in enumerate(self.pairs):
            # Subscribe to TRADES for instant price updates (stop-loss)
            try:
                await self.ws_manager.subscribe_trades(pair, self._on_trade)
                subscription_count += 1
                await asyncio.sleep(0.25)  # 4 subscriptions per second (Binance limit: 5/sec)
            except Exception as e:
                logger.error(f"Failed to subscribe to trades for {pair}: {e}")
            
            # Subscribe to CANDLES for real-time candle building (entries)
            for tf in TIMEFRAMES:
                try:
                    await self.ws_manager.subscribe_candles(
                        pair, tf, self.candle_builder.on_candle_update
                    )
                    subscription_count += 1
                    await asyncio.sleep(0.25)  # 4 subscriptions per second
                except Exception as e:
                    logger.error(f"Failed to subscribe to {pair} {tf}: {e}")
            
            # Progress update every 5 pairs
            if (i + 1) % 5 == 0:
                logger.info(f"📡 Subscribed {subscription_count}/{len(self.pairs) * (len(TIMEFRAMES) + 1)} streams...")
        
        self.subscribed_candles = True
        logger.info(f"✅ Subscription complete! Active streams: {subscription_count}")
        
        # === PRELOAD HISTORICAL DATA ===
        logger.info("📥 Preloading historical candles from CCXT...")
        await self._preload_history()
        
        while True:
            await asyncio.sleep(1)
    
    async def _preload_history(self):
        """Preload historical candles to fill WebSocket buffer."""
        import ccxt
        exchange = ccxt.binance()
        
        total_loads = len(self.pairs) * len(TIMEFRAMES)
        loaded = 0
        
        for pair in self.pairs:
            clean_symbol = pair.replace('_', '/')
            if '/' not in clean_symbol:
                clean_symbol = f"{pair[:-4]}/{pair[-4:]}"
            
            for tf in TIMEFRAMES:
                try:
                    # Fetch history
                    candles = exchange.fetch_ohlcv(clean_symbol, tf, limit=LOOKBACK)
                    if candles and len(candles) >= 50:
                        self.candle_builder.preload_history(pair, tf, candles)
                        loaded += 1
                    
                    # Rate limit
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    logger.warning(f"Failed to preload {pair} {tf}: {e}")
            
            # Progress update every 5 pairs
            if (self.pairs.index(pair) + 1) % 5 == 0:
                logger.info(f"📥 Loaded {loaded}/{total_loads} candle histories...")
        
        logger.info(f"✅ History preloaded! {loaded}/{total_loads} successful. WebSocket ready!")

    def _on_trade(self, trade):
        with self.lock:
            self.current_prices[trade.symbol] = trade.price
            
        # INSTANT EXIT CHECK
        if hasattr(self, 'on_price_update'):
            self.on_price_update(trade.symbol, trade.price)

    def get_candles_realtime(self, pair, timeframe):
        """Get real-time candles from WebSocket (faster than API!)."""
        return self.candle_builder.get_candles(pair, timeframe)

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
            scan_stats = {'total': 0, 'short': 0, 'side': 0, 'long': 0, 'errors': 0, 'ws_data': 0, 'api_data': 0}
            
            for pair in pairs:
                scan_stats['total'] += 1
                try:
                    # === TRY WEBSOCKET FIRST (FAST!) ===
                    data = {}
                    use_websocket = False
                    
                    if streamer.subscribed_candles:
                        # Try to get candles from WebSocket
                        ws_valid = True
                        for tf in TIMEFRAMES:
                            df = streamer.get_candles_realtime(pair, tf)
                            if df is None or len(df) < 50:
                                ws_valid = False
                                break
                            data[tf] = df
                        
                        if ws_valid:
                            use_websocket = True
                            scan_stats['ws_data'] += 1
                    
                    # === FALLBACK TO CCXT API (if WebSocket not ready) ===
                    if not use_websocket:
                        scan_stats['api_data'] += 1
                        clean_symbol = pair.replace('_', '/')
                        if '/' not in clean_symbol:
                            clean_symbol = f"{pair[:-4]}/{pair[-4:]}"
                        
                        valid = True
                        for tf in TIMEFRAMES:
                            # Request fresh data: last N candles from NOW (UTC!)
                            tf_minutes = int(tf[:-1])  # '5m' -> 5
                            now_utc = datetime.now(timezone.utc)
                            since_ms = int((now_utc - timedelta(minutes=LOOKBACK * tf_minutes)).timestamp() * 1000)
                            
                            candles = exchange.fetch_ohlcv(clean_symbol, tf, since=since_ms, limit=LOOKBACK)
                            
                            # Check data freshness (compare UTC times!)
                            if candles:
                                last_candle_time = pd.to_datetime(candles[-1][0], unit='ms', utc=True)
                                age_minutes = (now_utc - last_candle_time).total_seconds() / 60
                                
                                if age_minutes > 15:  # Data older than 15 minutes - skip
                                    logger.warning(f"{pair} {tf}: Data too old ({age_minutes:.0f}min), skipping")
                                    valid = False
                                    break
                            
                            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            data[tf] = df
                            if len(df) < 50: valid = False
                            
                            # Small delay to avoid rate limits
                            time.sleep(0.05)
                        
                        if not valid: continue
                    
                    df = prepare_features(data, mtf_fe)
                    if len(df) < 2: continue
                    
                    # Check Signal on last CLOSED candle (BACKTEST LOGIC)
                    # CRITICAL: Use second-to-last candle (fully closed), not last (incomplete)
                    if len(df) < 2:
                        continue
                    
                    row = df.iloc[[-2]]  # Предпоследняя свеча (закрытая!)
                    
                    # === DEBUG: Show what data model sees ===
                    last_candle_time = row.index[0]
                    last_close = row['close'].iloc[0]
                    prev_close = df['close'].iloc[-7] if len(df) >= 7 else df['close'].iloc[0]
                    price_change_5m = ((last_close - prev_close) / prev_close) * 100
                    
                    # Calculate time difference using UTC
                    now_utc = datetime.now(timezone.utc)
                    if last_candle_time.tzinfo is None:
                        last_candle_time_utc = last_candle_time.replace(tzinfo=timezone.utc)
                    else:
                        last_candle_time_utc = last_candle_time
                    time_ago = (now_utc - last_candle_time_utc).total_seconds() / 60
                    
                    # Get current live price (fallback to candle close if not available)
                    ws_price = streamer.current_prices.get(pair)
                    has_live_price = ws_price is not None and ws_price > 0
                    current_price = ws_price if has_live_price else last_close
                    price_diff = ((current_price - last_close) / last_close) * 100
                    
                    # Source indicator
                    data_source = "📡WS" if use_websocket else "🌐API"
                    price_source = "🔴Live" if has_live_price else "📊Candle"
                    
                    if pair == 'DOGE/USDT:USDT' or time_ago > 10:  # Always log DOGE or stale data
                        logger.warning(
                            f"⏰ {pair} {data_source}: Candle @ {last_candle_time} ({time_ago:.1f}min ago) | "
                            f"Close: {last_close:.6f} | Now {price_source}: {current_price:.6f} ({price_diff:+.2f}%) | "
                            f"5-bar: {price_change_5m:+.2f}%"
                        )
                    
                    # === VALIDATE FEATURES ===
                    missing_features = [f for f in models['features'] if f not in row.columns]
                    if missing_features:
                        if len(missing_features) < 10:
                            logger.error(f"{pair}: Missing {len(missing_features)} features: {missing_features}")
                        else:
                            logger.error(f"{pair}: Missing {len(missing_features)} features (first 10): {missing_features[:10]}")
                        # Don't skip - this is a critical error that needs fixing
                        continue
                    
                    # Extract features from the closed candle
                    try:
                        X = row[models['features']].values
                    except KeyError as e:
                        logger.error(f"{pair}: KeyError extracting features: {e}")
                        continue
                    
                    # Check for NaN
                    if pd.isna(X).any():
                        nan_count = pd.isna(X).sum()
                        logger.warning(f"{pair}: {nan_count} NaN values in features, skipping")
                        continue
                    
                    # NO STALE CHECK - Enter immediately on signal (like backtest)
                    
                    dir_proba = models['direction'].predict_proba(X)
                    dir_conf = np.max(dir_proba)
                    dir_pred = np.argmax(dir_proba)
                    
                    # === DETAILED LOGGING: Show ALL signal parameters ===
                    timing_prob = models['timing'].predict_proba(X)[0][1]
                    strength_pred = models['strength'].predict(X)[0]
                    
                    # Update stats
                    if dir_pred == 0:
                        scan_stats['short'] += 1
                    elif dir_pred == 1:
                        scan_stats['side'] += 1
                    else:
                        scan_stats['long'] += 1
                    
                    # Log ALL predictions (even sideways)
                    if dir_conf > 0.4 or dir_pred != 1:  # Log interesting signals
                        direction_name = ['SHORT', 'SIDE', 'LONG'][dir_pred]
                        logger.info(
                            f"{pair}: {direction_name} | "
                            f"Conf={dir_conf:.2f} | "
                            f"Timing={timing_prob:.2f} | "
                            f"Strength={strength_pred:.2f} | "
                            f"Proba: [{dir_proba[0][0]:.2f}, {dir_proba[0][1]:.2f}, {dir_proba[0][2]:.2f}]"
                        )
                    
                    # Apply filters
                    if dir_pred == 1:
                        continue  # Skip sideways
                    
                    if dir_conf < MIN_CONF:
                        logger.debug(f"{pair}: REJECTED - Low confidence ({dir_conf:.2f} < {MIN_CONF})")
                        continue
                    
                    if timing_prob < MIN_TIMING:
                        logger.debug(f"{pair}: REJECTED - Low timing ({timing_prob:.2f} < {MIN_TIMING})")
                        continue
                    
                    if strength_pred < MIN_STRENGTH:
                        logger.debug(f"{pair}: REJECTED - Low strength ({strength_pred:.2f} < {MIN_STRENGTH})")
                        continue
                    
                    # === SIGNAL FOUND - ENTER AT CURRENT PRICE ===
                    # Use current live price from WebSocket (faster entry, like backtest)
                    current_price = streamer.current_prices.get(pair, row['close'].iloc[0])
                    
                    signal = {
                        'pair': pair,
                        'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                        'price': current_price,  # Live price for faster entry
                        'atr': row['atr'].iloc[0],  # ATR from closed candle
                        'conf': dir_conf,
                        'timing': timing_prob,
                        'strength': strength_pred
                    }
                    portfolio.open_position(signal)
                    logger.info(f"🚀 Signal taken: {pair} {signal['direction']} @ {current_price:.4f}")
                    break  # Taken a slot
                    
                except Exception as e:
                    logger.error(f"Scan error {pair}: {e}")
                    scan_stats['errors'] += 1
            
            # Print scan summary
            logger.info(
                f"📊 Scan complete: {scan_stats['total']} pairs | "
                f"SHORT: {scan_stats['short']} | "
                f"SIDE: {scan_stats['side']} | "
                f"LONG: {scan_stats['long']} | "
                f"WS: {scan_stats['ws_data']} | "
                f"API: {scan_stats['api_data']} | "
                f"Errors: {scan_stats['errors']}"
            )
        
        # Sleep (shorter interval for faster trailing updates)
        time.sleep(10)

if __name__ == '__main__':
    main()

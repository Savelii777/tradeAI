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
# DATA STREAMER - WebSocket for live prices (SL monitoring)
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
        
        subscription_count = 0
        for i, pair in enumerate(self.pairs):
            # Subscribe to TRADES ONLY (for live prices & stop-loss)
            try:
                await self.ws_manager.subscribe_trades(pair, self._on_trade)
                subscription_count += 1
                await asyncio.sleep(0.25)  # 4 subscriptions per second (Binance limit: 5/sec)
            except Exception as e:
                logger.error(f"Failed to subscribe to trades for {pair}: {e}")
        
        logger.info(f"WebSocket subscribed to {subscription_count} trade streams")
        
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
        pred_strength = signal.get('pred_strength', 2.0)  # MATCH BACKTEST KEY
        conf = signal.get('conf', 0.5)
        timing = signal.get('timing_prob', 0.5)  # MATCH BACKTEST KEY
        
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
        
        # === CORRECT POSITION SIZING (MATCHES REAL EXCHANGE) ===
        # 1. Calculate position size based on risk
        stop_loss_pct = stop_distance / entry_price
        risk_amount = self.capital * risk_pct  # Dollar amount we're willing to risk
        position_value = risk_amount / stop_loss_pct  # Position size needed to risk this amount
        
        # 2. Calculate leverage (Position Size / Margin)
        leverage = position_value / self.capital
        
        # 3. Apply MAX_LEVERAGE limit
        if leverage > MAX_LEVERAGE:
            leverage = MAX_LEVERAGE
            position_value = self.capital * leverage
        
        # === BACKTEST LIMIT: Cap position size ===
        if position_value > MAX_POSITION_SIZE:
            position_value = MAX_POSITION_SIZE
            leverage = position_value / self.capital
        
        # Deduct Entry Fee from capital (like real exchange)
        entry_fee = position_value * ENTRY_FEE
        self.capital -= entry_fee
        
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
        
        # SHORT LOGIC
        else:
            # 1. Check Breakeven Trigger
            be_trigger_price = pos['entry_price'] - be_trigger_dist
            if not pos['breakeven_active'] and candle_low <= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] - (atr * 0.3)
            
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
    """Add volume features required by the model (MATCHES BACKTEST)."""
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # CRITICAL: OBV calculation (MUST match backtest)
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # Volume momentum
    df['vol_momentum'] = df['volume'].pct_change(5)
    
    return df

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe, pair=None):
    """
    Prepare features from multi-timeframe data.
    
    Args:
        data: Dict with '1m', '5m', '15m' DataFrames
        mtf_fe: MTFFeatureEngine instance
        pair: Pair name for logging
    
    Returns:
        Features DataFrame
    """
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    # Check data quality
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        if pair:
            logger.warning(f"{pair}: Insufficient data - 1m: {len(m1)}, 5m: {len(m5)}, 15m: {len(m15)}")
        return pd.DataFrame()
    
    # DEBUG: Check index types and ranges
    if pair and pair in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
        logger.debug(f"{pair}: Index types - m1: {type(m1.index)}, m5: {type(m5.index)}, m15: {type(m15.index)}")
        logger.debug(f"{pair}: Index ranges - m1: {m1.index[0]} to {m1.index[-1]}, m5: {m5.index[0]} to {m5.index[-1]}, m15: {m15.index[0]} to {m15.index[-1]}")
        logger.debug(f"{pair}: Index frequencies - m1: {m1.index.freq if hasattr(m1.index, 'freq') else 'None'}, m5: {m5.index.freq if hasattr(m5.index, 'freq') else 'None'}, m15: {m15.index.freq if hasattr(m15.index, 'freq') else 'None'}")
    
    # Ensure all DataFrames have DatetimeIndex and are sorted
    if not isinstance(m1.index, pd.DatetimeIndex):
        m1.index = pd.to_datetime(m1.index, utc=True)
    m1 = m1.sort_index()
    
    if not isinstance(m5.index, pd.DatetimeIndex):
        m5.index = pd.to_datetime(m5.index, utc=True)
    m5 = m5.sort_index()
    
    if not isinstance(m15.index, pd.DatetimeIndex):
        m15.index = pd.to_datetime(m15.index, utc=True)
    m15 = m15.sort_index()
    
    # Generate MTF features (166 features aligned to M5)
    try:
        # DEBUG: Check what generate_m5_signal_features returns
        if pair and pair in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
            m5_features_debug = mtf_fe.generate_m5_signal_features(m5)
            logger.debug(f"{pair}: m5_features shape: {m5_features_debug.shape}, NaN count: {m5_features_debug.isna().sum().sum()}, empty: {m5_features_debug.empty}")
            if len(m5_features_debug) > 0:
                logger.debug(f"{pair}: m5_features index: {m5_features_debug.index[0]} to {m5_features_debug.index[-1]}")
                logger.debug(f"{pair}: m5_features sample columns: {list(m5_features_debug.columns[:5])}")
        
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        
        if pair and pair in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
            logger.debug(f"{pair}: After align_timeframes - {len(ft)} rows, {len(ft.columns) if len(ft) > 0 else 0} cols")
            if len(ft) == 0:
                # Try to understand why it's empty
                logger.warning(f"{pair}: align_timeframes returned empty! m1: {len(m1)} rows, m5: {len(m5)} rows, m15: {len(m15)} rows")
                logger.warning(f"{pair}: m1 index: {m1.index[0]} to {m1.index[-1]}")
                logger.warning(f"{pair}: m5 index: {m5.index[0]} to {m5.index[-1]}")
                logger.warning(f"{pair}: m15 index: {m15.index[0]} to {m15.index[-1]}")
    except Exception as e:
        if pair:
            logger.error(f"{pair}: Error in align_timeframes: {e}", exc_info=True)
        return pd.DataFrame()
    
    if len(ft) == 0:
        if pair:
            logger.warning(f"{pair}: align_timeframes returned empty DataFrame")
        return pd.DataFrame()
    
    # === ADD OHLCV for volume features & ATR calculation ===
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    
    # Add volume features (6 features required by model)
    ft = add_volume_features(ft)
    
    # Add ATR for position sizing (not used by model)
    ft['atr'] = calculate_atr(ft)
    
    # Remove rows with NaN ONLY in critical columns (not all columns!)
    before_dropna = len(ft)
    
    # Определить критичные колонки - те которые используются моделью
    # НЕ удалять строки где есть NaN в не-критичных колонках
    critical_cols = ['close', 'atr']  # Минимум нужный для работы
    
    # Удалить строки только если в критичных колонках есть NaN
    ft = ft.dropna(subset=critical_cols)
    
    # ДОПОЛНИТЕЛЬНО: заполнить оставшиеся NaN в фичах (forward fill + backward fill)
    ft = ft.ffill().bfill()
    
    # Если всё ещё есть NaN (колонки где все значения NaN), заполнить нулями
    if ft.isna().any().any():
        # Найти колонки с NaN
        nan_cols = ft.columns[ft.isna().any()].tolist()
        if pair and pair in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
            logger.warning(f"{pair}: Filling remaining NaN with 0 in columns: {nan_cols}")
        ft = ft.fillna(0)
    
    after_dropna = len(ft)
    
    if pair and pair in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
        logger.debug(f"{pair}: After dropna (critical only): {before_dropna} -> {after_dropna} rows")
        if after_dropna > 0:
            nan_counts = ft.isna().sum()
            remaining_nans = nan_counts[nan_counts > 0]
            if len(remaining_nans) > 0:
                logger.debug(f"{pair}: Remaining NaNs after fillna: {len(remaining_nans)} columns")
    
    if after_dropna < 2 and pair:
        logger.warning(f"{pair}: After dropna: {before_dropna} -> {after_dropna} rows (need 2+)")
    
    return ft

def main():
    logger.info("Starting V8 Sniper (BACKTEST LOGIC + Real-time Exits)...")
    
    models = load_models()
    pairs = get_pairs()
    portfolio = PortfolioManager()
    mtf_fe = MTFFeatureEngine()
    exchange = ccxt.binance({
        'timeout': 5000,  # 5 seconds timeout
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'recvWindow': 10000
        }
    })
    
    # 1. Start WebSocket Streamer (Instant SL checks)
    streamer = DataStreamer(pairs)
    streamer.on_price_update = portfolio.check_instant_exit
    streamer.start() 
    
    logger.info(f"Monitoring {len(pairs)} pairs. Thresholds: MIN_CONF={MIN_CONF}, MIN_TIMING={MIN_TIMING}, MIN_STRENGTH={MIN_STRENGTH}")
    
    last_trailing_update = {}  # Track last trailing update per pair
    last_heartbeat = time.time()  # For periodic logging
    
    # 2. Main Loop (Entries + Trailing Updates)
    while True:
        current_time = time.time()
        
        # Check WS readiness
        if not streamer.current_prices:
            if current_time - last_heartbeat >= 10:  # Log every 10 seconds while waiting
                logger.warning("⏳ Waiting for WebSocket price data...")
                last_heartbeat = current_time
            time.sleep(2)
            continue
        
        # Periodic heartbeat log (every 60 seconds)
        if current_time - last_heartbeat >= 60:
            prices_count = len(streamer.current_prices)
            position_info = f"Active: {portfolio.position['pair']}" if portfolio.position else "No position"
            logger.info(f"💓 Heartbeat: {prices_count} prices tracked | {position_info} | Capital: ${portfolio.capital:.2f}")
            last_heartbeat = current_time
        
        # === UPDATE TRAILING STOP ON CANDLE CLOSE ===
        # Fetch last completed 5m candle for active position
        if portfolio.position:
            pair = portfolio.position['pair']
            try:
                clean_symbol = pair.replace('_', '/')
                if '/' not in clean_symbol:
                    clean_symbol = f"{pair[:-4]}/{pair[-4:]}"
                
                # Fetch last 2 candles (to get completed one)
                try:
                    candles_5m = exchange.fetch_ohlcv(clean_symbol, '5m', limit=2)
                except Exception as e:
                    logger.error(f"Error fetching trailing candles for {pair}: {e}")
                    continue
                    
                if len(candles_5m) >= 2:
                    # Use second-to-last (fully closed candle)
                    last_candle = candles_5m[-2]
                    candle_time = pd.to_datetime(last_candle[0], unit='ms')
                    
                    # Only update if it's a new candle
                    if last_trailing_update.get(pair) != candle_time:
                        last_trailing_update[pair] = candle_time
                        
                        # Update trailing stop (BACKTEST LOGIC)
                        old_sl = portfolio.position['stop_loss']
                        candle_close = last_candle[4]
                        portfolio.update_trailing_on_candle(
                            candle_high=last_candle[2],
                            candle_low=last_candle[3],
                            candle_close=candle_close,
                            candle_time=candle_time
                        )
                        new_sl = portfolio.position['stop_loss']
                        if old_sl != new_sl:
                            logger.info(f"📈 Trailing stop updated for {pair}: {old_sl:.6f} → {new_sl:.6f} | Price: {candle_close:.6f}")
            except Exception as e:
                logger.error(f"Error updating trailing for {pair}: {e}")
        
        # === SCAN FOR NEW SIGNALS ===
        # Only scan if no position (single slot)
        if portfolio.position is None:
            logger.debug("🔍 Starting signal scan...")
            scan_stats = {
                'total': 0, 'short': 0, 'side': 0, 'long': 0, 
                'errors': 0, 'fetched': 0,
                'rejected_conf': 0, 'rejected_timing': 0, 'rejected_strength': 0,
                'skipped_df_empty': 0, 'skipped_missing_features': 0, 'skipped_nan': 0
            }
            for pair in pairs:
                scan_stats['total'] += 1
                try:
                    # === FETCH FRESH DATA FROM CCXT ===
                    clean_symbol = pair.replace('_', '/')
                    if '/' not in clean_symbol:
                        clean_symbol = f"{pair[:-4]}/{pair[-4:]}"
                    
                    data = {}
                    valid = True
                    
                    # Fetch fresh data from exchange
                    scan_stats['fetched'] += 1
                    for tf in TIMEFRAMES:
                        # Fetch LOOKBACK candles for proper indicator calculation
                        try:
                            candles = exchange.fetch_ohlcv(clean_symbol, tf, limit=LOOKBACK)
                        except Exception as e:
                            if scan_stats['total'] <= 3:
                                logger.error(f"{pair}: CCXT fetch error for {tf}: {e}")
                            valid = False
                            break
                        
                        # DEBUG: Check data freshness for first 3 pairs
                        if scan_stats['total'] <= 3 and candles:
                            last_candle_ts = pd.to_datetime(candles[-1][0], unit='ms', utc=True)
                            now_utc = datetime.now(timezone.utc)
                            age_minutes = (now_utc - last_candle_ts).total_seconds() / 60
                            logger.info(f"{pair} {tf}: Last candle @ {last_candle_ts} ({age_minutes:.1f} min ago)")
                        
                        if not candles:
                            if scan_stats['total'] <= 3:
                                logger.warning(f"{pair}: No candles for {tf}")
                            valid = False
                            break
                        
                        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                        df.set_index('timestamp', inplace=True)
                        
                        # Ensure index is DatetimeIndex and sort
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index, utc=True)
                        df = df.sort_index()
                        
                        data[tf] = df
                        if len(df) < 50:
                            if scan_stats['total'] <= 3:
                                logger.warning(f"{pair}: {tf} has only {len(df)} candles (need 50+)")
                            valid = False
                        
                        # Small delay to avoid rate limits
                        time.sleep(0.02)
                    
                    if not valid:
                        if scan_stats['total'] <= 3:
                            logger.warning(f"{pair}: Data validation failed, skipping")
                        continue
                    
                    # Log data sizes before feature preparation
                    if scan_stats['total'] <= 3:
                        logger.debug(f"{pair}: Data sizes - 1m: {len(data.get('1m', []))}, "
                                   f"5m: {len(data.get('5m', []))}, 15m: {len(data.get('15m', []))}")
                    
                    df = prepare_features(data, mtf_fe, pair=pair)
                    
                    # Log result after feature preparation
                    if scan_stats['total'] <= 3:
                        logger.debug(f"{pair}: Features prepared - len={len(df) if df is not None else 0}")
                    
                    if df is None or len(df) < 2:
                        scan_stats['skipped_df_empty'] += 1
                        # Log first 3 pairs for debugging
                        if scan_stats['total'] <= 3:
                            logger.warning(f"{pair}: DataFrame is None or len < 2 (len={len(df) if df is not None else 0})")
                        continue
                    
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
                    
                    # Source indicators
                    price_source = "🔴Live" if has_live_price else "📊Candle"
                    
                    if pair == 'DOGE/USDT:USDT' or time_ago > 10:  # Always log DOGE or stale data
                        logger.warning(
                            f"⏰ {pair}: Candle @ {last_candle_time} ({time_ago:.1f}min ago) | "
                            f"Close: {last_close:.6f} | Now {price_source}: {current_price:.6f} ({price_diff:+.2f}%) | "
                            f"5-bar: {price_change_5m:+.2f}%"
                        )
                    
                    # === VALIDATE FEATURES ===
                    missing_features = [f for f in models['features'] if f not in row.columns]
                    if missing_features:
                        scan_stats['skipped_missing_features'] += 1
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
                        scan_stats['skipped_nan'] += 1
                        nan_count = pd.isna(X).sum()
                        logger.warning(f"{pair}: {nan_count} NaN values in features, skipping")
                        continue
                    
                    # NO STALE CHECK - Enter immediately on signal (like backtest)
                    
                    # === PREDICTIONS ===
                    try:
                        dir_proba = models['direction'].predict_proba(X)
                        dir_conf = float(np.max(dir_proba))
                        dir_pred = int(np.argmax(dir_proba))
                        
                        # === DETAILED LOGGING: Show ALL signal parameters ===
                        timing_prob = float(models['timing'].predict_proba(X)[0][1])
                        strength_pred = float(models['strength'].predict(X)[0])
                        
                        # DEBUG: Log prediction shape and values
                        if scan_stats['total'] <= 3:  # Log first 3 pairs
                            logger.debug(f"{pair}: Prediction - dir_pred={dir_pred}, dir_proba shape={dir_proba.shape}, dir_conf={dir_conf:.3f}")
                    except Exception as e:
                        logger.error(f"{pair}: Prediction error: {e}", exc_info=True)
                        scan_stats['errors'] += 1
                        continue
                    
                    # Update stats (ALWAYS update, even if signal is rejected)
                    # CRITICAL: This must execute for stats to be updated
                    if dir_pred == 0:
                        scan_stats['short'] += 1
                    elif dir_pred == 1:
                        scan_stats['side'] += 1
                    elif dir_pred == 2:
                        scan_stats['long'] += 1
                    else:
                        logger.warning(f"{pair}: Unexpected dir_pred value: {dir_pred} (type: {type(dir_pred)})")
                        scan_stats['errors'] += 1
                        continue
                    
                    # === LOG ALL SIGNALS (LONG/SHORT/SIDEWAYS) - ALWAYS LOG ===
                    direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
                    ws_price = streamer.current_prices.get(pair)
                    current_price = ws_price if ws_price else row['close'].iloc[0]
                    
                    # For LONG/SHORT: check if signal passes filters
                    if dir_pred != 1:  # LONG or SHORT
                        rejection_reasons = []
                        if dir_conf < MIN_CONF:
                            rejection_reasons.append(f"Conf({dir_conf:.2f}<{MIN_CONF})")
                        if timing_prob < MIN_TIMING:
                            rejection_reasons.append(f"Timing({timing_prob:.2f}<{MIN_TIMING})")
                        if strength_pred < MIN_STRENGTH:
                            rejection_reasons.append(f"Strength({strength_pred:.2f}<{MIN_STRENGTH})")
                        
                        status = "✅ ACCEPTED" if not rejection_reasons else f"❌ REJECTED: {', '.join(rejection_reasons)}"
                        
                        # ALWAYS LOG - this is critical for debugging
                        logger.info(
                            f"📊 {pair} {direction_str} | "
                            f"Conf: {dir_conf:.3f} | Timing: {timing_prob:.3f} | Strength: {strength_pred:.2f} | "
                            f"Price: {current_price:.6f} | {status}"
                        )
                    else:  # SIDEWAYS
                        # ALWAYS LOG - this is critical for debugging
                        logger.info(
                            f"📊 {pair} {direction_str} | "
                            f"Conf: {dir_conf:.3f} | Timing: {timing_prob:.3f} | Strength: {strength_pred:.2f} | "
                            f"Price: {current_price:.6f} | ⏸️  NO ACTION"
                        )
                    
                    # Apply filters
                    if dir_pred == 1:
                        continue  # Skip sideways
                    
                    rejection_reasons = []
                    if dir_conf < MIN_CONF:
                        rejection_reasons.append(f"Conf ({dir_conf:.2f} < {MIN_CONF})")
                        scan_stats['rejected_conf'] += 1
                    
                    if timing_prob < MIN_TIMING:
                        rejection_reasons.append(f"Timing ({timing_prob:.2f} < {MIN_TIMING})")
                        scan_stats['rejected_timing'] += 1
                    
                    if strength_pred < MIN_STRENGTH:
                        rejection_reasons.append(f"Strength ({strength_pred:.2f} < {MIN_STRENGTH})")
                        scan_stats['rejected_strength'] += 1
                    
                    if rejection_reasons:
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
                        'timing_prob': timing_prob,  # MATCH BACKTEST KEY NAME
                        'pred_strength': strength_pred  # MATCH BACKTEST KEY NAME
                    }
                    portfolio.open_position(signal)
                    logger.info(f"🚀 Signal taken: {pair} {signal['direction']} @ {current_price:.4f}")
                    break  # Taken a slot
                    
                except Exception as e:
                    logger.error(f"Scan error {pair}: {e}", exc_info=True)
                    scan_stats['errors'] += 1
            
            # Log scan statistics
            if scan_stats['total'] > 0:
                logger.info(
                    f"🔍 Scan complete: {scan_stats['total']} pairs | "
                    f"LONG: {scan_stats['long']} | SHORT: {scan_stats['short']} | SIDEWAYS: {scan_stats['side']} | "
                    f"Rejected: Conf({scan_stats['rejected_conf']}) Timing({scan_stats['rejected_timing']}) Strength({scan_stats['rejected_strength']}) | "
                    f"Skipped: DF-empty({scan_stats['skipped_df_empty']}) Missing-feat({scan_stats['skipped_missing_features']}) NaN({scan_stats['skipped_nan']}) | "
                    f"Fetched: {scan_stats['fetched']} | Errors: {scan_stats['errors']}"
                )
        
        # Sleep (shorter interval for faster trailing updates)
        time.sleep(10)

if __name__ == '__main__':
    main()

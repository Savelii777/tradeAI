#!/usr/bin/env python3
"""
Paper Trading Script V8 Improved - MEXC LIVE (Data from Binance)
- Uses v8_improved model
- Gets market data from Binance (free, no auth)
- Executes trades on MEXC (USDT-M futures)
- EXACT backtest logic (breakeven stop, trailing, etc.)
"""

import sys
import time
import json
import joblib
import ccxt
import requests
import pandas as pd
import numpy as np
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
TRADES_FILE = Path("active_trades_mexc.json")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 500

# V8 Thresholds (from backtest)
MIN_CONF = 0.50
MIN_TIMING = 0.55
MIN_STRENGTH = 1.4

# Risk Management (EXACT backtest settings)
RISK_PCT = 0.05
MAX_LEVERAGE = 20.0
MAX_HOLDING_BARS = 150  # 12.5 hours
ENTRY_FEE = 0.0002
EXIT_FEE = 0.0002
INITIAL_CAPITAL = 20.0
SL_ATR_BASE = 1.5
MAX_POSITION_SIZE = 50000.0
SLIPPAGE_PCT = 0.0001

# V8 Features
USE_ADAPTIVE_SL = True
USE_DYNAMIC_LEVERAGE = True
USE_AGGRESSIVE_TRAIL = True

# Telegram
TELEGRAM_TOKEN = "8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc"
TELEGRAM_CHAT_ID = "677822370"

# MEXC API
MEXC_API_KEY = "mx0vglp7RP0pQYiNA2"
MEXC_API_SECRET = "25817ec107364a55976a23ca6f19d470"

# ============================================================
# EXCHANGE MANAGERS
# ============================================================
class ExchangeManager:
    """Manages data fetching (Binance) and trading (MEXC)"""
    
    def __init__(self):
        # Binance for data (no auth needed)
        self.binance = ccxt.binance({
            'timeout': 10000,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # MEXC for trading
        self.mexc = ccxt.mexc({
            'apiKey': MEXC_API_KEY,
            'secret': MEXC_API_SECRET,
            'timeout': 10000,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # USDT-M futures
                'adjustForTimeDifference': True
            }
        })
        
        logger.info("✅ Exchanges initialized")
    
    def fetch_ohlcv_binance(self, symbol: str, timeframe: str, limit: int = 500):
        """Fetch candles from Binance (free data source)"""
        try:
            candles = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            return candles
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe} from Binance: {e}")
            return []
    
    def get_balance_mexc(self):
        """Get USDT balance from MEXC"""
        try:
            balance = self.mexc.fetch_balance()
            usdt_balance = balance['USDT']['free']
            return float(usdt_balance)
        except Exception as e:
            logger.error(f"Error fetching MEXC balance: {e}")
            return 0.0
    
    def create_order_mexc(self, symbol: str, side: str, amount: float, price: float = None, 
                         stop_loss: float = None, leverage: int = 20):
        """
        Create order on MEXC with stop loss.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            side: 'buy' or 'sell'
            amount: Position size in USD
            price: Limit price (None for market order)
            stop_loss: Stop loss price
            leverage: Leverage to use
        """
        try:
            # Set leverage
            self.mexc.set_leverage(leverage, symbol)
            
            # Calculate quantity in contracts
            ticker = self.mexc.fetch_ticker(symbol)
            current_price = ticker['last']
            quantity = amount / current_price
            
            # Round to exchange precision
            market = self.mexc.market(symbol)
            quantity = self.mexc.amount_to_precision(symbol, quantity)
            
            # Create market order
            order_type = 'market' if price is None else 'limit'
            order = self.mexc.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=price
            )
            
            logger.info(f"✅ Order created on MEXC: {side} {quantity} {symbol} @ {current_price}")
            
            # Set stop loss if provided
            if stop_loss:
                self.set_stop_loss_mexc(symbol, side, quantity, stop_loss)
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order on MEXC: {e}")
            return None
    
    def set_stop_loss_mexc(self, symbol: str, side: str, quantity: float, stop_price: float):
        """Set stop loss order on MEXC"""
        try:
            # Stop loss side is opposite of entry
            sl_side = 'sell' if side == 'buy' else 'buy'
            
            sl_order = self.mexc.create_order(
                symbol=symbol,
                type='stop_market',
                side=sl_side,
                amount=quantity,
                params={'stopPrice': stop_price}
            )
            
            logger.info(f"✅ Stop loss set: {sl_side} @ {stop_price}")
            return sl_order
            
        except Exception as e:
            logger.error(f"Error setting stop loss: {e}")
            return None
    
    def close_position_mexc(self, symbol: str, side: str):
        """Close position on MEXC"""
        try:
            # Get current position
            positions = self.mexc.fetch_positions([symbol])
            
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) > 0:
                    # Close position (opposite side)
                    close_side = 'sell' if side == 'LONG' else 'buy'
                    quantity = abs(float(pos['contracts']))
                    
                    order = self.mexc.create_order(
                        symbol=symbol,
                        type='market',
                        side=close_side,
                        amount=quantity,
                        params={'reduceOnly': True}
                    )
                    
                    logger.info(f"✅ Position closed: {close_side} {quantity} {symbol}")
                    return order
            
            logger.warning(f"No open position found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def update_stop_loss_mexc(self, symbol: str, side: str, new_stop: float):
        """Update stop loss (cancel old, create new)"""
        try:
            # Cancel all stop orders for this symbol
            open_orders = self.mexc.fetch_open_orders(symbol)
            for order in open_orders:
                if order['type'] == 'stop_market':
                    self.mexc.cancel_order(order['id'], symbol)
            
            # Get position size
            positions = self.mexc.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) > 0:
                    quantity = abs(float(pos['contracts']))
                    self.set_stop_loss_mexc(symbol, side, quantity, new_stop)
                    logger.info(f"✅ Stop loss updated: {new_stop}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return False


# ============================================================
# PORTFOLIO MANAGER (V8 BACKTEST LOGIC)
# ============================================================
class PortfolioManager:
    def __init__(self, exchange_manager):
        self.exchange = exchange_manager
        self.position = None
        self.trades_history = []
        
        # Load state from file (if exists) or get from exchange
        if self.load_state():
            logger.info(f"📂 State loaded from file. Capital: ${self.capital:.2f}")
        else:
            # Get real balance from MEXC
            self.capital = self.exchange.get_balance_mexc()
            if self.capital == 0:
                logger.warning("⚠️ Could not fetch MEXC balance, using INITIAL_CAPITAL")
                self.capital = INITIAL_CAPITAL
            else:
                logger.info(f"💰 Real balance from MEXC: ${self.capital:.2f}")
            self.save_state()
    
    def load_state(self):
        """Load state from file. Returns True if loaded, False if file doesn't exist."""
        if TRADES_FILE.exists():
            try:
                with open(TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    self.capital = data.get('capital', None)
                    self.position = data.get('position', None)
                    self.trades_history = data.get('history', [])
                    
                    if self.position and 'entry_time' in self.position:
                        if isinstance(self.position['entry_time'], str):
                            self.position['entry_time'] = datetime.fromisoformat(self.position['entry_time'])
                    
                    return self.capital is not None  # Return True if capital was loaded
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                return False
        return False
    
    def save_state(self):
        pos_copy = None
        if self.position:
            pos_copy = self.position.copy()
            if isinstance(pos_copy['entry_time'], datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
        
        data = {'capital': self.capital, 'position': pos_copy, 'history': self.trades_history}
        with open(TRADES_FILE, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    
    def open_position(self, signal):
        """Open position on MEXC (V8 backtest logic)"""
        if self.position is not None:
            return
        
        entry_price = signal['price']
        atr = signal['atr']
        pred_strength = signal.get('pred_strength', 2.0)
        conf = signal.get('conf', 0.5)
        timing = signal.get('timing_prob', 0.5)
        
        # === V8: ADAPTIVE STOP LOSS ===
        if USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:
                sl_mult = 1.6
            elif pred_strength >= 2.0:
                sl_mult = 1.5
            else:
                sl_mult = 1.2
        else:
            sl_mult = SL_ATR_BASE
        
        stop_distance = atr * sl_mult
        
        if signal['direction'] == 'LONG':
            stop_loss = entry_price - stop_distance
            side = 'buy'
        else:
            stop_loss = entry_price + stop_distance
            side = 'sell'
        
        # === V8: DYNAMIC BREAKEVEN TRIGGER ===
        if pred_strength >= 3.0:
            be_trigger_mult = 1.8
        elif pred_strength >= 2.0:
            be_trigger_mult = 1.5
        else:
            be_trigger_mult = 1.2
        
        # === V8: DYNAMIC RISK ===
        if USE_DYNAMIC_LEVERAGE:
            score = conf * timing
            quality = (score / 0.5) * (timing / 0.6) * (pred_strength / 2.0)
            quality_mult = np.clip(quality, 0.8, 1.5)
            risk_pct = RISK_PCT * quality_mult
        else:
            risk_pct = RISK_PCT
        
        # Position sizing
        stop_loss_pct = stop_distance / entry_price
        risk_amount = self.capital * risk_pct
        position_value = risk_amount / stop_loss_pct
        
        leverage = position_value / self.capital
        if leverage > MAX_LEVERAGE:
            leverage = MAX_LEVERAGE
            position_value = self.capital * leverage
        
        if position_value > MAX_POSITION_SIZE:
            position_value = MAX_POSITION_SIZE
            leverage = position_value / self.capital
        
        # Deduct entry fee
        entry_fee = position_value * ENTRY_FEE
        self.capital -= entry_fee
        
        # === CREATE ORDER ON MEXC ===
        order = self.exchange.create_order_mexc(
            symbol=signal['pair'],
            side=side,
            amount=position_value,
            stop_loss=stop_loss,
            leverage=int(leverage)
        )
        
        if order is None:
            logger.error(f"Failed to create order for {signal['pair']}")
            self.capital += entry_fee  # Refund fee
            return
        
        # Store position
        self.position = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'position_value': position_value,
            'leverage': leverage,
            'breakeven_active': False,
            'pred_strength': pred_strength,
            'be_trigger_mult': be_trigger_mult,
            'last_update_time': None,
            'bars_held': 0,
            'mexc_order_id': order['id']
        }
        self.save_state()
        
        self.send_alert(
            f"🟢 <b>{signal['direction']}</b> {signal['pair']}",
            f"Entry: {entry_price:.6f}\nSL: {stop_loss:.6f} ({sl_mult}ATR)\nLev: {leverage:.1f}x\nSize: ${position_value:.0f}\nStr: {pred_strength:.1f}"
        )
    
    def update_position(self, current_price, candle_high, candle_low):
        """
        Update position with V8 backtest logic:
        - Breakeven trigger
        - Progressive trailing stop
        """
        if self.position is None:
            return
        
        pos = self.position
        
        # Time limit check
        duration = datetime.now() - pos['entry_time']
        if duration > timedelta(minutes=MAX_HOLDING_BARS * 5):
            self.close_position(current_price, "Time Limit")
            return
        
        # Restore ATR
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
        
        old_sl = pos['stop_loss']
        updated = False
        
        # LONG LOGIC
        if pos['direction'] == 'LONG':
            # 1. Check Breakeven Trigger
            be_trigger_price = pos['entry_price'] + be_trigger_dist
            if not pos['breakeven_active'] and candle_high >= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] + (atr * 0.3)
                updated = True
                logger.info(f"✅ Breakeven activated for {pos['pair']}")
            
            # 2. Update Trailing Stop
            if pos['breakeven_active']:
                current_profit = candle_high - pos['entry_price']
                r_multiple = current_profit / pos['stop_distance']
                
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
                    updated = True
        
        # SHORT LOGIC
        else:
            be_trigger_price = pos['entry_price'] - be_trigger_dist
            if not pos['breakeven_active'] and candle_low <= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] - (atr * 0.3)
                updated = True
                logger.info(f"✅ Breakeven activated for {pos['pair']}")
            
            if pos['breakeven_active']:
                current_profit = pos['entry_price'] - candle_low
                r_multiple = current_profit / pos['stop_distance']
                
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
                    updated = True
        
        # Update stop loss on MEXC if changed
        if updated and old_sl != pos['stop_loss']:
            side = 'buy' if pos['direction'] == 'LONG' else 'sell'
            self.exchange.update_stop_loss_mexc(pos['pair'], side, pos['stop_loss'])
            logger.info(f"📈 Stop loss updated: {old_sl:.6f} → {pos['stop_loss']:.6f}")
        
        self.save_state()
    
    def close_position(self, price, reason):
        """Close position on MEXC"""
        pos = self.position
        
        if price is None or price <= 0:
            price = pos['entry_price']
        
        # Close on MEXC
        self.exchange.close_position_mexc(pos['pair'], pos['direction'])
        
        # Calculate PnL (V8 logic)
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
        self.send_alert(
            f"{emoji} <b>CLOSE {pos['direction']}</b> {pos['pair']}",
            f"Reason: {reason}\nPrice: {price:.4f}\nPnL: ${net:.2f} ({roe:.1f}%)\nBars: {pos.get('bars_held', 0)}\nCap: ${self.capital:.2f}"
        )
        
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
# UTILS
# ============================================================
def load_models():
    logger.info("Loading V8 Models...")
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    return models


def get_pairs():
    with open(PAIRS_FILE, 'r') as f:
        return [p['symbol'] for p in json.load(f)['pairs']][:20]


def add_volume_features(df):
    """Add volume features (V8 backtest)"""
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
    """Prepare features from multi-timeframe data"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Fill NaN
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        ft = ft.ffill().bfill()
        
        if ft.isna().any().any():
            ft = ft.fillna(0)
        
        return ft
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("Starting V8 MEXC Live Trading (Data from Binance)...")
    
    # Validate API keys
    if MEXC_API_KEY == "YOUR_MEXC_API_KEY":
        logger.error("⚠️ Please set your MEXC API keys in the script!")
        logger.error("   MEXC_API_KEY and MEXC_API_SECRET")
        return
    
    models = load_models()
    pairs = get_pairs()
    exchange = ExchangeManager()
    portfolio = PortfolioManager(exchange)
    mtf_fe = MTFFeatureEngine()
    
    logger.info(f"Monitoring {len(pairs)} pairs on MEXC")
    logger.info(f"Initial Capital: ${portfolio.capital:.2f}")
    
    last_scan = time.time()
    
    # Main loop
    while True:
        current_time = time.time()
        
        try:
            # Update position every 30 seconds
            if portfolio.position:
                pos = portfolio.position
                
                # Get current price from Binance
                ticker = exchange.binance.fetch_ticker(pos['pair'])
                current_price = ticker['last']
                
                # Get last 5m candle for trailing update
                candles_5m = exchange.fetch_ohlcv_binance(pos['pair'], '5m', limit=2)
                if len(candles_5m) >= 2:
                    last_candle = candles_5m[-2]  # Closed candle
                    candle_high = last_candle[2]
                    candle_low = last_candle[3]
                    
                    portfolio.update_position(current_price, candle_high, candle_low)
            
            # Scan for new signals every 60 seconds
            if current_time - last_scan >= 60:
                last_scan = current_time
                
                if portfolio.position is None:
                    logger.info("🔍 Scanning for new signals...")
                    
                    for pair in pairs:
                        try:
                            # Fetch data from Binance
                            data = {}
                            valid = True
                            
                            for tf in TIMEFRAMES:
                                candles = exchange.fetch_ohlcv_binance(pair, tf, LOOKBACK)
                                if not candles or len(candles) < 50:
                                    valid = False
                                    break
                                
                                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                                df.set_index('timestamp', inplace=True)
                                data[tf] = df
                            
                            if not valid:
                                continue
                            
                            # Prepare features
                            df = prepare_features(data, mtf_fe)
                            if df is None or len(df) < 2:
                                continue
                            
                            # Get last closed candle
                            row = df.iloc[[-2]]
                            
                            # Validate features
                            missing_features = [f for f in models['features'] if f not in row.columns]
                            if missing_features:
                                continue
                            
                            X = row[models['features']].values
                            
                            if pd.isna(X).any():
                                continue
                            
                            # Predictions
                            dir_proba = models['direction'].predict_proba(X)
                            dir_conf = float(np.max(dir_proba))
                            dir_pred = int(np.argmax(dir_proba))
                            
                            timing_prob = float(models['timing'].predict_proba(X)[0][1])
                            strength_pred = float(models['strength'].predict(X)[0])
                            
                            # Apply V8 filters
                            if dir_pred == 1:  # Sideways
                                continue
                            
                            if dir_conf < MIN_CONF:
                                continue
                            if timing_prob < MIN_TIMING:
                                continue
                            if strength_pred < MIN_STRENGTH:
                                continue
                            
                            # Get current price
                            ticker = exchange.binance.fetch_ticker(pair)
                            current_price = ticker['last']
                            
                            # Create signal
                            signal = {
                                'pair': pair,
                                'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                                'price': current_price,
                                'atr': row['atr'].iloc[0],
                                'conf': dir_conf,
                                'timing_prob': timing_prob,
                                'pred_strength': strength_pred
                            }
                            
                            portfolio.open_position(signal)
                            logger.info(f"🚀 Signal taken: {pair} {signal['direction']} @ {current_price:.6f}")
                            break  # Only one position
                            
                        except Exception as e:
                            logger.error(f"Error processing {pair}: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        
        # Sleep
        time.sleep(30)


if __name__ == '__main__':
    main()


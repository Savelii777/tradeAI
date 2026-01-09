#!/usr/bin/env python3
"""
Paper Trading Script V8 Improved - MEXC LIVE (Data from Binance)
- Uses v8_improved model
- Gets market data from Binance (free, no auth)
- Executes trades on MEXC (USDT-M futures) via DIRECT API
- EXACT backtest logic (breakeven stop, trailing, etc.)
- Loads credentials from config/secrets.yaml (no hardcoded secrets)
"""

import sys
import time
import json
import joblib
import ccxt
import requests
import hmac
import hashlib
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_engine import FeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES
)
from train_mtf import MTFFeatureEngine

# ============================================================
# LOGGING SETUP - Write to file
# ============================================================
LOG_FILE = Path("logs/live_trading.log")
LOG_FILE.parent.mkdir(exist_ok=True)

# Remove default handler and add file + console
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    rotation="10 MB",  # New file when reaches 10MB
    retention="7 days",  # Keep logs for 7 days
    level="DEBUG"  # Log everything to file
)
logger.info(f"📝 Logging to file: {LOG_FILE.absolute()}")

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
TRADES_FILE = Path("active_trades_mexc.json")
TIMEFRAMES = ['1m', '5m', '15m']

# ✅ FIX: LOOKBACK determines how many M1 bars to fetch from API
# After align_timeframes() and dropna(), we get fewer rows than LOOKBACK
# Example: LOOKBACK=3000 M1 bars → ~200-500 aligned rows (depends on M5/M15 alignment)
# CRITICAL: With too few bars, volume_ratio features differ significantly from backtest
LOOKBACK = 3000  # 3000 M1 bars → ~500 aligned rows → enough for analysis
WARMUP_BARS = 50  # ✅ First 50 bars may have unstable EMA values, skip them
MIN_ROWS_FOR_PREDICTION = 2  # Need at least 2 rows: current (forming) and last closed candle

# Timeframe multipliers for data alignment
M1_TO_M5_RATIO = 5  # M1 has 5x more candles than M5
M5_TO_M15_RATIO = 3  # M15 has 3x fewer candles than M5

# V8 IMPROVED Thresholds (UPDATED for new Timing model)
MIN_CONF = 0.50       # Direction confidence
MIN_TIMING = 0.8      # ✅ NEW! Timing now predicts ATR gain (0-5), threshold = 0.8 ATR minimum
MIN_STRENGTH = 1.4    # Strength prediction

# Risk Management (EXACT backtest settings)
RISK_PCT = 0.05
MAX_LEVERAGE = 50.0
MAX_HOLDING_BARS = 150  # 12.5 hours
ENTRY_FEE = 0.0002
EXIT_FEE = 0.0002
INITIAL_CAPITAL = 20.0
SL_ATR_BASE = 1.5
MAX_POSITION_SIZE = 200000.0
SLIPPAGE_PCT = 0.0001

# V8 Features
USE_ADAPTIVE_SL = True
USE_DYNAMIC_LEVERAGE = True
USE_AGGRESSIVE_TRAIL = True

# Telegram (load from config/secrets.yaml)
# IMPORTANT: Do not hardcode secrets! Use config/secrets.yaml
TELEGRAM_TOKEN = ""  # Loaded from secrets.yaml
TELEGRAM_CHAT_ID = ""  # Loaded from secrets.yaml

# MEXC API (load from config/secrets.yaml)
# IMPORTANT: Do not hardcode secrets! Use config/secrets.yaml
MEXC_API_KEY = ""  # Loaded from secrets.yaml
MEXC_API_SECRET = ""  # Loaded from secrets.yaml
MEXC_BASE_URL = "https://contract.mexc.com"

# Load secrets from config file
def _load_secrets():
    """Load secrets from config/secrets.yaml."""
    secrets_file = Path(__file__).parent.parent / "config" / "secrets.yaml"
    if not secrets_file.exists():
        logger.warning(
            f"⚠️ Secrets file not found: {secrets_file}\n"
            "Please copy config/secrets.yaml.example to config/secrets.yaml "
            "and fill in your credentials."
        )
        return {}
    
    with open(secrets_file, 'r') as f:
        return yaml.safe_load(f)

# Auto-load secrets on module import
try:
    _secrets = _load_secrets()
    TELEGRAM_TOKEN = _secrets.get('notifications', {}).get('telegram', {}).get('bot_token', '')
    TELEGRAM_CHAT_ID = _secrets.get('notifications', {}).get('telegram', {}).get('chat_id', '')
    MEXC_API_KEY = _secrets.get('mexc', {}).get('api_key', '')
    MEXC_API_SECRET = _secrets.get('mexc', {}).get('api_secret', '')
except Exception as e:
    logger.warning(f"Could not load secrets: {e}")

# ============================================================
# MEXC DIRECT API CLIENT
# ============================================================
class MEXCClient:
    """Direct MEXC Futures API client (no CCXT)"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = MEXC_BASE_URL
        logger.info("✅ MEXC Direct API Client initialized")
    
    def _generate_signature(self, sign_str: str) -> str:
        """Generate HMAC SHA256 signature for MEXC"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, params: dict = None):
        """Make authenticated request to MEXC"""
        timestamp = int(time.time() * 1000)
        
        if params is None:
            params = {}
        
        params['timestamp'] = timestamp
        
        # Sort parameters and build query string
        sorted_params = sorted(params.items())
        params_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Generate signature: API_KEY + timestamp + params_str
        sign_str = f"{self.api_key}{timestamp}{params_str}"
        signature = self._generate_signature(sign_str)
        
        # Headers with signature
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Signature': signature,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method == 'POST':
                # For POST, send params as query string (not JSON body)
                response = requests.post(url, params=params, headers=headers, timeout=30)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"MEXC API request failed: {e}")
            if 'response' in locals():
                logger.error(f"Response: {response.text if hasattr(response, 'text') else 'N/A'}")
            return None
    
    def get_account_assets(self):
        """Get account balance"""
        result = self._request('GET', '/api/v1/private/account/assets', {})
        if result and result.get('success'):
            # Find USDT balance
            for asset in result.get('data', []):
                if asset['currency'] == 'USDT':
                    return float(asset.get('availableBalance', 0))
        return 0.0
    
    def get_open_positions(self, symbol: str = None):
        """Get open positions"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        result = self._request('GET', '/api/v1/private/position/open_positions', params)
        if result and result.get('success'):
            return result.get('data', [])
        return []
    
    def place_order(self, symbol: str, side: int, volume: int, leverage: int, 
                   price: float = 0, order_type: int = 5, open_type: int = 1):
        """
        Place order on MEXC Futures.
        
        Args:
            symbol: Symbol (e.g., 'BTC_USDT', 'PIPPIN_USDT')
            side: 1=open long, 2=open short, 3=close long, 4=close short
            volume: Volume in contracts
            leverage: Leverage (1-125)
            price: Limit price (0 for market order)
            order_type: 5=market, 6=limit
            open_type: 1=isolated, 2=cross
        """
        params = {
            'symbol': symbol,
            'price': price,
            'vol': volume,
            'leverage': leverage,
            'side': side,
            'type': order_type,
            'openType': open_type
        }
        
        logger.info(f"📤 Placing MEXC order: {params}")
        result = self._request('POST', '/api/v1/private/order/submit', params)
        
        if result and result.get('success'):
            logger.info(f"✅ Order placed successfully! Order ID: {result.get('data')}")
            return result.get('data')
        else:
            logger.error(f"❌ Order failed: {result}")
            return None
    
    def place_stop_order(self, symbol: str, side: int, volume: int, stop_price: float, leverage: int):
        """
        Place STOP LOSS order on MEXC.
        
        Args:
            symbol: Symbol (e.g., 'BTC_USDT')
            side: 3=close long (stop), 4=close short (stop)
            volume: Volume in contracts
            stop_price: Trigger price for stop loss
            leverage: Leverage
        """
        # MEXC uses trigger price in planType
        params = {
            'symbol': symbol,
            'price': 0,  # Market order when triggered
            'vol': volume,
            'leverage': leverage,
            'side': side,
            'type': 5,  # Market order
            'openType': 1,  # Isolated
            'triggerPrice': stop_price,
            'triggerType': 1,  # Last price trigger
            'executeCycle': 1,  # Execute once
            'trend': 1 if side == 3 else 2  # 1=price down (long SL), 2=price up (short SL)
        }
        
        logger.info(f"📤 Placing STOP order @ {stop_price}: {params}")
        result = self._request('POST', '/api/v1/private/planorder/place', params)
        
        if result and result.get('success'):
            order_id = result.get('data')
            logger.info(f"✅ Stop order placed! Order ID: {order_id}")
            return order_id
        else:
            logger.error(f"❌ Stop order failed: {result}")
            return None
    
    def cancel_stop_orders(self, symbol: str):
        """Cancel all stop/plan orders for a symbol"""
        # Get all plan orders for symbol
        result = self._request('GET', '/api/v1/private/planorder/list', {'symbol': symbol})
        
        if not result or not result.get('success'):
            return False
        
        orders = result.get('data', [])
        cancelled = 0
        
        for order in orders:
            order_id = order.get('id')
            if order_id:
                cancel_result = self._request('POST', '/api/v1/private/planorder/cancel', {
                    'symbol': symbol,
                    'orderId': order_id
                })
                if cancel_result and cancel_result.get('success'):
                    cancelled += 1
        
        if cancelled > 0:
            logger.info(f"✅ Cancelled {cancelled} stop orders for {symbol}")
        
        return True

# ============================================================
# PORTFOLIO MANAGER (V8 BACKTEST LOGIC)
# ============================================================
class PortfolioManager:
    def __init__(self, mexc_client):
        self.mexc = mexc_client
        self.position = None
        self.trades_history = []
        
        # Load state from file (if exists) or get from exchange
        if self.load_state():
            logger.info(f"📂 State loaded from file. Capital: ${self.capital:.2f}")
        else:
            # Get real balance from MEXC
            self.capital = self.mexc.get_account_assets()
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
        
        # === V8: DYNAMIC RISK (MATCHES BACKTEST) ===
        if USE_DYNAMIC_LEVERAGE:
            # CORRECT: Use same formula as backtest
            score = conf * timing
            timing_norm = timing
            strength_norm = pred_strength
            
            # Quality multiplier: 0.8x to 1.5x based on signal quality
            # Formula from backtest: (score / 0.5) * (timing / 0.6) * (strength / 2.0)
            quality = (score / 0.5) * (timing_norm / 0.6) * (strength_norm / 2.0)
            quality_mult = np.clip(quality, 0.8, 1.5)
            risk_pct = RISK_PCT * quality_mult
        else:
            risk_pct = RISK_PCT
        
        # Position sizing (MATCHES BACKTEST)
        stop_loss_pct = stop_distance / entry_price
        risk_amount = self.capital * risk_pct
        position_value = risk_amount / stop_loss_pct
        
        # Calculate leverage
        leverage = position_value / self.capital
        if leverage > MAX_LEVERAGE:
            leverage = MAX_LEVERAGE
            position_value = self.capital * leverage
        
        # Cap position size (BACKTEST LIMIT)
        if position_value > MAX_POSITION_SIZE:
            position_value = MAX_POSITION_SIZE
            leverage = position_value / self.capital
        
        # Deduct entry fee
        entry_fee = position_value * ENTRY_FEE
        self.capital -= entry_fee
        
        # === PLACE ORDER ON MEXC ===
        # Convert symbol: BTC/USDT:USDT -> BTC_USDT
        mexc_symbol = signal['pair'].replace('/USDT:USDT', '_USDT').replace('/', '_')
        
        # Calculate volume in contracts
        volume = int(position_value / entry_price)
        if volume < 1:
            volume = 1
        
        # MEXC API: side 1=open long, 2=open short
        mexc_side = 1 if signal['direction'] == 'LONG' else 2
        
        logger.info(f"📊 Position sizing: ${position_value:.0f} | {volume} contracts | Lev: {leverage:.1f}x")
        
        order_id = self.mexc.place_order(
            symbol=mexc_symbol,
            side=mexc_side,
            volume=volume,
            leverage=int(leverage)
        )
        
        if order_id is None:
            logger.error(f"Failed to create order for {signal['pair']}")
            self.capital += entry_fee  # Refund fee
            return
        
        # === PLACE STOP LOSS ORDER ON MEXC ===
        # Side: 3=close long, 4=close short
        stop_side = 3 if signal['direction'] == 'LONG' else 4
        
        stop_order_id = self.mexc.place_stop_order(
            symbol=mexc_symbol,
            side=stop_side,
            volume=volume,
            stop_price=stop_loss,
            leverage=int(leverage)
        )
        
        if stop_order_id is None:
            logger.warning(f"⚠️ Failed to place stop order! Position is unprotected!")
            # Continue anyway - we'll manage SL manually if needed
        
        # Store position
        self.position = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': datetime.now(timezone.utc),
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'position_value': position_value,
            'leverage': leverage,
            'breakeven_active': False,
            'pred_strength': pred_strength,
            'be_trigger_mult': be_trigger_mult,
            'last_update_time': None,
            'bars_held': 0,
            'mexc_order_id': order_id,
            'mexc_stop_order_id': stop_order_id,  # Track stop order
            'mexc_symbol': mexc_symbol,
            'volume': volume
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
        duration = datetime.now(timezone.utc) - pos['entry_time']
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
            mexc_symbol = pos['mexc_symbol']
            volume = pos['volume']
            new_sl = pos['stop_loss']
            stop_side = 3 if pos['direction'] == 'LONG' else 4
            
            # Cancel old stop order
            if pos.get('mexc_stop_order_id'):
                logger.debug(f"Cancelling old stop order...")
                self.mexc.cancel_stop_orders(mexc_symbol)
            
            # Place new stop order
            logger.info(f"📈 Updating stop loss on MEXC: {old_sl:.6f} → {new_sl:.6f}")
            new_stop_order_id = self.mexc.place_stop_order(
                symbol=mexc_symbol,
                side=stop_side,
                volume=volume,
                stop_price=new_sl,
                leverage=int(pos['leverage'])
            )
            
            if new_stop_order_id:
                pos['mexc_stop_order_id'] = new_stop_order_id
            else:
                logger.warning(f"⚠️ Failed to update stop order on exchange!")
        
        self.save_state()
    
    def close_position(self, price, reason):
        """Close position on MEXC"""
        pos = self.position
        
        if price is None or price <= 0:
            price = pos['entry_price']
        
        # === CANCEL STOP ORDERS FIRST ===
        mexc_symbol = pos['mexc_symbol']
        logger.info(f"Cancelling stop orders for {mexc_symbol}...")
        self.mexc.cancel_stop_orders(mexc_symbol)
        
        # === CLOSE ON MEXC ===
        volume = pos['volume']
        
        # MEXC API: side 3=close long, 4=close short
        close_side = 3 if pos['direction'] == 'LONG' else 4
        
        logger.info(f"📤 Closing position: {mexc_symbol} | {volume} contracts | Side: {close_side}")
        
        close_order = self.mexc.place_order(
            symbol=mexc_symbol,
            side=close_side,
            volume=volume,
            leverage=int(pos['leverage'])
        )
        
        if close_order is None:
            logger.error(f"❌ Failed to close position on MEXC!")
            # Continue with PnL calculation anyway (for tracking)
        
        # Calculate PnL (V8 logic - MATCHES BACKTEST)
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
            'time': datetime.now(timezone.utc).isoformat()
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
    """Add volume features (V8 backtest) - OBV исключен (зависит от начала окна данных)"""
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # OBV УДАЛЕН: cumsum() зависит от начала окна данных
    # В бектесте данные могут начинаться с 2017 года, в лайве - с последних 1500 свечей
    # Это приводит к кардинально разным значениям OBV
    # OBV уже исключен из фичей модели, поэтому не вычисляем его вообще
    
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # ✅ FIX: Clip extreme spikes (PIPPIN had 431x volume spike)
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    
    return df


def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def wait_until_candle_close(timeframe_minutes=5):
    """
    Wait until the next candle close.
    
    5-minute candles close at: :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55
    We wait until that moment + 3 seconds (for data to update on exchange).
    
    Returns:
        datetime: The candle close time we waited for
    """
    now = datetime.now(timezone.utc)
    
    # Calculate next candle close time
    current_minute = now.minute
    current_second = now.second
    
    # Round up to next 5-minute mark
    next_close_minute = ((current_minute // timeframe_minutes) + 1) * timeframe_minutes
    
    if next_close_minute >= 60:
        # Next hour
        next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_close = now.replace(minute=next_close_minute, second=0, microsecond=0)
    
    # Calculate wait time
    wait_seconds = (next_close - now).total_seconds()
    
    # Add 3 seconds buffer for exchange data to update
    wait_seconds += 3
    
    if wait_seconds > 0:
        logger.info(f"⏰ Waiting {wait_seconds:.0f}s until candle close at {next_close.strftime('%H:%M:%S')} UTC...")
        time.sleep(wait_seconds)
    
    return next_close


def prepare_features(data, mtf_fe):
    """
    Prepare features from multi-timeframe data.
    
    ✅ FIXED: Matches backtest processing exactly:
    - Same NaN handling (dropna, not fillna)
    - Excludes cumsum-dependent features
    - Ensures sufficient data for indicators
    """
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    # 🔍 SUPER DEBUG: Log raw data stats BEFORE any processing
    logger.debug("=" * 70)
    logger.debug("🔍 [PREPARE_FEATURES] RAW DATA STATS BEFORE PROCESSING")
    logger.debug("=" * 70)
    for tf, df in [('M1', m1), ('M5', m5), ('M15', m15)]:
        logger.debug(f"[{tf}] Shape: {df.shape}")
        logger.debug(f"[{tf}] Index range: {df.index[0]} to {df.index[-1]}")
        logger.debug(f"[{tf}] Close range: {df['close'].min():.6f} to {df['close'].max():.6f}")
        logger.debug(f"[{tf}] Volume range: {df['volume'].min():.2f} to {df['volume'].max():.2f}")
        # Last 3 candles
        logger.debug(f"[{tf}] Last 3 candles:")
        for idx, row in df.tail(3).iterrows():
            logger.debug(f"    {idx} | C:{row['close']:.6f} V:{row['volume']:.2f}")
    
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 200:
        logger.warning(f"Insufficient data: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
        return pd.DataFrame()
    
    # Ensure DatetimeIndex (UTC, same as backtest)
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        # Remove duplicates (same as backtest)
        df = df[~df.index.duplicated(keep='first')]
    
    try:
        logger.debug("[PREPARE_FEATURES] Calling mtf_fe.align_timeframes()...")
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        
        if len(ft) == 0:
            logger.warning("[PREPARE_FEATURES] align_timeframes returned empty DataFrame!")
            return pd.DataFrame()
        
        logger.debug(f"[PREPARE_FEATURES] After align_timeframes: {ft.shape}")
        logger.debug(f"[PREPARE_FEATURES] Columns after align: {list(ft.columns[:20])}...")
        
        # 🔍 DEBUG: Check key M15 features BEFORE join
        m15_cols = [c for c in ft.columns if c.startswith('m15_')]
        if m15_cols:
            last_row = ft.iloc[-2] if len(ft) > 1 else ft.iloc[-1]
            logger.debug(f"[M15_FEATURES] Last closed candle ({ft.index[-2] if len(ft) > 1 else ft.index[-1]}):")
            for col in m15_cols[:10]:
                val = last_row[col]
                logger.debug(f"    {col}: {val:.6f}" if pd.notna(val) else f"    {col}: NaN")
            
            # 🔍 CRITICAL DEBUG: Check M15 source data timing
            logger.debug(f"[M15_SOURCE] Raw M15 last 3 candles:")
            for idx, row in m15.tail(3).iterrows():
                logger.debug(f"    {idx} | C:{row['close']:.6f}")
        
        # 🔍 DEBUG: Check key M1 features BEFORE join  
        m1_cols = [c for c in ft.columns if c.startswith('m1_')]
        if m1_cols:
            last_row = ft.iloc[-2] if len(ft) > 1 else ft.iloc[-1]
            logger.debug(f"[M1_FEATURES] Sample (last closed candle):")
            for col in m1_cols[:10]:
                val = last_row[col]
                logger.debug(f"    {col}: {val:.6f}" if pd.notna(val) else f"    {col}: NaN")
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        logger.debug(f"[PREPARE_FEATURES] After join with OHLCV: {ft.shape}")
        
        ft = add_volume_features(ft)
        logger.debug(f"[PREPARE_FEATURES] After add_volume_features: {ft.shape}")
        
        # 🔍 DEBUG: Check volume features
        last_row = ft.iloc[-2] if len(ft) > 1 else ft.iloc[-1]
        logger.debug(f"[VOLUME_FEATURES] vol_sma_20={last_row.get('vol_sma_20', 'N/A')}")
        logger.debug(f"[VOLUME_FEATURES] vol_ratio={last_row.get('vol_ratio', 'N/A')}")
        logger.debug(f"[VOLUME_FEATURES] vwap={last_row.get('vwap', 'N/A')}")
        logger.debug(f"[VOLUME_FEATURES] price_vs_vwap={last_row.get('price_vs_vwap', 'N/A')}")
        
        ft['atr'] = calculate_atr(ft)
        logger.debug(f"[PREPARE_FEATURES] ATR calculated. Last value: {ft['atr'].iloc[-2] if len(ft) > 1 else ft['atr'].iloc[-1]:.6f}")
        
        # ✅ FIXED: Match backtest NaN handling - DROP rows with NaN in critical cols
        # DO NOT fill NaN with 0 or ffill - this changes feature distributions!
        critical_cols = ['close', 'atr']
        before_dropna = len(ft)
        ft = ft.dropna(subset=critical_cols)
        logger.debug(f"[PREPARE_FEATURES] Dropped {before_dropna - len(ft)} rows with NaN in critical cols")
        
        # ✅ FIXED: Exclude ALL cumsum/window-dependent features (same as training)
        # Uses centralized constants from src/utils/constants.py
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        if cols_to_drop:
            logger.debug(f"Excluding cumsum-dependent features: {cols_to_drop}")
            ft = ft.drop(columns=cols_to_drop)
        
        # ✅ Exclude absolute price-based features (same as training)
        # Uses centralized constants from src/utils/constants.py
        absolute_cols_to_drop = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        if absolute_cols_to_drop:
            logger.debug(f"Excluding absolute price features: {absolute_cols_to_drop}")
            ft = ft.drop(columns=absolute_cols_to_drop)
        
        # Only forward-fill for non-critical columns
        non_critical = [c for c in ft.columns if c not in critical_cols]
        if non_critical:
            nan_before = ft[non_critical].isna().sum().sum()
            ft[non_critical] = ft[non_critical].ffill()
            nan_after = ft[non_critical].isna().sum().sum()
            logger.debug(f"[PREPARE_FEATURES] Forward-filled {nan_before - nan_after} NaN values")
        
        # Final check
        ft = ft.dropna()  # Remove warmup rows (e.g. first 200 for EMA-200)
        
        if len(ft) == 0:
            logger.warning("No valid rows after final NaN handling (all rows were warmup?)")
            return pd.DataFrame()
        
        # 🔍 FINAL DEBUG: Show final feature stats
        logger.debug("=" * 70)
        logger.debug(f"[PREPARE_FEATURES] FINAL: {ft.shape[0]} rows, {ft.shape[1]} columns")
        logger.debug(f"[PREPARE_FEATURES] Index range: {ft.index[0]} to {ft.index[-1]}")
        
        # Check for remaining NaN/Inf
        nan_count = ft.isna().sum().sum()
        inf_count = np.isinf(ft.select_dtypes(include=[np.number])).sum().sum()
        logger.debug(f"[PREPARE_FEATURES] Remaining NaN: {nan_count}, Inf: {inf_count}")
        
        if nan_count > 0:
            nan_cols = ft.columns[ft.isna().any()].tolist()
            logger.warning(f"[PREPARE_FEATURES] Columns with NaN: {nan_cols}")
        
        logger.debug("=" * 70)
        
        return ft
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("Starting V8 MEXC Live Trading (Data from Binance)...")
    
    # Validate API keys (now loaded from config/secrets.yaml)
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        logger.error("⚠️ MEXC API keys not configured!")
        logger.error("   Please copy config/secrets.yaml.example to config/secrets.yaml")
        logger.error("   and fill in your MEXC API credentials.")
        return
    
    models = load_models()
    pairs = get_pairs()
    
    # Initialize MEXC Direct API and Binance
    mexc_client = MEXCClient(MEXC_API_KEY, MEXC_API_SECRET)
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    portfolio = PortfolioManager(mexc_client)
    mtf_fe = MTFFeatureEngine()
    
    logger.info(f"Monitoring {len(pairs)} pairs on MEXC")
    logger.info(f"Initial Capital: ${portfolio.capital:.2f}")
    logger.info(f"🎯 SYNC MODE: Scanning at 5-minute candle closes (:00, :05, :10, ...)")
    
    # Main loop - synchronized with candle closes
    while True:
        try:
            # === WAIT FOR CANDLE CLOSE ===
            candle_close_time = wait_until_candle_close(timeframe_minutes=5)
            
            # === UPDATE POSITION IF EXISTS ===
            if portfolio.position:
                pos = portfolio.position
                
                logger.info(f"📍 Updating position: {pos['pair']} {pos['direction']}")
                
                # Get current price from Binance
                ticker = binance.fetch_ticker(pos['pair'])
                current_price = ticker['last']
                
                # Get last 5m candle for trailing update
                candles_5m = binance.fetch_ohlcv(pos['pair'], '5m', limit=2)
                if len(candles_5m) >= 2:
                    last_candle = candles_5m[-2]  # Closed candle
                    candle_high = last_candle[2]
                    candle_low = last_candle[3]
                    
                    portfolio.update_position(current_price, candle_high, candle_low)
            
            # === SCAN FOR NEW SIGNALS (only if no position) ===
            if portfolio.position is None:
                    logger.info("=" * 70)
                    logger.info("🔍 Scanning for new signals...")
                    logger.info(f"   Filters: Conf>{MIN_CONF}, Timing>{MIN_TIMING}, Strength>{MIN_STRENGTH}")
                    logger.info(f"   Available capital: ${portfolio.capital:.2f}")
                    
                    signals_checked = 0
                    signals_found = 0
                    
                    # 🔍 Собираем статистику по всем парам для анализа
                    scan_stats = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'pairs_scanned': 0,
                        'sideways_count': 0,
                        'rejected_conf': 0,
                        'rejected_timing': 0,
                        'rejected_strength': 0,
                        'signals_found': 0,
                        'predictions': []  # Все предсказания
                    }
                    
                    for pair in pairs:
                        signals_checked += 1
                        logger.info(f"   [{signals_checked}/{len(pairs)}] Checking {pair}...")
                        
                        try:
                            # Fetch data from Binance
                            data = {}
                            valid = True
                            
                            for tf in TIMEFRAMES:
                                # ✅ FIXED: Fetch MORE data if needed (Binance limit is 1000 per request)
                                # For LOOKBACK=3000, we need multiple requests
                                all_candles = []
                                limit_per_request = 1000
                                fetch_requests = 0
                                
                                # First request: get latest candles
                                try:
                                    logger.debug(f"      [{tf}] Fetching from Binance (limit={min(limit_per_request, LOOKBACK)})...")
                                    candles = binance.fetch_ohlcv(pair, tf, limit=min(limit_per_request, LOOKBACK))
                                    fetch_requests += 1
                                    
                                    if not candles or len(candles) < 50:
                                        logger.warning(f"      [{tf}] Got only {len(candles) if candles else 0} candles (need 50+)")
                                        valid = False
                                        break
                                    
                                    all_candles.extend(candles)
                                    logger.debug(f"      [{tf}] Got {len(candles)} candles in request #1")
                                    
                                    # Логируем первую и последнюю свечу
                                    first_ts = datetime.fromtimestamp(candles[0][0]/1000, tz=timezone.utc)
                                    last_ts = datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc)
                                    logger.debug(f"      [{tf}] Range: {first_ts} to {last_ts}")
                                    
                                except Exception as e:
                                    logger.warning(f"Error fetching {pair} {tf}: {e}")
                                    valid = False
                                    break
                                
                                # If we need more data, fetch older candles
                                if len(all_candles) < LOOKBACK:
                                    requests_needed = (LOOKBACK - len(all_candles) + limit_per_request - 1) // limit_per_request
                                    logger.debug(f"      [{tf}] Need more data, making {requests_needed} additional requests...")
                                    
                                    for req_num in range(requests_needed):
                                        try:
                                            # Get oldest timestamp from current data
                                            oldest_ts = all_candles[0][0]
                                            # Calculate timeframe milliseconds
                                            tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[tf]
                                            # Request older data
                                            since_ts = oldest_ts - (limit_per_request * tf_ms)
                                            candles = binance.fetch_ohlcv(pair, tf, since=since_ts, limit=limit_per_request)
                                            fetch_requests += 1
                                            
                                            if not candles:
                                                logger.debug(f"      [{tf}] No more data in request #{fetch_requests}")
                                                break
                                            
                                            # Remove duplicates and merge (oldest first)
                                            seen_timestamps = {c[0] for c in all_candles}
                                            new_candles = [c for c in candles if c[0] not in seen_timestamps]
                                            
                                            if not new_candles:
                                                logger.debug(f"      [{tf}] No new candles in request #{fetch_requests}")
                                                break
                                            
                                            logger.debug(f"      [{tf}] Got {len(new_candles)} new candles in request #{fetch_requests}")
                                            
                                            # Insert at beginning (older data)
                                            all_candles = new_candles + all_candles
                                            
                                            # Stop if we have enough
                                            if len(all_candles) >= LOOKBACK:
                                                # Keep only the NEWEST LOOKBACK candles (not oldest!)
                                                all_candles = sorted(all_candles, key=lambda x: x[0])[-LOOKBACK:]
                                                logger.debug(f"      [{tf}] Trimmed to {len(all_candles)} newest candles")
                                                break
                                            
                                            time.sleep(0.1)  # Rate limit
                                        except Exception as e:
                                            logger.debug(f"Error fetching {pair} {tf} batch {req_num}: {e}")
                                            break
                                
                                if not all_candles or len(all_candles) < 200:
                                    logger.warning(f"Insufficient data for {pair} {tf}: {len(all_candles)} candles (need 200+)")
                                    valid = False
                                    break
                                
                                # Sort by timestamp (oldest first)
                                all_candles = sorted(all_candles, key=lambda x: x[0])
                                
                                # 🔍 DEBUG: Логируем финальные данные перед DataFrame
                                logger.debug(f"      [{tf}] Final: {len(all_candles)} candles after {fetch_requests} API requests")
                                
                                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                                df.set_index('timestamp', inplace=True)
                                
                                # ✅ SAFETY: Ensure numeric types logic
                                for col in ['open', 'high', 'low', 'close', 'volume']:
                                    df[col] = df[col].astype(float)
                                
                                # ✅ FIX: Remove duplicates and sort (same as CSV loading)
                                # This ensures consistency with backtest data processing
                                dups_before = len(df)
                                df = df[~df.index.duplicated(keep='first')]
                                dups_removed = dups_before - len(df)
                                if dups_removed > 0:
                                    logger.debug(f"      [{tf}] Removed {dups_removed} duplicate timestamps")
                                df.sort_index(inplace=True)
                                
                                # 🔍 DEBUG: Log data range to verify we have fresh data
                                now_utc = datetime.now(timezone.utc)
                                oldest = df.index[0]
                                newest = df.index[-1]
                                age_seconds = (now_utc - newest.to_pydatetime()).total_seconds()
                                logger.debug(f"      [{tf}] Data: {len(df)} candles | Range: {oldest} to {newest} | Age: {age_seconds:.0f}s")
                                
                                # 🔍 DEBUG: Логируем последние 3 свечи этого ТФ
                                last3 = df.tail(3)
                                logger.debug(f"      [{tf}] Last 3 candles (RAW from API):")
                                for idx, row_data in last3.iterrows():
                                    logger.debug(f"         {idx} | O:{row_data['open']:.6f} H:{row_data['high']:.6f} L:{row_data['low']:.6f} C:{row_data['close']:.6f} V:{row_data['volume']:.2f}")
                                
                                data[tf] = df
                            
                            if not valid:
                                continue
                            
                            # Prepare features
                            logger.debug(f"      Preparing features from {len(data)} timeframes...")
                            df = prepare_features(data, mtf_fe)
                            if df is None or len(df) < MIN_ROWS_FOR_PREDICTION:
                                logger.warning(f"      Feature preparation failed or not enough rows")
                                continue
                            
                            logger.debug(f"      Features prepared: {len(df)} rows, {len(df.columns)} columns")
                            
                            # ✅ FIX: Skip warmup rows - first WARMUP_BARS have unstable features
                            # EMA-200, rolling stats, etc. need time to stabilize
                            if len(df) < WARMUP_BARS + MIN_ROWS_FOR_PREDICTION:
                                logger.warning(f"      Not enough data after warmup ({len(df)} < {WARMUP_BARS + MIN_ROWS_FOR_PREDICTION})")
                                continue
                            
                            # Only use data AFTER warmup period
                            df = df.iloc[WARMUP_BARS:]
                            logger.debug(f"      After warmup skip: {len(df)} rows")
                            
                            # Get last closed candle
                            row = df.iloc[[-2]]
                            
                            # ============================================================
                            # 🔍 FULL DEBUG LOGGING - все данные для диагностики
                            # ============================================================
                            logger.debug(f"=" * 60)
                            logger.debug(f"FULL DEBUG for {pair}")
                            
                            # 1. Данные о свечах
                            logger.debug(f"[CANDLES] 5m last 3 candles:")
                            last_3 = data['5m'].tail(3)
                            for idx, c in last_3.iterrows():
                                logger.debug(f"   {idx} | O:{c['open']:.6f} H:{c['high']:.6f} L:{c['low']:.6f} C:{c['close']:.6f} V:{c['volume']:.2f}")
                            
                            # 2. Какая свеча анализируется
                            logger.debug(f"[ANALYSIS] Analyzing candle: {row.index[0]}")
                            logger.debug(f"[ANALYSIS] Close={row['close'].iloc[0]:.6f}, ATR={row['atr'].iloc[0]:.6f}")
                            
                            # 3. Все фичи для этой свечи (в файл)
                            all_features_log = {}
                            for feat in models['features']:
                                if feat in row.columns:
                                    val = row[feat].iloc[0]
                                    all_features_log[feat] = float(val) if pd.notna(val) else "NaN"
                                else:
                                    all_features_log[feat] = "MISSING"
                            logger.debug(f"[FEATURES] All {len(models['features'])} features: {json.dumps(all_features_log, indent=2)}")
                            
                            # 4. Статистика фичей
                            feature_values = [v for v in all_features_log.values() if isinstance(v, (int, float))]
                            logger.debug(f"[FEATURES] Stats: min={min(feature_values):.4f}, max={max(feature_values):.4f}, mean={np.mean(feature_values):.4f}")
                            missing_count = sum(1 for v in all_features_log.values() if v == "MISSING")
                            nan_count = sum(1 for v in all_features_log.values() if v == "NaN")
                            logger.debug(f"[FEATURES] Missing: {missing_count}, NaN: {nan_count}")
                            logger.debug(f"=" * 60)

                            # Log candle timestamp and close to confirm we're on fresh data
                            last_candle_time = row.index[0]
                            candle_close = row['close'].iloc[0]
                            
                            # 🔍 DEBUG: Check candle age - should be < 10 minutes for 5m candles
                            now_utc = datetime.now(timezone.utc)
                            candle_age_sec = (now_utc - last_candle_time.to_pydatetime()).total_seconds()
                            candle_age_min = candle_age_sec / 60
                            
                            if candle_age_min > 10:
                                logger.warning(f"      ⚠️ STALE DATA! Candle age: {candle_age_min:.1f} min (should be < 10)")
                            
                            logger.info(
                                f"      Candle @ {last_candle_time} | Close: {candle_close:.6f} | Age: {candle_age_min:.1f} min"
                            )
                            
                            # Validate features and fill missing with 0
                            missing_features = [f for f in models['features'] if f not in row.columns]
                            if missing_features:
                                # ✅ Check if missing features are absolute price features (model needs retraining)
                                # Use centralized constant for checking
                                absolute_missing = [f for f in missing_features if f in ABSOLUTE_PRICE_FEATURES]
                                if absolute_missing:
                                    logger.warning(f"⚠️ Model expects ABSOLUTE price features: {absolute_missing}")
                                    logger.warning("⚠️ These features cause live/backtest discrepancy!")
                                    logger.warning("⚠️ Please RETRAIN model with: python scripts/train_v3_dynamic.py --days 60 --test_days 14")
                                    
                                logger.debug(f"Missing features for {pair}: {missing_features}")
                                # Fill missing features with 0 (these are typically volume-related)
                                for mf in missing_features:
                                    row[mf] = 0.0
                            
                            # ✅ FIXED: Use ALL features from model (cumsum already excluded during training)
                            # No additional filtering needed - models['features'] is already clean
                            features_to_use = models['features']
                            
                            # Extract features in EXACT order as model expects
                            X = row[features_to_use].values
                            
                            # ✅ FIX: Force convert to float64 to avoid object dtype issues
                            # This fixes "ufunc 'isinf' not supported" error
                            try:
                                X = X.astype(np.float64)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Could not convert features to float64: {e}")
                                # Try element-wise conversion
                                X = np.array([[float(v) if v is not None else 0.0 for v in X[0]]])
                            
                            # ✅ CRITICAL: Check for NaN/Inf and handle them
                            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Predictions (V8 IMPROVED)
                            dir_proba = models['direction'].predict_proba(X)
                            dir_conf = float(np.max(dir_proba))
                            dir_pred = int(np.argmax(dir_proba))
                            
                            # ✅ FIXED: Timing is now REGRESSOR (predicts ATR gain)
                            timing_pred = float(models['timing'].predict(X)[0])  # Returns 0-5 ATR gain
                            strength_pred = float(models['strength'].predict(X)[0])
                            
                            # Log prediction details with full probability distribution
                            p_down, p_sideways, p_up = dir_proba[0]
                            direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
                            logger.info(f"      → {direction_str} | Conf: {dir_conf:.2f} | Timing: {timing_pred:.2f} ATR | Strength: {strength_pred:.1f}")
                            
                            # 🔍 NEW: Log key indicators for manual verification
                            try:
                                # ✅ FIX: Calculate from REAL M5 data, not from normalized features
                                df_5m = data['5m']
                                close = df_5m['close'].iloc[-1]
                                
                                # Real RSI
                                delta = df_5m['close'].diff()
                                gain = delta.where(delta > 0, 0).rolling(14).mean()
                                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                                rs = gain / loss
                                rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]
                                
                                # Real EMAs
                                ema_21 = df_5m['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                                ema_50 = df_5m['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                                
                                # Vol ratio from features (already calculated)
                                vol_ratio = row.get('vol_ratio', pd.Series([0])).iloc[0]
                                
                                trend_txt = "BULL" if ema_21 > ema_50 else "BEAR"
                                logger.info(f"      📊 Indicators: RSI={rsi_14:.1f} | Trend={trend_txt} | VolRatio={vol_ratio:.1f}x")
                            except Exception as e:
                                logger.debug(f"Could not log indicators: {e}")

                            logger.debug(f"      Probabilities: DOWN={p_down:.3f}, SIDEWAYS={p_sideways:.3f}, UP={p_up:.3f}")
                            
                            # ============================================================
                            # 🔍 FULL PREDICTION DEBUG - ВСЕ детали предсказания
                            # ============================================================
                            logger.debug(f"[PREDICTION] Raw proba array: {dir_proba[0].tolist()}")
                            logger.debug(f"[PREDICTION] dir_pred={dir_pred} (0=SHORT, 1=SIDEWAYS, 2=LONG)")
                            logger.debug(f"[PREDICTION] Thresholds: MIN_CONF={MIN_CONF}, MIN_TIMING={MIN_TIMING}, MIN_STRENGTH={MIN_STRENGTH}")
                            
                            # Проверка фильтров
                            passes_conf = dir_conf >= MIN_CONF
                            passes_timing = timing_pred >= MIN_TIMING
                            passes_strength = strength_pred >= MIN_STRENGTH
                            passes_direction = dir_pred != 1
                            
                            logger.debug(f"[FILTERS] Direction != SIDEWAYS: {passes_direction}")
                            logger.debug(f"[FILTERS] Confidence >= {MIN_CONF}: {passes_conf} (actual: {dir_conf:.4f})")
                            logger.debug(f"[FILTERS] Timing >= {MIN_TIMING}: {passes_timing} (actual: {timing_pred:.4f})")
                            logger.debug(f"[FILTERS] Strength >= {MIN_STRENGTH}: {passes_strength} (actual: {strength_pred:.4f})")
                            logger.debug(f"[FILTERS] ALL PASS: {passes_conf and passes_timing and passes_strength and passes_direction}")
                            
                            # Log feature stats for ALL predictions (not just low confidence)
                            feature_stats = {
                                'mean': float(X.mean()),
                                'std': float(X.std()),
                                'min': float(X.min()),
                                'max': float(X.max()),
                                'nan_count': int(pd.isna(X).sum()),
                                'inf_count': int(np.isinf(X).sum() if hasattr(np, 'isinf') else 0)
                            }
                            logger.debug(f"[X_STATS] Feature matrix stats: {feature_stats}")
                            
                            # Log top 10 features with highest absolute values
                            feature_vals = {features_to_use[i]: float(X[0][i]) 
                                           for i in range(len(features_to_use))}
                            top_features = sorted(feature_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                            logger.debug(f"[X_TOP10] Top 10 features by abs value: {top_features}")
                            
                            # Важные индикаторы отдельно (используем фичи из МОДЕЛИ, не из row!)
                            key_features = ['m15_rsi', 'm5_rsi_14', 'm1_rsi_9_last', 'm5_atr_14', 'vol_ratio', 'price_vs_vwap']
                            key_vals = {k: feature_vals.get(k, 'N/A') for k in key_features}
                            logger.debug(f"[KEY_FEATURES] {key_vals}")
                            
                            # 🔍 Сохраняем предсказание для анализа
                            scan_stats['pairs_scanned'] += 1
                            pred_record = {
                                'pair': pair,
                                'candle_time': str(last_candle_time),
                                'close': float(candle_close),
                                'direction': direction_str,
                                'proba_down': float(p_down),
                                'proba_sideways': float(p_sideways),
                                'proba_up': float(p_up),
                                'confidence': float(dir_conf),
                                'timing': float(timing_pred),
                                'strength': float(strength_pred),
                                'passes_all': passes_conf and passes_timing and passes_strength and passes_direction,
                                'top5_features': {k: v for k, v in top_features[:5]}
                            }
                            scan_stats['predictions'].append(pred_record)
                            
                            # Apply V8 filters
                            if dir_pred == 1:  # Sideways
                                scan_stats['sideways_count'] += 1
                                current_volatility = row['atr'].iloc[0] / row['close'].iloc[0] * 100
                                logger.debug(f"      ✗ Skipped (SIDEWAYS) | Volatility: {current_volatility:.3f}% (Need > 0.5% move)")
                                continue
                            
                            rejected_reasons = []
                            if dir_conf < MIN_CONF:
                                rejected_reasons.append(f"Conf({dir_conf:.2f}<{MIN_CONF})")
                                scan_stats['rejected_conf'] += 1
                            if timing_pred < MIN_TIMING:  # ✅ FIXED: Compare timing_pred (not timing_prob)
                                rejected_reasons.append(f"Timing({timing_pred:.2f}<{MIN_TIMING})")
                                scan_stats['rejected_timing'] += 1
                            if strength_pred < MIN_STRENGTH:
                                rejected_reasons.append(f"Strength({strength_pred:.1f}<{MIN_STRENGTH})")
                                scan_stats['rejected_strength'] += 1
                            
                            if rejected_reasons:
                                logger.info(f"      ✗ Rejected: {', '.join(rejected_reasons)}")
                                continue
                            
                            # SIGNAL FOUND!
                            signals_found += 1
                            scan_stats['signals_found'] += 1
                            logger.info(f"      ✅ SIGNAL FOUND!")
                            
                            # Get current price
                            ticker = binance.fetch_ticker(pair)
                            current_price = ticker['last']
                            
                            # Create signal (MATCH BACKTEST KEYS)
                            signal = {
                                'pair': pair,
                                'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                                'price': current_price,
                                'atr': row['atr'].iloc[0],
                                'conf': dir_conf,
                                'timing_prob': timing_pred,  # ✅ FIXED: Store timing_pred (ATR gain)
                                'pred_strength': strength_pred
                            }
                            
                            portfolio.open_position(signal)
                            logger.info(f"🚀 Signal taken: {pair} {signal['direction']} @ {current_price:.6f}")
                            break  # Only one position
                            
                        except Exception as e:
                            logger.error(f"      ✗ Error processing {pair}: {e}")
                            continue
                    
                    # Summary after scan
                    logger.info("=" * 70)
                    logger.info(f"📊 Scan complete: {signals_checked} pairs checked, {signals_found} signals found")
                    
                    # 🆕 CONFIDENCE TABLE - показываем confidence для КАЖДОЙ пары
                    if scan_stats['predictions']:
                        logger.info("")
                        logger.info("📋 CONFIDENCE TABLE (all pairs):")
                        logger.info("-" * 70)
                        logger.info(f"{'Pair':<20} | {'Direction':<10} | {'Conf':>6} | {'Timing':>6} | {'Strength':>8} | Status")
                        logger.info("-" * 70)
                        
                        # Сортируем по confidence (от большего к меньшему)
                        sorted_preds = sorted(scan_stats['predictions'], 
                                             key=lambda x: x.get('confidence', 0), 
                                             reverse=True)
                        
                        for pred in sorted_preds:
                            pair_short = pred['pair'].replace('/USDT:USDT', '').replace('USDT:', '')
                            direction = pred.get('direction', 'N/A')
                            conf = pred.get('confidence', 0)
                            timing = pred.get('timing', 0)
                            strength = pred.get('strength', 0)
                            passes = pred.get('passes_all', False)
                            
                            # Определяем статус
                            if direction == 'SIDEWAYS':
                                status = "⏸️ SIDEWAYS"
                            elif passes:
                                status = "✅ SIGNAL"
                            elif conf < MIN_CONF:
                                status = f"❌ Conf<{MIN_CONF}"
                            elif timing < MIN_TIMING:
                                status = f"❌ Tim<{MIN_TIMING}"
                            elif strength < MIN_STRENGTH:
                                status = f"❌ Str<{MIN_STRENGTH}"
                            else:
                                status = "❌ Other"
                            
                            logger.info(f"{pair_short:<20} | {direction:<10} | {conf:>6.2f} | {timing:>6.2f} | {strength:>8.1f} | {status}")
                        
                        logger.info("-" * 70)
                    
                    # 🔍 Show rejection breakdown on INFO level for diagnostics
                    if scan_stats['pairs_scanned'] > 0 and signals_found == 0:
                        logger.info(f"   📈 SIDEWAYS: {scan_stats['sideways_count']}")
                        logger.info(f"   📈 Rejected by Confidence (<{MIN_CONF}): {scan_stats['rejected_conf']}")
                        logger.info(f"   📈 Rejected by Timing (<{MIN_TIMING}): {scan_stats['rejected_timing']}")
                        logger.info(f"   📈 Rejected by Strength (<{MIN_STRENGTH}): {scan_stats['rejected_strength']}")
                        
                        # Show best candidate for debugging
                        if scan_stats['predictions']:
                            try:
                                best = max(scan_stats['predictions'], key=lambda x: max(x.get('proba_up', 0), x.get('proba_down', 0)))
                                logger.info(f"   🏆 Best candidate: {best['pair']} {best['direction']} | Conf={best.get('confidence', 0):.2f} Tim={best.get('timing', 0):.2f} Str={best.get('strength', 0):.1f}")
                            except Exception as e:
                                logger.debug(f"Could not get best candidate: {e}")
                    
                    logger.info(f"⏰ Next scan at next 5-minute candle close")
                    logger.info("=" * 70)
                    
                    # 🔍 FULL SCAN SUMMARY для анализа
                    logger.debug("=" * 70)
                    logger.debug("[SCAN_SUMMARY] Full statistics:")
                    logger.debug(f"   Pairs scanned: {scan_stats['pairs_scanned']}")
                    logger.debug(f"   SIDEWAYS: {scan_stats['sideways_count']}")
                    logger.debug(f"   Rejected by Conf: {scan_stats['rejected_conf']}")
                    logger.debug(f"   Rejected by Timing: {scan_stats['rejected_timing']}")
                    logger.debug(f"   Rejected by Strength: {scan_stats['rejected_strength']}")
                    logger.debug(f"   Signals found: {scan_stats['signals_found']}")
                    
                    # Статистика по вероятностям
                    if scan_stats['predictions']:
                        up_probs = [p['proba_up'] for p in scan_stats['predictions']]
                        down_probs = [p['proba_down'] for p in scan_stats['predictions']]
                        sideways_probs = [p['proba_sideways'] for p in scan_stats['predictions']]
                        
                        logger.debug(f"   Proba UP: min={min(up_probs):.3f}, max={max(up_probs):.3f}, mean={np.mean(up_probs):.3f}")
                        logger.debug(f"   Proba DOWN: min={min(down_probs):.3f}, max={max(down_probs):.3f}, mean={np.mean(down_probs):.3f}")
                        logger.debug(f"   Proba SIDEWAYS: min={min(sideways_probs):.3f}, max={max(sideways_probs):.3f}, mean={np.mean(sideways_probs):.3f}")
                        
                        # Лучшие кандидаты (наибольшая вероятность UP или DOWN)
                        best_candidates = sorted(scan_stats['predictions'], 
                                                  key=lambda x: max(x['proba_up'], x['proba_down']), 
                                                  reverse=True)[:5]
                        logger.debug("[BEST_CANDIDATES] Top 5 by UP/DOWN probability:")
                        for c in best_candidates:
                            logger.debug(f"   {c['pair']}: {c['direction']} | UP={c['proba_up']:.3f} DOWN={c['proba_down']:.3f} SW={c['proba_sideways']:.3f} | T={c['timing']:.2f} S={c['strength']:.2f}")
                    
                    # Сохраняем в JSON для анализа
                    scan_log_file = Path("logs/scan_stats.json")
                    try:
                        # Читаем существующие данные
                        if scan_log_file.exists() and scan_log_file.stat().st_size > 0:
                            with open(scan_log_file, 'r') as f:
                                try:
                                    all_scans = json.load(f)
                                except json.JSONDecodeError:
                                    all_scans = []
                        else:
                            all_scans = []
                        
                        # Добавляем новый скан (храним последние 100)
                        all_scans.append(scan_stats)
                        all_scans = all_scans[-100:]
                        
                        with open(scan_log_file, 'w') as f:
                            json.dump(all_scans, f, indent=2, default=str)
                        logger.debug(f"[SAVED] Scan stats saved to {scan_log_file}")
                    except Exception as e:
                        logger.warning(f"Could not save scan stats: {e}")
                    
                    logger.debug("=" * 70)
        
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            # On error, wait 10 seconds before retrying
            time.sleep(10)


if __name__ == '__main__':
    main()


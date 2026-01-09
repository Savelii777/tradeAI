#!/usr/bin/env python3
"""
Live Trading Script V9 - MEXC Live Trading (Data from Binance)

Key Features:
- Uses v8_improved model (trained with V8 anti-overfitting parameters)
- Fetches market data from Binance (free, no auth required)
- Executes trades on MEXC Futures via direct API
- EXACT backtest logic (breakeven stop, trailing, adaptive SL)
- Loads credentials from config/secrets.yaml (no hardcoded secrets)

Usage:
    1. Copy config/secrets.yaml.example to config/secrets.yaml
    2. Fill in your MEXC API keys and Telegram credentials
    3. Run: python scripts/live_trading_v9.py

Based on train_v3_dynamic.py backtest logic.
"""

import sys
import time
import json
import hmac
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import joblib
import ccxt
import requests
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES
)
from train_mtf import MTFFeatureEngine


# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "live_trading_v9.log"

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
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """Configuration loaded from files and constants."""
    
    # Paths
    MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
    PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_20.json"
    SECRETS_FILE = Path(__file__).parent.parent / "config" / "secrets.yaml"
    TRADES_FILE = Path(__file__).parent.parent / "active_trades_v9.json"
    
    # Data settings
    TIMEFRAMES = ['1m', '5m', '15m']
    LOOKBACK = 3000  # M1 bars to fetch for indicators to stabilize
    WARMUP_BARS = 50  # Skip first N bars (indicator warmup)
    MIN_ROWS_FOR_PREDICTION = 2
    
    # V8 IMPROVED Signal Thresholds (match backtest)
    MIN_CONF = 0.50       # Direction confidence
    MIN_TIMING = 0.8      # Timing = ATR gain potential (regression output)
    MIN_STRENGTH = 1.4    # Strength prediction
    
    # Risk Management (EXACT backtest settings from train_v3_dynamic.py)
    RISK_PCT = 0.05           # 5% risk per trade
    MAX_LEVERAGE = 50.0       # Max leverage
    MAX_HOLDING_BARS = 150    # 12.5 hours on 5m
    ENTRY_FEE = 0.0002        # 0.02% maker fee
    EXIT_FEE = 0.0002         # 0.02% taker fee  
    SL_ATR_BASE = 1.5         # Base SL multiplier
    MAX_POSITION_SIZE = 200000.0  # Max $200k position
    SLIPPAGE_PCT = 0.0005     # 0.05% realistic slippage (matches backtest)
    
    # V8 Features (from backtest)
    USE_ADAPTIVE_SL = True
    USE_DYNAMIC_LEVERAGE = True
    USE_AGGRESSIVE_TRAIL = True
    
    # MEXC API
    MEXC_BASE_URL = "https://contract.mexc.com"
    
    @classmethod
    def load_secrets(cls) -> Dict[str, Any]:
        """Load secrets from config/secrets.yaml."""
        if not cls.SECRETS_FILE.exists():
            raise FileNotFoundError(
                f"Secrets file not found: {cls.SECRETS_FILE}\n"
                "Please copy config/secrets.yaml.example to config/secrets.yaml "
                "and fill in your credentials."
            )
        
        with open(cls.SECRETS_FILE, 'r') as f:
            secrets = yaml.safe_load(f)
        
        # Validate required keys
        required_keys = [
            ('mexc', 'api_key'),
            ('mexc', 'api_secret'),
            ('notifications', 'telegram', 'bot_token'),
            ('notifications', 'telegram', 'chat_id'),
        ]
        
        for key_path in required_keys:
            current = secrets
            for key in key_path:
                if key not in current:
                    raise ValueError(
                        f"Missing required key in secrets.yaml: {'.'.join(key_path)}"
                    )
                current = current[key]
            
            # Check for placeholder values
            if isinstance(current, str) and 'your_' in current.lower():
                raise ValueError(
                    f"Please replace placeholder value for: {'.'.join(key_path)}"
                )
        
        return secrets


# ============================================================
# MEXC DIRECT API CLIENT
# ============================================================
class MEXCClient:
    """Direct MEXC Futures API client."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = Config.MEXC_BASE_URL
        logger.info("✅ MEXC Direct API Client initialized")
    
    def _generate_signature(self, sign_str: str) -> str:
        """Generate HMAC SHA256 signature for MEXC."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to MEXC."""
        timestamp = int(time.time() * 1000)
        
        if params is None:
            params = {}
        
        params['timestamp'] = timestamp
        
        # Sort parameters and build query string
        sorted_params = sorted(params.items())
        params_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Generate signature
        sign_str = f"{self.api_key}{timestamp}{params_str}"
        signature = self._generate_signature(sign_str)
        
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Signature': signature,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30, verify=True)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers, timeout=30, verify=True)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"MEXC API request failed: {e}")
            return None
    
    def get_account_assets(self) -> float:
        """Get USDT balance."""
        result = self._request('GET', '/api/v1/private/account/assets', {})
        if result and result.get('success'):
            for asset in result.get('data', []):
                if asset['currency'] == 'USDT':
                    return float(asset.get('availableBalance', 0))
        return 0.0
    
    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        """Get open positions."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        result = self._request('GET', '/api/v1/private/position/open_positions', params)
        if result and result.get('success'):
            return result.get('data', [])
        return []
    
    def place_order(self, symbol: str, side: int, volume: int, leverage: int,
                   price: float = 0, order_type: int = 5, open_type: int = 1) -> Optional[str]:
        """
        Place order on MEXC Futures.
        
        Args:
            symbol: Symbol (e.g., 'BTC_USDT')
            side: 1=open long, 2=open short, 3=close long, 4=close short
            volume: Volume in contracts
            leverage: Leverage (1-125)
            price: Limit price (0 for market)
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
            order_id = result.get('data')
            logger.info(f"✅ Order placed! ID: {order_id}")
            return order_id
        else:
            logger.error(f"❌ Order failed: {result}")
            return None
    
    def place_stop_order(self, symbol: str, side: int, volume: int, 
                        stop_price: float, leverage: int) -> Optional[str]:
        """Place stop loss order on MEXC."""
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
        
        logger.info(f"📤 Placing STOP order @ {stop_price}")
        result = self._request('POST', '/api/v1/private/planorder/place', params)
        
        if result and result.get('success'):
            order_id = result.get('data')
            logger.info(f"✅ Stop order placed! ID: {order_id}")
            return order_id
        else:
            logger.error(f"❌ Stop order failed: {result}")
            return None
    
    def cancel_stop_orders(self, symbol: str) -> bool:
        """Cancel all stop orders for a symbol."""
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
# TELEGRAM NOTIFIER
# ============================================================
class TelegramNotifier:
    """Send alerts via Telegram."""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        
        if self.enabled:
            logger.info("✅ Telegram notifications enabled")
        else:
            logger.warning("⚠️ Telegram notifications disabled (missing credentials)")
    
    def send(self, title: str, body: str) -> bool:
        """Send a Telegram message."""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': f"{title}\n{body}",
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=5, verify=True)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.debug(f"Telegram send failed: {e}")
            return False


# ============================================================
# PORTFOLIO MANAGER (V8 BACKTEST LOGIC)
# ============================================================
class PortfolioManager:
    """Manages trading positions with V8 backtest-matched logic."""
    
    def __init__(self, mexc_client: MEXCClient, telegram: TelegramNotifier, 
                 initial_capital: Optional[float] = None):
        self.mexc = mexc_client
        self.telegram = telegram
        self.position = None
        self.trades_history = []
        
        # Load state or get from exchange
        if self.load_state():
            logger.info(f"📂 State loaded. Capital: ${self.capital:.2f}")
        else:
            # Get real balance from MEXC
            self.capital = initial_capital or self.mexc.get_account_assets()
            if self.capital == 0:
                logger.warning("⚠️ Could not fetch MEXC balance, using $20 default")
                self.capital = 20.0
            else:
                logger.info(f"💰 Balance from MEXC: ${self.capital:.2f}")
            self.save_state()
    
    def load_state(self) -> bool:
        """Load state from file."""
        if Config.TRADES_FILE.exists():
            try:
                with open(Config.TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    self.capital = data.get('capital')
                    self.position = data.get('position')
                    self.trades_history = data.get('history', [])
                    
                    if self.position and 'entry_time' in self.position:
                        if isinstance(self.position['entry_time'], str):
                            self.position['entry_time'] = datetime.fromisoformat(
                                self.position['entry_time']
                            )
                    
                    return self.capital is not None
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading state: {e}")
                return False
        return False
    
    def save_state(self):
        """Save state to file."""
        pos_copy = None
        if self.position:
            pos_copy = self.position.copy()
            if isinstance(pos_copy.get('entry_time'), datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
        
        data = {
            'capital': self.capital,
            'position': pos_copy,
            'history': self.trades_history
        }
        with open(Config.TRADES_FILE, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    
    def open_position(self, signal: Dict) -> bool:
        """Open position on MEXC (V8 backtest logic)."""
        if self.position is not None:
            logger.warning("Already in a position")
            return False
        
        entry_price = signal['price']
        atr = signal['atr']
        pred_strength = signal.get('pred_strength', 2.0)
        conf = signal.get('conf', 0.5)
        timing = signal.get('timing_prob', 0.5)
        
        # === V8: ADAPTIVE STOP LOSS ===
        if Config.USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:
                sl_mult = 1.6
            elif pred_strength >= 2.0:
                sl_mult = 1.5
            else:
                sl_mult = 1.2
        else:
            sl_mult = Config.SL_ATR_BASE
        
        stop_distance = atr * sl_mult
        
        if signal['direction'] == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # === V8: DYNAMIC BREAKEVEN TRIGGER ===
        if pred_strength >= 3.0:
            be_trigger_mult = 1.8
        elif pred_strength >= 2.0:
            be_trigger_mult = 1.5
        else:
            be_trigger_mult = 1.2
        
        # === V8: DYNAMIC RISK (MATCHES BACKTEST) ===
        if Config.USE_DYNAMIC_LEVERAGE:
            score = conf * timing
            quality = (score / 0.5) * (timing / 0.6) * (pred_strength / 2.0)
            quality_mult = np.clip(quality, 0.8, 1.5)
            risk_pct = Config.RISK_PCT * quality_mult
        else:
            risk_pct = Config.RISK_PCT
        
        # Position sizing (MATCHES BACKTEST)
        stop_loss_pct = stop_distance / entry_price
        risk_amount = self.capital * risk_pct
        position_value = risk_amount / stop_loss_pct
        
        # Calculate leverage
        leverage = position_value / self.capital
        if leverage > Config.MAX_LEVERAGE:
            leverage = Config.MAX_LEVERAGE
            position_value = self.capital * leverage
        
        # Cap position size
        if position_value > Config.MAX_POSITION_SIZE:
            position_value = Config.MAX_POSITION_SIZE
            leverage = position_value / self.capital
        
        # Deduct entry fee
        entry_fee = position_value * Config.ENTRY_FEE
        self.capital -= entry_fee
        
        # Convert symbol: BTC/USDT:USDT -> BTC_USDT
        mexc_symbol = signal['pair'].replace('/USDT:USDT', '_USDT').replace('/', '_')
        
        # Calculate volume in contracts
        volume = int(position_value / entry_price)
        if volume < 1:
            volume = 1
        
        # MEXC API: side 1=open long, 2=open short
        mexc_side = 1 if signal['direction'] == 'LONG' else 2
        
        logger.info(f"📊 Position: ${position_value:.0f} | {volume} contracts | Lev: {leverage:.1f}x")
        
        order_id = self.mexc.place_order(
            symbol=mexc_symbol,
            side=mexc_side,
            volume=volume,
            leverage=int(leverage)
        )
        
        if order_id is None:
            logger.error(f"Failed to open position for {signal['pair']}")
            self.capital += entry_fee  # Refund fee
            return False
        
        # Place stop loss order
        stop_side = 3 if signal['direction'] == 'LONG' else 4
        stop_order_id = self.mexc.place_stop_order(
            symbol=mexc_symbol,
            side=stop_side,
            volume=volume,
            stop_price=stop_loss,
            leverage=int(leverage)
        )
        
        if stop_order_id is None:
            logger.warning("⚠️ Failed to place stop order! Position unprotected!")
        
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
            'bars_held': 0,
            'mexc_order_id': order_id,
            'mexc_stop_order_id': stop_order_id,
            'mexc_symbol': mexc_symbol,
            'volume': volume
        }
        self.save_state()
        
        self.telegram.send(
            f"🟢 <b>{signal['direction']}</b> {signal['pair']}",
            f"Entry: {entry_price:.6f}\n"
            f"SL: {stop_loss:.6f} ({sl_mult}ATR)\n"
            f"Lev: {leverage:.1f}x\n"
            f"Size: ${position_value:.0f}\n"
            f"Str: {pred_strength:.1f}"
        )
        
        return True
    
    def update_position(self, current_price: float, candle_high: float, candle_low: float):
        """Update position with V8 trailing stop logic."""
        if self.position is None:
            return
        
        pos = self.position
        
        # Time limit check
        duration = datetime.now(timezone.utc) - pos['entry_time']
        if duration > timedelta(minutes=Config.MAX_HOLDING_BARS * 5):
            self.close_position(current_price, "Time Limit")
            return
        
        # Restore ATR and multipliers
        pred_strength = pos.get('pred_strength', 2.0)
        if Config.USE_ADAPTIVE_SL:
            if pred_strength >= 3.0:
                sl_mult = 1.6
            elif pred_strength >= 2.0:
                sl_mult = 1.5
            else:
                sl_mult = 1.2
        else:
            sl_mult = Config.SL_ATR_BASE
        
        atr = pos['stop_distance'] / sl_mult
        be_trigger_dist = atr * pos['be_trigger_mult']
        
        old_sl = pos['stop_loss']
        updated = False
        
        if pos['direction'] == 'LONG':
            be_trigger_price = pos['entry_price'] + be_trigger_dist
            
            # Check breakeven trigger
            if not pos['breakeven_active'] and candle_high >= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] + (atr * 0.3)
                updated = True
                logger.info(f"✅ Breakeven activated for {pos['pair']}")
            
            # Trailing stop
            if pos['breakeven_active']:
                current_profit = candle_high - pos['entry_price']
                r_multiple = current_profit / pos['stop_distance']
                
                if Config.USE_AGGRESSIVE_TRAIL:
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
                    if r_multiple > 5.0:
                        trail_mult = 0.5
                    elif r_multiple > 3.0:
                        trail_mult = 1.5
                
                new_sl = candle_high - (atr * trail_mult)
                if new_sl > pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    updated = True
        
        else:  # SHORT
            be_trigger_price = pos['entry_price'] - be_trigger_dist
            
            if not pos['breakeven_active'] and candle_low <= be_trigger_price:
                pos['breakeven_active'] = True
                pos['stop_loss'] = pos['entry_price'] - (atr * 0.3)
                updated = True
                logger.info(f"✅ Breakeven activated for {pos['pair']}")
            
            if pos['breakeven_active']:
                current_profit = pos['entry_price'] - candle_low
                r_multiple = current_profit / pos['stop_distance']
                
                if Config.USE_AGGRESSIVE_TRAIL:
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
                    if r_multiple > 5.0:
                        trail_mult = 0.5
                    elif r_multiple > 3.0:
                        trail_mult = 1.5
                
                new_sl = candle_low + (atr * trail_mult)
                if new_sl < pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    updated = True
        
        # Update stop loss on MEXC if changed
        if updated and old_sl != pos['stop_loss']:
            mexc_symbol = pos['mexc_symbol']
            volume = pos['volume']
            stop_side = 3 if pos['direction'] == 'LONG' else 4
            
            # Cancel old stop order
            self.mexc.cancel_stop_orders(mexc_symbol)
            
            # Place new stop order
            logger.info(f"📈 Updating SL: {old_sl:.6f} → {pos['stop_loss']:.6f}")
            new_stop_order_id = self.mexc.place_stop_order(
                symbol=mexc_symbol,
                side=stop_side,
                volume=volume,
                stop_price=pos['stop_loss'],
                leverage=int(pos['leverage'])
            )
            
            if new_stop_order_id:
                pos['mexc_stop_order_id'] = new_stop_order_id
            else:
                logger.warning("⚠️ Failed to update stop order!")
        
        self.save_state()
    
    def close_position(self, price: float, reason: str):
        """Close position on MEXC."""
        pos = self.position
        
        if price is None or price <= 0:
            price = pos['entry_price']
        
        mexc_symbol = pos['mexc_symbol']
        
        # Cancel stop orders first
        self.mexc.cancel_stop_orders(mexc_symbol)
        
        # Close position
        volume = pos['volume']
        close_side = 3 if pos['direction'] == 'LONG' else 4
        
        logger.info(f"📤 Closing: {mexc_symbol} | {volume} contracts")
        
        close_order = self.mexc.place_order(
            symbol=mexc_symbol,
            side=close_side,
            volume=volume,
            leverage=int(pos['leverage'])
        )
        
        if close_order is None:
            logger.error("❌ Failed to close position on MEXC!")
        
        # Calculate PnL (V8 logic with slippage)
        if pos['direction'] == 'LONG':
            effective_entry = pos['entry_price'] * (1 + Config.SLIPPAGE_PCT)
            effective_exit = price * (1 - Config.SLIPPAGE_PCT)
            pnl_pct = (effective_exit - effective_entry) / effective_entry
        else:
            effective_entry = pos['entry_price'] * (1 - Config.SLIPPAGE_PCT)
            effective_exit = price * (1 + Config.SLIPPAGE_PCT)
            pnl_pct = (effective_entry - effective_exit) / effective_entry
        
        gross = pos['position_value'] * pnl_pct
        fees = pos['position_value'] * Config.EXIT_FEE
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
        self.telegram.send(
            f"{emoji} <b>CLOSE {pos['direction']}</b> {pos['pair']}",
            f"Reason: {reason}\n"
            f"Price: {price:.4f}\n"
            f"PnL: ${net:.2f} ({roe:.1f}%)\n"
            f"Capital: ${self.capital:.2f}"
        )
        
        self.position = None
        self.save_state()


# ============================================================
# FEATURE ENGINEERING (MATCHES BACKTEST)
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features (matches train_v3_dynamic.py)."""
    df = df.copy()
    
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # VWAP (rolling, not cumsum - for live/backtest consistency)
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # Volume momentum (clipped to prevent extreme values)
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features(data: Dict[str, pd.DataFrame], mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Prepare features from multi-timeframe data (matches backtest)."""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 200:
        logger.warning(f"Insufficient data: M1={len(m1)}, M5={len(m5)}, M15={len(m15)}")
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        # Align timeframes
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        
        if len(ft) == 0:
            logger.warning("align_timeframes returned empty DataFrame")
            return pd.DataFrame()
        
        # Add OHLCV from M5
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        
        # Add volume features
        ft = add_volume_features(ft)
        
        # Add ATR
        ft['atr'] = calculate_atr(ft)
        
        # Drop rows with NaN in critical columns (match backtest)
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        
        # Exclude cumsum-dependent features
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        if cols_to_drop:
            ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        # Exclude absolute price features
        absolute_cols_to_drop = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        if absolute_cols_to_drop:
            ft = ft.drop(columns=absolute_cols_to_drop, errors='ignore')
        
        # Forward-fill non-critical columns
        non_critical = [c for c in ft.columns if c not in critical_cols]
        if non_critical:
            ft[non_critical] = ft[non_critical].ffill()
        
        # Final dropna
        ft = ft.dropna()
        
        return ft
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()


# ============================================================
# MODEL LOADING
# ============================================================
def load_models() -> Dict:
    """Load V8 models."""
    logger.info("Loading V8 models...")
    
    if not Config.MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {Config.MODEL_DIR}\n"
            "Please train models first with:\n"
            "python scripts/train_v3_dynamic.py --days 90 --test_days 30"
        )
    
    models = {
        'direction': joblib.load(Config.MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(Config.MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(Config.MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(Config.MODEL_DIR / 'feature_names.joblib')
    }
    
    logger.info(f"✅ Loaded models with {len(models['features'])} features")
    return models


def get_pairs() -> list:
    """Load trading pairs from config."""
    if not Config.PAIRS_FILE.exists():
        # Fallback to pairs_list.json
        fallback = Config.PAIRS_FILE.parent / "pairs_list.json"
        if fallback.exists():
            with open(fallback, 'r') as f:
                return [p['symbol'] for p in json.load(f)['pairs']][:20]
        raise FileNotFoundError(f"Pairs file not found: {Config.PAIRS_FILE}")
    
    with open(Config.PAIRS_FILE, 'r') as f:
        return [p['symbol'] for p in json.load(f)['pairs']][:20]


# ============================================================
# MAIN TRADING LOOP
# ============================================================
def wait_until_candle_close(timeframe_minutes: int = 5) -> datetime:
    """Wait until next candle close."""
    now = datetime.now(timezone.utc)
    
    current_minute = now.minute
    next_close_minute = ((current_minute // timeframe_minutes) + 1) * timeframe_minutes
    
    if next_close_minute >= 60:
        next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_close = now.replace(minute=next_close_minute, second=0, microsecond=0)
    
    wait_seconds = (next_close - now).total_seconds() + 3  # 3 second buffer
    
    if wait_seconds > 0:
        logger.info(f"⏰ Waiting {wait_seconds:.0f}s until candle close at {next_close.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_seconds)
    
    return next_close


def fetch_ohlcv_data(binance: ccxt.Exchange, pair: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data from Binance."""
    try:
        all_candles = []
        
        # First request
        candles = binance.fetch_ohlcv(pair, timeframe, limit=min(limit, 1000))
        if not candles:
            return pd.DataFrame()
        
        all_candles.extend(candles)
        
        # Fetch more if needed
        while len(all_candles) < limit:
            oldest_ts = all_candles[0][0]
            tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[timeframe]
            since_ts = oldest_ts - (1000 * tf_ms)
            
            candles = binance.fetch_ohlcv(pair, timeframe, since=since_ts, limit=1000)
            if not candles:
                break
            
            seen = {c[0] for c in all_candles}
            new_candles = [c for c in candles if c[0] not in seen]
            
            if not new_candles:
                break
            
            all_candles = new_candles + all_candles
            time.sleep(0.1)
        
        # Keep newest candles
        all_candles = sorted(all_candles, key=lambda x: x[0])[-limit:]
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {pair} {timeframe}: {e}")
        return pd.DataFrame()


def main():
    """Main trading loop."""
    logger.info("=" * 70)
    logger.info("V9 LIVE TRADING - MEXC (Data from Binance)")
    logger.info("=" * 70)
    
    # Load secrets
    try:
        secrets = Config.load_secrets()
        logger.info("✅ Secrets loaded from config/secrets.yaml")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"❌ {e}")
        return
    
    # Initialize clients
    mexc_client = MEXCClient(
        api_key=secrets['mexc']['api_key'],
        api_secret=secrets['mexc']['api_secret']
    )
    
    telegram = TelegramNotifier(
        bot_token=secrets['notifications']['telegram']['bot_token'],
        chat_id=secrets['notifications']['telegram']['chat_id']
    )
    
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Load models and pairs
    try:
        models = load_models()
        pairs = get_pairs()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    # Initialize portfolio
    portfolio = PortfolioManager(mexc_client, telegram)
    mtf_fe = MTFFeatureEngine()
    
    logger.info(f"Monitoring {len(pairs)} pairs")
    logger.info(f"Initial Capital: ${portfolio.capital:.2f}")
    logger.info(f"🎯 Scanning at 5-minute candle closes")
    logger.info("=" * 70)
    
    # Send startup notification
    telegram.send(
        "🚀 <b>V9 Live Trading Started</b>",
        f"Capital: ${portfolio.capital:.2f}\n"
        f"Pairs: {len(pairs)}\n"
        f"Thresholds: Conf>{Config.MIN_CONF}, Tim>{Config.MIN_TIMING}, Str>{Config.MIN_STRENGTH}"
    )
    
    # Main loop
    while True:
        try:
            # Wait for candle close
            wait_until_candle_close(timeframe_minutes=5)
            
            # Update position if exists
            if portfolio.position:
                pos = portfolio.position
                logger.info(f"📍 Updating position: {pos['pair']} {pos['direction']}")
                
                ticker = binance.fetch_ticker(pos['pair'])
                current_price = ticker['last']
                
                candles_5m = binance.fetch_ohlcv(pos['pair'], '5m', limit=2)
                if len(candles_5m) >= 2:
                    last_candle = candles_5m[-2]
                    candle_high = last_candle[2]
                    candle_low = last_candle[3]
                    portfolio.update_position(current_price, candle_high, candle_low)
            
            # Scan for new signals (only if no position)
            if portfolio.position is None:
                logger.info("=" * 70)
                logger.info("🔍 Scanning for signals...")
                
                for pair in pairs:
                    try:
                        # Fetch data from Binance
                        data = {}
                        valid = True
                        
                        for tf in Config.TIMEFRAMES:
                            df = fetch_ohlcv_data(binance, pair, tf, Config.LOOKBACK)
                            if df.empty or len(df) < 200:
                                valid = False
                                break
                            data[tf] = df
                        
                        if not valid:
                            continue
                        
                        # Prepare features
                        df = prepare_features(data, mtf_fe)
                        if df is None or len(df) < Config.WARMUP_BARS + Config.MIN_ROWS_FOR_PREDICTION:
                            continue
                        
                        # Skip warmup rows
                        df = df.iloc[Config.WARMUP_BARS:]
                        
                        # Get last closed candle
                        row = df.iloc[[-2]]
                        
                        # Validate features
                        missing_features = [f for f in models['features'] if f not in row.columns]
                        for mf in missing_features:
                            row[mf] = 0.0
                        
                        # Extract features
                        X = row[models['features']].values
                        X = X.astype(np.float64)
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Predictions
                        dir_proba = models['direction'].predict_proba(X)
                        dir_conf = float(np.max(dir_proba))
                        dir_pred = int(np.argmax(dir_proba))
                        
                        timing_pred = float(models['timing'].predict(X)[0])
                        strength_pred = float(models['strength'].predict(X)[0])
                        
                        direction_str = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
                        logger.info(f"   {pair}: {direction_str} | Conf: {dir_conf:.2f} | Tim: {timing_pred:.2f} | Str: {strength_pred:.1f}")
                        
                        # Apply filters
                        if dir_pred == 1:  # Sideways
                            continue
                        
                        if dir_conf < Config.MIN_CONF or timing_pred < Config.MIN_TIMING or strength_pred < Config.MIN_STRENGTH:
                            continue
                        
                        # SIGNAL FOUND!
                        logger.info(f"   ✅ SIGNAL FOUND!")
                        
                        ticker = binance.fetch_ticker(pair)
                        current_price = ticker['last']
                        
                        signal = {
                            'pair': pair,
                            'direction': 'LONG' if dir_pred == 2 else 'SHORT',
                            'price': current_price,
                            'atr': row['atr'].iloc[0],
                            'conf': dir_conf,
                            'timing_prob': timing_pred,
                            'pred_strength': strength_pred
                        }
                        
                        if portfolio.open_position(signal):
                            logger.info(f"🚀 Position opened: {pair} {signal['direction']} @ {current_price:.6f}")
                            break  # Only one position
                        
                    except Exception as e:
                        logger.debug(f"Error processing {pair}: {e}")
                        continue
                
                logger.info(f"⏰ Next scan at next 5-minute candle close")
                logger.info("=" * 70)
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            telegram.send("🛑 <b>V9 Live Trading Stopped</b>", f"Capital: ${portfolio.capital:.2f}")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    main()

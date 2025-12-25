#!/usr/bin/env python3
"""
Paper Trading with Telegram Notifications

Runs V1 model in real-time paper trading mode with Telegram alerts.
Supports both single model and multi-period ensemble.

Usage:
    # Single model:
    python scripts/paper_trading.py --model-path ./models/v1_fresh --capital 100
    
    # Ensemble (recommended):
    python scripts/paper_trading.py --ensemble --capital 100
    
    # Or in Docker:
    docker-compose -f docker/docker-compose.yml run --rm trading-bot \
        python scripts/paper_trading.py --ensemble --capital 100
"""

import sys
import argparse
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from loguru import logger
import joblib
import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from src.risk.risk_manager import RiskManager, load_risk_config
from src.models.multi_period_ensemble import MultiPeriodEnsemble, EnsembleSignal

# Import MTF feature generator from train_mtf
from train_mtf import MTFFeatureEngine


def generate_live_features(
    mtf_engine: MTFFeatureEngine,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate features for live trading.
    
    Unlike align_timeframes, this doesn't rely on perfect timestamp alignment.
    It takes the latest available data from each timeframe.
    """
    # Generate features for each TF
    m15_features = mtf_engine.generate_m15_trend_features(df_15m)
    m5_features = mtf_engine.generate_m5_signal_features(df_5m)
    m1_features = mtf_engine.generate_m1_timing_features(df_1m)
    
    # Take only the last row from each
    if m5_features.empty or m15_features.empty or m1_features.empty:
        return pd.DataFrame()
    
    # Start with M5 features (last row)
    result = m5_features.iloc[[-1]].copy()
    
    # Add M15 features (last available)
    m15_last = m15_features.iloc[-1]
    for col in m15_features.columns:
        result[col] = m15_last[col]
    
    # Add M1 features - aggregate last 5 candles
    m1_last_5 = m1_features.iloc[-5:] if len(m1_features) >= 5 else m1_features
    
    for col in m1_features.columns:
        if 'momentum' in col or 'rsi' in col:
            # For momentum/RSI: add last, mean, std
            result[f'{col}_last'] = m1_last_5[col].iloc[-1]
            result[f'{col}_mean'] = m1_last_5[col].mean()
            result[f'{col}_std'] = m1_last_5[col].std()
        else:
            # For other features: just last
            result[f'{col}_last'] = m1_last_5[col].iloc[-1]
    
    # Fill any NaN values
    result = result.fillna(0)
    
    # Convert object columns to numeric
    for col in result.columns:
        if result[col].dtype == 'object':
            result[col] = pd.Categorical(result[col]).codes
    
    return result


# ============================================================
# TELEGRAM NOTIFIER
# ============================================================

class TelegramNotifier:
    """Send notifications to Telegram."""
    
    def __init__(self, bot_token: str, chat_id: str = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_chat_id(self) -> Optional[str]:
        """Get chat ID from recent messages."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/getUpdates") as resp:
                data = await resp.json()
                if data.get('ok') and data.get('result'):
                    for update in data['result']:
                        if 'message' in update:
                            chat_id = str(update['message']['chat']['id'])
                            logger.info(f"Found chat_id: {chat_id}")
                            return chat_id
        except Exception as e:
            logger.error(f"Failed to get chat_id: {e}")
        return None
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram."""
        if not self.chat_id:
            logger.warning("No chat_id set, trying to get it...")
            self.chat_id = await self.get_chat_id()
            if not self.chat_id:
                logger.error("Cannot send message: no chat_id")
                return False
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode
                }
            ) as resp:
                result = await resp.json()
                if result.get('ok'):
                    return True
                else:
                    logger.error(f"Telegram error: {result}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False


# ============================================================
# V1 MODEL WRAPPER
# ============================================================

class V1FreshModel:
    """Simple wrapper for V1 fresh models (raw LightGBM)."""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.direction_model = None
        self.timing_model = None
        self.strength_model = None
        self.volatility_model = None
        self._is_trained = False
        
    def load(self):
        """Load all models from directory."""
        logger.info(f"Loading V1 Fresh models from {self.model_dir}")
        
        self.direction_model = joblib.load(self.model_dir / 'direction_model.joblib')
        self.timing_model = joblib.load(self.model_dir / 'timing_model.joblib')
        self.strength_model = joblib.load(self.model_dir / 'strength_model.joblib')
        self.volatility_model = joblib.load(self.model_dir / 'volatility_model.joblib')
        
        self._is_trained = True
        logger.info("V1 Fresh models loaded successfully")
        
    def get_trading_signal(
        self,
        X: pd.DataFrame,
        min_direction_prob: float = 0.50,
        min_strength: float = 0.30,
        min_timing: float = 0.01
    ) -> Dict:
        """Generate trading signal from features."""
        if not self._is_trained:
            raise RuntimeError("Models not loaded")
        
        # Get direction prediction
        direction_proba = self.direction_model.predict_proba(X)[0]
        direction_pred = np.argmax(direction_proba)
        
        # Get timing prediction
        timing_proba = self.timing_model.predict_proba(X)[0]
        is_good_timing = timing_proba[1] > min_timing
        
        # Get strength prediction
        strength_pred = self.strength_model.predict(X)[0]
        
        # Get volatility prediction
        volatility_pred = self.volatility_model.predict(X)[0]
        
        # Map direction
        direction_map = {0: -1, 1: 0, 2: 1}  # down, sideways, up
        direction = direction_map.get(direction_pred, 0)
        
        # Calculate confidence
        max_prob = max(direction_proba)
        confidence = max_prob * timing_proba[1]
        
        # Generate signal
        signal = 0
        if is_good_timing and max_prob >= min_direction_prob:
            if direction == 1:  # up
                signal = 1
            elif direction == -1:  # down
                signal = -1
        
        return {
            'signal': signal,
            'direction': direction,
            'direction_proba': direction_proba.tolist(),
            'timing_proba': timing_proba[1],
            'strength': strength_pred,
            'volatility': volatility_pred,
            'confidence': confidence,
            'is_good_timing': is_good_timing
        }


# ============================================================
# PAPER POSITION
# ============================================================

@dataclass
class PaperPosition:
    """Represents a paper trading position."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_time: datetime
    entry_price: float
    position_size: float
    leverage: float
    stop_loss: float
    take_profit: float
    margin: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTrader:
    """Paper trading engine with real-time data."""
    
    # MEXC fees
    MEXC_MAKER_FEE = 0.0       # 0%
    MEXC_TAKER_FEE = 0.0002    # 0.02%
    
    def __init__(
        self,
        model_path: str,
        telegram_token: str,
        capital: float = 100.0,
        risk_pct: float = 0.05,
        rr_ratio: float = 2.0,
        max_hold_candles: int = 60,
        pairs: List[str] = None,
        use_ensemble: bool = False
    ):
        self.model_path = model_path
        self.capital = capital
        self.initial_capital = capital
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.max_hold_candles = max_hold_candles
        self.use_ensemble = use_ensemble
        
        # Default pairs
        self.pairs = pairs or [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT",
            "AVAX/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
            "UNI/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
            "OP/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
            "BNB/USDT:USDT", "SUI/USDT:USDT", "AAVE/USDT:USDT"
        ]
        
        # Components
        self.model: Optional[V1FreshModel] = None
        self.ensemble: Optional[MultiPeriodEnsemble] = None
        self.feature_engine: Optional[MTFFeatureEngine] = None
        self.exchange: Optional[ccxt.binance] = None
        self.telegram: Optional[TelegramNotifier] = None
        
        # State
        self.position: Optional[PaperPosition] = None
        self.trades: List[PaperPosition] = []
        self.position_candles: int = 0
        self._running = False
        
        # Telegram
        self.telegram_token = telegram_token
        
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Paper Trader...")
        
        # Load model or ensemble
        if self.use_ensemble:
            logger.info("🔀 Using Multi-Period Ensemble (30d + 90d + 365d)")
            self.ensemble = MultiPeriodEnsemble()
            model_paths = {
                '30d': './models/v1_fresh',
                '90d': './models/v1_90d',
                '365d': './models/v1_365d'
            }
            if not self.ensemble.load(model_paths):
                raise RuntimeError("Failed to load ensemble models")
        else:
            logger.info(f"Using single model: {self.model_path}")
            self.model = V1FreshModel(self.model_path)
            self.model.load()
        
        # Feature engine
        self.feature_engine = MTFFeatureEngine()
        
        # Exchange (read-only, for price data)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Telegram
        self.telegram = TelegramNotifier(self.telegram_token)
        
        # Get chat_id
        logger.info("Waiting for Telegram message to get chat_id...")
        logger.info("Please send /start to your bot in Telegram!")
        
        for i in range(30):  # Wait up to 30 seconds
            chat_id = await self.telegram.get_chat_id()
            if chat_id:
                self.telegram.chat_id = chat_id
                break
            await asyncio.sleep(1)
        
        if not self.telegram.chat_id:
            logger.warning("No chat_id found. Telegram notifications disabled.")
        else:
            # Send startup message
            mode = "🔀 Ensemble (30d+90d+365d)" if self.use_ensemble else f"📅 {self.model_path}"
            await self.telegram.send_message(
                "🤖 <b>Paper Trading Started</b>\n\n"
                f"💰 Capital: ${self.capital:.2f}\n"
                f"📊 Risk per trade: {self.risk_pct*100:.1f}%\n"
                f"📈 RR Ratio: 1:{self.rr_ratio:.1f}\n"
                f"🔢 Pairs: {len(self.pairs)}\n"
                f"{mode}\n\n"
                "Scanning for opportunities..."
            )
        
        logger.info("Paper Trader initialized")
        
    async def close(self):
        """Cleanup resources."""
        if self.exchange:
            await self.exchange.close()
        if self.telegram:
            await self.telegram.close()
            
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data from exchange."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_leverage(self, stop_loss_pct: float) -> float:
        """Calculate leverage: leverage = risk_pct / stop_loss_pct.
        
        Rounds to integer for better compatibility with exchanges.
        """
        if stop_loss_pct <= 0:
            return 1.0
        leverage = self.risk_pct / stop_loss_pct
        # Round to nearest integer for exchange compatibility
        leverage = round(leverage)
        return min(max(leverage, 1.0), 20.0)  # Clamp 1x-20x
    
    def calculate_position(
        self,
        entry_price: float,
        stop_loss_pct: float
    ) -> Tuple[float, float, float, float]:
        """Calculate position size with leverage."""
        leverage = self.calculate_leverage(stop_loss_pct)
        margin = self.capital  # 100% of capital
        position_value = margin * leverage
        size = position_value / entry_price
        entry_fee = position_value * self.MEXC_TAKER_FEE
        
        return margin, position_value, size, entry_fee
    
    async def scan_for_signals(self) -> Optional[Tuple[str, Dict, pd.DataFrame, float]]:
        """Scan all pairs for trading signals."""
        best_signal = None
        best_pair = None
        best_df = None
        best_score = 0
        best_position_mult = 1.0
        
        for pair in self.pairs:
            try:
                # Fetch data for multiple timeframes
                df_1m = await self.fetch_ohlcv(pair, '1m', 100)
                df_5m = await self.fetch_ohlcv(pair, '5m', 100)
                df_15m = await self.fetch_ohlcv(pair, '15m', 100)
                
                if df_5m.empty or df_1m.empty or df_15m.empty:
                    continue
                
                # Generate features using live feature generator
                features = generate_live_features(
                    self.feature_engine, df_1m, df_5m, df_15m
                )
                
                if features.empty:
                    continue
                
                # Get expected features (from ensemble or single model)
                if self.use_ensemble:
                    expected_features = self.ensemble.feature_names
                else:
                    expected_features = self.model.direction_model.feature_name_
                
                # Add missing features as 0
                for feat in expected_features:
                    if feat not in features.columns:
                        features[feat] = 0
                
                # Select only expected features in correct order
                X = features[expected_features]
                
                # Get signal (ensemble or single model)
                if self.use_ensemble:
                    ensemble_signal = self.ensemble.get_trading_signal(X)
                    
                    # Skip if protected (no consensus)
                    if ensemble_signal.is_protected:
                        logger.debug(f"{pair}: Protected - {ensemble_signal.protection_reason}")
                        continue
                    
                    signal = {
                        'signal': ensemble_signal.signal,
                        'direction_proba': ensemble_signal.direction_proba,
                        'timing_proba': ensemble_signal.timing_proba,
                        'confidence': ensemble_signal.confidence,
                        'strength': ensemble_signal.strength,
                        'volatility': ensemble_signal.volatility,
                        'agreement_level': ensemble_signal.agreement_level,
                        'individual_signals': ensemble_signal.individual_signals
                    }
                    position_mult = ensemble_signal.position_size_multiplier
                    score = ensemble_signal.confidence
                else:
                    signal = self.model.get_trading_signal(X)
                    position_mult = 1.0
                    score = max(signal['direction_proba']) * signal['timing_proba']
                
                if signal['signal'] != 0 and score > best_score:
                    best_score = score
                    best_signal = signal
                    best_pair = pair
                    best_df = df_5m
                    best_position_mult = position_mult
                    
            except Exception as e:
                logger.debug(f"Error scanning {pair}: {e}")
                continue
        
        if best_signal:
            if self.use_ensemble:
                agreement = best_signal.get('agreement_level', 0)
                indiv = best_signal.get('individual_signals', {})
                logger.info(
                    f"🎯 Ensemble signal for {best_pair}: {best_signal['signal']}, "
                    f"score={best_score:.3f}, agreement={agreement}/3, "
                    f"votes={indiv}"
                )
            else:
                logger.info(f"🎯 Found signal for {best_pair}: {best_signal['signal']}, score={best_score:.3f}")
            return best_pair, best_signal, best_df, best_position_mult
        
        # Log scan summary every minute
        logger.info(f"📊 Scan complete: No signals found. Best score={best_score:.3f}")
        return None
    
    async def open_position(self, pair: str, signal: Dict, df: pd.DataFrame, 
                             position_mult: float = 1.0):
        """Open a paper position.
        
        Args:
            pair: Trading pair symbol
            signal: Signal dictionary
            df: OHLCV dataframe
            position_mult: Position size multiplier (0.7 for 2/3 consensus, 1.0 for full)
        """
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        
        direction = signal['signal']
        
        # Calculate SL/TP
        stop_distance = atr * 1.5
        stop_loss_pct = stop_distance / current_price
        
        if direction == 1:  # Long
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * self.rr_ratio)
        else:  # Short
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * self.rr_ratio)
        
        # Calculate position size (apply multiplier for reduced consensus)
        margin, position_value, size, entry_fee = self.calculate_position(
            current_price, stop_loss_pct
        )
        # Apply position multiplier (e.g., 0.7x for 2/3 consensus)
        margin *= position_mult
        size *= position_mult
        entry_fee *= position_mult
        
        leverage = self.calculate_leverage(stop_loss_pct)
        
        # Create position
        self.position = PaperPosition(
            symbol=pair,
            direction=direction,
            entry_time=datetime.utcnow(),
            entry_price=current_price,
            position_size=size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin=margin
        )
        self.position_candles = 0
        
        # Deduct entry fee from capital
        self.capital -= entry_fee
        
        # Log
        side = "LONG" if direction == 1 else "SHORT"
        
        # Build consensus info for ensemble
        consensus_info = ""
        if self.use_ensemble and 'agreement_level' in signal:
            agreement = signal['agreement_level']
            indiv = signal.get('individual_signals', {})
            consensus_info = f"\n🗳 Consensus: {agreement}/3 ({position_mult*100:.0f}% size)"
            if indiv:
                votes = ", ".join([f"{k}:{v}" for k, v in indiv.items()])
                consensus_info += f"\n📊 Votes: {votes}"
        
        logger.info(
            f"📈 OPENED {side} {pair} @ ${current_price:.4f}\n"
            f"   SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}\n"
            f"   Leverage: {leverage:.1f}x | Size: {size:.4f}{consensus_info}"
        )
        
        # Telegram
        if self.telegram:
            tg_consensus = ""
            if self.use_ensemble and 'agreement_level' in signal:
                tg_consensus = f"\n🗳 Ensemble: {signal['agreement_level']}/3 ({position_mult*100:.0f}% pos)"
            
            # Calculate potential PnL for display
            sl_pct = abs(stop_loss - current_price) / current_price * 100
            tp_pct = abs(take_profit - current_price) / current_price * 100
            risk_usd = margin * sl_pct / 100 * leverage
            reward_usd = margin * tp_pct / 100 * leverage
            
            await self.telegram.send_message(
                f"{'🟢' if direction == 1 else '🔴'} <b>{side} {pair.split(':')[0]}</b>\n\n"
                f"💵 Entry: ${current_price:.4f}\n"
                f"🛡 SL: ${stop_loss:.4f} ({sl_pct:.2f}%)\n"
                f"🎯 TP: ${take_profit:.4f} ({tp_pct:.2f}%)\n\n"
                f"⚡️ Leverage: {leverage:.1f}x\n"
                f"💰 Margin: ${margin:.2f}\n"
                f"📊 Position: ${margin * leverage:.2f}\n\n"
                f"⚠️ Risk: -${risk_usd:.2f}\n"
                f"✅ Reward: +${reward_usd:.2f}\n"
                f"📈 R:R = 1:{self.rr_ratio:.1f}{tg_consensus}\n"
                f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
            )
    
    async def check_position(self):
        """Check and update current position."""
        if not self.position:
            return
        
        try:
            # Fetch current price
            df = await self.fetch_ohlcv(self.position.symbol, '1m', 10)
            if df.empty:
                return
            
            current_price = df['close'].iloc[-1]
            self.position_candles += 1
            
            # Check exit conditions
            exit_reason = None
            exit_price = None
            
            if self.position.direction == 1:  # Long
                if current_price <= self.position.stop_loss:
                    exit_reason = 'stop_loss'
                    exit_price = self.position.stop_loss
                elif current_price >= self.position.take_profit:
                    exit_reason = 'take_profit'
                    exit_price = self.position.take_profit
            else:  # Short
                if current_price >= self.position.stop_loss:
                    exit_reason = 'stop_loss'
                    exit_price = self.position.stop_loss
                elif current_price <= self.position.take_profit:
                    exit_reason = 'take_profit'
                    exit_price = self.position.take_profit
            
            # Time exit
            if self.position_candles >= self.max_hold_candles and not exit_reason:
                exit_reason = 'time_exit'
                exit_price = current_price
            
            if exit_reason:
                await self.close_position(exit_reason, exit_price)
                
        except Exception as e:
            logger.error(f"Error checking position: {e}")
    
    async def close_position(self, reason: str, exit_price: float):
        """Close current position."""
        if not self.position:
            return
        
        pos = self.position
        
        # Calculate PnL
        if pos.direction == 1:  # Long
            price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:  # Short
            price_change_pct = (pos.entry_price - exit_price) / pos.entry_price
        
        position_value = pos.position_size * pos.entry_price
        pnl_before_fees = position_value * price_change_pct
        
        # Exit fee
        exit_fee = pos.position_size * exit_price * self.MEXC_TAKER_FEE
        
        # Final PnL
        pnl_usd = pnl_before_fees - exit_fee
        pnl_pct = pnl_usd / pos.margin * 100
        
        # Update position
        pos.exit_time = datetime.utcnow()
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.pnl_usd = pnl_usd
        pos.pnl_pct = pnl_pct
        
        # Update capital
        self.capital += pos.margin + pnl_usd
        
        # Save to history
        self.trades.append(pos)
        
        # Log
        emoji = "✅" if pnl_usd > 0 else "❌"
        side = "LONG" if pos.direction == 1 else "SHORT"
        logger.info(
            f"{emoji} CLOSED {side} {pos.symbol} @ ${exit_price:.4f}\n"
            f"   Reason: {reason}\n"
            f"   PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)\n"
            f"   Capital: ${self.capital:.2f}"
        )
        
        # Telegram
        if self.telegram:
            total_pnl = self.capital - self.initial_capital
            total_pnl_pct = (total_pnl / self.initial_capital) * 100
            wins = sum(1 for t in self.trades if t.pnl_usd > 0)
            total = len(self.trades)
            wr = wins / total * 100 if total > 0 else 0
            
            reason_emoji = {
                'stop_loss': '🛑 Stop Loss',
                'take_profit': '🎯 Take Profit',
                'time_exit': '⏰ Timeout'
            }.get(reason, reason)
            
            await self.telegram.send_message(
                f"{emoji} <b>Position Closed</b>\n\n"
                f"{'🟢' if pos.direction == 1 else '🔴'} {side} {pos.symbol.split(':')[0]}\n"
                f"📍 {reason_emoji}\n\n"
                f"💵 Entry: ${pos.entry_price:.4f}\n"
                f"💵 Exit: ${exit_price:.4f}\n"
                f"{'📈' if pnl_usd > 0 else '📉'} <b>PnL: ${pnl_usd:+.2f} ({pnl_pct:+.1f}%)</b>\n\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 Capital: ${self.capital:.2f}\n"
                f"📊 Session: ${total_pnl:+.2f} ({total_pnl_pct:+.1f}%)\n"
                f"🎯 W/L: {wins}/{total-wins} ({wr:.0f}%)\n"
                f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
            )
        
        # Clear position
        self.position = None
        self.position_candles = 0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else (high - low).mean()
    
    async def run(self):
        """Main trading loop."""
        logger.info("Starting paper trading loop...")
        self._running = True
        
        scan_interval = 60  # seconds
        position_check_interval = 10  # seconds
        last_scan = 0
        
        while self._running:
            try:
                now = time.time()
                
                if self.position:
                    # Check position every 10 seconds
                    await self.check_position()
                    await asyncio.sleep(position_check_interval)
                else:
                    # Scan for signals every minute
                    if now - last_scan >= scan_interval:
                        result = await self.scan_for_signals()
                        if result:
                            pair, signal, df, position_mult = result
                            await self.open_position(pair, signal, df, position_mult)
                        last_scan = now
                    await asyncio.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)
        
        # Final summary
        await self.send_summary()
    
    async def send_summary(self):
        """Send final trading summary."""
        if not self.trades:
            logger.info("No trades executed")
            return
        
        total_pnl = self.capital - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        losses = len(self.trades) - wins
        wr = wins / len(self.trades) * 100
        
        avg_win = np.mean([t.pnl_usd for t in self.trades if t.pnl_usd > 0]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_usd for t in self.trades if t.pnl_usd <= 0]) if losses > 0 else 0
        
        summary = (
            f"📊 <b>Paper Trading Summary</b>\n\n"
            f"💰 Final Capital: ${self.capital:.2f}\n"
            f"📈 Total PnL: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)\n\n"
            f"<b>Statistics:</b>\n"
            f"🔢 Total Trades: {len(self.trades)}\n"
            f"✅ Wins: {wins}\n"
            f"❌ Losses: {losses}\n"
            f"🎯 Win Rate: {wr:.1f}%\n"
            f"📈 Avg Win: ${avg_win:.2f}\n"
            f"📉 Avg Loss: ${avg_loss:.2f}\n"
        )
        
        logger.info(summary.replace('<b>', '').replace('</b>', ''))
        
        if self.telegram:
            await self.telegram.send_message(summary)


# ============================================================
# MAIN
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description='Paper Trading with Telegram')
    parser.add_argument('--model-path', type=str, default='./models/v1_fresh',
                        help='Path to model directory')
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Initial capital in USDT')
    parser.add_argument('--risk-pct', type=float, default=0.05,
                        help='Risk per trade (0.05 = 5%%)')
    parser.add_argument('--rr-ratio', type=float, default=2.0,
                        help='Risk-reward ratio')
    parser.add_argument('--telegram-token', type=str, 
                        default='8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc',
                        help='Telegram bot token')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use multi-period ensemble (30d + 90d + 365d models)')
    
    args = parser.parse_args()
    
    trader = PaperTrader(
        model_path=args.model_path,
        telegram_token=args.telegram_token,
        capital=args.capital,
        risk_pct=args.risk_pct,
        rr_ratio=args.rr_ratio,
        use_ensemble=args.ensemble
    )
    
    try:
        await trader.initialize()
        await trader.run()
    finally:
        await trader.close()


if __name__ == '__main__':
    asyncio.run(main())

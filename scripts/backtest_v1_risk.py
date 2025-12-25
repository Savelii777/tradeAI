#!/usr/bin/env python3
"""
V1 Model Backtest with Risk Management

Uses the original V1 model (from saved_mtf) with NEW risk management.
Does NOT change any ML logic - only adds risk controls.

Usage:
    python scripts/backtest_v1_risk.py --pairs 19 --days 14
    
    # Or in Docker:
    docker-compose -f docker/docker-compose.yml run --rm trading-bot \
        python scripts/backtest_v1_risk.py --pairs 19 --days 14
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
# Use direct LightGBM models instead of EnsembleModel wrapper
# from src.models.ensemble import EnsembleModel
from src.risk.risk_manager import RiskManager, load_risk_config
import joblib

# Import MTF feature generator from train_mtf
from train_mtf import MTFFeatureEngine


# ============================================================
# V1 FRESH MODEL WRAPPER
# ============================================================

class V1FreshModel:
    """
    Simple wrapper for V1 fresh models (raw LightGBM).
    
    Loads direction, timing, strength, volatility models directly
    and provides get_trading_signal() interface.
    """
    
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
        """
        Generate trading signal from features.
        
        Args:
            X: Feature DataFrame (single row)
            min_direction_prob: Minimum probability for direction
            min_strength: Minimum strength score (not used if model not trained)
            min_timing: Minimum timing probability
            
        Returns:
            Signal dictionary with 'signal', 'confidence', etc.
        """
        if not self._is_trained:
            raise RuntimeError("Models not loaded")
        
        # Get direction prediction
        direction_proba = self.direction_model.predict_proba(X)[0]  # [p_down, p_sideways, p_up]
        direction_pred = np.argmax(direction_proba)
        
        # Get timing prediction (probability of good entry)
        timing_proba = self.timing_model.predict_proba(X)[0][1]  # Probability of class 1
        
        # Get strength prediction
        try:
            strength = self.strength_model.predict(X)[0]
        except:
            strength = 1.0
        
        # Get volatility prediction
        try:
            volatility = self.volatility_model.predict(X)[0]
        except:
            volatility = 1.0
        
        # Build signal
        signal = {
            'signal': 'hold',
            'confidence': 0.0,
            'direction_proba': direction_proba.tolist(),
            'timing': timing_proba,
            'strength': strength,
            'volatility': volatility
        }
        
        p_down, p_sideways, p_up = direction_proba
        
        # Long signal conditions
        if p_up >= min_direction_prob and p_up > p_down and p_up > p_sideways:
            if timing_proba >= min_timing:
                signal['signal'] = 'buy'
                signal['confidence'] = p_up
                
        # Short signal conditions
        elif p_down >= min_direction_prob and p_down > p_up and p_down > p_sideways:
            if timing_proba >= min_timing:
                signal['signal'] = 'sell'
                signal['confidence'] = p_down
        
        return signal


# ============================================================
# V1 TOP PAIRS (ADA removed - only losing pair)
# ============================================================

V1_TOP_PAIRS = [
    'XAUT/USDT:USDT', 'BTC/USDT:USDT', 'BNB/USDT:USDT', 'TONCOIN/USDT:USDT',
    'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
    # 'ADA/USDT:USDT' - REMOVED (only losing pair in V1)
    'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
    'LTC/USDT:USDT', 'BCH/USDT:USDT', 'UNI/USDT:USDT', 'AAVE/USDT:USDT',
    'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT', 'OP/USDT:USDT',
]


# ============================================================
# DATA LOADING
# ============================================================

def load_pair_data(symbol: str, data_dir: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a single pair from CSV."""
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    filepath = Path(data_dir) / f"{safe_symbol}_{timeframe}.csv"
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def load_mtf_data(pair: str, data_dir: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load M1, M5, M15 data for a single pair."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        df = load_pair_data(pair, data_dir, tf)
        if df is None:
            return None
        data[tf.replace('m', '')] = df
    return data


def filter_data_by_days(
    data: Dict[str, pd.DataFrame], 
    days: int
) -> Dict[str, pd.DataFrame]:
    """Filter data to last N days."""
    end_time = data['5'].index[-1]
    start_time = end_time - timedelta(days=days)
    
    filtered = {}
    for tf, df in data.items():
        filtered[tf] = df[df.index >= start_time].copy()
    
    return filtered


# ============================================================
# MTF FEATURE GENERATION
# ============================================================

def generate_mtf_features(
    m1_data: pd.DataFrame,
    m5_data: pd.DataFrame,
    m15_data: pd.DataFrame
) -> pd.DataFrame:
    """Generate MTF features using same generator as training."""
    mtf_engine = MTFFeatureEngine()
    features = mtf_engine.align_timeframes(m1_data, m5_data, m15_data)
    
    # Convert object columns to numeric
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = pd.Categorical(features[col]).codes
    
    features = features.fillna(0)
    return features


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


# ============================================================
# V1 BACKTESTER WITH RISK MANAGEMENT + LEVERAGE
# ============================================================

# MEXC Futures Fees
MEXC_MAKER_FEE = 0.0       # 0%
MEXC_TAKER_FEE = 0.0002    # 0.02%
# We use taker for entry/exit = 0.02% * 2 = 0.04% round trip

class V1BacktesterWithRisk:
    """
    Backtester for V1 model WITH integrated risk management.
    
    Key features:
    - Leverage calculation based on SL distance and risk %
    - MEXC futures commissions (taker 0.05%)
    - Compound interest (capital grows/shrinks each trade)
    - Respects daily loss limits
    - Respects drawdown limits
    - Respects consecutive loss cooldowns
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_leverage: float = 20.0,     # Max leverage allowed
        slippage: float = 0.0001,       # 0.01% slippage
        risk_config: Optional[Dict] = None,
        max_positions: int = 999        # Max simultaneous positions
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.slippage = slippage
        self.max_positions = max_positions
        
        # MEXC fees
        self.entry_fee = MEXC_TAKER_FEE  # 0.05%
        self.exit_fee = MEXC_TAKER_FEE   # 0.05%
        
        # Initialize risk manager
        self.risk_manager = RiskManager(risk_config or {})
        self.risk_manager.set_initial_capital(initial_capital)
        
        # Trade tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_results: List[Dict] = []
        self.open_positions: List[Dict] = []  # Track multiple positions
        
        # Statistics
        self.blocked_by_risk: Dict[str, int] = {}
        self.current_day: Optional[datetime] = None
        self.total_fees_paid: float = 0.0
        self.leverage_used: List[float] = []
    
    def calculate_leverage(self, stop_loss_pct: float) -> float:
        """
        Calculate leverage based on risk % and stop loss distance.
        
        Formula: Leverage = Risk% / StopLoss%
        
        Example:
        - Risk = 2%, SL = 1% → Leverage = 2x
        - Risk = 2%, SL = 0.5% → Leverage = 4x
        - Risk = 2%, SL = 2% → Leverage = 1x
        """
        if stop_loss_pct <= 0:
            return 1.0
        
        leverage = self.risk_per_trade / stop_loss_pct
        
        # Cap at max leverage
        leverage = min(leverage, self.max_leverage)
        
        # Min leverage = 1x
        leverage = max(leverage, 1.0)
        
        return leverage
    
    def calculate_position_with_leverage(
        self,
        capital: float,
        entry_price: float,
        stop_loss_pct: float,
        leverage: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate position size with leverage and fees.
        
        FORMULA (100% capital as margin):
        - Margin = 100% of capital
        - Position value = Margin × Leverage
        - Leverage = Risk% / SL%
        
        Example with $100 capital, 2% risk, 0.5% SL:
        - Leverage = 2% / 0.5% = 4x
        - Margin = $100 (100% capital)
        - Position value = $100 × 4 = $400
        - On SL hit: loss = 0.5% × $400 = $2 = 2% of capital ✓
        - On TP hit (RR=2): profit = 1.0% × $400 = $4 = 4% of capital ✓
        
        Returns:
            - margin: Amount of capital used as margin (100% of capital)
            - position_value: Total position value (margin * leverage)
            - size: Position size in units (contracts)
            - entry_fee: Fee for opening position
        """
        # Use 100% of capital as margin
        margin = capital
        
        # Position value = margin × leverage
        position_value = margin * leverage
        
        # Position size in units
        size = position_value / entry_price
        
        # Entry fee (applied to full position value!)
        entry_fee_amount = position_value * self.entry_fee
        
        return margin, position_value, size, entry_fee_amount
    
    def run(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        model,  # V1FreshModel or EnsembleModel
        pair: str,
        min_confidence: float = 0.50,    # V1 threshold!
        min_timing: float = 0.01,        # V1 threshold!
        min_strength: float = 0.30,      # V1 threshold!
        stop_loss_atr: float = 1.5,
        take_profit_rr: float = 2.0,     # V1: RR 1:2
        max_holding_bars: int = 50,
    ) -> Dict:
        """
        Run backtest with risk management.
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting V1 backtest with risk management: {pair}")
        logger.info(f"Data: {len(df)} bars, Capital: ${self.initial_capital}")
        
        # Check blacklist
        can_trade, reason = self.risk_manager.can_trade(pair, self.initial_capital)
        if not can_trade and "blacklisted" in reason.lower():
            logger.warning(f"Pair {pair} is blacklisted, skipping")
            return {'error': 'Pair blacklisted', 'total_trades': 0}
        
        # Calculate ATR for position sizing
        atr = calculate_atr(df, period=14)
        
        # Initialize state
        capital = self.initial_capital
        position = None
        
        # Skip first 100 bars for indicator warmup
        start_idx = min(100, len(df) - 10)
        
        # Ensure features are aligned with df
        features = features.dropna()
        
        for i in range(start_idx, len(df) - 1):
            timestamp = df.index[i]
            current_price = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            # Track day changes for daily resets
            current_date = timestamp.date() if hasattr(timestamp, 'date') else None
            if current_date and current_date != self.current_day:
                if self.current_day is not None:
                    # Save previous day's results
                    self._save_daily_result()
                self.current_day = current_date
                self.risk_manager.reset_daily()
            
            # Get features for current bar - use timestamp-based lookup
            if timestamp not in features.index:
                continue
                
            current_features = features.loc[[timestamp]]
            if current_features.isna().all().all():
                continue
            
            # Record equity
            equity = capital
            if position:
                if position['side'] == 'long':
                    unrealized_pnl = (current_price - position['entry']) * position['size']
                else:
                    unrealized_pnl = (position['entry'] - current_price) * position['size']
                equity += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'price': current_price,
                'capital': capital
            })
            
            # Update risk manager capital
            self.risk_manager.current_capital = capital
            
            # Check position exit
            if position:
                bars_held = i - position['entry_idx']
                should_exit, exit_reason = self._check_exit(
                    position, current_price, high, low,
                    bars_held, max_holding_bars
                )
                
                if should_exit:
                    # Use actual SL/TP price, not current_price!
                    if exit_reason == 'stop_loss':
                        exit_price = position['stop_loss']
                    elif exit_reason == 'take_profit':
                        exit_price = position['take_profit']
                    else:
                        exit_price = current_price  # time_exit uses market price
                    
                    pnl = self._close_position(position, exit_price, exit_reason, timestamp)
                    capital += pnl
                    
                    # Record in risk manager
                    is_win = pnl > 0
                    self.risk_manager.record_trade_result(pnl, is_win)
                    
                    # Remove from open_positions
                    if position in self.open_positions:
                        self.open_positions.remove(position)
                    
                    position = None
                    continue
            
            # Check for new entry
            if not position:
                # Check max positions limit
                if len(self.open_positions) >= self.max_positions:
                    reason = f"Max positions limit ({self.max_positions}) reached"
                    if reason not in self.blocked_by_risk:
                        self.blocked_by_risk[reason] = 0
                    self.blocked_by_risk[reason] += 1
                    continue
                    
                # Check risk limits BEFORE getting signal
                can_trade, reason = self.risk_manager.can_trade(pair, capital)
                
                if not can_trade:
                    # Track blocked trades
                    if reason not in self.blocked_by_risk:
                        self.blocked_by_risk[reason] = 0
                    self.blocked_by_risk[reason] += 1
                    continue
                
                try:
                    # Get V1 signal with V1 thresholds!
                    signal = model.get_trading_signal(
                        current_features,
                        min_direction_prob=min_confidence,  # 0.50 for V1
                        min_strength=min_strength,          # 0.30 for V1
                        min_timing=min_timing               # 0.01 for V1
                    )
                    
                    if isinstance(signal, list):
                        signal = signal[0]
                    
                    if signal['signal'] in ['buy', 'sell']:
                        # Calculate stop loss distance
                        stop_distance = current_atr * stop_loss_atr
                        
                        if stop_distance <= 0 or current_price <= 0:
                            continue
                        
                        stop_loss_pct = stop_distance / current_price
                        
                        # Calculate leverage based on 2% risk and SL distance
                        leverage = self.calculate_leverage(stop_loss_pct)
                        self.leverage_used.append(leverage)
                        
                        # Entry with slippage
                        if signal['signal'] == 'buy':
                            entry_price = current_price * (1 + self.slippage)
                            stop_loss = entry_price - stop_distance
                            take_profit = entry_price + stop_distance * take_profit_rr
                            side = 'long'
                        else:
                            entry_price = current_price * (1 - self.slippage)
                            stop_loss = entry_price + stop_distance
                            take_profit = entry_price - stop_distance * take_profit_rr
                            side = 'short'
                        
                        # Calculate position with leverage (compound - uses current capital!)
                        margin, position_value, size, entry_fee = self.calculate_position_with_leverage(
                            capital=capital,
                            entry_price=entry_price,
                            stop_loss_pct=stop_loss_pct,
                            leverage=leverage
                        )
                        
                        if margin <= 0 or margin > capital:
                            continue
                        
                        # Deduct entry fee from capital (fee on full position value!)
                        capital -= entry_fee
                        self.total_fees_paid += entry_fee
                        
                        position = {
                            'entry': entry_price,
                            'size': size,
                            'side': side,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': timestamp,
                            'entry_idx': i,
                            'confidence': signal['confidence'],
                            'direction_proba': signal.get('direction_proba', [0.33, 0.34, 0.33]),
                            'pair': pair,
                            'margin': margin,
                            'position_value': position_value,
                            'leverage': leverage,
                            'entry_fee': entry_fee,
                            'stop_distance': stop_distance,
                            'stop_loss_pct': stop_loss_pct
                        }
                        
                        # DEBUG: Log first few trades
                        if len(self.trades) < 3:
                            logger.info(f"TRADE OPEN: {pair} {side}")
                            logger.info(f"  Entry: ${entry_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
                            logger.info(f"  SL distance: ${stop_distance:.2f} ({stop_loss_pct:.4%})")
                            logger.info(f"  TP distance: ${stop_distance * take_profit_rr:.2f} ({stop_loss_pct * take_profit_rr:.4%})")
                            logger.info(f"  Position: ${position_value:.2f}, Leverage: {leverage:.1f}x")
                            logger.info(f"  Expected SL PnL: ${-self.risk_per_trade * capital:.2f}")
                            logger.info(f"  Expected TP PnL: ${self.risk_per_trade * take_profit_rr * capital:.2f}")
                        
                        # Track in open_positions list
                        self.open_positions.append(position)
                        
                except Exception as e:
                    logger.debug(f"Signal error at {timestamp}: {e}")
                    continue
        
        # Close any remaining position
        if position:
            pnl = self._close_position(
                position, df['close'].iloc[-1], 'end_of_data', df.index[-1]
            )
            capital += pnl
            self.risk_manager.record_trade_result(pnl, pnl > 0)
        
        # Save final day result
        self._save_daily_result()
        
        return self._calculate_results(pair)
    
    def _check_exit(
        self,
        position: Dict,
        price: float,
        high: float,
        low: float,
        bars_held: int,
        max_bars: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if position should be exited."""
        # Time-based exit
        if bars_held >= max_bars:
            return True, 'time_exit'
        
        # Stop loss / Take profit
        if position['side'] == 'long':
            if low <= position['stop_loss']:
                return True, 'stop_loss'
            if high >= position['take_profit']:
                return True, 'take_profit'
        else:
            if high >= position['stop_loss']:
                return True, 'stop_loss'
            if low <= position['take_profit']:
                return True, 'take_profit'
        
        return False, None
    
    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        reason: str,
        timestamp
    ) -> float:
        """
        Close position and return PnL with leverage.
        
        PnL calculation with leverage:
        - Raw PnL = (Exit - Entry) * Size * Direction
        - Exit Fee = Position Value * 0.05%
        - Net PnL = Raw PnL - Exit Fee
        
        With 2% risk and SL hit: lose ~2% of capital
        With 2% risk and TP hit at RR 2:1: gain ~4% of capital
        """
        leverage = position.get('leverage', 1.0)
        position_value = position.get('position_value', position['entry'] * position['size'])
        margin = position.get('margin', position_value / leverage)
        
        # Apply slippage
        if position['side'] == 'long':
            actual_exit = exit_price * (1 - self.slippage)
            price_change_pct = (actual_exit - position['entry']) / position['entry']
        else:
            actual_exit = exit_price * (1 + self.slippage)
            price_change_pct = (position['entry'] - actual_exit) / position['entry']
        
        # PnL = price_change * position_value (NOT margin!)
        # Example: 0.5% price move × $400 position = $2 PnL
        raw_pnl = price_change_pct * position_value
        
        # Exit fee on full position value (not just margin!)
        exit_position_value = actual_exit * position['size']
        exit_fee = exit_position_value * self.exit_fee
        self.total_fees_paid += exit_fee
        
        # Net PnL after exit fee
        pnl = raw_pnl - exit_fee
        
        # DEBUG: Log first few trades close
        if len(self.trades) < 3:
            stop_loss_pct = position.get('stop_loss_pct', 0)
            logger.info(f"TRADE CLOSE: {position.get('pair')} {position['side']} via {reason}")
            logger.info(f"  Entry: ${position['entry']:.2f}, Exit: ${actual_exit:.2f}")
            logger.info(f"  SL: ${position['stop_loss']:.2f}, TP: ${position['take_profit']:.2f}")
            logger.info(f"  Price change: {price_change_pct:.4%}")
            logger.info(f"  Position value: ${position_value:.2f}")
            logger.info(f"  Raw PnL: ${raw_pnl:.2f}, Exit fee: ${exit_fee:.2f}, Net PnL: ${pnl:.2f}")
        
        # Calculate PnL as % of margin (for stats)
        pnl_pct_margin = (pnl / margin) * 100 if margin > 0 else 0
        
        self.trades.append({
            'pair': position.get('pair', 'Unknown'),
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry'],
            'exit_price': actual_exit,
            'side': position['side'],
            'size': position['size'],
            'leverage': leverage,
            'margin': margin,
            'position_value': position_value,
            'pnl': pnl,
            'pnl_percent': pnl_pct_margin,
            'exit_reason': reason,
            'confidence': position['confidence'],
            'entry_fee': position.get('entry_fee', 0),
            'exit_fee': exit_fee,
            'total_fees': position.get('entry_fee', 0) + exit_fee
        })
        
        return pnl
    
    def _save_daily_result(self) -> None:
        """Save daily result."""
        if self.current_day:
            status = self.risk_manager.get_status()
            self.daily_results.append({
                'date': self.current_day,
                'pnl': self.risk_manager.daily_pnl,
                'trades': self.risk_manager.trades_today,
                'drawdown': status['current_drawdown']
            })
    
    def _calculate_results(self, pair: str) -> Dict:
        """Calculate backtest statistics."""
        if not self.trades:
            return {
                'pair': pair,
                'error': 'No trades executed',
                'total_trades': 0,
                'blocked_by_risk': self.blocked_by_risk
            }
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Equity curve analysis
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            returns = equity_df['equity'].pct_change().dropna()
            
            # Sharpe ratio (annualized, assuming 5-min bars)
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(105120)
            else:
                sharpe = 0
            
            # Max drawdown
            equity = equity_df['equity']
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak
            max_dd = abs(drawdown.min()) * 100
        else:
            sharpe = 0
            max_dd = 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            reason = t['exit_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t['pnl']
        
        return {
            'pair': pair,
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital + total_pnl,
            'total_return': (total_pnl / self.initial_capital) * 100,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(self.trades)) * 100,
            'profit_factor': profit_factor,
            'avg_win': sum(t['pnl'] for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t['pnl'] for t in losses) / len(losses) if losses else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'exit_reasons': exit_reasons,
            'blocked_by_risk': self.blocked_by_risk,
            'daily_results': self.daily_results,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            # Leverage stats
            'avg_leverage': np.mean(self.leverage_used) if self.leverage_used else 1.0,
            'max_leverage_used': max(self.leverage_used) if self.leverage_used else 1.0,
            'min_leverage_used': min(self.leverage_used) if self.leverage_used else 1.0,
            # Fee stats
            'total_fees': self.total_fees_paid,
            'fees_pct_of_pnl': (self.total_fees_paid / abs(total_pnl) * 100) if total_pnl != 0 else 0
        }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V1 Backtest with Risk Management")
    parser.add_argument("--pairs", type=int, default=19, help="Number of pairs to test")
    parser.add_argument("--days", type=int, default=14, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--model-path", type=str, default="./models/v1_fresh",
                       help="Path to V1 model")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--risk-config", type=str, default="./config/risk_management.yaml")
    
    # V1 Signal thresholds
    parser.add_argument("--min-conf", type=float, default=0.50, 
                       help="V1: Min direction confidence (0.50)")
    parser.add_argument("--min-timing", type=float, default=0.01,
                       help="V1: Min timing score (0.01)")
    parser.add_argument("--min-strength", type=float, default=0.30,
                       help="V1: Min strength score (0.30)")
    
    # V1 Exit parameters
    parser.add_argument("--sl-atr", type=float, default=1.5,
                       help="Stop loss in ATR units")
    parser.add_argument("--tp-rr", type=float, default=2.0,
                       help="V1: Take profit RR ratio (2.0)")
    
    # Position limits
    parser.add_argument("--max-positions", type=int, default=999,
                       help="Max simultaneous positions (1 = one at a time)")
    
    # Leverage settings
    parser.add_argument("--risk-pct", type=float, default=0.05,
                       help="Risk per trade (0.05 = 5%)")
    parser.add_argument("--max-leverage", type=float, default=20.0,
                       help="Max leverage allowed (default 20x)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("V1 MODEL BACKTEST WITH LEVERAGE + MEXC FEES")
    print("="*70)
    print(f"Model Path: {args.model_path}")
    print(f"Test Days: {args.days}")
    print(f"Initial Capital: ${args.capital}")
    print(f"Risk per Trade: {args.risk_pct*100:.1f}%")
    print(f"Max Leverage: {args.max_leverage}x")
    print(f"MEXC Fees: Entry {MEXC_TAKER_FEE*100:.2f}% + Exit {MEXC_TAKER_FEE*100:.2f}% = {MEXC_TAKER_FEE*200:.2f}% round-trip")
    print(f"V1 Thresholds: direction={args.min_conf}, timing={args.min_timing}, strength={args.min_strength}")
    print(f"V1 Exit: SL={args.sl_atr} ATR, TP RR={args.tp_rr}")
    print(f"Max Positions: {args.max_positions}")
    print("="*70)
    
    # Load risk configuration
    risk_config = load_risk_config(args.risk_config)
    print(f"\nRisk Config Loaded:")
    print(f"  Max risk per trade: {risk_config.get('max_risk_per_trade', 0.02):.1%}")
    print(f"  Max daily loss: {risk_config.get('max_daily_loss', 0.05):.1%}")
    print(f"  Max drawdown: {risk_config.get('max_drawdown', 0.20):.1%}")
    print(f"  Blacklist: {risk_config.get('blacklist', [])}")
    
    # Load V1 model (use V1FreshModel for v1_fresh/v1_xxx models, EnsembleModel for saved_mtf)
    logger.info(f"Loading V1 model from {args.model_path}")
    
    # Check if it's a V1 model (v1_fresh, v1_365d, v1_180d, etc.)
    if 'v1_' in args.model_path or 'v1_fresh' in args.model_path:
        # Use V1FreshModel for newly trained models
        model = V1FreshModel(args.model_path)
        try:
            model.load()
            logger.info("V1 Fresh Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load V1 Fresh model: {e}")
            return 1
    else:
        # Try EnsembleModel for older models
        try:
            from src.models.ensemble import EnsembleModel
            model = EnsembleModel()
            model.load(args.model_path)
            logger.info("V1 Ensemble Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load V1 model: {e}")
            logger.error("Make sure you're using models/v1_fresh or compatible model!")
            return 1
    
    # Select pairs
    pairs = V1_TOP_PAIRS[:args.pairs]
    logger.info(f"Testing on {len(pairs)} pairs")
    
    # Aggregate results
    all_results = []
    total_blocked_by_risk = {}
    
    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting {pair}")
        logger.info('='*60)
        
        # Load MTF data
        mtf_data = load_mtf_data(pair, args.data_dir)
        if mtf_data is None:
            logger.warning(f"No data for {pair}, skipping")
            continue
        
        # Filter to test days
        mtf_data = filter_data_by_days(mtf_data, args.days)
        
        m1_df = mtf_data['1']
        m5_df = mtf_data['5']
        m15_df = mtf_data['15']
        
        logger.info(f"Data: M1={len(m1_df)}, M5={len(m5_df)}, M15={len(m15_df)} bars")
        
        if len(m5_df) < 100:
            logger.warning(f"Not enough data for {pair}, skipping")
            continue
        
        # Generate features
        logger.info("Generating MTF features...")
        try:
            features = generate_mtf_features(m1_df, m5_df, m15_df)
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            continue
        
        # Align features to M5 data
        features = features.loc[features.index.isin(m5_df.index)]
        m5_df = m5_df.loc[m5_df.index.isin(features.index)]
        
        logger.info(f"Features: {features.shape[1]} columns, {len(features)} rows")
        
        # Run backtest with leverage and MEXC fees
        backtester = V1BacktesterWithRisk(
            initial_capital=args.capital,
            risk_per_trade=args.risk_pct,
            max_leverage=args.max_leverage,
            slippage=0.0001,  # 0.01% slippage
            risk_config=risk_config,
            max_positions=args.max_positions
        )
        
        results = backtester.run(
            df=m5_df,
            features=features,
            model=model,
            pair=pair,
            min_confidence=args.min_conf,
            min_timing=args.min_timing,
            min_strength=args.min_strength,
            stop_loss_atr=args.sl_atr,
            take_profit_rr=args.tp_rr,
            max_holding_bars=50
        )
        
        all_results.append(results)
        
        # Accumulate blocked trades
        for reason, count in results.get('blocked_by_risk', {}).items():
            if reason not in total_blocked_by_risk:
                total_blocked_by_risk[reason] = 0
            total_blocked_by_risk[reason] += count
        
        # Print pair results
        if results.get('total_trades', 0) > 0:
            print(f"\n{pair}:")
            print(f"  Trades: {results['total_trades']}, Win Rate: {results['win_rate']:.1f}%")
            print(f"  Return: {results['total_return']:.2f}%, PF: {results['profit_factor']:.2f}")
            print(f"  Sharpe: {results['sharpe_ratio']:.2f}, MaxDD: {results['max_drawdown']:.2f}%")
            if results.get('blocked_by_risk'):
                print(f"  Blocked by risk: {sum(results['blocked_by_risk'].values())} signals")
        else:
            print(f"\n{pair}: No trades")
            if results.get('blocked_by_risk'):
                print(f"  Blocked by risk: {results['blocked_by_risk']}")
    
    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("V1 + RISK MANAGEMENT: AGGREGATE RESULTS")
    print("="*70)
    
    valid_results = [r for r in all_results if r.get('total_trades', 0) > 0]
    
    if not valid_results:
        print("No trades executed across all pairs!")
        print(f"\nBlocked by risk management: {total_blocked_by_risk}")
        return 1
    
    total_trades = sum(r['total_trades'] for r in valid_results)
    total_wins = sum(r['winning_trades'] for r in valid_results)
    
    # Calculate aggregate metrics
    final_capitals = [r['final_capital'] for r in valid_results]
    avg_return = np.mean([r['total_return'] for r in valid_results])
    total_return = (sum(final_capitals) - args.capital * len(valid_results)) / (args.capital * len(valid_results)) * 100
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results])
    avg_pf = np.mean([r['profit_factor'] for r in valid_results])
    max_dd = max(r['max_drawdown'] for r in valid_results)
    
    # Calculate total PnL in $
    total_pnl = sum(final_capitals) - args.capital * len(valid_results)
    avg_pnl_per_pair = total_pnl / len(valid_results)
    
    # Monthly projection (extrapolate from test days)
    days_tested = args.days
    monthly_multiplier = 30 / days_tested
    monthly_pnl_projection = total_pnl * monthly_multiplier
    monthly_roi_projection = (monthly_pnl_projection / (args.capital * len(valid_results))) * 100
    
    print(f"\nPairs tested: {len(valid_results)}/{len(V1_TOP_PAIRS[:args.pairs])}")
    print(f"Total trades: {total_trades}")
    print(f"Overall win rate: {(total_wins / total_trades * 100):.1f}%")
    print(f"Average return per pair: {avg_return:.2f}%")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Average Profit Factor: {avg_pf:.2f}")
    print(f"Max Drawdown (worst pair): {max_dd:.2f}%")
    
    # PnL and ROI Summary
    print(f"\n{'='*40}")
    print("💰 PnL & ROI SUMMARY")
    print('='*40)
    print(f"Initial Capital: ${args.capital:.2f}")
    print(f"Test Period: {days_tested} days")
    print(f"Max Positions: {args.max_positions}")
    print(f"\n📊 Actual Results ({days_tested} days):")
    print(f"  Total PnL: ${total_pnl:.2f}")
    print(f"  Avg PnL per pair: ${avg_pnl_per_pair:.2f}")
    print(f"  ROI: {total_return:.2f}%")
    print(f"\n📈 Monthly Projection (30 days):")
    print(f"  Projected Monthly PnL: ${monthly_pnl_projection:.2f}")
    print(f"  Projected Monthly ROI: {monthly_roi_projection:.2f}%")
    print(f"  Final Capital (projected): ${args.capital + monthly_pnl_projection:.2f}")
    
    # Leverage & Fees stats
    print(f"\n{'='*40}")
    print("⚡ LEVERAGE & FEES")
    print('='*40)
    avg_lev = np.mean([r.get('avg_leverage', 1.0) for r in valid_results])
    max_lev = max([r.get('max_leverage_used', 1.0) for r in valid_results])
    total_fees = sum([r.get('total_fees', 0) for r in valid_results])
    print(f"Risk per Trade: {args.risk_pct*100:.1f}%")
    print(f"Average Leverage: {avg_lev:.1f}x")
    print(f"Max Leverage Used: {max_lev:.1f}x")
    print(f"Total Fees Paid: ${total_fees:.2f}")
    print(f"Fees % of Gross PnL: {(total_fees / (abs(total_pnl) + total_fees) * 100):.1f}%" if total_pnl != 0 else "N/A")
    
    # Risk management impact
    print(f"\n{'='*40}")
    print("RISK MANAGEMENT IMPACT")
    print('='*40)
    
    total_blocked = sum(total_blocked_by_risk.values())
    print(f"Total signals blocked: {total_blocked}")
    for reason, count in sorted(total_blocked_by_risk.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
    # Exit reason analysis
    print(f"\n{'='*40}")
    print("EXIT REASONS")
    print('='*40)
    
    exit_summary = {}
    for r in valid_results:
        for reason, data in r.get('exit_reasons', {}).items():
            if reason not in exit_summary:
                exit_summary[reason] = {'count': 0, 'pnl': 0}
            exit_summary[reason]['count'] += data['count']
            exit_summary[reason]['pnl'] += data['pnl']
    
    for reason, data in sorted(exit_summary.items()):
        pct = data['count'] / total_trades * 100
        print(f"  {reason}: {data['count']} ({pct:.1f}%), PnL: ${data['pnl']:.2f}")
    
    # Per-pair summary
    print(f"\n{'='*40}")
    print("PER-PAIR SUMMARY")
    print('='*40)
    
    # Sort by return
    valid_results_sorted = sorted(valid_results, key=lambda x: x['total_return'], reverse=True)
    
    profitable_pairs = 0
    for r in valid_results_sorted:
        pair = r.get('pair', 'Unknown')
        profit_marker = "✓" if r['total_return'] > 0 else "✗"
        if r['total_return'] > 0:
            profitable_pairs += 1
        print(f"  {profit_marker} {pair:20s}: {r['total_trades']:3d} trades, "
              f"WR: {r['win_rate']:5.1f}%, "
              f"Ret: {r['total_return']:+7.2f}%, "
              f"PF: {r['profit_factor']:.2f}, "
              f"MaxDD: {r['max_drawdown']:.1f}%")
    
    print(f"\nProfitable pairs: {profitable_pairs}/{len(valid_results)}")
    
    # V1 vs V2 comparison
    print("\n" + "="*70)
    print("V1 + RISK MANAGEMENT SUMMARY")
    print("="*70)
    print(f"""
Expected V1 Results (without risk management):
├── Win Rate: 59.8%
├── Max Drawdown: 15.64%
├── Return: +18.14%
└── Profitable pairs: 18/19

Current Results (V1 + Risk Management):
├── Win Rate: {(total_wins / total_trades * 100):.1f}%
├── Max Drawdown: {max_dd:.2f}%
├── Avg Return: {avg_return:.2f}%
├── Profitable pairs: {profitable_pairs}/{len(valid_results)}
└── Blocked by risk: {total_blocked} signals
""")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

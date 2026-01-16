#!/usr/bin/env python3
"""
Backtest MEXC Live - ТОЧНАЯ копия live_trading_v10_csv.py

Этот бэктест симулирует ТОЧНО то что происходит на MEXC:
- Те же пороги сигналов (0.50, 1.5, 1.8)
- Те же комиссии (0.02% entry + 0.02% exit)
- Тот же slippage (0.05%)
- Та же логика position sizing (risk_amount / sl_pct, max leverage 50x)
- Та же логика breakeven (entry + 1.0*ATR после триггера)
- Та же логика trailing (R-based multipliers)
- Capital compounding (как на бирже - баланс меняется после каждой сделки)

Usage:
    python scripts/backtest_mexc_live.py --days 14 --capital 61
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES

# ============================================================
# КОНФИГ - ТОЧНО КАК В live_trading_v10_csv.py Config class
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_20.json"

# Signal thresholds - ТОЧНО КАК В LIVE (Config class line 121-126)
MIN_CONF = 0.58
MIN_TIMING = 1.8
MIN_STRENGTH = 2.5

# Risk Management - ТОЧНО КАК В LIVE (Config class line 127-137)
# RISK_PCT is set via --risk argument (default 5%)
RISK_PCT = 0.05           # Will be overridden by args
MAX_LEVERAGE = 50.0       # Maximum leverage
ENTRY_FEE = 0.0002        # 0.02% entry fee
EXIT_FEE = 0.0002         # 0.02% exit fee  
SLIPPAGE_PCT = 0.0005     # 0.05% slippage
MAX_POSITION_SIZE = 4_000_000.0
MAX_HOLDING_BARS = 150

# V8 Features - ТОЧНО КАК В LIVE (Config class line 139-141)
USE_ADAPTIVE_SL = True
USE_DYNAMIC_LEVERAGE = True
USE_AGGRESSIVE_TRAIL = True


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Position:
    """Позиция - точная копия self.position в PortfolioManager"""
    pair: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    entry_bar: int
    stop_loss: float
    stop_distance: float
    position_value: float
    leverage: float
    atr: float
    pred_strength: float
    breakeven_active: bool = False
    be_trigger_mult: float = 2.2


@dataclass
class Trade:
    """Закрытая сделка"""
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_pct: float
    pnl_dollar: float
    position_value: float
    leverage: float
    exit_reason: str
    bars_held: int


# ============================================================
# DATA LOADING (same as live)
# ============================================================
def load_pair_data(pair: str, data_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
    """Load data from CSV/Parquet - same as CSVDataManager"""
    safe_symbol = pair.replace('/', '_').replace(':', '_')
    
    # Try parquet first
    parquet_path = data_dir / f"{safe_symbol}_{timeframe}.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)
            return df
        except:
            pass
    
    # Fall back to CSV
    csv_path = data_dir / f"{safe_symbol}_{timeframe}.csv"
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same as live"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Same as live"""
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame,
                     mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Same as live prepare_features()"""
    # Use lookback windows like live
    LOOKBACK_M1 = 7500
    LOOKBACK_M5 = 1500
    LOOKBACK_M15 = 500
    
    m1 = m1.tail(LOOKBACK_M1)
    m5 = m5.tail(LOOKBACK_M5)
    m15 = m15.tail(LOOKBACK_M15)
    
    if len(m1) < 200 or len(m5) < 200 or len(m15) < 200:
        return pd.DataFrame()
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        ft = ft.dropna(subset=['close', 'atr'])
        
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        ft = ft.ffill().dropna()
        return ft
    except:
        return pd.DataFrame()


# ============================================================
# MEXC LIVE BACKTESTER
# ============================================================
class MEXCLiveBacktester:
    """
    Бэктест который ТОЧНО копирует логику PortfolioManager из live_trading_v10_csv.py
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.total_signals = 0
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.total_signals = 0
    
    def _get_sl_mult(self, pred_strength: float) -> float:
        """
        Adaptive SL multiplier - ТОЧНО КАК В LIVE open_position() line 1055-1060
        """
        if pred_strength >= 3.0:
            return 1.6
        elif pred_strength >= 2.0:
            return 1.5
        else:
            return 1.2
    
    def _get_be_trigger_mult(self, pred_strength: float) -> float:
        """
        BE trigger multiplier - ТОЧНО КАК В LIVE open_position() line 1134
        """
        if pred_strength >= 3.0:
            return 2.5
        elif pred_strength >= 2.0:
            return 2.2
        else:
            return 1.8
    
    def _get_trail_mult(self, r_mult: float) -> float:
        """
        Trailing multiplier - ТОЧНО КАК В LIVE update_position() line 1178
        """
        if r_mult > 5:
            return 0.6
        elif r_mult > 3:
            return 1.2
        elif r_mult > 2:
            return 1.8
        else:
            return 2.5
    
    def open_position(self, signal: Dict, bar_idx: int) -> bool:
        """
        Open position - ТОЧНО КАК В LIVE PortfolioManager.open_position()
        """
        if self.position is not None:
            return False
        
        if self.capital <= 0:
            return False
        
        entry_price = signal['price']
        atr = signal['atr']
        pred_strength = signal.get('strength', 2.0)
        
        # Apply slippage to entry (WORSE entry like LIVE)
        if signal['direction'] == 'LONG':
            entry_price = entry_price * (1 + SLIPPAGE_PCT)  # Pay more for LONG
        else:
            entry_price = entry_price * (1 - SLIPPAGE_PCT)  # Get less for SHORT
        
        # === ATR-BASED STOP LOSS (далеко, стабильно) ===
        sl_mult = self._get_sl_mult(pred_strength)  # 1.2-1.6x ATR
        stop_distance = atr * sl_mult
        
        if signal['direction'] == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # === RISK-BASED POSITION SIZING ===
        # Размер позиции рассчитан так, чтобы при стопе терять ровно 5% от депозита
        RISK_PCT = 0.05  # 5% риска от депозита
        risk_amount = self.capital * RISK_PCT  # Сколько готовы потерять
        sl_pct = stop_distance / entry_price  # Стоп в % от цены
        
        # position_size * sl_pct = risk_amount
        # position_size = risk_amount / sl_pct
        position_value = risk_amount / sl_pct
        
        # Ограничения
        max_position_by_leverage = self.capital * MAX_LEVERAGE
        original_position = position_value
        position_value = min(position_value, max_position_by_leverage, MAX_POSITION_SIZE)
        
        # CRITICAL: Если position урезан, нужно уменьшить стоп чтобы сохранить 5% риск!
        if position_value < original_position:
            # Пересчитываем стоп для сохранения риска
            # risk_amount = position_value * new_sl_pct
            # new_sl_pct = risk_amount / position_value
            new_sl_pct = risk_amount / position_value
            new_stop_distance = new_sl_pct * entry_price
            
            # Обновляем стоп
            if signal['direction'] == 'LONG':
                stop_loss = entry_price - new_stop_distance
            else:
                stop_loss = entry_price + new_stop_distance
            stop_distance = new_stop_distance
        
        # Рассчитываем итоговое плечо
        leverage = position_value / self.capital
        
        # Минимальная проверка
        min_position_value = 1.0
        if position_value < min_position_value:
            return False
        
        # Entry fee (вычитается из capital)
        entry_fee = position_value * ENTRY_FEE
        self.capital -= entry_fee
        
        self.position = Position(
            pair=signal['pair'],
            direction=signal['direction'],
            entry_price=entry_price,
            entry_time=signal['timestamp'],
            entry_bar=bar_idx,
            stop_loss=stop_loss,
            stop_distance=stop_distance,
            position_value=position_value,
            leverage=leverage,
            atr=atr,
            pred_strength=pred_strength,
            breakeven_active=False,
            be_trigger_mult=self._get_be_trigger_mult(pred_strength)
        )
        
        return True
    
    def update_position(self, candle_high: float, candle_low: float) -> Optional[str]:
        """
        Update position with breakeven and trailing - ATR-BASED
        Returns exit reason if stop hit, None otherwise.
        """
        if self.position is None:
            return None
        
        pos = self.position
        atr = pos.atr
        
        # === ATR-BASED BREAKEVEN TRIGGER ===
        be_trigger_dist = atr * pos.be_trigger_mult  # 1.8-2.5x ATR
        
        # === ATR-BASED BE MARGIN = 1.0 ATR locked ===
        be_margin_dist = atr * 1.0
        
        if pos.direction == 'LONG':
            # 1. CHECK STOP FIRST
            if candle_low <= pos.stop_loss:
                return 'stop_loss' if not pos.breakeven_active else 'breakeven_stop'
            
            # 2. Breakeven trigger
            if not pos.breakeven_active and candle_high >= pos.entry_price + be_trigger_dist:
                pos.breakeven_active = True
                pos.stop_loss = pos.entry_price + be_margin_dist  # Lock 1.0 ATR profit
            
            # 3. Trailing - R-based
            if pos.breakeven_active:
                r_mult = (candle_high - pos.entry_price) / pos.stop_distance
                trail_mult = self._get_trail_mult(r_mult)
                new_sl = candle_high - atr * trail_mult
                
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
        
        else:  # SHORT
            # 1. CHECK STOP FIRST
            if candle_high >= pos.stop_loss:
                return 'stop_loss' if not pos.breakeven_active else 'breakeven_stop'
            
            # 2. Breakeven trigger
            if not pos.breakeven_active and candle_low <= pos.entry_price - be_trigger_dist:
                pos.breakeven_active = True
                pos.stop_loss = pos.entry_price - be_margin_dist  # Lock 1.0 ATR profit
            
            # 3. Trailing - R-based
            if pos.breakeven_active:
                r_mult = (pos.entry_price - candle_low) / pos.stop_distance
                trail_mult = self._get_trail_mult(r_mult)
                new_sl = candle_low + atr * trail_mult
                
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl
        
        return None  # No exit
    
    def close_position(self, exit_price: float, exit_time: datetime, 
                       bar_idx: int, reason: str):
        """
        Close position and calculate PnL - like LIVE _handle_closed_position()
        """
        if self.position is None:
            return
        
        pos = self.position
        
        # Apply slippage to exit (WORSE exit like LIVE)
        if pos.direction == 'LONG':
            exit_price = exit_price * (1 - SLIPPAGE_PCT)  # Get less for LONG exit
        else:
            exit_price = exit_price * (1 + SLIPPAGE_PCT)  # Pay more for SHORT exit
        
        # Calculate PnL - same logic as LIVE (using position_value)
        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100
        
        # Gross profit (on full position value)
        gross_pnl = pos.position_value * (pnl_pct / 100)
        
        # Exit fee
        exit_fee = pos.position_value * EXIT_FEE
        
        # Net PnL
        pnl_dollar = gross_pnl - exit_fee
        
        # Update capital (compounding like LIVE)
        self.capital += pnl_dollar
        
        # Determine exit reason more precisely
        if reason == 'breakeven_stop' and pnl_pct > 0.5:
            reason = 'trailing_stop'
        
        trade = Trade(
            pair=pos.pair,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl_pct=pnl_pct,
            pnl_dollar=pnl_dollar,
            position_value=pos.position_value,
            leverage=pos.leverage,
            exit_reason=reason,
            bars_held=bar_idx - pos.entry_bar
        )
        
        self.trades.append(trade)
        self.position = None
    
    def run(self, signals_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest bar by bar.
        """
        self.reset()
        
        if len(signals_df) == 0:
            return {'error': 'No signals'}
        
        # Sort signals by timestamp, then by score
        signals_df = signals_df.sort_values(['timestamp', 'score'], ascending=[True, False])
        
        # Get all timestamps
        all_timestamps = set()
        for pair, df in price_data.items():
            all_timestamps.update(df.index.tolist())
        all_timestamps = sorted(all_timestamps)
        
        # Create signal lookup
        signals_by_time = {}
        for _, row in signals_df.iterrows():
            ts = row['timestamp']
            if ts not in signals_by_time:
                signals_by_time[ts] = []
            signals_by_time[ts].append(row.to_dict())
        
        logger.info(f"Processing {len(all_timestamps)} bars, {len(signals_df)} signals")
        
        # Main loop
        for bar_idx, ts in enumerate(all_timestamps):
            
            # If we have position, update it
            if self.position is not None:
                pair = self.position.pair
                
                if pair in price_data and ts in price_data[pair].index:
                    candle = price_data[pair].loc[ts]
                    high = candle['high']
                    low = candle['low']
                    close = candle['close']
                    
                    # Check max holding
                    bars_held = bar_idx - self.position.entry_bar
                    if bars_held >= MAX_HOLDING_BARS:
                        self.close_position(close, ts, bar_idx, 'max_bars')
                        continue
                    
                    # Update position and check for exit
                    exit_reason = self.update_position(high, low)
                    
                    if exit_reason:
                        # Exit at stop loss price
                        self.close_position(self.position.stop_loss, ts, bar_idx, exit_reason)
                        continue
            
            # Look for new signals (only if no position)
            if self.position is None and ts in signals_by_time:
                candidates = signals_by_time[ts]
                self.total_signals += len(candidates)
                
                # Filter and sort by score (best first)
                valid = []
                for sig in candidates:
                    if (sig['confidence'] >= MIN_CONF and 
                        sig['timing'] >= MIN_TIMING and 
                        sig['strength'] >= MIN_STRENGTH):
                        valid.append(sig)
                
                valid.sort(key=lambda x: x['score'], reverse=True)
                
                # Open best signal
                for sig in valid:
                    if self.open_position(sig, bar_idx):
                        break
        
        # Force close any open position
        if self.position is not None:
            pair = self.position.pair
            if pair in price_data and len(price_data[pair]) > 0:
                last_candle = price_data[pair].iloc[-1]
                self.close_position(last_candle['close'], price_data[pair].index[-1],
                                   len(all_timestamps), 'end_of_data')
        
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate final results."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_signals': self.total_signals,
                'final_capital': self.capital,
            }
        
        wins = [t for t in self.trades if t.pnl_dollar > 0]
        losses = [t for t in self.trades if t.pnl_dollar <= 0]
        
        total_pnl = sum(t.pnl_dollar for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        gross_wins = sum(t.pnl_dollar for t in wins) if wins else 0
        gross_losses = abs(sum(t.pnl_dollar for t in losses)) if losses else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        
        # Exit reasons
        sl_exits = len([t for t in self.trades if t.exit_reason == 'stop_loss'])
        be_exits = len([t for t in self.trades if t.exit_reason == 'breakeven_stop'])
        trail_exits = len([t for t in self.trades if t.exit_reason == 'trailing_stop'])
        
        avg_bars = np.mean([t.bars_held for t in self.trades])
        
        # Calculate avg ROE (profit as % of margin used)
        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
        avg_leverage = np.mean([t.leverage for t in self.trades])
        avg_win_roe = avg_win_pct * avg_leverage
        avg_loss_roe = avg_loss_pct * avg_leverage
        
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS (MEXC Live Simulation)")
        logger.info("=" * 60)
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")
        logger.info(f"Final capital: ${self.capital:.2f}")
        logger.info(f"Total return: {(self.capital - self.initial_capital) / self.initial_capital * 100:.1f}%")
        logger.info("-" * 60)
        logger.info(f"Total signals: {self.total_signals}")
        logger.info(f"Total trades: {len(self.trades)}")
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info("-" * 60)
        logger.info(f"Avg leverage: {avg_leverage:.0f}x")
        logger.info(f"Avg WIN:  {avg_win_pct:+.2f}% price move → {avg_win_roe:+.1f}% ROE")
        logger.info(f"Avg LOSS: {avg_loss_pct:+.2f}% price move → {avg_loss_roe:+.1f}% ROE")
        logger.info("-" * 60)
        logger.info(f"Exits - SL: {sl_exits}, BE: {be_exits}, Trail: {trail_exits}")
        logger.info(f"Avg holding: {avg_bars:.0f} bars ({avg_bars * 5 / 60:.1f} hours)")
        logger.info("=" * 60)
        
        return {
            'total_trades': len(self.trades),
            'total_signals': self.total_signals,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'final_capital': self.capital,
            'initial_capital': self.initial_capital,
            'trades': self.trades,
            'sl_exits': sl_exits,
            'be_exits': be_exits,
            'trail_exits': trail_exits,
        }


# ============================================================
# MAIN
# ============================================================
def main():
    global RISK_PCT
    
    parser = argparse.ArgumentParser(description='Backtest MEXC Live')
    parser.add_argument('--days', type=int, default=14, help='Days to backtest')
    parser.add_argument('--capital', type=float, default=61.0, help='Starting capital')
    parser.add_argument('--risk', type=float, default=5.0, help='Risk per trade in %% (default 5)')
    args = parser.parse_args()
    
    RISK_PCT = args.risk / 100.0  # Convert to decimal
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)
    
    logger.info("=" * 70)
    logger.info("MEXC LIVE BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Capital: ${args.capital:.2f}")
    logger.info(f"Risk per trade: {args.risk}%")
    logger.info(f"Thresholds: Conf={MIN_CONF}, Tim={MIN_TIMING}, Str={MIN_STRENGTH}")
    logger.info(f"Fees: Entry={ENTRY_FEE*100:.3f}%, Exit={EXIT_FEE*100:.3f}%, Slippage={SLIPPAGE_PCT*100:.3f}%")
    
    # Load pairs
    import json
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            pairs = [p['symbol'] for p in json.load(f)['pairs']][:20]
    else:
        pairs = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    
    logger.info(f"Pairs: {len(pairs)}")
    
    # Load models
    logger.info("\nLoading model...")
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
    }
    scaler_path = MODEL_DIR / 'scaler.joblib'
    if scaler_path.exists():
        models['scaler'] = joblib.load(scaler_path)
    
    logger.info(f"  Features: {len(models['features'])}")
    
    # Load data and generate signals
    mtf_engine = MTFFeatureEngine()
    
    logger.info("\nLoading data...")
    all_signals = []
    price_data = {}
    
    for pair in pairs:
        m1 = load_pair_data(pair, DATA_DIR, '1m')
        m5 = load_pair_data(pair, DATA_DIR, '5m')
        m15 = load_pair_data(pair, DATA_DIR, '15m')
        
        if m1 is None or m5 is None or m15 is None:
            continue
        
        # Filter to date range
        m1 = m1[(m1.index >= start_date) & (m1.index < end_date)]
        m5 = m5[(m5.index >= start_date) & (m5.index < end_date)]
        m15 = m15[(m15.index >= start_date) & (m15.index < end_date)]
        
        if len(m5) < 100:
            continue
        
        price_data[pair] = m5
        
        # Prepare features
        features = prepare_features(m1, m5, m15, mtf_engine)
        if len(features) < 10:
            continue
        
        # Align with model
        for f in models['features']:
            if f not in features.columns:
                features[f] = 0
        
        X = features[models['features']].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if models.get('scaler') is not None:
            X = models['scaler'].transform(X)
        
        # Get predictions
        dir_proba = models['direction'].predict_proba(X)
        dir_pred = models['direction'].predict(X)
        timing_pred = models['timing'].predict(X)
        strength_pred = models['strength'].predict(X)
        
        # Generate signals (direction: 0=Down, 1=Sideways, 2=Up)
        pair_signals = 0
        for i, (ts, row) in enumerate(features.iterrows()):
            direction = int(dir_pred[i])
            if direction == 1:  # Skip sideways
                continue
            
            conf = float(np.max(dir_proba[i]))
            timing = float(timing_pred[i])
            strength = float(strength_pred[i])
            
            if conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH:
                signal = {
                    'timestamp': ts,
                    'pair': pair,
                    'direction': 'LONG' if direction == 2 else 'SHORT',
                    'confidence': conf,
                    'timing': timing,
                    'strength': strength,
                    'price': row['close'],
                    'atr': row['atr'],
                    'score': conf * timing * strength,
                }
                all_signals.append(signal)
                pair_signals += 1
        
        logger.info(f"  {pair}: {len(m5)} candles, {pair_signals} signals")
    
    logger.info(f"\nTotal: {len(all_signals)} signals from {len(price_data)} pairs")
    
    if not all_signals:
        logger.error("No signals generated!")
        return
    
    # Run backtest
    logger.info("\nRunning MEXC live simulation...")
    
    backtester = MEXCLiveBacktester(args.capital)
    signals_df = pd.DataFrame(all_signals)
    results = backtester.run(signals_df, price_data)
    
    # Per-pair breakdown
    if results.get('total_trades', 0) > 0:
        trades = results.get('trades', [])
        
        logger.info("\nPER-PAIR BREAKDOWN:")
        logger.info("-" * 60)
        
        pair_stats = {}
        for t in trades:
            p = t.pair
            if p not in pair_stats:
                pair_stats[p] = {'trades': 0, 'wins': 0, 'pnl': 0}
            pair_stats[p]['trades'] += 1
            if t.pnl_dollar > 0:
                pair_stats[p]['wins'] += 1
            pair_stats[p]['pnl'] += t.pnl_dollar
        
        sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        
        logger.info(f"{'Pair':<20} {'Trades':<8} {'WR%':<10} {'PnL$':<12}")
        logger.info("-" * 60)
        for pair, stats in sorted_pairs:
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            logger.info(f"{pair:<20} {stats['trades']:<8} {wr:<10.1f} ${stats['pnl']:<12.2f}")
        
        # Direction breakdown
        longs = [t for t in trades if t.direction == 'LONG']
        shorts = [t for t in trades if t.direction == 'SHORT']
        
        logger.info("\nDIRECTION BREAKDOWN:")
        logger.info(f"  LONG:  {len(longs)} trades, ${sum(t.pnl_dollar for t in longs):.2f} PnL")
        logger.info(f"  SHORT: {len(shorts)} trades, ${sum(t.pnl_dollar for t in shorts):.2f} PnL")
        
        # BE analysis
        be_trades = [t for t in trades if t.exit_reason in ['breakeven_stop', 'trailing_stop']]
        logger.info(f"\nBREAKEVEN/TRAILING ANALYSIS:")
        logger.info(f"  Trades that used BE/Trail: {len(be_trades)}/{len(trades)} ({len(be_trades)/len(trades)*100:.0f}%)")
        logger.info(f"  PnL from BE/Trail trades: ${sum(t.pnl_dollar for t in be_trades):.2f}")


if __name__ == "__main__":
    main()

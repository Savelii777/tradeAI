#!/usr/bin/env python3
"""
Fetch & Backtest V2 - Realistic Single-Position Backtest

This script:
1. Fetches fresh data from Binance for a specific date
2. Runs REALISTIC backtest with ONLY ONE position at a time
3. Uses proper RR 1:3 with 5% risk per trade

Key differences from previous version:
- ONE position at a time (like real trading)
- Chronological order across all pairs
- Exact PnL: SL=-5%, TP=+15%

Usage:
    python scripts/fetch_and_backtest_v2.py --date 2025-12-25 --model-path models/v2_improved
"""

import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from loguru import logger
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine


# ============================================================
# CONFIGURATION
# ============================================================

RISK_PCT = 0.05      # 5% риск на сделку
RR_RATIO = 3.0       # RR 1:3
SL_ATR_MULT = 1.5    # SL = 1.5 * ATR
MAX_LEVERAGE = 20.0  # Максимальное плечо
MAX_BARS = 50        # Максимум баров в сделке
FEE_PCT = 0.0002     # 0.02% комиссия (MEXC taker)


# ============================================================
# MODEL WRAPPER
# ============================================================

class BacktestModel:
    """Model wrapper for backtest."""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.direction_model = None
        self.timing_model = None
        self.entry_quality_model = None
        self.long_quality_model = None
        self.short_quality_model = None
        self.feature_names = None
        
    def load(self):
        """Load models."""
        logger.info(f"Loading models from {self.model_dir}")
        
        self.direction_model = joblib.load(self.model_dir / 'direction_model.joblib')
        self.timing_model = joblib.load(self.model_dir / 'timing_model.joblib')
        
        fn_path = self.model_dir / 'feature_names.joblib'
        if fn_path.exists():
            self.feature_names = joblib.load(fn_path)
        else:
            self.feature_names = self.direction_model.feature_name_
        
        eq_path = self.model_dir / 'entry_quality_model.joblib'
        if eq_path.exists():
            self.entry_quality_model = joblib.load(eq_path)
            self.long_quality_model = joblib.load(self.model_dir / 'long_quality_model.joblib')
            self.short_quality_model = joblib.load(self.model_dir / 'short_quality_model.joblib')
            logger.info("Entry quality models loaded")
        
        logger.info("Models loaded successfully")
    
    def get_signal(self, X: pd.DataFrame) -> Dict:
        """Get trading signal."""
        direction_proba = self.direction_model.predict_proba(X)[0]
        p_down, p_sideways, p_up = direction_proba
        
        timing = self.timing_model.predict_proba(X)[0][1]
        
        entry_quality = 0.5
        long_quality = None
        short_quality = None
        
        if self.entry_quality_model is not None:
            entry_quality = self.entry_quality_model.predict_proba(X)[0][1]
            long_quality = self.long_quality_model.predict_proba(X)[0][1]
            short_quality = self.short_quality_model.predict_proba(X)[0][1]
        
        return {
            'p_up': p_up,
            'p_down': p_down,
            'p_sideways': p_sideways,
            'timing': timing,
            'entry_quality': entry_quality,
            'long_quality': long_quality,
            'short_quality': short_quality
        }


# ============================================================
# DATA FETCHING
# ============================================================

async def fetch_binance_data(exchange, symbol: str, timeframe: str, since: int, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data from Binance."""
    try:
        # Convert symbol format
        binance_symbol = symbol.replace(':USDT', '').replace('/', '')
        
        ohlcv = await exchange.fetch_ohlcv(binance_symbol, timeframe, since=since, limit=limit)
        
        if not ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Binance: {e}")
        return pd.DataFrame()


async def fetch_pair_mtf(exchange, pair: str, start_date: str, end_date: str) -> Optional[Dict]:
    """Fetch multi-timeframe data for a pair."""
    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    logger.info(f"  Fetching {pair} from Binance...")
    
    # Fetch all timeframes
    df_1m = await fetch_binance_data(exchange, pair, '1m', since, limit=10000)
    if df_1m.empty:
        logger.warning(f"    No 1m data for {pair}")
        return None
    logger.info(f"    1m: {len(df_1m)} bars")
    
    df_5m = await fetch_binance_data(exchange, pair, '5m', since, limit=2000)
    logger.info(f"    5m: {len(df_5m)} bars")
    
    df_15m = await fetch_binance_data(exchange, pair, '15m', since, limit=700)
    logger.info(f"    15m: {len(df_15m)} bars")
    
    if df_5m.empty or df_15m.empty:
        return None
    
    return {'1': df_1m, '5': df_5m, '15': df_15m}


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
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
# SIGNAL DATACLASS
# ============================================================

@dataclass
class Signal:
    """Trading signal with all info needed for backtest."""
    pair: str
    timestamp: pd.Timestamp
    side: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    confidence: float
    entry_quality: float
    # References to data
    m5_df: pd.DataFrame = None
    entry_idx: int = 0


# ============================================================
# REALISTIC BACKTEST
# ============================================================

def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    max_bars: int = MAX_BARS
) -> Dict:
    """Simulate a trade and return result with EXACT exit price."""
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        
        if side == 'long':
            # Check SL first (conservative)
            if low <= stop_loss:
                return {
                    'outcome': 'stop_loss',
                    'exit_price': stop_loss,
                    'bars_held': i - entry_idx,
                    'exit_time': df.index[i]
                }
            if high >= take_profit:
                return {
                    'outcome': 'take_profit',
                    'exit_price': take_profit,
                    'bars_held': i - entry_idx,
                    'exit_time': df.index[i]
                }
        else:  # short
            # Check SL first (conservative)
            if high >= stop_loss:
                return {
                    'outcome': 'stop_loss',
                    'exit_price': stop_loss,
                    'bars_held': i - entry_idx,
                    'exit_time': df.index[i]
                }
            if low <= take_profit:
                return {
                    'outcome': 'take_profit',
                    'exit_price': take_profit,
                    'bars_held': i - entry_idx,
                    'exit_time': df.index[i]
                }
    
    # Time exit
    exit_idx = min(entry_idx + max_bars, len(df) - 1)
    exit_price = df['close'].iloc[exit_idx]
    
    return {
        'outcome': 'time_exit',
        'exit_price': exit_price,
        'bars_held': exit_idx - entry_idx,
        'exit_time': df.index[exit_idx]
    }


def calculate_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    outcome: str,
    sl_pct: float,
    tp_pct: float
) -> Tuple[float, float, float]:
    """
    Calculate PnL with proper leverage.
    
    Returns: (leverage, raw_pnl_pct, net_pnl_pct)
    
    For SL: net_pnl ≈ -5% (RISK_PCT)
    For TP: net_pnl ≈ +15% (RISK_PCT * RR_RATIO)
    """
    # Calculate leverage to achieve exact risk
    leverage = RISK_PCT / sl_pct
    leverage = min(round(leverage), MAX_LEVERAGE)
    leverage = max(leverage, 1.0)
    
    # Calculate raw PnL
    if side == 'long':
        raw_pnl_pct = (exit_price - entry_price) / entry_price
    else:
        raw_pnl_pct = (entry_price - exit_price) / entry_price
    
    # Apply leverage
    leveraged_pnl = raw_pnl_pct * leverage
    
    # Deduct fees (entry + exit)
    fee_total = FEE_PCT * leverage * 2
    net_pnl = leveraged_pnl - fee_total
    
    return leverage, raw_pnl_pct, net_pnl


async def run_backtest(
    model: BacktestModel,
    pairs: List[str],
    target_date: str,
    min_confidence: float = 0.50,
    min_timing: float = 0.50,
    min_entry_quality: float = 0.50
) -> List[Dict]:
    """
    Run REALISTIC backtest with ONE position at a time.
    
    Process:
    1. Fetch all data
    2. Generate all features
    3. Collect all potential signals
    4. Sort by time
    5. Execute one at a time, waiting for exit before next entry
    """
    mtf_engine = MTFFeatureEngine()
    
    # Need extra days before for warmup
    target = datetime.strptime(target_date, '%Y-%m-%d')
    start_date = (target - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (target + timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # ============================================================
    # PHASE 1: Collect all potential signals from all pairs
    # ============================================================
    
    all_signals: List[Signal] = []
    
    for pair in pairs:
        logger.info(f"\nProcessing {pair}...")
        
        try:
            # Fetch data
            data = await fetch_pair_mtf(exchange, pair, start_date, end_date)
            if data is None:
                continue
            
            m1_df = data['1']
            m5_df = data['5']
            m15_df = data['15']
            
            if len(m5_df) < 200:
                logger.warning(f"Not enough data for {pair}")
                continue
            
            # Generate features
            features = mtf_engine.align_timeframes(m1_df, m5_df, m15_df)
            
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = pd.Categorical(features[col]).codes
            features = features.fillna(0)
            
            # Add missing features
            for feat in model.feature_names:
                if feat not in features.columns:
                    features[feat] = 0
            
            X = features[model.feature_names]
            
            # Calculate ATR
            atr = calculate_atr(m5_df)
            
            # Filter to target date only
            date_mask = m5_df.index.date == target.date()
            target_indices = m5_df.index[date_mask]
            
            if len(target_indices) == 0:
                continue
            
            logger.info(f"  {len(target_indices)} bars on {target_date}")
            
            # Scan for signals (every 15 bars = ~75 minutes)
            for idx, timestamp in enumerate(target_indices):
                if idx % 15 != 0:
                    continue
                if timestamp not in X.index:
                    continue
                
                current_features = X.loc[[timestamp]]
                current_price = m5_df.loc[timestamp, 'close']
                current_atr = atr.loc[timestamp]
                
                signal = model.get_signal(current_features)
                
                side = None
                confidence = 0
                quality = 0
                
                # Check for LONG
                if signal['p_up'] >= min_confidence and signal['p_up'] > signal['p_down']:
                    if signal['timing'] >= min_timing:
                        quality = signal['long_quality'] if signal['long_quality'] else signal['entry_quality']
                        if quality >= min_entry_quality:
                            side = 'long'
                            confidence = signal['p_up']
                
                # Check for SHORT
                elif signal['p_down'] >= min_confidence and signal['p_down'] > signal['p_up']:
                    if signal['timing'] >= min_timing:
                        quality = signal['short_quality'] if signal['short_quality'] else signal['entry_quality']
                        if quality >= min_entry_quality:
                            side = 'short'
                            confidence = signal['p_down']
                
                if side is None:
                    continue
                
                # Calculate SL/TP
                sl_distance = current_atr * SL_ATR_MULT
                tp_distance = sl_distance * RR_RATIO
                
                if side == 'long':
                    stop_loss = current_price - sl_distance
                    take_profit = current_price + tp_distance
                else:
                    stop_loss = current_price + sl_distance
                    take_profit = current_price - tp_distance
                
                try:
                    entry_idx = m5_df.index.get_loc(timestamp)
                except:
                    continue
                
                # Create signal object
                sig = Signal(
                    pair=pair,
                    timestamp=timestamp,
                    side=side,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    atr=current_atr,
                    confidence=confidence,
                    entry_quality=quality,
                    m5_df=m5_df,
                    entry_idx=entry_idx
                )
                all_signals.append(sig)
                
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
            continue
    
    await exchange.close()
    
    # ============================================================
    # PHASE 2: Sort signals by time and execute ONE at a time
    # ============================================================
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Collected {len(all_signals)} potential signals")
    logger.info(f"{'='*60}")
    
    # Sort by timestamp
    all_signals.sort(key=lambda s: s.timestamp)
    
    trades = []
    current_position = None
    position_exit_time = None
    
    for sig in all_signals:
        # Skip if we have an open position
        if position_exit_time is not None and sig.timestamp < position_exit_time:
            continue
        
        # Simulate this trade
        result = simulate_trade(
            sig.m5_df,
            sig.entry_idx,
            sig.side,
            sig.entry_price,
            sig.stop_loss,
            sig.take_profit,
            max_bars=MAX_BARS
        )
        
        # Calculate PnL
        sl_pct = abs(sig.stop_loss - sig.entry_price) / sig.entry_price
        tp_pct = abs(sig.take_profit - sig.entry_price) / sig.entry_price
        
        leverage, raw_pnl, net_pnl = calculate_pnl(
            sig.side,
            sig.entry_price,
            result['exit_price'],
            result['outcome'],
            sl_pct,
            tp_pct
        )
        
        # Record trade
        trade = {
            'pair': sig.pair.split('/')[0],  # Just symbol
            'entry_time': sig.timestamp,
            'exit_time': result['exit_time'],
            'side': sig.side.upper(),
            'entry_price': sig.entry_price,
            'exit_price': result['exit_price'],
            'stop_loss': sig.stop_loss,
            'take_profit': sig.take_profit,
            'confidence': sig.confidence,
            'entry_quality': sig.entry_quality,
            'outcome': result['outcome'],
            'leverage': leverage,
            'raw_pnl_pct': raw_pnl * 100,
            'net_pnl_pct': net_pnl * 100,
            'bars_held': result['bars_held']
        }
        trades.append(trade)
        
        # Set exit time to prevent overlapping trades
        position_exit_time = result['exit_time']
        
        # Log
        emoji = "🟢" if trade['net_pnl_pct'] > 0 else "🔴"
        logger.info(
            f"{emoji} {trade['side']} {trade['pair']} @ {sig.timestamp.strftime('%H:%M')} | "
            f"Exit: {result['outcome']} @ {result['exit_time'].strftime('%H:%M')} | "
            f"Lev: {leverage:.0f}x | "
            f"PnL: {trade['net_pnl_pct']:+.2f}%"
        )
    
    return trades


def print_summary(trades: List[Dict]):
    """Print backtest summary."""
    print("\n" + "=" * 70)
    print("TRADES SUMMARY (ONE POSITION AT A TIME)")
    print("=" * 70)
    print(f"{'Pair':<8} {'Entry':>6} {'Exit':>6} {'Side':<6} {'Conf':>5} {'Outcome':<12} {'Lev':>4} {'PnL':>8}")
    print("-" * 70)
    
    for t in trades:
        entry_time = t['entry_time'].strftime('%H:%M')
        exit_time = t['exit_time'].strftime('%H:%M')
        print(f"{t['pair']:<8} {entry_time:>6} {exit_time:>6} {t['side']:<6} "
              f"{t['confidence']:.0%}  {t['outcome']:<12} {t['leverage']:>3.0f}x {t['net_pnl_pct']:>+7.2f}%")
    
    print("=" * 70)
    
    if not trades:
        print("No trades executed")
        return
    
    total_pnl = sum(t['net_pnl_pct'] for t in trades)
    wins = sum(1 for t in trades if t['net_pnl_pct'] > 0)
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100 if trades else 0
    
    avg_win = np.mean([t['net_pnl_pct'] for t in trades if t['net_pnl_pct'] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t['net_pnl_pct'] for t in trades if t['net_pnl_pct'] <= 0]) if losses > 0 else 0
    
    # By outcome
    tp_trades = [t for t in trades if t['outcome'] == 'take_profit']
    sl_trades = [t for t in trades if t['outcome'] == 'stop_loss']
    te_trades = [t for t in trades if t['outcome'] == 'time_exit']
    
    print("\nSTATISTICS")
    print("=" * 70)
    print(f"Total Trades:     {len(trades)}")
    print(f"Wins / Losses:    {wins} / {losses}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Total PnL:        {total_pnl:+.2f}%")
    print(f"Avg Win:          {avg_win:+.2f}%")
    print(f"Avg Loss:         {avg_loss:+.2f}%")
    
    print("\nBy Outcome:")
    if tp_trades:
        tp_pnl = np.mean([t['net_pnl_pct'] for t in tp_trades])
        print(f"  Take Profit:    {len(tp_trades)} trades, avg PnL: {tp_pnl:+.2f}%")
    if sl_trades:
        sl_pnl = np.mean([t['net_pnl_pct'] for t in sl_trades])
        print(f"  Stop Loss:      {len(sl_trades)} trades, avg PnL: {sl_pnl:+.2f}%")
    if te_trades:
        te_pnl = np.mean([t['net_pnl_pct'] for t in te_trades])
        print(f"  Time Exit:      {len(te_trades)} trades, avg PnL: {te_pnl:+.2f}%")
    
    print("=" * 70)
    
    # Verify RR logic
    print("\nRR 1:3 VERIFICATION (Risk 5%, Reward 15%)")
    print("-" * 70)
    for t in trades:
        expected = ""
        if t['outcome'] == 'take_profit':
            expected = f"Expected: ~+{RISK_PCT * RR_RATIO * 100:.0f}%"
        elif t['outcome'] == 'stop_loss':
            expected = f"Expected: ~-{RISK_PCT * 100:.0f}%"
        print(f"  {t['pair']:<6} {t['outcome']:<12} {t['net_pnl_pct']:+.2f}% {expected}")


async def main():
    parser = argparse.ArgumentParser(description="Realistic Backtest with ONE position at a time")
    parser.add_argument("--date", type=str, default="2025-12-25",
                        help="Date to backtest (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=str, default="./models/v2_improved",
                        help="Path to model")
    parser.add_argument("--pairs", type=int, default=20,
                        help="Number of pairs to test")
    parser.add_argument("--min-confidence", type=float, default=0.50,
                        help="Minimum direction confidence")
    parser.add_argument("--min-entry-quality", type=float, default=0.50,
                        help="Minimum entry quality")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("REALISTIC BACKTEST - ONE POSITION AT A TIME")
    print("=" * 70)
    print(f"Date:             {args.date}")
    print(f"Model:            {args.model_path}")
    print(f"Risk per trade:   {RISK_PCT*100:.0f}%")
    print(f"RR Ratio:         1:{RR_RATIO:.0f}")
    print(f"SL:               {SL_ATR_MULT} ATR")
    print(f"Expected PnL:")
    print(f"  At SL:          -{RISK_PCT*100:.0f}%")
    print(f"  At TP:          +{RISK_PCT*RR_RATIO*100:.0f}%")
    print("=" * 70)
    
    # Load pairs
    import json
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_list.json'
    with open(pairs_file) as f:
        pairs_data = json.load(f)
    all_pairs = [p['symbol'] for p in pairs_data['pairs']]
    pairs = all_pairs[:args.pairs]
    
    # Load model
    model = BacktestModel(args.model_path)
    model.load()
    
    # Run backtest
    trades = await run_backtest(
        model=model,
        pairs=pairs,
        target_date=args.date,
        min_confidence=args.min_confidence,
        min_entry_quality=args.min_entry_quality
    )
    
    # Print summary
    print_summary(trades)


if __name__ == '__main__':
    asyncio.run(main())

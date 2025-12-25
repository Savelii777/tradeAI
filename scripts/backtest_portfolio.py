#!/usr/bin/env python3
"""
Portfolio Backtest - TRUE Single Position Across All Pairs

This backtest enforces a GLOBAL limit of 1 position at a time,
meaning only one trade can be open across ALL pairs simultaneously.

Usage:
    docker-compose -f docker/docker-compose.yml run --rm trading-bot \
        python scripts/backtest_portfolio.py --pairs 20 --days 1 --capital 100
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine


# ============================================================
# V1 MODEL
# ============================================================

class V1Model:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.direction_model = None
        self.timing_model = None
        self.feature_names = None
        
    def load(self):
        self.direction_model = joblib.load(self.model_dir / 'direction_model.joblib')
        self.timing_model = joblib.load(self.model_dir / 'timing_model.joblib')
        self.feature_names = self.direction_model.feature_name_
        print(f"✅ Model loaded from {self.model_dir}")
        
    def predict(self, X: pd.DataFrame) -> Dict:
        # Ensure all features exist
        for feat in self.feature_names:
            if feat not in X.columns:
                X[feat] = 0
        
        X_model = X[self.feature_names]
        
        dir_proba = self.direction_model.predict_proba(X_model)[0]
        timing_proba = self.timing_model.predict_proba(X_model)[0][1]
        
        return {
            'direction_proba': dir_proba,  # [down, sideways, up]
            'timing_proba': timing_proba
        }


# ============================================================
# PAIRS
# ============================================================

V1_PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT',
    'DOT/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT', 'UNI/USDT:USDT',
    'AAVE/USDT:USDT', 'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT',
    'OP/USDT:USDT', 'TONCOIN/USDT:USDT', 'ARB/USDT:USDT', 'MATIC/USDT:USDT'
]


# ============================================================
# DATA LOADING
# ============================================================

def load_pair_data(pair: str, data_dir: str, days: int) -> Optional[pd.DataFrame]:
    """Load 5m data for a pair and filter to last N days."""
    safe_symbol = pair.replace('/', '_').replace(':', '_')
    filepath = Path(data_dir) / f"{safe_symbol}_5m.csv"
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Filter to last N days
    end_time = df.index[-1]
    start_time = end_time - timedelta(days=days)
    df = df[df.index >= start_time].copy()
    
    if len(df) < 50:
        return None
    
    return df


def load_mtf_data(pair: str, data_dir: str, days: int) -> Optional[Dict[str, pd.DataFrame]]:
    """Load M1, M5, M15 data for feature generation."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        safe_symbol = pair.replace('/', '_').replace(':', '_')
        filepath = Path(data_dir) / f"{safe_symbol}_{tf}.csv"
        
        if not filepath.exists():
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Filter to last N days + buffer for feature calculation
        end_time = df.index[-1]
        start_time = end_time - timedelta(days=days + 2)  # Extra buffer
        df = df[df.index >= start_time].copy()
        
        data[tf.replace('m', '')] = df
    
    return data


# ============================================================
# PORTFOLIO BACKTESTER
# ============================================================

class PortfolioBacktester:
    def __init__(
        self,
        capital: float = 100,
        risk_pct: float = 0.05,
        max_leverage: float = 20.0,
        sl_atr: float = 1.5,
        tp_rr: float = 2.0,
        min_confidence: float = 0.50,
        min_timing: float = 0.01,
        max_holding_bars: int = 50,
        entry_fee: float = 0.0002,
        exit_fee: float = 0.0005
    ):
        self.initial_capital = capital
        self.capital = capital
        self.risk_pct = risk_pct
        self.max_leverage = max_leverage
        self.sl_atr = sl_atr
        self.tp_rr = tp_rr
        self.min_confidence = min_confidence
        self.min_timing = min_timing
        self.max_holding_bars = max_holding_bars
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee
        
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def run(
        self,
        all_data: Dict[str, pd.DataFrame],
        all_features: Dict[str, pd.DataFrame],
        model: V1Model
    ):
        """
        Run portfolio backtest with GLOBAL 1 position limit.
        
        Args:
            all_data: Dict of pair -> DataFrame with OHLCV
            all_features: Dict of pair -> DataFrame with features
            model: V1Model instance
        """
        print(f"\n{'='*70}")
        print(f"🚀 PORTFOLIO BACKTEST - Global 1 Position Limit")
        print(f"{'='*70}")
        print(f"Capital: ${self.capital:.2f}")
        print(f"Pairs: {len(all_data)}")
        print(f"Risk: {self.risk_pct*100:.1f}% per trade")
        print(f"{'='*70}\n")
        
        # Build unified timeline of all bars across all pairs
        timeline = []
        for pair, df in all_data.items():
            if pair not in all_features:
                continue
            features = all_features[pair]
            
            for timestamp in df.index:
                if timestamp in features.index:
                    timeline.append({
                        'timestamp': timestamp,
                        'pair': pair,
                        'bar': df.loc[timestamp],
                        'features': features.loc[[timestamp]]
                    })
        
        # Sort by timestamp
        timeline = sorted(timeline, key=lambda x: x['timestamp'])
        print(f"📊 Total bars in timeline: {len(timeline)}")
        
        # Track position entry bar for holding time
        position_entry_idx = None
        
        # Iterate through timeline
        for i, point in enumerate(timeline):
            timestamp = point['timestamp']
            pair = point['pair']
            bar = point['bar']
            features = point['features']
            
            # Check exit if we have a position
            if self.position is not None:
                # Only check exit on the same pair
                if self.position['pair'] == pair:
                    bars_held = i - position_entry_idx
                    
                    should_exit, reason = self._check_exit(
                        bar['high'], bar['low'], bars_held
                    )
                    
                    if should_exit:
                        self._close_position(bar, reason, timestamp)
                        position_entry_idx = None
            
            # Try to open new position if none exists
            if self.position is None:
                try:
                    pred = model.predict(features)
                    
                    dir_proba = pred['direction_proba']
                    timing = pred['timing_proba']
                    
                    p_down, p_sideways, p_up = dir_proba
                    max_prob = max(dir_proba)
                    direction_idx = np.argmax(dir_proba)
                    
                    # Check signal conditions
                    if max_prob >= self.min_confidence and timing >= self.min_timing:
                        if direction_idx == 2:  # LONG
                            self._open_position(pair, 'long', bar, timestamp, p_up, timing, all_data[pair], i)
                            position_entry_idx = i
                        elif direction_idx == 0:  # SHORT
                            self._open_position(pair, 'short', bar, timestamp, p_down, timing, all_data[pair], i)
                            position_entry_idx = i
                            
                except Exception as e:
                    continue
            
            # Track equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.capital + (self._unrealized_pnl(bar) if self.position and self.position['pair'] == pair else 0)
            })
        
        # Close any remaining position
        if self.position is not None:
            last_pair = self.position['pair']
            if last_pair in all_data:
                last_bar = all_data[last_pair].iloc[-1]
                self._close_position(last_bar, 'end_of_data', all_data[last_pair].index[-1])
        
        return self._calculate_results()
    
    def _open_position(self, pair: str, side: str, bar, timestamp, confidence: float, timing: float, df: pd.DataFrame, bar_idx: int):
        """Open a new position."""
        entry_price = bar['close']
        
        # Calculate ATR for stop loss
        lookback = min(14, bar_idx)
        if lookback > 0:
            recent_idx = df.index.get_loc(timestamp)
            if recent_idx >= lookback:
                recent = df.iloc[recent_idx-lookback:recent_idx+1]
                atr = (recent['high'] - recent['low']).mean()
            else:
                atr = bar['high'] - bar['low']
        else:
            atr = bar['high'] - bar['low']
        
        # Calculate SL/TP
        stop_distance = atr * self.sl_atr
        stop_loss_pct = stop_distance / entry_price
        
        if side == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * self.tp_rr)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * self.tp_rr)
        
        # Calculate position size
        leverage = min(self.risk_pct / stop_loss_pct, self.max_leverage)
        position_value = self.capital * leverage
        size = position_value / entry_price
        
        # Entry fee
        entry_fee = position_value * self.entry_fee
        
        self.position = {
            'pair': pair,
            'side': side,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
            'leverage': leverage,
            'position_value': position_value,
            'confidence': confidence,
            'timing': timing,
            'entry_fee': entry_fee
        }
        
        print(f"📈 OPEN {side.upper()} {pair} @ ${entry_price:.4f} | SL: ${stop_loss:.4f} TP: ${take_profit:.4f} | Lev: {leverage:.1f}x")
    
    def _check_exit(self, high: float, low: float, bars_held: int) -> tuple:
        """Check if position should be closed."""
        if bars_held >= self.max_holding_bars:
            return True, 'time_exit'
        
        if self.position['side'] == 'long':
            if low <= self.position['stop_loss']:
                return True, 'stop_loss'
            if high >= self.position['take_profit']:
                return True, 'take_profit'
        else:
            if high >= self.position['stop_loss']:
                return True, 'stop_loss'
            if low <= self.position['take_profit']:
                return True, 'take_profit'
        
        return False, None
    
    def _close_position(self, bar, reason: str, timestamp):
        """Close current position."""
        if reason == 'stop_loss':
            exit_price = self.position['stop_loss']
        elif reason == 'take_profit':
            exit_price = self.position['take_profit']
        else:
            exit_price = bar['close']
        
        # Calculate PnL
        if self.position['side'] == 'long':
            price_change = (exit_price - self.position['entry_price']) / self.position['entry_price']
        else:
            price_change = (self.position['entry_price'] - exit_price) / self.position['entry_price']
        
        raw_pnl = price_change * self.position['position_value']
        exit_fee = self.position['position_value'] * self.exit_fee
        pnl = raw_pnl - self.position['entry_fee'] - exit_fee
        
        self.capital += pnl
        
        result_emoji = "✅" if pnl > 0 else "❌"
        print(f"{result_emoji} CLOSE {self.position['side'].upper()} {self.position['pair']} @ ${exit_price:.4f} | {reason} | PnL: ${pnl:+.2f}")
        
        self.trades.append({
            'pair': self.position['pair'],
            'side': self.position['side'],
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': self.position['stop_loss'],
            'take_profit': self.position['take_profit'],
            'leverage': self.position['leverage'],
            'position_value': self.position['position_value'],
            'pnl': pnl,
            'exit_reason': reason,
            'confidence': self.position['confidence'],
            'timing': self.position['timing']
        })
        
        self.position = None
    
    def _unrealized_pnl(self, bar) -> float:
        """Calculate unrealized PnL for current position."""
        if self.position is None:
            return 0
        
        current_price = bar['close']
        
        if self.position['side'] == 'long':
            price_change = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:
            price_change = (self.position['entry_price'] - current_price) / self.position['entry_price']
        
        return price_change * self.position['position_value']
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if not self.trades:
            return {'total_trades': 0}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Calculate max drawdown
        equity = pd.Series([e['equity'] for e in self.equity_curve])
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': total_pnl,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'final_capital': self.capital,
            'max_drawdown': max_dd,
            'avg_pnl_per_trade': total_pnl / len(self.trades) if self.trades else 0,
            'trades': self.trades
        }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Portfolio Backtest - Global 1 Position")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs")
    parser.add_argument("--days", type=int, default=1, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=100, help="Initial capital")
    parser.add_argument("--model-path", type=str, default="./models/v1_fresh")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--risk-pct", type=float, default=0.05, help="Risk per trade")
    parser.add_argument("--max-leverage", type=float, default=20.0)
    parser.add_argument("--min-conf", type=float, default=0.50)
    parser.add_argument("--min-timing", type=float, default=0.01)
    
    args = parser.parse_args()
    
    # Load model
    model = V1Model(args.model_path)
    model.load()
    
    # Load MTF feature engine
    mtf_engine = MTFFeatureEngine()
    
    # Load data for all pairs
    pairs = V1_PAIRS[:args.pairs]
    
    all_data = {}
    all_features = {}
    
    print(f"\n📂 Loading data for {len(pairs)} pairs...")
    
    for pair in pairs:
        # Load MTF data
        mtf_data = load_mtf_data(pair, args.data_dir, args.days)
        if mtf_data is None:
            print(f"  ❌ {pair}: No data")
            continue
        
        # Generate features
        try:
            features = mtf_engine.align_timeframes(
                mtf_data['1'], mtf_data['5'], mtf_data['15']
            )
            
            if len(features) < 10:
                print(f"  ❌ {pair}: Not enough features")
                continue
            
            # Get 5m data aligned with features
            m5_df = mtf_data['5'].loc[mtf_data['5'].index.isin(features.index)]
            
            # Filter to actual test period
            end_time = m5_df.index[-1]
            start_time = end_time - timedelta(days=args.days)
            m5_df = m5_df[m5_df.index >= start_time]
            features = features[features.index >= start_time]
            
            all_data[pair] = m5_df
            all_features[pair] = features
            print(f"  ✅ {pair}: {len(m5_df)} bars")
            
        except Exception as e:
            print(f"  ❌ {pair}: Error - {e}")
            continue
    
    print(f"\n✅ Loaded {len(all_data)} pairs")
    
    if not all_data:
        print("❌ No data loaded!")
        return 1
    
    # Run backtest
    backtester = PortfolioBacktester(
        capital=args.capital,
        risk_pct=args.risk_pct,
        max_leverage=args.max_leverage,
        min_confidence=args.min_conf,
        min_timing=args.min_timing
    )
    
    results = backtester.run(all_data, all_features, model)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 PORTFOLIO BACKTEST RESULTS")
    print(f"{'='*70}")
    
    if results['total_trades'] == 0:
        print("No trades executed!")
        return 1
    
    print(f"\n💰 CAPITAL:")
    print(f"  Initial:  ${args.capital:.2f}")
    print(f"  Final:    ${results['final_capital']:.2f}")
    print(f"  P&L:      ${results['total_pnl']:+.2f}")
    print(f"  Return:   {results['return_pct']:+.2f}%")
    
    print(f"\n📈 TRADES:")
    print(f"  Total:    {results['total_trades']}")
    print(f"  Wins:     {results['wins']} ({results['win_rate']:.1f}%)")
    print(f"  Losses:   {results['losses']}")
    print(f"  Avg PnL:  ${results['avg_pnl_per_trade']:.2f}")
    
    print(f"\n⚠️  RISK:")
    print(f"  Max DD:   {results['max_drawdown']:.2f}%")
    
    print(f"\n📋 TRADE LOG:")
    print(f"{'─'*90}")
    
    for i, t in enumerate(results['trades'], 1):
        emoji = "✅" if t['pnl'] > 0 else "❌"
        side_emoji = "🟢" if t['side'] == 'long' else "🔴"
        pair_short = t['pair'].replace('/USDT:USDT', '')
        
        entry_time = t['entry_time'].strftime('%H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[-8:-3]
        exit_time = t['exit_time'].strftime('%H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[-8:-3]
        
        print(f"  {i:2}. {emoji} {side_emoji} {t['side'].upper():5} {pair_short:8} | {entry_time}→{exit_time} | Entry: ${t['entry_price']:.2f} Exit: ${t['exit_price']:.2f} | {t['exit_reason']:11} | PnL: ${t['pnl']:+.2f}")
    
    print(f"{'─'*90}")
    
    # Exit reason breakdown
    print(f"\n📊 EXIT REASONS:")
    reasons = {}
    for t in results['trades']:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl']
    
    for reason, data in reasons.items():
        print(f"  {reason}: {data['count']} trades, PnL: ${data['pnl']:+.2f}")
    
    # Pair breakdown
    print(f"\n📊 BY PAIR:")
    pairs_stats = {}
    for t in results['trades']:
        p = t['pair'].replace('/USDT:USDT', '')
        if p not in pairs_stats:
            pairs_stats[p] = {'count': 0, 'pnl': 0, 'wins': 0}
        pairs_stats[p]['count'] += 1
        pairs_stats[p]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            pairs_stats[p]['wins'] += 1
    
    for pair, data in sorted(pairs_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  {pair:8}: {data['count']} trades, WR: {wr:.0f}%, PnL: ${data['pnl']:+.2f}")
    
    print(f"\n{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

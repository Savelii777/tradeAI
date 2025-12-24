"""
Single Position Backtester for Multi-Pair Trading.
Only one position at a time across all pairs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade."""
    trade_id: int
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_time: datetime
    entry_price: float
    position_size: float
    leverage: float
    stop_loss: float
    take_profit: float
    signal_score: float
    skipped_signals: int = 0
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    balance_after: Optional[float] = None


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    probability: float
    expected_strength: float
    score: float  # probability * expected_strength
    timestamp: datetime
    entry_price: float


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.05  # 5% SL
    reward_ratio: float = 3.0  # 1:3 RR -> 15% TP
    commission: float = 0.0004  # 0.04%
    slippage: float = 0.0001  # 0.01%
    min_probability: float = 0.55
    min_score: float = 0.3


class SinglePositionBacktester:
    """
    Backtester with single position constraint.
    Only one position open at a time across all pairs.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # State
        self.balance = self.config.initial_capital
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.skipped_signals: List[Signal] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_counter = 0
        
        # Statistics
        self.stats = {
            'total_signals': 0,
            'filtered_signals': 0,
            'executed_trades': 0,
            'skipped_due_to_position': 0,
            'signals_per_symbol': {},
            'trades_per_symbol': {}
        }
    
    def reset(self):
        """Reset backtester state."""
        self.balance = self.config.initial_capital
        self.current_trade = None
        self.trades = []
        self.skipped_signals = []
        self.equity_curve = []
        self.trade_counter = 0
        self.stats = {
            'total_signals': 0,
            'filtered_signals': 0,
            'executed_trades': 0,
            'skipped_due_to_position': 0,
            'signals_per_symbol': {},
            'trades_per_symbol': {}
        }
    
    def _calculate_leverage(self, stop_distance_pct: float) -> float:
        """Calculate leverage to achieve desired risk per trade."""
        # risk_per_trade = stop_distance * leverage
        # leverage = risk_per_trade / stop_distance
        leverage = self.config.risk_per_trade / stop_distance_pct
        return min(max(leverage, 1.0), 20.0)  # Clamp 1-20x
    
    def _apply_slippage(self, price: float, direction: int, is_entry: bool) -> float:
        """Apply slippage to price."""
        slip = self.config.slippage
        if is_entry:
            # Entry: buy higher, sell lower
            return price * (1 + slip * direction)
        else:
            # Exit: buy lower, sell higher (opposite)
            return price * (1 - slip * direction)
    
    def _get_signals_at_time(
        self,
        timestamp: datetime,
        all_data: Dict[str, pd.DataFrame],
        all_predictions: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """Get all signals at a specific timestamp."""
        signals = []
        
        for symbol, predictions in all_predictions.items():
            if timestamp not in predictions.index:
                continue
            
            pred = predictions.loc[timestamp]
            data = all_data[symbol]
            
            if timestamp not in data.index:
                continue
            
            candle = data.loc[timestamp]
            
            # Get prediction values
            direction = int(pred.get('direction', 0))
            probability = float(pred.get('direction_prob', 0.5))
            strength = float(pred.get('strength', 1.0))
            
            if direction == 0:
                continue
            
            self.stats['total_signals'] += 1
            self.stats['signals_per_symbol'][symbol] = \
                self.stats['signals_per_symbol'].get(symbol, 0) + 1
            
            # Filter by probability threshold
            if probability < self.config.min_probability:
                self.stats['filtered_signals'] += 1
                continue
            
            score = probability * abs(strength)
            
            if score < self.config.min_score:
                self.stats['filtered_signals'] += 1
                continue
            
            signals.append(Signal(
                symbol=symbol,
                direction=direction,
                probability=probability,
                expected_strength=strength,
                score=score,
                timestamp=timestamp,
                entry_price=float(candle['close'])
            ))
        
        return signals
    
    def _open_position(self, signal: Signal, skipped_count: int) -> Trade:
        """Open a new position."""
        self.trade_counter += 1
        
        # Calculate stop loss distance (use ATR-based or fixed)
        stop_distance_pct = 0.005  # 0.5% default stop
        
        leverage = self._calculate_leverage(stop_distance_pct)
        
        # Apply slippage to entry
        entry_price = self._apply_slippage(signal.entry_price, signal.direction, is_entry=True)
        
        # Calculate SL and TP prices
        if signal.direction == 1:  # Long
            stop_loss = entry_price * (1 - stop_distance_pct)
            take_profit = entry_price * (1 + stop_distance_pct * self.config.reward_ratio)
        else:  # Short
            stop_loss = entry_price * (1 + stop_distance_pct)
            take_profit = entry_price * (1 - stop_distance_pct * self.config.reward_ratio)
        
        # Calculate position size (100% of balance * leverage)
        position_value = self.balance * leverage
        position_size = position_value / entry_price
        
        # Apply entry commission
        commission = self.balance * self.config.commission
        self.balance -= commission
        
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_time=signal.timestamp,
            entry_price=entry_price,
            position_size=position_size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_score=signal.score,
            skipped_signals=skipped_count
        )
        
        self.current_trade = trade
        self.stats['executed_trades'] += 1
        self.stats['trades_per_symbol'][signal.symbol] = \
            self.stats['trades_per_symbol'].get(signal.symbol, 0) + 1
        
        logger.debug(f"Opened {signal.direction} {signal.symbol} @ {entry_price:.4f}, "
                     f"SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
        
        return trade
    
    def _check_exit(self, trade: Trade, candle: pd.Series, timestamp: datetime) -> Optional[str]:
        """Check if position should be closed."""
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        if trade.direction == 1:  # Long
            # Check stop loss
            if low <= trade.stop_loss:
                return 'SL'
            # Check take profit
            if high >= trade.take_profit:
                return 'TP'
        else:  # Short
            # Check stop loss
            if high >= trade.stop_loss:
                return 'SL'
            # Check take profit
            if low <= trade.take_profit:
                return 'TP'
        
        return None
    
    def _close_position(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """Close current position."""
        # Apply slippage
        exit_price = self._apply_slippage(exit_price, trade.direction, is_entry=False)
        
        # Calculate PnL
        if trade.direction == 1:  # Long
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:  # Short
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        # Apply leverage
        pnl_pct *= trade.leverage
        
        # Calculate USD PnL
        pnl_usd = self.balance * pnl_pct
        
        # Apply exit commission
        commission = self.balance * self.config.commission
        
        # Update balance
        self.balance += pnl_usd - commission
        
        # Update trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = pnl_pct * 100  # As percentage
        trade.pnl_usd = pnl_usd - commission
        trade.balance_after = self.balance
        
        self.trades.append(trade)
        self.current_trade = None
        
        logger.debug(f"Closed {trade.symbol} @ {exit_price:.4f}, "
                     f"PnL: {trade.pnl_pct:.2f}%, Balance: ${self.balance:.2f}")
    
    def run(
        self,
        all_data: Dict[str, pd.DataFrame],
        all_predictions: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Run backtest on multiple pairs.
        
        Args:
            all_data: Dict mapping symbol to OHLCV DataFrame
            all_predictions: Dict mapping symbol to predictions DataFrame
        
        Returns:
            Backtest results dictionary
        """
        self.reset()
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in all_data.values():
            all_timestamps.update(df.index.tolist())
        
        timestamps = sorted(all_timestamps)
        logger.info(f"Running backtest on {len(timestamps)} timestamps, {len(all_data)} pairs")
        
        for i, ts in enumerate(timestamps):
            # Record equity
            self.equity_curve.append((ts, self.balance))
            
            # Check if we have an open position
            if self.current_trade is not None:
                symbol = self.current_trade.symbol
                if symbol in all_data and ts in all_data[symbol].index:
                    candle = all_data[symbol].loc[ts]
                    
                    # Check for exit
                    exit_reason = self._check_exit(self.current_trade, candle, ts)
                    if exit_reason:
                        if exit_reason == 'SL':
                            exit_price = self.current_trade.stop_loss
                        else:  # TP
                            exit_price = self.current_trade.take_profit
                        
                        self._close_position(self.current_trade, exit_price, ts, exit_reason)
            
            # If no position, look for signals
            if self.current_trade is None:
                signals = self._get_signals_at_time(ts, all_data, all_predictions)
                
                if signals:
                    # Sort by score descending
                    signals.sort(key=lambda s: s.score, reverse=True)
                    
                    # Take best signal
                    best_signal = signals[0]
                    skipped = len(signals) - 1
                    
                    # Record skipped signals
                    for sig in signals[1:]:
                        self.skipped_signals.append(sig)
                        self.stats['skipped_due_to_position'] += 1
                    
                    # Open position
                    self._open_position(best_signal, skipped)
            else:
                # Position is open, record any signals we're missing
                signals = self._get_signals_at_time(ts, all_data, all_predictions)
                for sig in signals:
                    self.skipped_signals.append(sig)
                    self.stats['skipped_due_to_position'] += 1
            
            # Progress logging
            if (i + 1) % 10000 == 0:
                logger.info(f"Progress: {i+1}/{len(timestamps)} ({(i+1)/len(timestamps)*100:.1f}%)")
        
        # Close any remaining position at last price
        if self.current_trade is not None:
            symbol = self.current_trade.symbol
            last_ts = timestamps[-1]
            if symbol in all_data:
                last_price = all_data[symbol].iloc[-1]['close']
                self._close_position(self.current_trade, last_price, last_ts, 'END')
        
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results and statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'error': 'No trades executed'
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl_pct > 0]
        losing_trades = [t for t in self.trades if t.pnl_pct <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        # PnL
        total_pnl = sum(t.pnl_usd for t in self.trades)
        total_return = (self.balance - self.config.initial_capital) / self.config.initial_capital * 100
        
        # Profit factor
        gross_profit = sum(t.pnl_usd for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl_usd for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity = [e[1] for e in self.equity_curve]
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio (assuming daily returns)
        if len(self.trades) > 1:
            returns = [t.pnl_pct for t in self.trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Average trade duration
        durations = []
        for t in self.trades:
            if t.entry_time and t.exit_time:
                dur = (t.exit_time - t.entry_time).total_seconds() / 3600  # Hours
                durations.append(dur)
        avg_duration = np.mean(durations) if durations else 0
        
        # Trades per day
        if self.trades:
            first_trade = min(t.entry_time for t in self.trades)
            last_trade = max(t.exit_time for t in self.trades if t.exit_time)
            days = (last_trade - first_trade).days or 1
            trades_per_day = total_trades / days
        else:
            trades_per_day = 0
        
        # Per-symbol stats
        symbol_stats = {}
        for symbol in self.stats['trades_per_symbol'].keys():
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            if symbol_trades:
                symbol_wins = [t for t in symbol_trades if t.pnl_pct > 0]
                symbol_stats[symbol] = {
                    'trades': len(symbol_trades),
                    'win_rate': len(symbol_wins) / len(symbol_trades) * 100,
                    'total_pnl': sum(t.pnl_usd for t in symbol_trades)
                }
        
        return {
            'initial_capital': self.config.initial_capital,
            'final_capital': self.balance,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'avg_trade_duration_hours': avg_duration,
            'trades_per_day': trades_per_day,
            'total_signals': self.stats['total_signals'],
            'filtered_signals': self.stats['filtered_signals'],
            'skipped_due_to_position': self.stats['skipped_due_to_position'],
            'symbol_stats': symbol_stats,
            'equity_curve': self.equity_curve
        }
    
    def save_trades_csv(self, filepath: str = 'results/backtest_trades.csv'):
        """Save trades to CSV file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for t in self.trades:
            rows.append({
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'leverage': t.leverage,
                'pnl_pct': t.pnl_pct,
                'pnl_usd': t.pnl_usd,
                'balance_after': t.balance_after,
                'exit_reason': t.exit_reason,
                'signal_score': t.signal_score,
                'skipped_signals': t.skipped_signals
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(rows)} trades to {filepath}")
    
    def print_symbol_stats(self, top_n: int = 10):
        """Print statistics by symbol."""
        symbol_stats = {}
        for t in self.trades:
            if t.symbol not in symbol_stats:
                symbol_stats[t.symbol] = {'trades': 0, 'wins': 0, 'pnl': 0}
            symbol_stats[t.symbol]['trades'] += 1
            symbol_stats[t.symbol]['pnl'] += t.pnl_usd or 0
            if t.pnl_pct and t.pnl_pct > 0:
                symbol_stats[t.symbol]['wins'] += 1
        
        # Calculate win rates
        for s in symbol_stats:
            symbol_stats[s]['win_rate'] = symbol_stats[s]['wins'] / symbol_stats[s]['trades'] * 100
        
        print("\n" + "=" * 60)
        print(f"Top {top_n} pairs by number of trades:")
        print("-" * 60)
        sorted_by_trades = sorted(symbol_stats.items(), key=lambda x: x[1]['trades'], reverse=True)
        for symbol, stats in sorted_by_trades[:top_n]:
            print(f"  {symbol:<20} Trades: {stats['trades']:>3}  "
                  f"Win: {stats['win_rate']:>5.1f}%  PnL: ${stats['pnl']:>8.2f}")
        
        print(f"\nTop {top_n} pairs by profit:")
        print("-" * 60)
        sorted_by_pnl = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for symbol, stats in sorted_by_pnl[:top_n]:
            print(f"  {symbol:<20} PnL: ${stats['pnl']:>8.2f}  "
                  f"Trades: {stats['trades']:>3}  Win: {stats['win_rate']:>5.1f}%")
        
        print(f"\nBottom {top_n} pairs by profit:")
        print("-" * 60)
        for symbol, stats in sorted_by_pnl[-top_n:]:
            print(f"  {symbol:<20} PnL: ${stats['pnl']:>8.2f}  "
                  f"Trades: {stats['trades']:>3}  Win: {stats['win_rate']:>5.1f}%")
        
        # Pairs with no trades
        all_symbols = set(self.stats.get('signals_per_symbol', {}).keys())
        traded_symbols = set(symbol_stats.keys())
        no_trades = all_symbols - traded_symbols
        
        if no_trades:
            print(f"\nPairs with signals but no trades ({len(no_trades)}):")
            for s in list(no_trades)[:10]:
                print(f"  {s}")
            if len(no_trades) > 10:
                print(f"  ... and {len(no_trades) - 10} more")
        
        print("=" * 60)

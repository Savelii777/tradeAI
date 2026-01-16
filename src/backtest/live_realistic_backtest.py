"""
Live-Realistic Backtester

This backtester uses IDENTICAL logic to live trading:
- ATR-based stop loss with adaptive multiplier
- Breakeven trigger at 2.2x ATR profit
- Aggressive trailing stop after breakeven
- Single position constraint
- Same signal thresholds as live

This ensures backtest results match live performance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class LiveRealisticConfig:
    """Config matching live_trading_v10_csv.py EXACTLY."""
    
    # Capital
    initial_capital: float = 10000.0
    
    # Signal thresholds (matching train_v3_dynamic.py for comparison)
    min_conf: float = 0.58
    min_timing: float = 1.8
    min_strength: float = 2.5
    
    # Risk management (from Config class in live)
    risk_pct: float = 0.05  # 5% risk per trade
    max_leverage: float = 50.0
    max_position_size: float = 4_000_000.0
    
    # Fees
    entry_fee: float = 0.0002  # 0.02%
    exit_fee: float = 0.0002
    slippage_pct: float = 0.0005  # 0.05%
    
    # ATR-based SL (from live)
    sl_atr_base: float = 1.5  # Base SL multiplier
    
    # Breakeven settings (from live update_position)
    # be_trigger_mult depends on strength: 2.5 if >=3.0, 2.2 if >=2.0, else 1.8
    be_sl_offset_atr: float = 1.0  # Move SL to entry + 1.0*ATR after BE trigger
    
    # Trailing settings (from live, R-based trailing)
    # trail multiplier: 0.6 if R>5, 1.2 if R>3, 1.8 if R>2, else 2.5
    
    # Max holding (not used in live, but useful for backtest)
    max_holding_bars: int = 150


@dataclass
class Position:
    """Active position state - mirrors live PortfolioManager.position."""
    pair: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    entry_bar: int
    stop_loss: float
    stop_distance: float  # ATR * sl_mult at entry
    position_value: float
    leverage: float
    atr: float
    pred_strength: float
    
    # Breakeven state
    breakeven_active: bool = False
    be_trigger_mult: float = 2.2
    
    # For tracking
    highest_price: float = 0.0  # For LONG trailing
    lowest_price: float = float('inf')  # For SHORT trailing


@dataclass 
class ClosedTrade:
    """Completed trade record."""
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    entry_bar: int
    exit_bar: int
    pnl_pct: float  # Price change %
    pnl_roe: float  # ROE % (with leverage)
    pnl_dollar: float
    position_value: float
    leverage: float
    exit_reason: str  # 'stop_loss', 'breakeven_stop', 'trailing_stop', 'max_bars', 'end_of_data'
    bars_held: int
    pred_strength: float
    breakeven_was_active: bool


class LiveRealisticBacktester:
    """
    Backtester with IDENTICAL logic to live trading.
    
    Key features matching live:
    1. ATR-based stop loss with adaptive multiplier based on strength
    2. Breakeven trigger when price moves be_trigger_mult * ATR in profit
    3. Aggressive R-based trailing after breakeven
    4. Single position across all pairs
    5. Best signal selection by score (conf * strength * timing)
    """
    
    def __init__(self, config: Optional[LiveRealisticConfig] = None):
        self.config = config or LiveRealisticConfig()
        self.reset()
    
    def reset(self):
        """Reset state for new backtest."""
        self.capital = self.config.initial_capital
        self.position: Optional[Position] = None
        self.trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.skipped_signals = 0
        self.total_signals = 0
    
    def _get_sl_multiplier(self, pred_strength: float) -> float:
        """Get SL ATR multiplier based on prediction strength - SAME AS LIVE."""
        if pred_strength >= 3.0:
            return 1.6
        elif pred_strength >= 2.0:
            return 1.5
        else:
            return 1.2
    
    def _get_be_trigger_mult(self, pred_strength: float) -> float:
        """Get breakeven trigger multiplier - SAME AS LIVE."""
        if pred_strength >= 3.0:
            return 2.5
        elif pred_strength >= 2.0:
            return 2.2
        else:
            return 1.8
    
    def _get_trailing_mult(self, r_mult: float) -> float:
        """Get trailing distance multiplier based on R - SAME AS LIVE."""
        if r_mult > 5:
            return 0.6
        elif r_mult > 3:
            return 1.2
        elif r_mult > 2:
            return 1.8
        else:
            return 2.5
    
    def _apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """Apply slippage to price."""
        slip = self.config.slippage_pct
        if is_entry:
            # Entry: pay more for LONG, less for SHORT
            if direction == 'LONG':
                return price * (1 + slip)
            else:
                return price * (1 - slip)
        else:
            # Exit: get less for LONG, more for SHORT
            if direction == 'LONG':
                return price * (1 - slip)
            else:
                return price * (1 + slip)
    
    def _open_position(self, signal: Dict, bar_idx: int) -> bool:
        """Open position - logic from PortfolioManager.open_position."""
        if self.position is not None:
            return False
        
        entry_price = signal['price']
        atr = signal['atr']
        pred_strength = signal.get('pred_strength', signal.get('strength', 2.0))
        direction = signal['direction']
        
        # Apply slippage
        entry_price = self._apply_slippage(entry_price, direction, is_entry=True)
        
        # Adaptive SL multiplier - SAME AS LIVE
        sl_mult = self._get_sl_multiplier(pred_strength)
        stop_distance = atr * sl_mult
        
        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # Position sizing - SAME AS LIVE
        stop_loss_pct = stop_distance / entry_price
        risk_amount = self.capital * self.config.risk_pct
        position_value = min(risk_amount / stop_loss_pct, self.config.max_position_size)
        leverage = min(position_value / self.capital, self.config.max_leverage)
        
        # Adjust for margin constraints
        max_margin = self.capital * 0.95  # 95% max margin usage
        required_margin = position_value / leverage
        if required_margin > max_margin:
            position_value = max_margin * leverage
        
        # Apply entry fee
        fee = position_value * self.config.entry_fee
        
        self.position = Position(
            pair=signal['pair'],
            direction=direction,
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
            be_trigger_mult=self._get_be_trigger_mult(pred_strength),
            highest_price=entry_price,
            lowest_price=entry_price
        )
        
        # Deduct fee from capital
        self.capital -= fee
        
        return True
    
    def _update_position_and_check_exit(self, candle_high: float, candle_low: float) -> Optional[str]:
        """
        Update position with breakeven and trailing, check for exit.
        
        MATCHES train_v3_dynamic.py simulate_trade() EXACTLY:
        1. First check if stop was hit (before any updates)
        2. Then check breakeven trigger
        3. Then update trailing if breakeven active
        
        Returns exit reason if position should close, None otherwise.
        """
        if self.position is None:
            return None
        
        pos = self.position
        atr = pos.atr
        be_trigger_dist = atr * pos.be_trigger_mult
        
        if pos.direction == 'LONG':
            # 1. CHECK STOP FIRST (before any updates)
            if candle_low <= pos.stop_loss:
                reason = 'stop_loss' if not pos.breakeven_active else 'breakeven_stop'
                return reason
            
            # 2. BREAKEVEN TRIGGER
            if not pos.breakeven_active and candle_high >= pos.entry_price + be_trigger_dist:
                pos.breakeven_active = True
                pos.stop_loss = pos.entry_price + atr * self.config.be_sl_offset_atr
            
            # 3. TRAILING (update SL based on current bar's high)
            if pos.breakeven_active:
                current_profit = candle_high - pos.entry_price
                r_mult = current_profit / pos.stop_distance
                trail_mult = self._get_trailing_mult(r_mult)
                new_sl = candle_high - atr * trail_mult
                
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
        
        else:  # SHORT
            # 1. CHECK STOP FIRST
            if candle_high >= pos.stop_loss:
                reason = 'stop_loss' if not pos.breakeven_active else 'breakeven_stop'
                return reason
            
            # 2. BREAKEVEN TRIGGER
            if not pos.breakeven_active and candle_low <= pos.entry_price - be_trigger_dist:
                pos.breakeven_active = True
                pos.stop_loss = pos.entry_price - atr * self.config.be_sl_offset_atr
            
            # 3. TRAILING (update SL based on current bar's low)
            if pos.breakeven_active:
                current_profit = pos.entry_price - candle_low
                r_mult = current_profit / pos.stop_distance
                trail_mult = self._get_trailing_mult(r_mult)
                new_sl = candle_low + atr * trail_mult
                
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl
        
        return None  # No exit
    
    def _close_position(self, exit_price: float, exit_time: datetime, 
                        bar_idx: int, reason: str):
        """Close position and record trade."""
        if self.position is None:
            return
        
        pos = self.position
        
        # Apply slippage
        exit_price = self._apply_slippage(exit_price, pos.direction, is_entry=False)
        
        # Calculate PnL
        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100
        
        # ROE (with leverage)
        pnl_roe = pnl_pct * pos.leverage
        
        # Dollar PnL
        pnl_dollar = pos.position_value * (pnl_pct / 100)
        
        # Exit fee
        fee = pos.position_value * self.config.exit_fee
        pnl_dollar -= fee
        
        # Update capital
        margin_used = pos.position_value / pos.leverage
        self.capital = self.capital + pnl_dollar
        
        # Determine exit reason more precisely  
        # reason comes from _update_position_and_check_exit as 'stop_loss' or 'breakeven_stop'
        # If breakeven was active and PnL is positive, it's trailing_stop
        if reason == 'breakeven_stop' and pnl_pct > 0.3:
            reason = 'trailing_stop'
        
        trade = ClosedTrade(
            pair=pos.pair,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_bar=pos.entry_bar,
            exit_bar=bar_idx,
            pnl_pct=pnl_pct,
            pnl_roe=pnl_roe,
            pnl_dollar=pnl_dollar,
            position_value=pos.position_value,
            leverage=pos.leverage,
            exit_reason=reason,
            bars_held=bar_idx - pos.entry_bar,
            pred_strength=pos.pred_strength,
            breakeven_was_active=pos.breakeven_active
        )
        
        self.trades.append(trade)
        self.position = None
    
    def run(
        self,
        signals_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> Dict:
        """
        Run backtest with live-realistic logic.
        
        Args:
            signals_df: DataFrame with columns:
                - timestamp: datetime
                - pair: str
                - direction: 'LONG' or 'SHORT'
                - confidence: float
                - timing: float
                - strength: float (pred_strength)
                - price: float (entry price)
                - atr: float
                - score: float (optional, will be calculated)
                
            price_data: Dict mapping pair -> OHLCV DataFrame with index=timestamp
            
        Returns:
            Results dictionary with metrics
        """
        self.reset()
        
        if len(signals_df) == 0:
            return {'error': 'No signals provided', 'total_trades': 0}
        
        # Ensure signals are sorted by timestamp
        signals_df = signals_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate score if not present
        if 'score' not in signals_df.columns:
            signals_df['score'] = (
                signals_df['confidence'] * 
                signals_df['timing'] * 
                signals_df['strength']
            )
        
        # Get all timestamps from price data
        all_timestamps = set()
        for pair, df in price_data.items():
            all_timestamps.update(df.index.tolist())
        all_timestamps = sorted(all_timestamps)
        
        # Create signal lookup by timestamp
        signals_by_time = {}
        for _, row in signals_df.iterrows():
            ts = row['timestamp']
            if ts not in signals_by_time:
                signals_by_time[ts] = []
            signals_by_time[ts].append(row.to_dict())
        
        if verbose:
            logger.info(f"Running live-realistic backtest: {len(all_timestamps)} bars, {len(signals_df)} signals")
        
        # Main loop - bar by bar
        for bar_idx, ts in enumerate(all_timestamps):
            # Record equity
            self.equity_curve.append((ts, self.capital))
            
            # If we have a position, update it
            if self.position is not None:
                pair = self.position.pair
                
                if pair in price_data and ts in price_data[pair].index:
                    candle = price_data[pair].loc[ts]
                    high = candle['high']
                    low = candle['low']
                    close = candle['close']
                    
                    # Check max holding
                    bars_held = bar_idx - self.position.entry_bar
                    if bars_held >= self.config.max_holding_bars:
                        self._close_position(close, ts, bar_idx, 'max_bars')
                        continue
                    
                    # Combined: check stop, update BE, update trailing (in correct order)
                    exit_reason = self._update_position_and_check_exit(high, low)
                    
                    if exit_reason:
                        exit_price = self.position.stop_loss
                        self._close_position(exit_price, ts, bar_idx, exit_reason)
                        continue
            
            # Look for new signals (only if no position)
            if self.position is None and ts in signals_by_time:
                candidates = signals_by_time[ts]
                self.total_signals += len(candidates)
                
                # Filter by thresholds - SAME AS LIVE
                valid = []
                for sig in candidates:
                    conf = sig.get('confidence', 0)
                    timing = sig.get('timing', 0)
                    strength = sig.get('strength', sig.get('pred_strength', 0))
                    
                    if (conf >= self.config.min_conf and 
                        timing >= self.config.min_timing and 
                        strength >= self.config.min_strength):
                        valid.append(sig)
                
                if valid:
                    # Sort by score descending - SAME AS LIVE (best signal first)
                    valid.sort(key=lambda x: x.get('score', 0), reverse=True)
                    best = valid[0]
                    
                    # Open position
                    self._open_position(best, bar_idx)
                    
                    # Count skipped
                    self.skipped_signals += len(valid) - 1
            
            elif self.position is not None and ts in signals_by_time:
                # Had signals but couldn't take them
                self.skipped_signals += len(signals_by_time[ts])
        
        # Close any remaining position
        if self.position is not None:
            pair = self.position.pair
            if pair in price_data and len(price_data[pair]) > 0:
                last_price = price_data[pair].iloc[-1]['close']
                last_ts = price_data[pair].index[-1]
                self._close_position(last_price, last_ts, len(all_timestamps)-1, 'end_of_data')
        
        return self._calculate_results(verbose)
    
    def _calculate_results(self, verbose: bool = True) -> Dict:
        """Calculate comprehensive results."""
        if not self.trades:
            return {
                'total_trades': 0,
                'error': 'No trades executed',
                'total_signals': self.total_signals,
                'skipped_signals': self.skipped_signals
            }
        
        # Basic counts
        total = len(self.trades)
        wins = [t for t in self.trades if t.pnl_dollar > 0]
        losses = [t for t in self.trades if t.pnl_dollar <= 0]
        
        # Exit reasons breakdown
        sl_trades = [t for t in self.trades if t.exit_reason == 'stop_loss']
        be_trades = [t for t in self.trades if t.exit_reason == 'breakeven_stop']
        trail_trades = [t for t in self.trades if t.exit_reason == 'trailing_stop']
        
        # PnL stats
        total_pnl = sum(t.pnl_dollar for t in self.trades)
        gross_profit = sum(t.pnl_dollar for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_dollar for t in losses)) if losses else 0
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_pnl = total_pnl / total
        avg_win = np.mean([t.pnl_dollar for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_dollar for t in losses]) if losses else 0
        
        # ROE stats
        avg_roe = np.mean([t.pnl_roe for t in self.trades])
        avg_roe_win = np.mean([t.pnl_roe for t in wins]) if wins else 0
        avg_roe_loss = np.mean([t.pnl_roe for t in losses]) if losses else 0
        
        # Holding time
        avg_bars = np.mean([t.bars_held for t in self.trades])
        
        # Drawdown
        equity = [eq[1] for eq in self.equity_curve]
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Return
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        results = {
            'total_trades': total,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / total * 100,
            
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            
            'avg_roe': avg_roe,
            'avg_roe_win': avg_roe_win,
            'avg_roe_loss': avg_roe_loss,
            
            'avg_bars_held': avg_bars,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'final_capital': self.capital,
            
            # Exit reasons
            'stop_loss_exits': len(sl_trades),
            'breakeven_exits': len(be_trades),
            'trailing_exits': len(trail_trades),
            
            # Signal stats
            'total_signals': self.total_signals,
            'skipped_signals': self.skipped_signals,
            
            # Trades list for detailed analysis
            'trades': self.trades,
            'equity_curve': self.equity_curve,
        }
        
        if verbose:
            logger.info("=" * 60)
            logger.info("BACKTEST RESULTS (Live-Realistic)")
            logger.info("=" * 60)
            logger.info(f"Total trades: {total}")
            logger.info(f"Win rate: {results['win_rate']:.1f}%")
            logger.info(f"Profit factor: {profit_factor:.2f}")
            logger.info(f"Total PnL: ${total_pnl:.2f}")
            logger.info(f"Total return: {total_return:.1f}%")
            logger.info(f"Max drawdown: {max_drawdown:.1f}%")
            logger.info("-" * 40)
            logger.info(f"Avg ROE: {avg_roe:.1f}%")
            logger.info(f"Avg ROE (wins): {avg_roe_win:.1f}%")
            logger.info(f"Avg ROE (losses): {avg_roe_loss:.1f}%")
            logger.info("-" * 40)
            logger.info(f"Exits - SL: {len(sl_trades)}, BE: {len(be_trades)}, Trail: {len(trail_trades)}")
            logger.info(f"Avg holding: {avg_bars:.0f} bars")
            logger.info("=" * 60)
        
        return results


def run_quick_backtest(
    signals: List[Dict],
    price_data: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0,
    verbose: bool = True
) -> Dict:
    """
    Quick helper to run backtest from signal list.
    
    Args:
        signals: List of signal dicts with keys:
            timestamp, pair, direction, confidence, timing, strength, price, atr
        price_data: Dict[pair] -> OHLCV DataFrame
        initial_capital: Starting capital
        verbose: Print results
        
    Returns:
        Results dict
    """
    if not signals:
        return {'error': 'No signals', 'total_trades': 0}
    
    signals_df = pd.DataFrame(signals)
    
    config = LiveRealisticConfig(initial_capital=initial_capital)
    backtester = LiveRealisticBacktester(config)
    
    return backtester.run(signals_df, price_data, verbose)

#!/usr/bin/env python
"""
MTF Trading Bot - Backtesting Script V2
=======================================

Enhanced backtesting with V2 improvements:
- RR 1:3 optimization (SL=1.5 ATR, TP=4.5 ATR)
- EnsembleModelV2 with TradingSignal dataclass
- Dynamic thresholds based on volatility
- Blacklist filtering
- Position size multipliers
- Walk-forward validation mode
- Per-pair performance tracking

Usage:
    python scripts/backtest_mtf_v2.py --pairs "BTC,ETH,SOL" --days 30
    python scripts/backtest_mtf_v2.py --all-pairs --days 60 --test-only
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.utils.helpers import load_yaml_config
from src.features.feature_engine import FeatureEngine
from src.models.ensemble_v2 import EnsembleModelV2, TradingSignal
from src.utils.blacklist import PairBlacklist


@dataclass
class BacktestConfig:
    """Configuration for V2 backtesting."""
    
    # Data settings
    pairs: List[str] = field(default_factory=list)
    days: int = 30
    timeframe: str = "5m"
    
    # Capital and risk
    initial_capital: float = 10000
    risk_per_trade: float = 0.05  # 5% per trade
    max_trades_per_pair_per_day: int = 3
    min_time_between_trades: int = 300  # 5 minutes
    
    # RR 1:3 settings
    rr_ratio: float = 3.0
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 4.5  # 3x SL
    
    # Signal thresholds (V2: lowered for calibrated probs)
    # Timing model outputs very low scores (mean ~0.20, max ~0.35)
    # Direction: ~43% of candles have p_up or p_down > 0.30
    min_direction_prob: float = 0.32  # Slightly above base 0.33
    min_timing_score: float = 0.15    # Model outputs 0.08-0.35, mean 0.20
    min_strength_score: float = 0.20  # Lower threshold
    use_dynamic_thresholds: bool = False  # Disable dynamic to avoid over-filtering
    
    # Costs
    commission_pct: float = 0.0004  # 0.04%
    slippage_pct: float = 0.0002  # 0.02%
    
    # Mode
    test_only: bool = False
    test_ratio: float = 0.15
    
    # Blacklist
    use_blacklist: bool = True
    
    # Output
    output_dir: str = "results/backtest_v2"


@dataclass
class Trade:
    """Represents a completed trade."""
    
    pair: str
    entry_time: datetime
    exit_time: datetime
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'end_of_data'
    direction_prob: float
    timing_score: float
    position_mult: float


@dataclass
class PairResults:
    """Results for a single pair."""
    
    pair: str
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_pnl: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0


@dataclass
class BacktestResults:
    """Complete backtest results."""
    
    # Overall metrics
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Per-pair results
    pair_results: Dict[str, PairResults] = field(default_factory=dict)
    
    # RR analysis
    avg_win_r: float = 0.0  # Average win in R multiples
    avg_loss_r: float = 0.0  # Average loss in R multiples
    expectancy_r: float = 0.0  # Expected value in R
    
    # Trade lists
    all_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    
    # Config used
    config: Optional[BacktestConfig] = None


class BacktesterV2:
    """
    Enhanced backtester with V2 improvements.
    
    Features:
    - RR 1:3 optimization
    - TradingSignal dataclass support
    - Dynamic thresholds
    - Blacklist integration
    - Per-pair tracking
    - Position multipliers
    """
    
    DEFAULT_PAIRS = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
        "DOGE/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT",
        "LINK/USDT:USDT", "ATOM/USDT:USDT", "LTC/USDT:USDT", "UNI/USDT:USDT",
        "APT/USDT:USDT", "NEAR/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
        "FIL/USDT:USDT", "INJ/USDT:USDT", "BNB/USDT:USDT"
    ]
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.blacklist = PairBlacklist()
        self.feature_engine = FeatureEngine({})
        
        # Model
        self.model: Optional[EnsembleModelV2] = None
        
        # State
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.pair_results: Dict[str, PairResults] = {}
    
    def load_model(self, model_path: str):
        """
        Load the ensemble model.
        
        Args:
            model_path: Path to model directory or pickle file
        """
        if os.path.isdir(model_path):
            ensemble_file = os.path.join(model_path, "ensemble_v2.pkl")
        else:
            ensemble_file = model_path
        
        if not os.path.exists(ensemble_file):
            raise FileNotFoundError(f"Model not found: {ensemble_file}")
        
        with open(ensemble_file, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Loaded model from {ensemble_file}")
    
    async def fetch_pair_data(
        self,
        pair: str,
        timeframe: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data for a pair."""
        import ccxt.async_support as ccxt
        
        logger.info(f"Fetching {days} days of {timeframe} data for {pair}")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        try:
            formatted_symbol = pair.replace(':USDT', '').replace('/', '')
            if 'USDT' not in formatted_symbol:
                formatted_symbol += 'USDT'
            formatted_symbol = formatted_symbol.replace('USDT', '/USDT')
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            since = int(start_time.timestamp() * 1000)
            
            all_ohlcv = []
            while True:
                try:
                    ohlcv = await exchange.fetch_ohlcv(
                        formatted_symbol, timeframe, since=since, limit=1000
                    )
                except Exception as e:
                    logger.warning(f"Error fetching {formatted_symbol}: {e}")
                    break
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if since > int(end_time.timestamp() * 1000):
                    break
                
                await asyncio.sleep(0.1)
            
            if not all_ohlcv:
                return None
            
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {pair}: {e}")
            return None
        finally:
            await exchange.close()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    
    def run_pair_backtest(
        self,
        pair: str,
        df: pd.DataFrame,
        features: pd.DataFrame,
        capital_allocation: float
    ) -> PairResults:
        """
        Run backtest for a single pair.
        
        Args:
            pair: Trading pair
            df: OHLCV DataFrame
            features: Feature DataFrame
            capital_allocation: Capital allocated to this pair
            
        Returns:
            PairResults object
        """
        logger.info(f"Running backtest for {pair}...")
        
        results = PairResults(pair=pair)
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # State
        capital = capital_allocation
        position = None
        last_trade_time = None
        daily_trades = {}
        
        # DEBUG: Log sizes
        logger.debug(f"df size: {len(df)}, features size: {len(features)}, atr size: {len(atr)}")
        
        # Iterate through data
        skip_start = 100  # Skip first 100 for indicators
        skip_end = 5  # Skip last 5 for forward data
        
        # Counters
        signal_buy_count = 0
        signal_sell_count = 0
        signal_hold_count = 0
        
        for i in range(skip_start, len(df) - skip_end):
            timestamp = df.index[i]
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_atr = atr.iloc[i]
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # Get current features using timestamp (not iloc!) because features may have fewer rows after dropna
            if timestamp not in features.index:
                continue
            
            current_features = features.loc[[timestamp]].fillna(0)
            
            # Check position exit
            if position:
                should_exit, exit_reason, exit_price = self._check_exit(
                    position, current_price, current_high, current_low
                )
                
                if should_exit:
                    trade = self._close_position(
                        position, exit_price, exit_reason, timestamp
                    )
                    capital += trade.pnl
                    results.trades.append(trade)
                    self.trades.append(trade)
                    position = None
                    continue
            
            # Check for new position
            if not position:
                # Check trade limits
                date_key = timestamp.date()
                if date_key not in daily_trades:
                    daily_trades[date_key] = 0
                
                if daily_trades[date_key] >= self.config.max_trades_per_pair_per_day:
                    continue
                
                # Check time between trades
                if last_trade_time:
                    time_diff = (timestamp - last_trade_time).total_seconds()
                    if time_diff < self.config.min_time_between_trades:
                        continue
                
                # Get signal
                try:
                    signals = self.model.get_trading_signal(
                        current_features,
                        min_direction_prob=self.config.min_direction_prob,
                        min_timing=self.config.min_timing_score,
                        min_strength=self.config.min_strength_score,
                        use_dynamic_thresholds=self.config.use_dynamic_thresholds
                    )
                    
                    if not signals:
                        continue
                    
                    signal = signals[0]
                    
                    # DEBUG: count signals
                    if signal.signal == 'buy':
                        signal_buy_count += 1
                    elif signal.signal == 'sell':
                        signal_sell_count += 1
                    else:
                        signal_hold_count += 1
                    
                    if signal.signal in ['buy', 'sell']:
                        # Calculate position size with multiplier
                        risk_amount = capital * self.config.risk_per_trade
                        risk_amount *= signal.position_size_mult
                        
                        stop_distance = current_atr * self.config.sl_atr_mult
                        size = risk_amount / stop_distance
                        
                        # Apply slippage
                        if signal.signal == 'buy':
                            entry_price = current_price * (1 + self.config.slippage_pct)
                            stop_loss = entry_price - stop_distance
                            take_profit = entry_price + (stop_distance * self.config.rr_ratio)
                            side = 'long'
                        else:
                            entry_price = current_price * (1 - self.config.slippage_pct)
                            stop_loss = entry_price + stop_distance
                            take_profit = entry_price - (stop_distance * self.config.rr_ratio)
                            side = 'short'
                        
                        # Commission on entry
                        capital -= entry_price * size * self.config.commission_pct
                        
                        position = {
                            'pair': pair,
                            'entry': entry_price,
                            'size': size,
                            'side': side,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': timestamp,
                            'direction_prob': max(signal.direction_proba),
                            'timing_score': signal.timing,
                            'position_mult': signal.position_size_mult
                        }
                        
                        daily_trades[date_key] += 1
                        last_trade_time = timestamp
                        
                except Exception as e:
                    logger.debug(f"Signal error at {timestamp}: {e}")
                    continue
            
            # Record equity
            equity = capital
            if position:
                if position['side'] == 'long':
                    equity += (current_price - position['entry']) * position['size']
                else:
                    equity += (position['entry'] - current_price) * position['size']
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'pair': pair,
                'equity': equity
            })
        
        # Close remaining position
        if position:
            trade = self._close_position(
                position, df['close'].iloc[-1], 'end_of_data', df.index[-1]
            )
            capital += trade.pnl
            results.trades.append(trade)
            self.trades.append(trade)
        
        # Log signal counts
        logger.info(f"{pair} signals: buy={signal_buy_count}, sell={signal_sell_count}, hold={signal_hold_count}")
        
        # Calculate pair metrics
        self._calculate_pair_metrics(results, capital_allocation)
        
        return results
    
    def _check_exit(
        self,
        position: Dict,
        current_price: float,
        high: float,
        low: float
    ) -> Tuple[bool, str, float]:
        """Check if position should be exited."""
        if position['side'] == 'long':
            # Check stop loss first (worst case)
            if low <= position['stop_loss']:
                return True, 'stop_loss', position['stop_loss']
            # Then take profit
            if high >= position['take_profit']:
                return True, 'take_profit', position['take_profit']
        else:
            if high >= position['stop_loss']:
                return True, 'stop_loss', position['stop_loss']
            if low <= position['take_profit']:
                return True, 'take_profit', position['take_profit']
        
        return False, '', 0.0
    
    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        reason: str,
        exit_time: datetime
    ) -> Trade:
        """Close position and create Trade object."""
        # Apply slippage
        if position['side'] == 'long':
            actual_exit = exit_price * (1 - self.config.slippage_pct)
            pnl = (actual_exit - position['entry']) * position['size']
        else:
            actual_exit = exit_price * (1 + self.config.slippage_pct)
            pnl = (position['entry'] - actual_exit) * position['size']
        
        # Commission on exit
        pnl -= actual_exit * position['size'] * self.config.commission_pct
        
        pnl_percent = pnl / (position['entry'] * position['size']) * 100
        
        return Trade(
            pair=position['pair'],
            entry_time=position['entry_time'],
            exit_time=exit_time,
            side=position['side'],
            entry_price=position['entry'],
            exit_price=actual_exit,
            size=position['size'],
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=reason,
            direction_prob=position['direction_prob'],
            timing_score=position['timing_score'],
            position_mult=position['position_mult']
        )
    
    def _calculate_pair_metrics(
        self,
        results: PairResults,
        initial_capital: float
    ):
        """Calculate metrics for a pair."""
        trades = results.trades
        
        if not trades:
            return
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        results.n_trades = len(trades)
        results.n_wins = len(wins)
        results.n_losses = len(losses)
        results.total_pnl = sum(t.pnl for t in trades)
        results.total_return_pct = results.total_pnl / initial_capital * 100
        results.win_rate = len(wins) / len(trades) * 100 if trades else 0
        results.avg_trade_pnl = results.total_pnl / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    async def run(self) -> BacktestResults:
        """
        Run full backtest across all pairs.
        
        Returns:
            BacktestResults object
        """
        logger.info("Starting V2 backtest...")
        
        results = BacktestResults(
            initial_capital=self.config.initial_capital,
            config=self.config
        )
        
        # Get pairs
        pairs = self.config.pairs or self.DEFAULT_PAIRS
        
        # Filter blacklisted pairs
        if self.config.use_blacklist:
            pairs = [p for p in pairs if not self.blacklist.is_blacklisted(p)]
            logger.info(f"After blacklist filter: {len(pairs)} pairs")
        
        # Allocate capital evenly
        capital_per_pair = self.config.initial_capital / len(pairs)
        
        # Run backtest for each pair
        for pair in pairs:
            try:
                # Fetch data
                df = await self.fetch_pair_data(
                    pair, self.config.timeframe, self.config.days
                )
                
                if df is None or len(df) < 500:
                    logger.warning(f"Insufficient data for {pair}")
                    continue
                
                # Use test data only if configured
                if self.config.test_only:
                    test_start = int(len(df) * (1 - self.config.test_ratio))
                    df = df.iloc[test_start:]
                    logger.info(f"Using test split for {pair}: {len(df)} candles")
                
                # Generate features
                features = self.feature_engine.generate_all_features(df, normalize=True)
                
                # Drop rows with NaN (first ~120 rows due to indicators lookback)
                features = features.dropna()
                logger.debug(f"Features after dropna: {len(features)} rows")
                
                # Convert categorical to numeric
                for col in features.columns:
                    if features[col].dtype == 'object':
                        features[col] = pd.Categorical(features[col]).codes
                
                # Run pair backtest
                pair_results = self.run_pair_backtest(
                    pair, df, features, capital_per_pair
                )
                
                results.pair_results[pair] = pair_results
                
                logger.info(
                    f"{pair}: {pair_results.n_trades} trades, "
                    f"WR={pair_results.win_rate:.1f}%, "
                    f"Return={pair_results.total_return_pct:.2f}%"
                )
                
            except Exception as e:
                logger.error(f"Error backtesting {pair}: {e}")
                continue
        
        # Calculate overall results
        self._calculate_overall_results(results)
        
        return results
    
    def _calculate_overall_results(self, results: BacktestResults):
        """Calculate overall backtest results."""
        all_trades = self.trades
        results.all_trades = all_trades
        results.equity_curve = self.equity_curve
        
        if not all_trades:
            logger.warning("No trades executed")
            return
        
        # Basic metrics
        results.total_trades = len(all_trades)
        
        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        
        results.win_rate = len(wins) / len(all_trades) * 100
        
        total_pnl = sum(t.pnl for t in all_trades)
        results.final_capital = results.initial_capital + total_pnl
        results.total_return_pct = total_pnl / results.initial_capital * 100
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # R-multiple analysis
        if wins:
            # For RR 1:3, a win should be ~3R
            results.avg_win_r = np.mean([t.pnl_percent / (100 / self.config.rr_ratio) for t in wins])
        if losses:
            results.avg_loss_r = np.mean([abs(t.pnl_percent) / (100 / self.config.rr_ratio) for t in losses])
        
        # Expectancy in R
        win_rate_decimal = len(wins) / len(all_trades)
        results.expectancy_r = (win_rate_decimal * results.avg_win_r) - ((1 - win_rate_decimal) * results.avg_loss_r)
        
        # Equity curve analysis
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df = equity_df.groupby('timestamp')['equity'].sum().reset_index()
            equity_df.set_index('timestamp', inplace=True)
            
            # Max drawdown
            running_max = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - running_max) / running_max
            results.max_drawdown_pct = abs(drawdown.min()) * 100
            
            # Sharpe ratio
            returns = equity_df['equity'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Annualized (assuming 5m candles)
                periods_per_year = 365 * 24 * 12  # 5min periods
                results.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def print_results(results: BacktestResults):
    """Print backtest results summary."""
    print("\n" + "=" * 70)
    print("MTF V2 BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\nCapital: ${results.initial_capital:.2f} → ${results.final_capital:.2f}")
    print(f"Total Return: {results.total_return_pct:.2f}%")
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.1f}%")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
    
    print("\n" + "-" * 40)
    print("RR 1:3 ANALYSIS")
    print("-" * 40)
    print(f"Average Win: {results.avg_win_r:.2f}R")
    print(f"Average Loss: {results.avg_loss_r:.2f}R")
    print(f"Expectancy: {results.expectancy_r:.3f}R per trade")
    
    # Expected value check
    expected_ev = results.win_rate/100 * 3.0 - (1 - results.win_rate/100) * 1.0
    print(f"Theoretical EV (WR={results.win_rate:.1f}%, RR=3): {expected_ev:.2f}R")
    
    if results.expectancy_r > 0:
        print("✓ Positive expectancy achieved")
    else:
        print("⚠ Negative expectancy - review strategy")
    
    # Per-pair breakdown
    print("\n" + "-" * 40)
    print("PER-PAIR RESULTS")
    print("-" * 40)
    
    # Sort by return
    sorted_pairs = sorted(
        results.pair_results.items(),
        key=lambda x: x[1].total_return_pct,
        reverse=True
    )
    
    profitable_count = 0
    for pair, pr in sorted_pairs:
        status = "✓" if pr.total_return_pct > 0 else "✗"
        if pr.total_return_pct > 0:
            profitable_count += 1
        print(
            f"{status} {pair}: "
            f"{pr.n_trades} trades, "
            f"WR={pr.win_rate:.1f}%, "
            f"PF={pr.profit_factor:.2f}, "
            f"Return={pr.total_return_pct:+.2f}%"
        )
    
    print(f"\nProfitable pairs: {profitable_count}/{len(sorted_pairs)}")
    
    # Trade breakdown by exit reason
    print("\n" + "-" * 40)
    print("EXIT REASONS")
    print("-" * 40)
    
    exit_reasons = {}
    for trade in results.all_trades:
        reason = trade.exit_reason
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += trade.pnl
    
    for reason, data in exit_reasons.items():
        print(f"{reason}: {data['count']} trades, PnL=${data['pnl']:.2f}")
    
    print("\n" + "=" * 70)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MTF V2 backtest")
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs (e.g., 'BTC,ETH,SOL')"
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Use all default pairs"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mtf_v2",
        help="Path to model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/trading_params.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Use only test data split"
    )
    parser.add_argument(
        "--no-blacklist",
        action="store_true",
        help="Disable blacklist filtering"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/backtest_v2",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = BacktestConfig()
    config.initial_capital = args.capital
    config.days = args.days
    config.test_only = args.test_only
    config.use_blacklist = not args.no_blacklist
    config.output_dir = args.output
    
    if args.pairs:
        config.pairs = [f"{p.strip()}/USDT:USDT" for p in args.pairs.split(",")]
    elif args.all_pairs:
        config.pairs = []  # Use defaults
    
    # Load trading params
    if os.path.exists(args.config):
        trading_config = load_yaml_config(args.config)
        entry_cfg = trading_config.get('entry', {})
        # Use V2 model calibrated thresholds as defaults
        config.min_direction_prob = entry_cfg.get('min_direction_probability', 0.32)
        config.min_timing_score = entry_cfg.get('min_timing_score', 0.15)
        config.min_strength_score = entry_cfg.get('min_strength_score', 0.20)
        config.use_dynamic_thresholds = entry_cfg.get('use_dynamic_thresholds', False)
        
        exit_cfg = trading_config.get('exit', {})
        config.sl_atr_mult = exit_cfg.get('stop_loss_atr_multiplier', 1.5)
        config.tp_atr_mult = exit_cfg.get('take_profit_atr_multiplier', 4.5)
        config.rr_ratio = trading_config.get('backtest', {}).get('rr_ratio', 3.0)
    
    # Initialize backtester
    backtester = BacktesterV2(config)
    
    # Load model
    try:
        backtester.load_model(args.model_path)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.info("Train models first using: python scripts/train_mtf_v2.py")
        return 1
    
    # Run backtest
    try:
        results = await backtester.run()
        print_results(results)
        
        # Save results
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(config.output_dir, "backtest_summary.json")
        summary = {
            'initial_capital': results.initial_capital,
            'final_capital': results.final_capital,
            'total_return_pct': results.total_return_pct,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown_pct': results.max_drawdown_pct,
            'avg_win_r': results.avg_win_r,
            'avg_loss_r': results.avg_loss_r,
            'expectancy_r': results.expectancy_r,
            'pair_results': {
                pair: {
                    'n_trades': pr.n_trades,
                    'win_rate': pr.win_rate,
                    'profit_factor': pr.profit_factor,
                    'total_return_pct': pr.total_return_pct
                }
                for pair, pr in results.pair_results.items()
            },
            'config': asdict(config)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {summary_path}")
        
        # Save trades
        trades_path = os.path.join(config.output_dir, "trades.json")
        trades_data = [asdict(t) for t in results.all_trades]
        with open(trades_path, 'w') as f:
            json.dump(trades_data, f, indent=2, default=str)
        
        logger.info(f"Trades saved to {trades_path}")
        
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

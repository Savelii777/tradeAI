#!/usr/bin/env python3
"""
Multi-Timeframe (MTF) Backtest Script

Uses the trained MTF model to backtest on historical data.
Supports both single pair and multi-pair backtesting.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from src.models.ensemble import EnsembleModel
from src.utils import load_yaml_config

# Import MTF feature generator from train_mtf (same features as training!)
from train_mtf import MTFFeatureEngine


# ============================================================
# TOP PAIRS (same as training)
# ============================================================

TOP_20_PAIRS = [
    'XAUT/USDT:USDT', 'BTC/USDT:USDT', 'BNB/USDT:USDT', 'TONCOIN/USDT:USDT',
    'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
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
        data[tf.replace('m', '')] = df  # '1m' -> '1', etc.
    return data


# ============================================================
# MTF FEATURE GENERATION (using same generator as training)
# ============================================================

def generate_mtf_features(
    m1_data: pd.DataFrame,
    m5_data: pd.DataFrame,
    m15_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate MTF features using same generator as training.
    This ensures feature consistency between training and inference.
    """
    mtf_engine = MTFFeatureEngine()
    features = mtf_engine.align_timeframes(m1_data, m5_data, m15_data)
    
    # Convert object columns to numeric (same as training)
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = pd.Categorical(features[col]).codes
    
    features = features.fillna(0)
    return features


# ============================================================
# BACKTESTER CLASS
# ============================================================

class MTFBacktester:
    """Multi-timeframe backtester with realistic simulation."""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.0004,  # 0.04% taker fee
        slippage: float = 0.0002,    # 0.02% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
    def run(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        model: EnsembleModel,
        min_confidence: float = 0.60,
        risk_per_trade: float = 0.02,
        stop_loss_atr: float = 1.5,
        take_profit_rr: float = 2.0,
        max_holding_bars: int = 50,
    ) -> Dict:
        """
        Run backtest on prepared data.
        
        Args:
            df: OHLCV data (M5 timeframe)
            features: MTF features aligned to M5
            model: Trained ensemble model
            min_confidence: Min direction probability for entry
            risk_per_trade: Risk per trade as fraction of capital
            stop_loss_atr: Stop loss in ATR units
            take_profit_rr: Take profit R:R ratio
            max_holding_bars: Max bars to hold position
        """
        logger.info(f"Starting backtest: {len(df)} bars, capital ${self.initial_capital}")
        
        # Calculate ATR for position sizing
        atr = self._calculate_atr(df, period=14)
        
        # Initialize state
        capital = self.initial_capital
        position = None
        
        # Skip first 100 bars for indicator warmup
        start_idx = 100
        
        for i in range(start_idx, len(df) - 1):
            timestamp = df.index[i]
            current_price = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            # Get features for current bar
            if i >= len(features):
                continue
                
            current_features = features.iloc[[i]]
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
                'price': current_price
            })
            
            # Check position exit
            if position:
                bars_held = i - position['entry_idx']
                should_exit, exit_reason = self._check_exit(
                    position, current_price, high, low,
                    bars_held, max_holding_bars
                )
                
                if should_exit:
                    pnl = self._close_position(position, current_price, exit_reason, timestamp)
                    capital += pnl
                    position = None
                    continue
            
            # Check for new entry
            if not position:
                try:
                    signal = model.get_trading_signal(
                        current_features,
                        min_direction_prob=min_confidence,
                        min_strength=0.3,  # Lower: models predict ~1.2 avg
                        min_timing=0.01    # Lower: models predict ~0.03 avg
                    )
                    
                    if isinstance(signal, list):
                        signal = signal[0]
                    
                    if signal['signal'] in ['buy', 'sell']:
                        # Calculate position size based on ATR
                        risk_amount = capital * risk_per_trade
                        stop_distance = current_atr * stop_loss_atr
                        
                        if stop_distance <= 0:
                            continue
                            
                        size = risk_amount / stop_distance
                        
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
                        
                        # Deduct commission
                        capital -= entry_price * size * self.commission
                        
                        position = {
                            'entry': entry_price,
                            'size': size,
                            'side': side,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': timestamp,
                            'entry_idx': i,
                            'confidence': signal['confidence'],
                            'direction_proba': signal.get('direction_proba', [0.33, 0.34, 0.33])
                        }
                        
                except Exception as e:
                    logger.debug(f"Signal error at {timestamp}: {e}")
                    continue
        
        # Close any remaining position
        if position:
            pnl = self._close_position(
                position, df['close'].iloc[-1], 'end_of_data', df.index[-1]
            )
            capital += pnl
        
        return self._calculate_results()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        
        # Stop loss
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
        """Close position and return PnL."""
        # Apply slippage
        if position['side'] == 'long':
            actual_exit = exit_price * (1 - self.slippage)
            pnl = (actual_exit - position['entry']) * position['size']
        else:
            actual_exit = exit_price * (1 + self.slippage)
            pnl = (position['entry'] - actual_exit) * position['size']
        
        # Deduct exit commission
        pnl -= actual_exit * position['size'] * self.commission
        
        self.trades.append({
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry'],
            'exit_price': actual_exit,
            'side': position['side'],
            'size': position['size'],
            'pnl': pnl,
            'pnl_percent': (pnl / (position['entry'] * position['size'])) * 100,
            'exit_reason': reason,
            'confidence': position['confidence']
        })
        
        return pnl
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest statistics."""
        if not self.trades:
            return {
                'error': 'No trades executed',
                'total_trades': 0
            }
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Equity curve analysis
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Sharpe ratio (annualized, assuming 5-min bars)
        if len(returns) > 0 and returns.std() > 0:
            # ~105,120 5-min bars per year
            sharpe = returns.mean() / returns.std() * np.sqrt(105120)
        else:
            sharpe = 0
        
        # Max drawdown
        equity = equity_df['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min()) * 100
        
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
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MTF Backtest")
    parser.add_argument("--pairs", type=int, default=5, help="Number of top pairs to test")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--model-path", type=str, default="./models/saved_mtf")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Min direction confidence")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade")
    parser.add_argument("--test-split", type=float, default=0.15, help="Use only last X% as test")
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = EnsembleModel()
    try:
        model.load(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Select pairs
    pairs = TOP_20_PAIRS[:args.pairs]
    logger.info(f"Testing on {len(pairs)} pairs: {pairs}")
    
    # Aggregate results
    all_results = []
    combined_trades = []
    
    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting {pair}")
        logger.info('='*60)
        
        # Load MTF data
        mtf_data = load_mtf_data(pair, args.data_dir)
        if mtf_data is None:
            logger.warning(f"No data for {pair}, skipping")
            continue
        
        m1_df = mtf_data['1']
        m5_df = mtf_data['5']
        m15_df = mtf_data['15']
        
        logger.info(f"Data: M1={len(m1_df)}, M5={len(m5_df)}, M15={len(m15_df)} bars")
        
        # Use only test split (last X%)
        if args.test_split > 0:
            test_start = int(len(m5_df) * (1 - args.test_split))
            
            # Also slice M1 and M15 proportionally
            m1_test_start = int(len(m1_df) * (1 - args.test_split))
            m15_test_start = int(len(m15_df) * (1 - args.test_split))
            
            m1_df = m1_df.iloc[m1_test_start:]
            m5_df = m5_df.iloc[test_start:]
            m15_df = m15_df.iloc[m15_test_start:]
            
            logger.info(f"Test split: M5={len(m5_df)} bars (last {args.test_split:.0%})")
        
        # Generate features using SAME generator as training
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
        
        # Run backtest
        backtester = MTFBacktester(
            initial_capital=args.capital,
            commission=0.0004,
            slippage=0.0002
        )
        
        results = backtester.run(
            df=m5_df,
            features=features,
            model=model,
            min_confidence=args.min_conf,
            risk_per_trade=args.risk,
            stop_loss_atr=1.5,
            take_profit_rr=2.0,
            max_holding_bars=50
        )
        
        results['pair'] = pair
        all_results.append(results)
        
        if 'trades' in results:
            for t in results['trades']:
                t['pair'] = pair
            combined_trades.extend(results['trades'])
        
        # Print pair results
        if results.get('total_trades', 0) > 0:
            print(f"\n{pair}:")
            print(f"  Trades: {results['total_trades']}, Win Rate: {results['win_rate']:.1f}%")
            print(f"  Return: {results['total_return']:.2f}%, PF: {results['profit_factor']:.2f}")
            print(f"  Sharpe: {results['sharpe_ratio']:.2f}, MaxDD: {results['max_drawdown']:.2f}%")
        else:
            print(f"\n{pair}: No trades")
    
    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("AGGREGATE BACKTEST RESULTS")
    print("="*60)
    
    valid_results = [r for r in all_results if r.get('total_trades', 0) > 0]
    
    if not valid_results:
        print("No trades executed across all pairs!")
        return 1
    
    total_trades = sum(r['total_trades'] for r in valid_results)
    total_wins = sum(r['winning_trades'] for r in valid_results)
    
    # Weighted average return
    avg_return = np.mean([r['total_return'] for r in valid_results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results])
    avg_pf = np.mean([r['profit_factor'] for r in valid_results])
    max_dd = max(r['max_drawdown'] for r in valid_results)
    
    print(f"\nPairs tested: {len(valid_results)}")
    print(f"Total trades: {total_trades}")
    print(f"Overall win rate: {(total_wins / total_trades * 100):.1f}%")
    print(f"Average return: {avg_return:.2f}%")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Average Profit Factor: {avg_pf:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    
    # Exit reason analysis
    print("\nExit Reasons:")
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
    print("\nPer-Pair Summary:")
    print("-" * 60)
    for r in valid_results:
        pair = r.get('pair', 'Unknown')
        print(f"  {pair:20s}: {r['total_trades']:3d} trades, "
              f"WR: {r['win_rate']:5.1f}%, "
              f"Ret: {r['total_return']:+6.2f}%, "
              f"PF: {r['profit_factor']:.2f}")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

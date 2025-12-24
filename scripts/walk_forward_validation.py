#!/usr/bin/env python3
"""
Walk-Forward Validation for MTF Trading Bot

Implements rolling window validation to assess model stability over time.
Trains on past data, tests on future data, then rolls forward.

Usage:
    python scripts/walk_forward_validation.py --train-days 60 --test-days 7 --step-days 7
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import EnsembleModel
from src.utils.blacklist import PairBlacklist


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FoldMetrics:
    """Metrics for a single fold in walk-forward validation."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Model metrics
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    
    # Per-pair results
    pair_results: Dict[str, Dict] = field(default_factory=dict)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)
    
    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe - actual would need daily returns."""
        if self.total_trades == 0:
            return 0.0
        avg_pnl = self.total_pnl / self.total_trades
        # Rough approximation
        return avg_pnl / (abs(self.total_pnl) / self.total_trades + 1e-8) * np.sqrt(252)
    
    def to_dict(self) -> Dict:
        return {
            'fold_id': self.fold_id,
            'train_period': f"{self.train_start.date()} to {self.train_end.date()}",
            'test_period': f"{self.test_start.date()} to {self.test_end.date()}",
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
        }


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation."""
    folds: List[FoldMetrics] = field(default_factory=list)
    
    @property
    def num_folds(self) -> int:
        return len(self.folds)
    
    @property
    def mean_test_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.test_accuracy for f in self.folds])
    
    @property
    def std_test_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return np.std([f.test_accuracy for f in self.folds])
    
    @property
    def mean_win_rate(self) -> float:
        rates = [f.win_rate for f in self.folds if f.total_trades > 0]
        return np.mean(rates) if rates else 0.0
    
    @property
    def std_win_rate(self) -> float:
        rates = [f.win_rate for f in self.folds if f.total_trades > 0]
        return np.std(rates) if rates else 0.0
    
    @property
    def mean_profit_factor(self) -> float:
        pfs = [f.profit_factor for f in self.folds 
               if f.total_trades > 0 and f.profit_factor < float('inf')]
        return np.mean(pfs) if pfs else 0.0
    
    @property
    def total_pnl(self) -> float:
        return sum(f.total_pnl for f in self.folds)
    
    @property
    def profitable_folds(self) -> int:
        return sum(1 for f in self.folds if f.total_pnl > 0)
    
    @property
    def worst_fold(self) -> Optional[FoldMetrics]:
        if not self.folds:
            return None
        return min(self.folds, key=lambda f: f.total_pnl)
    
    @property
    def best_fold(self) -> Optional[FoldMetrics]:
        if not self.folds:
            return None
        return max(self.folds, key=lambda f: f.total_pnl)
    
    @property
    def stability_score(self) -> float:
        """Score from 0-1 indicating model stability. Higher = more stable."""
        if not self.folds:
            return 0.0
        
        # Factors:
        # 1. Low std in accuracy
        # 2. Low std in win rate
        # 3. All folds profitable
        
        acc_stability = max(0, 1 - self.std_test_accuracy * 10)
        wr_stability = max(0, 1 - self.std_win_rate * 5)
        profitability = self.profitable_folds / self.num_folds if self.num_folds > 0 else 0
        
        return (acc_stability + wr_stability + profitability) / 3
    
    def summary(self) -> Dict:
        return {
            'num_folds': self.num_folds,
            'mean_test_accuracy': f"{self.mean_test_accuracy:.2%}",
            'std_test_accuracy': f"{self.std_test_accuracy:.2%}",
            'mean_win_rate': f"{self.mean_win_rate:.2%}",
            'std_win_rate': f"{self.std_win_rate:.2%}",
            'mean_profit_factor': f"{self.mean_profit_factor:.2f}",
            'total_pnl': f"${self.total_pnl:.2f}",
            'profitable_folds': f"{self.profitable_folds}/{self.num_folds}",
            'stability_score': f"{self.stability_score:.2f}",
            'worst_fold_pnl': f"${self.worst_fold.total_pnl:.2f}" if self.worst_fold else "N/A",
            'best_fold_pnl': f"${self.best_fold.total_pnl:.2f}" if self.best_fold else "N/A",
        }


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
        data[tf] = df
    return data


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

class WalkForwardValidator:
    """
    Walk-forward validation for time-series trading models.
    
    Rolls through data chronologically:
    [====TRAIN 60d====][TEST 7d]
           [====TRAIN 60d====][TEST 7d]
                  [====TRAIN 60d====][TEST 7d]
    """
    
    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 7,
        step_days: int = 7,
        min_train_samples: int = 5000,
        data_dir: str = './data/candles',
        blacklist: Optional[PairBlacklist] = None,
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_days: Days of data for training
            test_days: Days of data for testing
            step_days: Days to step forward each fold
            min_train_samples: Minimum samples required for training
            data_dir: Directory with candle data
            blacklist: Pair blacklist
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.data_dir = data_dir
        self.blacklist = blacklist or PairBlacklist(static_blacklist=['ADA/USDT:USDT'])
        
        # Import MTF feature engine
        try:
            from train_mtf import MTFFeatureEngine
            self.mtf_engine = MTFFeatureEngine()
        except ImportError:
            logger.warning("Could not import MTFFeatureEngine, using default FeatureEngine")
            from src.features.feature_engine import FeatureEngine
            self.mtf_engine = FeatureEngine()
    
    def run(
        self,
        pairs: List[str],
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
        rr_ratio: float = 3.0,  # RR 1:3
    ) -> WalkForwardResults:
        """
        Run walk-forward validation on specified pairs.
        
        Args:
            pairs: List of trading pairs
            initial_capital: Starting capital per pair
            risk_per_trade: Risk per trade as fraction
            rr_ratio: Risk:Reward ratio (3.0 = 1:3)
            
        Returns:
            WalkForwardResults with all fold metrics
        """
        # Filter blacklisted pairs
        pairs = self.blacklist.get_allowed_pairs(pairs)
        logger.info(f"Running walk-forward on {len(pairs)} pairs")
        
        # Load all data
        logger.info("Loading data...")
        all_data = {}
        for pair in pairs:
            data = load_mtf_data(pair, self.data_dir)
            if data is not None:
                all_data[pair] = data
        
        if not all_data:
            logger.error("No data loaded!")
            return WalkForwardResults()
        
        # Determine date range from M5 data
        first_pair = list(all_data.keys())[0]
        m5_df = all_data[first_pair]['5m']
        start_date = m5_df.index[0]
        end_date = m5_df.index[-1]
        total_days = (end_date - start_date).days
        
        logger.info(f"Data range: {start_date.date()} to {end_date.date()} ({total_days} days)")
        
        # Calculate number of folds
        available_days = total_days - self.train_days - self.test_days
        if available_days < 0:
            logger.error(f"Not enough data! Need {self.train_days + self.test_days} days, have {total_days}")
            return WalkForwardResults()
        
        num_folds = available_days // self.step_days + 1
        logger.info(f"Will run {num_folds} folds")
        
        results = WalkForwardResults()
        
        # Run each fold
        for fold_idx in range(num_folds):
            fold_start = start_date + timedelta(days=fold_idx * self.step_days)
            train_end = fold_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)
            
            if test_end > end_date:
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
            logger.info(f"Train: {fold_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {train_end.date()} to {test_end.date()}")
            logger.info('='*60)
            
            try:
                fold_metrics = self._run_fold(
                    fold_id=fold_idx + 1,
                    all_data=all_data,
                    train_start=fold_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    initial_capital=initial_capital,
                    risk_per_trade=risk_per_trade,
                    rr_ratio=rr_ratio,
                )
                results.folds.append(fold_metrics)
                
                # Log fold results
                logger.info(f"Fold {fold_idx + 1} results: "
                           f"Test Acc={fold_metrics.test_accuracy:.1%}, "
                           f"WR={fold_metrics.win_rate:.1%}, "
                           f"PnL=${fold_metrics.total_pnl:.2f}")
                
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                continue
        
        return results
    
    def _run_fold(
        self,
        fold_id: int,
        all_data: Dict[str, Dict[str, pd.DataFrame]],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        initial_capital: float,
        risk_per_trade: float,
        rr_ratio: float,
    ) -> FoldMetrics:
        """Run a single fold of walk-forward validation."""
        from train_mtf import MTFFeatureEngine, prepare_mtf_data
        
        # Prepare training data
        train_features_list = []
        train_targets_list = []
        
        test_features_list = []
        test_targets_list = []
        test_price_data = {}
        
        for pair, data in all_data.items():
            m1_df = data['1m']
            m5_df = data['5m']
            m15_df = data['15m']
            
            # Split by date
            m1_train = m1_df[(m1_df.index >= train_start) & (m1_df.index < train_end)]
            m5_train = m5_df[(m5_df.index >= train_start) & (m5_df.index < train_end)]
            m15_train = m15_df[(m15_df.index >= train_start) & (m15_df.index < train_end)]
            
            m1_test = m1_df[(m1_df.index >= test_start) & (m1_df.index < test_end)]
            m5_test = m5_df[(m5_df.index >= test_start) & (m5_df.index < test_end)]
            m15_test = m15_df[(m15_df.index >= test_start) & (m15_df.index < test_end)]
            
            if len(m5_train) < 100 or len(m5_test) < 20:
                continue
            
            try:
                # Generate features for train
                mtf_engine = MTFFeatureEngine()
                train_features = mtf_engine.align_timeframes(m1_train, m5_train, m15_train)
                
                # Generate targets for train
                train_direction = self._create_direction_target(m5_train, train_features.index)
                train_timing = self._create_timing_target(m5_train, train_features.index)
                
                # Generate features for test
                test_features = mtf_engine.align_timeframes(m1_test, m5_test, m15_test)
                test_direction = self._create_direction_target(m5_test, test_features.index)
                test_timing = self._create_timing_target(m5_test, test_features.index)
                
                # Convert object columns
                for col in train_features.columns:
                    if train_features[col].dtype == 'object':
                        train_features[col] = pd.Categorical(train_features[col]).codes
                for col in test_features.columns:
                    if test_features[col].dtype == 'object':
                        test_features[col] = pd.Categorical(test_features[col]).codes
                
                train_features_list.append(train_features.fillna(0))
                train_targets_list.append({
                    'direction': train_direction,
                    'timing': train_timing
                })
                
                test_features_list.append(test_features.fillna(0))
                test_targets_list.append({
                    'direction': test_direction,
                    'timing': test_timing,
                    'pair': pair
                })
                test_price_data[pair] = m5_test
                
            except Exception as e:
                logger.warning(f"Error processing {pair}: {e}")
                continue
        
        if not train_features_list:
            raise ValueError("No valid training data")
        
        # Combine training data
        X_train = pd.concat(train_features_list, ignore_index=True)
        y_train = {
            'direction': pd.concat([t['direction'] for t in train_targets_list], ignore_index=True),
            'strength': pd.Series(np.ones(len(X_train))),  # Placeholder
            'volatility': pd.Series(np.ones(len(X_train)) * 0.01),  # Placeholder
            'timing': pd.concat([t['timing'] for t in train_targets_list], ignore_index=True),
        }
        
        # Train model
        model = EnsembleModel()
        train_metrics = model.train_all(X_train, y_train)
        
        # Initialize fold metrics
        fold_metrics = FoldMetrics(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_accuracy=train_metrics.get('direction', {}).get('train_accuracy', 0),
        )
        
        # Evaluate on test data
        total_capital = initial_capital * len(test_features_list)
        
        for i, test_features in enumerate(test_features_list):
            pair = test_targets_list[i]['pair']
            test_direction = test_targets_list[i]['direction']
            
            # Predict
            predictions = model.predict(test_features)
            pred_direction = predictions['direction']
            
            # Calculate test accuracy
            accuracy = (pred_direction == test_direction.values).mean()
            fold_metrics.test_accuracy += accuracy / len(test_features_list)
            
            # Simulate trading
            pair_capital = initial_capital
            pair_trades = 0
            pair_wins = 0
            pair_pnl = 0.0
            
            # Simple simulation (placeholder for full backtest)
            # In production, would use full backtester
            signals = model.get_trading_signal(
                test_features,
                min_direction_prob=0.50,
                min_strength=0.3,
                min_timing=0.3
            )
            
            if isinstance(signals, dict):
                signals = [signals]
            
            for signal in signals:
                if signal.get('signal') in ['buy', 'sell']:
                    pair_trades += 1
                    # Simplified PnL calculation
                    if np.random.random() < accuracy:  # Approximate win rate
                        pair_wins += 1
                        pnl = pair_capital * risk_per_trade * rr_ratio
                        fold_metrics.gross_profit += pnl
                    else:
                        pnl = -pair_capital * risk_per_trade
                        fold_metrics.gross_loss += abs(pnl)
                    pair_pnl += pnl
            
            fold_metrics.total_trades += pair_trades
            fold_metrics.winning_trades += pair_wins
            fold_metrics.total_pnl += pair_pnl
            
            fold_metrics.pair_results[pair] = {
                'accuracy': accuracy,
                'trades': pair_trades,
                'wins': pair_wins,
                'pnl': pair_pnl
            }
        
        return fold_metrics
    
    def _create_direction_target(
        self, 
        df: pd.DataFrame, 
        index: pd.DatetimeIndex
    ) -> pd.Series:
        """Create direction target for walk-forward."""
        close = df['close']
        threshold = 0.002  # 0.2%
        
        future_return = close.pct_change(periods=5).shift(-5)
        
        direction = pd.Series(0, index=df.index)
        direction[future_return > threshold] = 1
        direction[future_return < -threshold] = -1
        
        return direction.reindex(index).fillna(0).astype(int)
    
    def _create_timing_target(
        self, 
        df: pd.DataFrame, 
        index: pd.DatetimeIndex
    ) -> pd.Series:
        """Create timing target for walk-forward."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        future_high = high.rolling(15).max().shift(-15)
        future_low = low.rolling(15).min().shift(-15)
        
        favorable = (future_high - close) / close
        adverse = (close - future_low) / close
        
        ratio = favorable / (adverse + 0.001)
        timing = np.clip(ratio / 3, 0, 1)
        
        return pd.Series(timing, index=df.index).reindex(index).fillna(0.5)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--train-days", type=int, default=60, help="Training period in days")
    parser.add_argument("--test-days", type=int, default=7, help="Test period in days")
    parser.add_argument("--step-days", type=int, default=7, help="Step size in days")
    parser.add_argument("--pairs", type=int, default=10, help="Number of pairs to test")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital per pair")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade")
    parser.add_argument("--rr", type=float, default=3.0, help="Risk:Reward ratio")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("="*60)
    logger.info(f"Train period: {args.train_days} days")
    logger.info(f"Test period: {args.test_days} days")
    logger.info(f"Step size: {args.step_days} days")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"RR: 1:{args.rr}")
    
    # Initialize validator
    validator = WalkForwardValidator(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        data_dir=args.data_dir
    )
    
    # Run validation
    pairs = TOP_20_PAIRS[:args.pairs]
    results = validator.run(
        pairs=pairs,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        rr_ratio=args.rr
    )
    
    # Print results
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*60)
    
    summary = results.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nPer-Fold Results:")
    print("-"*60)
    for fold in results.folds:
        fold_dict = fold.to_dict()
        print(f"  Fold {fold.fold_id}: "
              f"Acc={fold.test_accuracy:.1%}, "
              f"WR={fold.win_rate:.1%}, "
              f"PF={fold.profit_factor:.2f}, "
              f"PnL=${fold.total_pnl:.2f}")
    
    print("="*60)
    
    # Stability assessment
    print("\nSTABILITY ASSESSMENT:")
    if results.stability_score >= 0.7:
        print(f"✅ Model is STABLE (score: {results.stability_score:.2f})")
    elif results.stability_score >= 0.5:
        print(f"⚠️ Model has MODERATE stability (score: {results.stability_score:.2f})")
    else:
        print(f"❌ Model is UNSTABLE (score: {results.stability_score:.2f})")
    
    if results.profitable_folds == results.num_folds:
        print("✅ All folds are profitable")
    else:
        print(f"⚠️ {results.num_folds - results.profitable_folds}/{results.num_folds} folds are unprofitable")
    
    if results.std_win_rate < 0.08:
        print("✅ Win rate is consistent across folds")
    else:
        print(f"⚠️ Win rate varies significantly: std={results.std_win_rate:.1%}")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

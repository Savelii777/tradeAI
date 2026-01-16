#!/usr/bin/env python3
"""
Walk-Forward Validation V2 - Live-Realistic

Uses IDENTICAL backtesting logic to live trading:
- ATR-based stop loss with adaptive multiplier
- Breakeven trigger at 2.2x ATR
- Aggressive trailing stop after breakeven
- Single position constraint
- Same signal thresholds as live

This ensures walk-forward results closely match live performance.

Usage:
    python scripts/walk_forward_validation_v2.py --train-days 60 --test-days 14 --step-days 7
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
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.live_realistic_backtest import LiveRealisticBacktester, LiveRealisticConfig
from src.models.ensemble import EnsembleModel
from src.utils.blacklist import PairBlacklist
from train_mtf import MTFFeatureEngine


# ============================================================
# CONFIG
# ============================================================

# Pairs to test (same as live)
TOP_20_PAIRS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT',
    'DOT/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT', 'UNI/USDT:USDT',
    'AAVE/USDT:USDT', 'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT',
    'OP/USDT:USDT', 'BNB/USDT:USDT', 'TONCOIN/USDT:USDT', 'MATIC/USDT:USDT',
]

# Signal thresholds (SAME AS LIVE)
MIN_CONF = 0.50
MIN_TIMING = 1.5  
MIN_STRENGTH = 1.8

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"


# ============================================================
# DATA LOADING
# ============================================================

def load_pair_data(symbol: str, data_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a single pair."""
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    
    # Try parquet first (faster)
    parquet_path = data_dir / f"{safe_symbol}_{timeframe}.parquet"
    csv_path = data_dir / f"{safe_symbol}_{timeframe}.csv"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    
    return None


def load_mtf_data(pair: str, data_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    """Load M1, M5, M15 data for a pair."""
    data = {}
    for tf in ['1m', '5m', '15m']:
        df = load_pair_data(pair, data_dir, tf)
        if df is None:
            return None
        data[tf] = df
    return data


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features - SAME AS LIVE."""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR - SAME AS LIVE."""
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features_for_period(
    data: Dict[str, pd.DataFrame],
    mtf_engine: MTFFeatureEngine,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Prepare features for a time period - SAME AS LIVE."""
    m1 = data['1m'][(data['1m'].index >= start_date) & (data['1m'].index < end_date)]
    m5 = data['5m'][(data['5m'].index >= start_date) & (data['5m'].index < end_date)]
    m15 = data['15m'][(data['15m'].index >= start_date) & (data['15m'].index < end_date)]
    
    if len(m1) < 200 or len(m5) < 50 or len(m15) < 20:
        return pd.DataFrame()
    
    try:
        ft = mtf_engine.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        ft = ft.dropna(subset=['close', 'atr'])
        ft = ft.ffill().dropna()
        
        return ft
    except Exception as e:
        logger.warning(f"Feature preparation error: {e}")
        return pd.DataFrame()


# ============================================================
# FOLD RESULTS
# ============================================================

@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Trading results
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    
    # Exit breakdown
    stop_loss_exits: int = 0
    breakeven_exits: int = 0
    trailing_exits: int = 0
    
    # Signal stats
    total_signals: int = 0
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0.0


# ============================================================
# WALK-FORWARD VALIDATOR V2
# ============================================================

class WalkForwardValidatorV2:
    """
    Walk-forward validation with live-realistic backtesting.
    
    Process:
    1. Split data into train/test windows
    2. Train model on train data
    3. Generate signals on test data
    4. Run live-realistic backtest on test signals
    5. Roll forward and repeat
    """
    
    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 14,
        step_days: int = 7,
        data_dir: Path = DATA_DIR,
        initial_capital: float = 10000.0,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        
        self.mtf_engine = MTFFeatureEngine()
        
        # Backtest config matching live
        self.backtest_config = LiveRealisticConfig(
            initial_capital=initial_capital,
            min_conf=MIN_CONF,
            min_timing=MIN_TIMING,
            min_strength=MIN_STRENGTH,
        )
    
    def run(self, pairs: List[str]) -> List[FoldResult]:
        """Run walk-forward validation."""
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION V2 (Live-Realistic)")
        logger.info("=" * 60)
        logger.info(f"Train: {self.train_days}d | Test: {self.test_days}d | Step: {self.step_days}d")
        logger.info(f"Pairs: {len(pairs)}")
        
        # Load all data
        logger.info("\nLoading data...")
        all_data = {}
        for pair in pairs:
            data = load_mtf_data(pair, self.data_dir)
            if data is not None:
                all_data[pair] = data
                logger.debug(f"  {pair}: {len(data['5m'])} candles")
        
        if not all_data:
            logger.error("No data loaded!")
            return []
        
        logger.info(f"Loaded {len(all_data)} pairs")
        
        # Determine date range from M5 data
        all_dates = []
        for pair, data in all_data.items():
            all_dates.extend(data['5m'].index.tolist())
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        total_days = (end_date - start_date).days
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({total_days} days)")
        
        # Calculate folds
        min_days_needed = self.train_days + self.test_days
        if total_days < min_days_needed:
            logger.error(f"Not enough data! Need {min_days_needed} days, have {total_days}")
            return []
        
        num_folds = (total_days - min_days_needed) // self.step_days + 1
        logger.info(f"Will run {num_folds} folds")
        
        results = []
        
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
                fold_result = self._run_fold(
                    fold_id=fold_idx + 1,
                    all_data=all_data,
                    train_start=fold_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                )
                results.append(fold_result)
                
                logger.info(f"\nFold {fold_idx + 1} Results:")
                logger.info(f"  Trades: {fold_result.total_trades}")
                logger.info(f"  Win rate: {fold_result.win_rate:.1f}%")
                logger.info(f"  PnL: ${fold_result.total_pnl:.2f}")
                logger.info(f"  PF: {fold_result.profit_factor:.2f}")
                logger.info(f"  Exits - SL: {fold_result.stop_loss_exits}, "
                           f"BE: {fold_result.breakeven_exits}, "
                           f"Trail: {fold_result.trailing_exits}")
                
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _run_fold(
        self,
        fold_id: int,
        all_data: Dict[str, Dict[str, pd.DataFrame]],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
    ) -> FoldResult:
        """Run a single fold."""
        
        # ========================================
        # STEP 1: Prepare training data
        # ========================================
        logger.info("Preparing training data...")
        
        train_features_list = []
        train_targets_list = []
        
        for pair, data in all_data.items():
            features = prepare_features_for_period(
                data, self.mtf_engine, train_start, train_end
            )
            if len(features) < 100:
                continue
            
            # Create targets
            direction = self._create_direction_target(features)
            timing = self._create_timing_target(features)
            strength = self._create_strength_target(features)
            
            # Add pair identifier
            features['_pair'] = pair
            features['_direction'] = direction
            features['_timing'] = timing
            features['_strength'] = strength
            
            train_features_list.append(features)
        
        if not train_features_list:
            raise ValueError("No training data prepared")
        
        # Combine training data
        train_df = pd.concat(train_features_list)
        
        # Remove target columns from features
        feature_cols = [c for c in train_df.columns if not c.startswith('_')]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = {
            'direction': train_df['_direction'],
            'timing': train_df['_timing'],
            'strength': train_df['_strength'],
            'volatility': pd.Series(np.ones(len(train_df)) * 0.01),
        }
        
        logger.info(f"Training samples: {len(X_train)}")
        
        # ========================================
        # STEP 2: Train model
        # ========================================
        logger.info("Training model...")
        model = EnsembleModel()
        model.train_all(X_train, y_train)
        
        # ========================================
        # STEP 3: Generate signals on test data
        # ========================================
        logger.info("Generating test signals...")
        
        all_signals = []
        test_price_data = {}
        
        for pair, data in all_data.items():
            features = prepare_features_for_period(
                data, self.mtf_engine, test_start, test_end
            )
            if len(features) < 10:
                continue
            
            # Get M5 price data for backtesting
            m5 = data['5m'][(data['5m'].index >= test_start) & (data['5m'].index < test_end)]
            if len(m5) < 10:
                continue
            test_price_data[pair] = m5
            
            # Predict
            X_test = features[feature_cols].fillna(0)
            
            try:
                predictions = model.predict(X_test)
                
                # Generate signals for each bar
                for i, (ts, row) in enumerate(features.iterrows()):
                    direction_proba = predictions.get('direction_prob', [0.5])[i] if 'direction_prob' in predictions else 0.5
                    direction = predictions.get('direction', [0])[i]
                    timing = predictions.get('timing', [0])[i]
                    strength = predictions.get('strength', [1])[i]
                    
                    # Skip neutral
                    if direction == 0:
                        continue
                    
                    # Get confidence (probability of predicted class)
                    conf = direction_proba if isinstance(direction_proba, float) else float(np.max(direction_proba))
                    
                    # Filter by thresholds (same as live)
                    if conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH:
                        signal = {
                            'timestamp': ts,
                            'pair': pair,
                            'direction': 'LONG' if direction == 1 else 'SHORT',
                            'confidence': conf,
                            'timing': timing,
                            'strength': strength,
                            'price': row['close'],
                            'atr': row['atr'],
                            'score': conf * timing * strength,
                        }
                        all_signals.append(signal)
                        
            except Exception as e:
                logger.warning(f"Prediction error for {pair}: {e}")
                continue
        
        logger.info(f"Generated {len(all_signals)} signals for {len(test_price_data)} pairs")
        
        if not all_signals:
            return FoldResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                total_signals=0,
            )
        
        # ========================================
        # STEP 4: Run live-realistic backtest
        # ========================================
        logger.info("Running live-realistic backtest...")
        
        signals_df = pd.DataFrame(all_signals)
        
        backtester = LiveRealisticBacktester(self.backtest_config)
        bt_results = backtester.run(signals_df, test_price_data, verbose=False)
        
        # ========================================
        # STEP 5: Create fold result
        # ========================================
        return FoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            total_trades=bt_results.get('total_trades', 0),
            winning_trades=bt_results.get('winning_trades', 0),
            total_pnl=bt_results.get('total_pnl', 0),
            profit_factor=bt_results.get('profit_factor', 0),
            max_drawdown=bt_results.get('max_drawdown', 0),
            stop_loss_exits=bt_results.get('stop_loss_exits', 0),
            breakeven_exits=bt_results.get('breakeven_exits', 0),
            trailing_exits=bt_results.get('trailing_exits', 0),
            total_signals=len(all_signals),
        )
    
    def _create_direction_target(self, df: pd.DataFrame) -> pd.Series:
        """Create direction target."""
        close = df['close']
        threshold = 0.003  # 0.3%
        
        future_return = close.pct_change(periods=10).shift(-10)
        
        direction = pd.Series(0, index=df.index)
        direction[future_return > threshold] = 1
        direction[future_return < -threshold] = -1
        
        return direction
    
    def _create_timing_target(self, df: pd.DataFrame) -> pd.Series:
        """Create timing target (0-3 scale like live)."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        future_high = high.rolling(15).max().shift(-15)
        future_low = low.rolling(15).min().shift(-15)
        
        favorable = (future_high - close) / close
        adverse = (close - future_low) / close
        
        ratio = favorable / (adverse + 0.001)
        timing = np.clip(ratio, 0, 3)
        
        return pd.Series(timing, index=df.index).fillna(1.5)
    
    def _create_strength_target(self, df: pd.DataFrame) -> pd.Series:
        """Create strength target (1-4 scale like live)."""
        close = df['close']
        atr = df['atr']
        
        future_move = abs(close.pct_change(periods=15).shift(-15))
        atr_normalized = atr / close
        
        # Strength = how many ATRs the move covers
        strength = future_move / (atr_normalized + 0.0001)
        strength = np.clip(strength, 1, 4)
        
        return pd.Series(strength, index=df.index).fillna(2.0)
    
    def _print_summary(self, results: List[FoldResult]):
        """Print summary of all folds."""
        if not results:
            logger.warning("No results to summarize")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        total_trades = sum(r.total_trades for r in results)
        total_wins = sum(r.winning_trades for r in results)
        total_pnl = sum(r.total_pnl for r in results)
        
        total_sl = sum(r.stop_loss_exits for r in results)
        total_be = sum(r.breakeven_exits for r in results)
        total_trail = sum(r.trailing_exits for r in results)
        
        avg_pf = np.mean([r.profit_factor for r in results if r.profit_factor > 0 and r.profit_factor < 100])
        max_dd = max(r.max_drawdown for r in results)
        
        profitable_folds = sum(1 for r in results if r.total_pnl > 0)
        
        logger.info(f"Folds: {len(results)} | Profitable: {profitable_folds}/{len(results)}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Overall win rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Average PF: {avg_pf:.2f}")
        logger.info(f"Max drawdown: {max_dd:.1f}%")
        logger.info("-" * 40)
        logger.info(f"Exits - SL: {total_sl}, BE: {total_be}, Trail: {total_trail}")
        if total_trades > 0:
            logger.info(f"Exit %  - SL: {total_sl/total_trades*100:.0f}%, "
                       f"BE: {total_be/total_trades*100:.0f}%, "
                       f"Trail: {total_trail/total_trades*100:.0f}%")
        
        # Per-fold table
        logger.info("\nPer-Fold Results:")
        logger.info("-" * 70)
        logger.info(f"{'Fold':<5} {'Trades':<8} {'WR%':<8} {'PnL$':<12} {'PF':<8} {'DD%':<8}")
        logger.info("-" * 70)
        for r in results:
            logger.info(f"{r.fold_id:<5} {r.total_trades:<8} {r.win_rate:<8.1f} "
                       f"{r.total_pnl:<12.2f} {r.profit_factor:<8.2f} {r.max_drawdown:<8.1f}")
        logger.info("-" * 70)
        
        # Stability assessment
        pnls = [r.total_pnl for r in results]
        win_rates = [r.win_rate for r in results if r.total_trades > 0]
        
        pnl_std = np.std(pnls) if pnls else 0
        wr_std = np.std(win_rates) if win_rates else 0
        
        logger.info("\nSTABILITY ASSESSMENT:")
        
        if profitable_folds >= len(results) * 0.7:
            logger.info("✅ Model is STABLE (70%+ profitable folds)")
        elif profitable_folds >= len(results) * 0.5:
            logger.info("⚠️ Model has MODERATE stability (50-70% profitable)")
        else:
            logger.info("❌ Model is UNSTABLE (<50% profitable folds)")
        
        if wr_std < 10:
            logger.info("✅ Win rate is consistent (std < 10%)")
        else:
            logger.info(f"⚠️ Win rate varies significantly (std = {wr_std:.1f}%)")
        
        # Expected live performance
        avg_pnl_per_fold = total_pnl / len(results)
        trades_per_day = total_trades / (len(results) * self.test_days)
        
        logger.info("\nEXPECTED LIVE PERFORMANCE:")
        logger.info(f"Avg PnL per {self.test_days}-day period: ${avg_pnl_per_fold:.2f}")
        logger.info(f"Trades per day: {trades_per_day:.2f}")
        logger.info(f"Monthly projection: ${avg_pnl_per_fold * (30/self.test_days):.2f}")
        
        logger.info("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation V2 (Live-Realistic)")
    parser.add_argument("--train-days", type=int, default=60, help="Training period")
    parser.add_argument("--test-days", type=int, default=14, help="Test period")
    parser.add_argument("--step-days", type=int, default=7, help="Step size")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    
    args = parser.parse_args()
    
    # Select pairs
    pairs = TOP_20_PAIRS[:args.pairs]
    
    # Run validation
    validator = WalkForwardValidatorV2(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        data_dir=Path(args.data_dir),
        initial_capital=args.capital,
    )
    
    results = validator.run(pairs)
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())

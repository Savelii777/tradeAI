#!/usr/bin/env python3
"""
Quick Backtest with Live-Realistic Logic

Uses the TRAINED model (v8_improved) and runs backtest with
IDENTICAL logic to live trading:
- ATR-based stop loss
- Breakeven trigger
- Aggressive trailing stop
- Same signal thresholds

This shows what to expect from live trading.

Usage:
    python scripts/quick_backtest_v2.py --days 30
    python scripts/quick_backtest_v2.py --start-date 2025-12-01 --end-date 2026-01-01
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from loguru import logger

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.live_realistic_backtest import (
    LiveRealisticBacktester, 
    LiveRealisticConfig,
    ClosedTrade
)
from train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES

# ============================================================
# CONFIGURATION - SAME AS LIVE
# ============================================================

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_20.json"

# Signal thresholds (MATCHING train_v3_dynamic.py for fair comparison)
MIN_CONF = 0.58
MIN_TIMING = 1.8
MIN_STRENGTH = 2.5


# ============================================================
# DATA LOADING
# ============================================================

def load_pair_data(symbol: str, data_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data."""
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    
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


def get_pairs() -> List[str]:
    """Get trading pairs from config."""
    import json
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            return [p['symbol'] for p in json.load(f)['pairs']][:20]
    return ['BTC/USDT:USDT', 'ETH/USDT:USDT']


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
    """Calculate ATR."""
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def prepare_features(
    m1: pd.DataFrame, 
    m5: pd.DataFrame, 
    m15: pd.DataFrame,
    mtf_engine: MTFFeatureEngine
) -> pd.DataFrame:
    """Prepare features - SAME AS LIVE."""
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
        
        # Exclude cumsum features
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        # Exclude absolute price features
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        ft = ft.ffill().dropna()
        return ft
        
    except Exception as e:
        logger.warning(f"Feature error: {e}")
        return pd.DataFrame()


# ============================================================
# MAIN BACKTEST LOGIC
# ============================================================

def run_backtest(
    start_date: datetime,
    end_date: datetime,
    pairs: List[str],
    initial_capital: float = 10000.0,
    verbose: bool = True
) -> Dict:
    """
    Run backtest with trained model using live-realistic logic.
    """
    logger.info("=" * 70)
    logger.info("QUICK BACKTEST V2 (Live-Realistic)")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Pairs: {len(pairs)}")
    logger.info(f"Capital: ${initial_capital:,.0f}")
    logger.info(f"Thresholds: Conf={MIN_CONF}, Tim={MIN_TIMING}, Str={MIN_STRENGTH}")
    
    # Load models
    logger.info("\nLoading trained model...")
    
    if not MODEL_DIR.exists():
        logger.error(f"Model directory not found: {MODEL_DIR}")
        return {'error': 'Model not found'}
    
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
    }
    
    scaler_path = MODEL_DIR / 'scaler.joblib'
    if scaler_path.exists():
        models['scaler'] = joblib.load(scaler_path)
    
    logger.info(f"  Model features: {len(models['features'])}")
    
    # Load MTF engine
    mtf_engine = MTFFeatureEngine()
    
    # Load data and generate signals
    logger.info("\nLoading data and generating signals...")
    
    all_signals = []
    price_data = {}
    
    for pair in pairs:
        # Load data
        m1 = load_pair_data(pair, DATA_DIR, '1m')
        m5 = load_pair_data(pair, DATA_DIR, '5m')
        m15 = load_pair_data(pair, DATA_DIR, '15m')
        
        if m1 is None or m5 is None or m15 is None:
            logger.warning(f"  {pair}: No data")
            continue
        
        # Filter to date range
        m1 = m1[(m1.index >= start_date) & (m1.index < end_date)]
        m5 = m5[(m5.index >= start_date) & (m5.index < end_date)]
        m15 = m15[(m15.index >= start_date) & (m15.index < end_date)]
        
        if len(m5) < 100:
            logger.debug(f"  {pair}: Insufficient data ({len(m5)} candles)")
            continue
        
        # Store M5 for backtesting
        price_data[pair] = m5
        
        # Prepare features
        features = prepare_features(m1, m5, m15, mtf_engine)
        if len(features) < 10:
            continue
        
        # Align features with model
        for f in models['features']:
            if f not in features.columns:
                features[f] = 0
        
        X = features[models['features']].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if models.get('scaler') is not None:
            X = models['scaler'].transform(X)
        
        # Get predictions
        try:
            dir_proba = models['direction'].predict_proba(X)
            dir_pred = models['direction'].predict(X)
            timing_pred = models['timing'].predict(X)
            strength_pred = models['strength'].predict(X)
        except Exception as e:
            logger.warning(f"  {pair}: Prediction error - {e}")
            continue
        
        # Generate signals
        # Direction labels: 0=Down, 1=Sideways, 2=Up (from create_targets_v1)
        pair_signals = 0
        for i, (ts, row) in enumerate(features.iterrows()):
            direction = int(dir_pred[i])
            if direction == 1:  # Skip sideways (1), allow Down (0) and Up (2)
                continue
            
            conf = float(np.max(dir_proba[i]))
            timing = float(timing_pred[i])
            strength = float(strength_pred[i])
            
            # Apply thresholds - SAME AS train_v3_dynamic.py
            if conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH:
                signal = {
                    'timestamp': ts,
                    'pair': pair,
                    'direction': 'LONG' if direction == 2 else 'SHORT',  # 2=Up=LONG, 0=Down=SHORT
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
        return {
            'error': 'No signals generated',
            'total_trades': 0,
            'pairs_loaded': len(price_data),
        }
    
    # Run backtest
    logger.info("\nRunning live-realistic backtest...")
    
    config = LiveRealisticConfig(
        initial_capital=initial_capital,
        min_conf=MIN_CONF,
        min_timing=MIN_TIMING,
        min_strength=MIN_STRENGTH,
    )
    
    backtester = LiveRealisticBacktester(config)
    signals_df = pd.DataFrame(all_signals)
    
    results = backtester.run(signals_df, price_data, verbose=True)
    
    # Additional analysis
    if results.get('total_trades', 0) > 0:
        trades = results.get('trades', [])
        
        # Per-pair breakdown
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
        
        logger.info(f"{'Pair':<20} {'Trades':<8} {'WR%':<10} {'PnL$':<12}")
        logger.info("-" * 60)
        for pair, stats in sorted(pair_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            logger.info(f"{pair:<20} {stats['trades']:<8} {wr:<10.1f} ${stats['pnl']:<12.2f}")
        
        # Direction breakdown
        long_trades = [t for t in trades if t.direction == 'LONG']
        short_trades = [t for t in trades if t.direction == 'SHORT']
        
        logger.info("\nDIRECTION BREAKDOWN:")
        logger.info(f"  LONG:  {len(long_trades)} trades, ${sum(t.pnl_dollar for t in long_trades):.2f} PnL")
        logger.info(f"  SHORT: {len(short_trades)} trades, ${sum(t.pnl_dollar for t in short_trades):.2f} PnL")
        
        # Time analysis
        avg_hold = np.mean([t.bars_held for t in trades])
        logger.info(f"\nHOLDING TIME:")
        logger.info(f"  Average: {avg_hold:.0f} bars ({avg_hold*5/60:.1f} hours)")
        
        # Breakeven effectiveness
        be_active_trades = [t for t in trades if t.breakeven_was_active]
        if be_active_trades:
            be_pnl = sum(t.pnl_dollar for t in be_active_trades)
            logger.info(f"\nBREAKEVEN ANALYSIS:")
            logger.info(f"  Trades that reached BE: {len(be_active_trades)}/{len(trades)} ({len(be_active_trades)/len(trades)*100:.0f}%)")
            logger.info(f"  PnL from BE trades: ${be_pnl:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick Backtest V2 (Live-Realistic)")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)
    
    # Get pairs
    pairs = get_pairs()[:args.pairs]
    
    # Run backtest
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        pairs=pairs,
        initial_capital=args.capital,
    )
    
    return 0 if results.get('total_trades', 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

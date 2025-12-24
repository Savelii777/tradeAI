#!/usr/bin/env python3
"""
Multi-pair backtest script.
Runs backtest with single position constraint across all pairs.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.single_position_backtest import SinglePositionBacktester, BacktestConfig
from src.features.feature_engine import FeatureEngine
from src.models.ensemble import EnsembleModel
from src.models.validator import ModelValidator

warnings.filterwarnings('ignore')


def load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs_list(path: str = 'config/pairs_list.json') -> List[str]:
    """Load pairs list from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get('symbols', [])


def load_pair_data(
    symbol: str,
    data_dir: str = 'data/candles',
    timeframe: str = '5m'
) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a single pair."""
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    filepath = Path(data_dir) / f"{safe_symbol}_{timeframe}.csv"
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def generate_predictions(
    model: EnsembleModel,
    data: pd.DataFrame,
    feature_engine: FeatureEngine
) -> pd.DataFrame:
    """Generate predictions for a single pair."""
    # Generate features
    features = feature_engine.generate_all_features(data, normalize=True)
    
    # Encode categorical columns
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = pd.Categorical(features[col]).codes
    
    # Handle NaN/Inf
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get predictions
    predictions = model.predict(features)
    predictions.index = features.index
    
    return predictions


def run_multi_pair_backtest(
    symbols: List[str],
    model_path: str = 'models/saved_multi',
    data_dir: str = 'data/candles',
    timeframe: str = '5m',
    config_path: str = './config/trading_params.yaml',
    test_only: bool = False,
    test_ratio: float = 0.15
) -> Dict:
    """
    Run backtest on multiple pairs.
    
    Args:
        symbols: List of trading pair symbols
        model_path: Path to saved model
        data_dir: Directory with candle data
        timeframe: Timeframe to use
        config_path: Path to config file
        test_only: If True, use only test portion of data
        test_ratio: Ratio of data to use for test
    
    Returns:
        Backtest results dictionary
    """
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = EnsembleModel()
    model.load(model_path)
    
    # Load config
    config = load_yaml_config(config_path)
    feature_engine = FeatureEngine(config.get('features', {}))
    
    # Load and process all data
    all_data = {}
    all_predictions = {}
    
    logger.info(f"Loading data for {len(symbols)} pairs...")
    
    for symbol in symbols:
        data = load_pair_data(symbol, data_dir, timeframe)
        
        if data is None or len(data) < 500:
            logger.warning(f"Skipping {symbol}: insufficient data")
            continue
        
        try:
            # Use only test portion if requested
            if test_only:
                test_start = int(len(data) * (1 - test_ratio))
                data = data.iloc[test_start:]
            
            # Generate predictions
            predictions = generate_predictions(model, data, feature_engine)
            
            all_data[symbol] = data
            all_predictions[symbol] = predictions
            
            logger.debug(f"✓ {symbol}: {len(data)} candles")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    logger.info(f"Loaded {len(all_data)} pairs successfully")
    
    if not all_data:
        raise ValueError("No valid data loaded")
    
    # Create backtester
    bt_config = BacktestConfig(
        initial_capital=10000.0,
        risk_per_trade=0.05,  # 5% SL
        reward_ratio=3.0,  # 1:3 RR
        commission=0.0004,  # 0.04%
        slippage=0.0001,  # 0.01%
        min_probability=config.get('entry', {}).get('min_direction_probability', 0.55),
        min_score=0.3
    )
    
    backtester = SinglePositionBacktester(bt_config)
    
    # Run backtest
    logger.info("Running backtest...")
    results = backtester.run(all_data, all_predictions)
    
    # Save trades to CSV
    backtester.save_trades_csv('results/backtest_trades.csv')
    
    # Print symbol statistics
    backtester.print_symbol_stats(top_n=10)
    
    # Generate charts
    save_charts(backtester, results, 'results/charts')
    
    return results, backtester


def save_charts(backtester: SinglePositionBacktester, results: Dict, output_dir: str):
    """Save analysis charts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Equity curve
    if results.get('equity_curve'):
        fig, ax = plt.subplots(figsize=(12, 6))
        equity = results['equity_curve']
        times = [e[0] for e in equity]
        values = [e[1] for e in equity]
        ax.plot(times, values, 'b-', linewidth=1)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Time')
        ax.set_ylabel('Balance ($)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curve.png', dpi=150)
        plt.close()
    
    # 2. Trades distribution by symbol (pie chart)
    if results.get('symbol_stats'):
        fig, ax = plt.subplots(figsize=(10, 10))
        symbol_trades = {s: v['trades'] for s, v in results['symbol_stats'].items()}
        
        # Take top 15 symbols
        sorted_symbols = sorted(symbol_trades.items(), key=lambda x: x[1], reverse=True)[:15]
        labels = [s[0].replace('/USDT:USDT', '') for s in sorted_symbols]
        values = [s[1] for s in sorted_symbols]
        
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title('Trade Distribution by Symbol (Top 15)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trades_by_symbol.png', dpi=150)
        plt.close()
    
    # 3. PnL distribution histogram
    if backtester.trades:
        fig, ax = plt.subplots(figsize=(10, 6))
        pnls = [t.pnl_pct for t in backtester.trades if t.pnl_pct is not None]
        ax.hist(pnls, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_title('Trade PnL Distribution')
        ax.set_xlabel('PnL (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pnl_distribution.png', dpi=150)
        plt.close()
    
    # 4. Win rate by day of week
    if backtester.trades:
        trades_by_dow = {i: {'wins': 0, 'total': 0} for i in range(7)}
        for t in backtester.trades:
            if t.entry_time:
                dow = t.entry_time.weekday()
                trades_by_dow[dow]['total'] += 1
                if t.pnl_pct and t.pnl_pct > 0:
                    trades_by_dow[dow]['wins'] += 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        win_rates = []
        for i in range(7):
            if trades_by_dow[i]['total'] > 0:
                wr = trades_by_dow[i]['wins'] / trades_by_dow[i]['total'] * 100
            else:
                wr = 0
            win_rates.append(wr)
        
        colors = ['green' if wr >= 50 else 'red' for wr in win_rates]
        ax.bar(days, win_rates, color=colors, alpha=0.7)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
        ax.set_title('Win Rate by Day of Week')
        ax.set_xlabel('Day')
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/winrate_by_day.png', dpi=150)
        plt.close()
    
    # 5. Win rate by hour
    if backtester.trades:
        trades_by_hour = {i: {'wins': 0, 'total': 0} for i in range(24)}
        for t in backtester.trades:
            if t.entry_time:
                hour = t.entry_time.hour
                trades_by_hour[hour]['total'] += 1
                if t.pnl_pct and t.pnl_pct > 0:
                    trades_by_hour[hour]['wins'] += 1
        
        fig, ax = plt.subplots(figsize=(12, 6))
        hours = list(range(24))
        win_rates = []
        for h in hours:
            if trades_by_hour[h]['total'] > 0:
                wr = trades_by_hour[h]['wins'] / trades_by_hour[h]['total'] * 100
            else:
                wr = 0
            win_rates.append(wr)
        
        colors = ['green' if wr >= 50 else 'red' for wr in win_rates]
        ax.bar(hours, win_rates, color=colors, alpha=0.7)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
        ax.set_title('Win Rate by Hour (UTC)')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/winrate_by_hour.png', dpi=150)
        plt.close()
    
    logger.info(f"Charts saved to {output_dir}/")


def print_results(results: Dict, test_only: bool = False):
    """Print backtest results."""
    print("\n" + "=" * 70)
    print("MULTI-PAIR BACKTEST RESULTS" + (" (TEST DATA ONLY)" if test_only else ""))
    print("=" * 70)
    
    print(f"\n{'Capital:':<30}")
    print(f"  Initial:           ${results.get('initial_capital', 0):,.2f}")
    print(f"  Final:             ${results.get('final_capital', 0):,.2f}")
    print(f"  Total Return:      {results.get('total_return', 0):,.2f}%")
    print(f"  Total PnL:         ${results.get('total_pnl', 0):,.2f}")
    
    print(f"\n{'Trades:':<30}")
    print(f"  Total:             {results.get('total_trades', 0)}")
    print(f"  Winning:           {results.get('winning_trades', 0)}")
    print(f"  Losing:            {results.get('losing_trades', 0)}")
    print(f"  Win Rate:          {results.get('win_rate', 0):.2f}%")
    print(f"  Profit Factor:     {results.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio:      {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:      {results.get('max_drawdown', 0):.2f}%")
    
    print(f"\n{'Activity:':<30}")
    print(f"  Avg Duration:      {results.get('avg_trade_duration_hours', 0):.1f} hours")
    print(f"  Trades/Day:        {results.get('trades_per_day', 0):.2f}")
    
    print(f"\n{'Signals:':<30}")
    print(f"  Total Signals:     {results.get('total_signals', 0)}")
    print(f"  Filtered:          {results.get('filtered_signals', 0)}")
    print(f"  Skipped (position):{results.get('skipped_due_to_position', 0)}")
    
    # Validate results
    validator = ModelValidator()
    is_realistic, warnings = validator.validate_results(results)
    
    if not is_realistic:
        print("\n" + "=" * 70)
        print("⚠️  VALIDATION WARNINGS")
        print("=" * 70)
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✓ Results appear realistic")
    
    print("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-pair backtest')
    parser.add_argument('--pairs-file', type=str, default='config/pairs_list.json',
                        help='Path to pairs list JSON')
    parser.add_argument('--model-path', type=str, default='models/saved_multi',
                        help='Path to saved model')
    parser.add_argument('--data-dir', type=str, default='data/candles',
                        help='Directory with candle data')
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Timeframe to use (default: 5m)')
    parser.add_argument('--config', type=str, default='./config/trading_params.yaml',
                        help='Path to config file')
    parser.add_argument('--test-only', action='store_true',
                        help='Use only test portion of data')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test data ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Load pairs
    try:
        symbols = load_pairs_list(args.pairs_file)
        logger.info(f"Loaded {len(symbols)} pairs from {args.pairs_file}")
    except FileNotFoundError:
        logger.error(f"Pairs file not found: {args.pairs_file}")
        return 1
    
    # Run backtest
    try:
        results, backtester = run_multi_pair_backtest(
            symbols=symbols,
            model_path=args.model_path,
            data_dir=args.data_dir,
            timeframe=args.timeframe,
            config_path=args.config,
            test_only=args.test_only,
            test_ratio=args.test_ratio
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print results
    print_results(results, args.test_only)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

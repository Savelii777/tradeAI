#!/usr/bin/env python
"""
AI Trading Bot - Backtesting Script
Runs backtests on historical data.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from src.utils import load_yaml_config, calculate_sharpe_ratio, calculate_profit_factor, calculate_max_drawdown
from src.data import DataCollector, DataPreprocessor
from src.features import FeatureEngine
from src.models import EnsembleModel


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    Features:
    - Realistic simulation with slippage and fees
    - Position sizing
    - Stop-loss and take-profit simulation
    - Performance metrics calculation
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        config: Optional[Dict] = None
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital.
            commission: Commission rate per trade.
            slippage: Slippage rate.
            config: Configuration dictionary.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.config = config or {}
        
        # Results storage
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
    def run(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        model: EnsembleModel,
        min_confidence: float = 0.6,
        risk_per_trade: float = 0.02,
        stop_loss_atr: float = 1.5,
        take_profit_rr: float = 2.0
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            df: OHLCV data.
            features: Feature DataFrame.
            model: Trained model.
            min_confidence: Minimum confidence for entry.
            risk_per_trade: Risk per trade as fraction of capital.
            stop_loss_atr: Stop loss in ATR units.
            take_profit_rr: Take profit risk:reward ratio.
            
        Returns:
            Backtest results dictionary.
        """
        logger.info("Starting backtest...")
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Initialize state
        capital = self.initial_capital
        position = None
        
        # Iterate through data
        for i in range(100, len(df) - 5):  # Skip first 100 for indicators, last 5 for forward data
            timestamp = df.index[i]
            current_price = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            
            # Get current features
            if i >= len(features):
                continue
            current_features = features.iloc[[i]].fillna(0)
            
            # Record equity
            equity = capital
            if position:
                if position['side'] == 'long':
                    equity += (current_price - position['entry']) * position['size']
                else:
                    equity += (position['entry'] - current_price) * position['size']
                    
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity
            })
            
            # Check for position exit
            if position:
                should_exit, exit_reason = self._check_exit(
                    position, current_price, df['high'].iloc[i], df['low'].iloc[i]
                )
                
                if should_exit:
                    pnl = self._close_position(position, current_price, exit_reason)
                    capital += pnl
                    position = None
                    continue
                    
            # Check for new position
            if not position:
                try:
                    signal = model.get_trading_signal(
                        current_features,
                        min_direction_prob=min_confidence
                    )
                    
                    if isinstance(signal, list):
                        signal = signal[0]
                        
                    if signal['signal'] in ['buy', 'sell']:
                        # Calculate position size
                        risk_amount = capital * risk_per_trade
                        stop_distance = current_atr * stop_loss_atr
                        size = risk_amount / stop_distance
                        
                        # Apply slippage to entry
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
                            'confidence': signal['confidence']
                        }
                        
                except Exception as e:
                    logger.debug(f"Signal generation error at {timestamp}: {e}")
                    continue
                    
        # Close any remaining position
        if position:
            pnl = self._close_position(position, df['close'].iloc[-1], 'end_of_data')
            capital += pnl
            
        # Calculate results
        return self._calculate_results()
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        
    def _check_exit(
        self,
        position: Dict,
        current_price: float,
        high: float,
        low: float
    ) -> tuple:
        """Check if position should be exited."""
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
        reason: str
    ) -> float:
        """Close position and calculate PnL."""
        # Apply slippage
        if position['side'] == 'long':
            actual_exit = exit_price * (1 - self.slippage)
            pnl = (actual_exit - position['entry']) * position['size']
        else:
            actual_exit = exit_price * (1 + self.slippage)
            pnl = (position['entry'] - actual_exit) * position['size']
            
        # Deduct commission
        pnl -= actual_exit * position['size'] * self.commission
        
        self.trades.append({
            'entry_time': position['entry_time'],
            'entry_price': position['entry'],
            'exit_price': actual_exit,
            'side': position['side'],
            'size': position['size'],
            'pnl': pnl,
            'pnl_percent': pnl / (position['entry'] * position['size']) * 100,
            'exit_reason': reason,
            'confidence': position['confidence']
        })
        
        return pnl
        
    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if not self.trades:
            return {'error': 'No trades executed'}
            
        # Trade statistics
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Equity curve analysis
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['equity'].pct_change().dropna()
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital + total_pnl,
            'total_return': total_pnl / self.initial_capital * 100,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100,
            'profit_factor': calculate_profit_factor(self.trades),
            'avg_win': sum(t['pnl'] for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t['pnl'] for t in losses) / len(losses) if losses else 0,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(equity_df['equity'])['max_drawdown'] * 100,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
        
        return results


async def fetch_data(symbol: str, timeframe: str, days: int):
    """Fetch historical data."""
    collector = DataCollector(
        exchange_id="binance",
        symbol=symbol.replace('USDT', '/USDT'),
        testnet=False
    )
    
    await collector.start()
    
    try:
        return await collector.fetch_historical_ohlcv(timeframe, days)
    finally:
        await collector.stop()


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--model-path", type=str, default="./models/saved")
    parser.add_argument("--config", type=str, default="./config/trading_params.yaml")
    
    args = parser.parse_args()
    
    # Load model
    model = EnsembleModel()
    try:
        model.load(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Training a new model first...")
        return 1
        
    # Load config
    config = load_yaml_config(args.config)
    
    # Fetch data
    logger.info(f"Fetching {args.days} days of data...")
    df = asyncio.run(fetch_data(args.symbol, args.timeframe, args.days))
    
    if df is None or df.empty:
        logger.error("Failed to fetch data")
        return 1
        
    # Generate features
    logger.info("Generating features...")
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_ohlcv(df)
    
    feature_engine = FeatureEngine(config.get('features', {}))
    features = feature_engine.generate_all_features(df, normalize=True)
    
    # Run backtest
    backtester = Backtester(
        initial_capital=args.capital,
        commission=0.001,
        slippage=0.0005
    )
    
    results = backtester.run(
        df, features, model,
        min_confidence=config.get('entry', {}).get('min_direction_probability', 0.6),
        risk_per_trade=config.get('risk', {}).get('max_risk_per_trade', 0.02)
    )
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"\nTotal Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

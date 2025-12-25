#!/usr/bin/env python3
"""
Verify Backtest Trades Against MEXC Real Data

Exports trade details and verifies prices against actual MEXC candles.

Usage:
    python scripts/verify_trades.py --pairs 20 --days 1 --capital 100 --output trades_report.csv
    
    # In Docker:
    docker-compose -f docker/docker-compose.yml run --rm trading-bot \
        python scripts/verify_trades.py --pairs 20 --days 1 --capital 100
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from src.risk.risk_manager import RiskManager, load_risk_config
import joblib

# Import MTF feature generator
from train_mtf import MTFFeatureEngine


# ============================================================
# TRADE DETAILS COLLECTOR
# ============================================================

class TradeCollector:
    """
    Collects detailed trade information for verification.
    """
    
    def __init__(self):
        self.trades = []
        
    def add_trade(
        self,
        pair: str,
        side: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        leverage: float,
        position_value: float,
        confidence: float,
        timing_score: float,
        candle_data: Dict = None
    ):
        """Add a trade with full details."""
        trade = {
            # Basic info
            'pair': pair,
            'side': side,
            
            # Timing
            'entry_time': entry_time.isoformat() if isinstance(entry_time, datetime) else str(entry_time),
            'exit_time': exit_time.isoformat() if isinstance(exit_time, datetime) else str(exit_time),
            
            # Prices
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            
            # Result
            'exit_reason': exit_reason,
            'pnl_usd': pnl,
            'pnl_pct': pnl_pct,
            
            # Position details
            'leverage': leverage,
            'position_value': position_value,
            
            # Model scores
            'direction_confidence': confidence,
            'timing_score': timing_score,
            
            # Candle data for verification
            'entry_candle_open': candle_data.get('entry_open') if candle_data else None,
            'entry_candle_high': candle_data.get('entry_high') if candle_data else None,
            'entry_candle_low': candle_data.get('entry_low') if candle_data else None,
            'entry_candle_close': candle_data.get('entry_close') if candle_data else None,
            'exit_candle_open': candle_data.get('exit_open') if candle_data else None,
            'exit_candle_high': candle_data.get('exit_high') if candle_data else None,
            'exit_candle_low': candle_data.get('exit_low') if candle_data else None,
            'exit_candle_close': candle_data.get('exit_close') if candle_data else None,
        }
        
        self.trades.append(trade)
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        return pd.DataFrame(self.trades)
    
    def to_csv(self, filepath: str):
        """Export trades to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(self.trades)} trades to {filepath}")
        
    def to_json(self, filepath: str):
        """Export trades to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)
        print(f"✅ Exported {len(self.trades)} trades to {filepath}")
        
    def print_summary(self):
        """Print trade summary."""
        if not self.trades:
            print("No trades to display")
            return
            
        print("\n" + "="*100)
        print("📊 DETAILED TRADE LOG FOR VERIFICATION")
        print("="*100)
        
        for i, t in enumerate(self.trades, 1):
            result_emoji = "✅" if t['pnl_usd'] > 0 else "❌"
            side_emoji = "🟢" if t['side'] == 'long' else "🔴"
            
            print(f"\n{'─'*100}")
            print(f"Trade #{i} {result_emoji} {side_emoji} {t['side'].upper()} {t['pair']}")
            print(f"{'─'*100}")
            
            print(f"\n📅 TIMING:")
            print(f"  Entry: {t['entry_time']}")
            print(f"  Exit:  {t['exit_time']}")
            
            print(f"\n💰 PRICES:")
            print(f"  Entry Price:  ${t['entry_price']:.6f}")
            print(f"  Exit Price:   ${t['exit_price']:.6f}")
            print(f"  Stop Loss:    ${t['stop_loss']:.6f}")
            print(f"  Take Profit:  ${t['take_profit']:.6f}")
            
            print(f"\n📊 RESULT:")
            print(f"  Exit Reason:  {t['exit_reason']}")
            print(f"  PnL:          ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2f}%)")
            print(f"  Leverage:     {t['leverage']:.1f}x")
            print(f"  Position:     ${t['position_value']:.2f}")
            
            print(f"\n🤖 MODEL SCORES:")
            print(f"  Direction:    {t['direction_confidence']:.2%}")
            print(f"  Timing:       {t['timing_score']:.2%}")
            
            if t.get('entry_candle_open'):
                print(f"\n🕯️ ENTRY CANDLE (for MEXC verification):")
                print(f"  Open:  ${t['entry_candle_open']:.6f}")
                print(f"  High:  ${t['entry_candle_high']:.6f}")
                print(f"  Low:   ${t['entry_candle_low']:.6f}")
                print(f"  Close: ${t['entry_candle_close']:.6f}")
                
            if t.get('exit_candle_open'):
                print(f"\n🕯️ EXIT CANDLE (for MEXC verification):")
                print(f"  Open:  ${t['exit_candle_open']:.6f}")
                print(f"  High:  ${t['exit_candle_high']:.6f}")
                print(f"  Low:   ${t['exit_candle_low']:.6f}")
                print(f"  Close: ${t['exit_candle_close']:.6f}")


# ============================================================
# V1 MODEL WRAPPER (simplified from backtest_v1_risk.py)
# ============================================================

class V1FreshModel:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.direction_model = None
        self.timing_model = None
        self._is_trained = False
        
    def load(self):
        self.direction_model = joblib.load(self.model_dir / 'direction_model.joblib')
        self.timing_model = joblib.load(self.model_dir / 'timing_model.joblib')
        self._is_trained = True
        
    def get_trading_signal(self, X: pd.DataFrame, min_direction_prob: float = 0.50, min_timing: float = 0.01) -> Dict:
        if not self._is_trained:
            raise RuntimeError("Models not loaded")
        
        direction_proba = self.direction_model.predict_proba(X)[0]
        timing_proba = self.timing_model.predict_proba(X)[0][1]
        direction_pred = np.argmax(direction_proba)
        
        signal = {
            'signal': 'hold',
            'confidence': 0.0,
            'timing': timing_proba,
            'direction_proba': direction_proba.tolist()
        }
        
        p_down, p_sideways, p_up = direction_proba
        
        if p_up >= min_direction_prob and p_up > p_down and p_up > p_sideways:
            if timing_proba >= min_timing:
                signal['signal'] = 'buy'
                signal['confidence'] = p_up
                
        elif p_down >= min_direction_prob and p_down > p_up and p_down > p_sideways:
            if timing_proba >= min_timing:
                signal['signal'] = 'sell'
                signal['confidence'] = p_down
        
        return signal


# ============================================================
# V1 TOP PAIRS
# ============================================================

V1_TOP_PAIRS = [
    'XAUT/USDT:USDT', 'BTC/USDT:USDT', 'BNB/USDT:USDT', 'TONCOIN/USDT:USDT',
    'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
    'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
    'LTC/USDT:USDT', 'BCH/USDT:USDT', 'UNI/USDT:USDT', 'AAVE/USDT:USDT',
    'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT', 'OP/USDT:USDT',
]


# ============================================================
# DATA LOADING
# ============================================================

def load_pair_data(symbol: str, data_dir: str, timeframe: str) -> Optional[pd.DataFrame]:
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    filepath = Path(data_dir) / f"{safe_symbol}_{timeframe}.csv"
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def load_mtf_data(pair: str, data_dir: str) -> Optional[Dict[str, pd.DataFrame]]:
    data = {}
    for tf in ['1m', '5m', '15m']:
        df = load_pair_data(pair, data_dir, tf)
        if df is None:
            return None
        data[tf.replace('m', '')] = df
    return data


def filter_data_by_days(data: Dict[str, pd.DataFrame], days: int) -> Dict[str, pd.DataFrame]:
    end_time = data['5'].index[-1]
    start_time = end_time - timedelta(days=days)
    
    filtered = {}
    for tf, df in data.items():
        filtered[tf] = df[df.index >= start_time].copy()
    
    return filtered


# ============================================================
# BACKTESTER WITH TRADE COLLECTION
# ============================================================

class VerifyBacktester:
    def __init__(
        self,
        initial_capital: float = 100,
        risk_per_trade: float = 0.05,
        max_leverage: float = 20.0,
        slippage: float = 0.0001,
        entry_fee: float = 0.0002,
        exit_fee: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.slippage = slippage
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee
        
        self.trade_collector = TradeCollector()
        
    def run(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        model,
        pair: str,
        min_confidence: float = 0.50,
        min_timing: float = 0.01,
        stop_loss_atr: float = 1.5,
        take_profit_rr: float = 2.0,
        max_holding_bars: int = 50
    ):
        """Run backtest and collect detailed trade data."""
        capital = self.initial_capital
        position = None
        trades_count = 0
        
        feature_names = model.direction_model.feature_name_
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Check if we have features for this timestamp
            if timestamp not in features.index:
                continue
                
            feat_row = features.loc[[timestamp]]
            
            # Fill missing features
            for feat in feature_names:
                if feat not in feat_row.columns:
                    feat_row[feat] = 0
            
            X = feat_row[feature_names]
            
            # Check exit if in position
            if position is not None:
                bars_held = i - position['entry_idx']
                
                should_exit, reason = self._check_exit(
                    position, row['close'], row['high'], row['low'],
                    bars_held, max_holding_bars
                )
                
                if should_exit:
                    exit_price = self._get_exit_price(position, row, reason)
                    pnl = self._close_position(position, exit_price, reason, timestamp, row, pair)
                    capital += pnl
                    position = None
                    
            # Generate signal if no position
            if position is None:
                try:
                    signal = model.get_trading_signal(X, min_confidence, min_timing)
                    
                    if signal['signal'] in ['buy', 'sell']:
                        # Calculate ATR for stop loss
                        lookback = min(14, i)
                        if lookback > 0:
                            recent = df.iloc[max(0, i-lookback):i+1]
                            atr = (recent['high'] - recent['low']).mean()
                        else:
                            atr = row['high'] - row['low']
                        
                        entry_price = row['close']
                        side = 'long' if signal['signal'] == 'buy' else 'short'
                        
                        # Calculate stop/take profit
                        stop_distance = atr * stop_loss_atr
                        stop_loss_pct = stop_distance / entry_price
                        
                        if side == 'long':
                            stop_loss = entry_price - stop_distance
                            take_profit = entry_price + (stop_distance * take_profit_rr)
                        else:
                            stop_loss = entry_price + stop_distance
                            take_profit = entry_price - (stop_distance * take_profit_rr)
                        
                        # Calculate position size
                        risk_amount = capital * self.risk_per_trade
                        leverage = self.risk_per_trade / stop_loss_pct
                        leverage = min(leverage, self.max_leverage)
                        
                        position_value = capital * leverage
                        size = position_value / entry_price
                        
                        position = {
                            'side': side,
                            'entry': entry_price,
                            'entry_time': timestamp,
                            'entry_idx': i,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': size,
                            'leverage': leverage,
                            'position_value': position_value,
                            'confidence': signal['confidence'],
                            'timing': signal['timing'],
                            'entry_candle': {
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close']
                            }
                        }
                        trades_count += 1
                        
                except Exception as e:
                    continue
        
        # Close remaining position
        if position is not None:
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            pnl = self._close_position(position, exit_price, 'end_of_data', df.index[-1], last_row, pair)
            capital += pnl
            
        return {
            'trades': trades_count,
            'final_capital': capital,
            'pnl': capital - self.initial_capital,
            'return_pct': (capital - self.initial_capital) / self.initial_capital * 100
        }
    
    def _check_exit(self, position, price, high, low, bars_held, max_bars):
        if bars_held >= max_bars:
            return True, 'time_exit'
        
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
    
    def _get_exit_price(self, position, row, reason):
        if reason == 'stop_loss':
            return position['stop_loss']
        elif reason == 'take_profit':
            return position['take_profit']
        else:
            return row['close']
    
    def _close_position(self, position, exit_price, reason, timestamp, exit_row, pair):
        if position['side'] == 'long':
            actual_exit = exit_price * (1 - self.slippage)
            price_change_pct = (actual_exit - position['entry']) / position['entry']
        else:
            actual_exit = exit_price * (1 + self.slippage)
            price_change_pct = (position['entry'] - actual_exit) / position['entry']
        
        raw_pnl = price_change_pct * position['position_value']
        exit_fee = position['position_value'] * self.exit_fee
        pnl = raw_pnl - exit_fee
        pnl_pct = (pnl / (position['position_value'] / position['leverage'])) * 100
        
        # Collect trade with full details
        self.trade_collector.add_trade(
            pair=pair,
            side=position['side'],
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_price=position['entry'],
            exit_price=actual_exit,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            exit_reason=reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            leverage=position['leverage'],
            position_value=position['position_value'],
            confidence=position['confidence'],
            timing_score=position['timing'],
            candle_data={
                'entry_open': position['entry_candle']['open'],
                'entry_high': position['entry_candle']['high'],
                'entry_low': position['entry_candle']['low'],
                'entry_close': position['entry_candle']['close'],
                'exit_open': exit_row['open'],
                'exit_high': exit_row['high'],
                'exit_low': exit_row['low'],
                'exit_close': exit_row['close']
            }
        )
        
        return pnl


# ============================================================
# MEXC VERIFICATION
# ============================================================

def verify_with_mexc(trades_df: pd.DataFrame, data_dir: str = "./data/candles"):
    """
    Verify trade prices against local candle data.
    Returns verification report.
    """
    print("\n" + "="*100)
    print("🔍 VERIFICATION AGAINST LOCAL CANDLE DATA")
    print("="*100)
    
    verification_results = []
    
    for idx, trade in trades_df.iterrows():
        pair = trade['pair']
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        # Load 5m data for this pair
        safe_symbol = pair.replace('/', '_').replace(':', '_')
        filepath = Path(data_dir) / f"{safe_symbol}_5m.csv"
        
        if not filepath.exists():
            print(f"⚠️  No data file for {pair}")
            continue
            
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Find entry candle
        entry_candles = df[df.index == entry_time]
        exit_candles = df[df.index == exit_time]
        
        entry_match = "❌ NOT FOUND"
        exit_match = "❌ NOT FOUND"
        
        if len(entry_candles) > 0:
            entry_candle = entry_candles.iloc[0]
            price_diff = abs(entry_candle['close'] - trade['entry_price'])
            if price_diff < 0.01 * trade['entry_price']:  # Within 1%
                entry_match = f"✅ MATCH (diff: ${price_diff:.6f})"
            else:
                entry_match = f"⚠️  DIFF: ${price_diff:.6f}"
                
        if len(exit_candles) > 0:
            exit_candle = exit_candles.iloc[0]
            # For TP/SL, check if price was within candle range
            if trade['exit_reason'] == 'take_profit':
                target = trade['take_profit']
            elif trade['exit_reason'] == 'stop_loss':
                target = trade['stop_loss']
            else:
                target = trade['exit_price']
                
            if exit_candle['low'] <= target <= exit_candle['high']:
                exit_match = f"✅ MATCH (within candle range)"
            else:
                exit_match = f"⚠️  Price ${target:.6f} outside [{exit_candle['low']:.6f}, {exit_candle['high']:.6f}]"
        
        result = {
            'trade_num': idx + 1,
            'pair': pair,
            'side': trade['side'],
            'entry_time': entry_time,
            'entry_match': entry_match,
            'exit_time': exit_time,
            'exit_match': exit_match,
            'exit_reason': trade['exit_reason'],
            'pnl': trade['pnl_usd']
        }
        verification_results.append(result)
        
        # Print result
        print(f"\nTrade #{idx+1}: {trade['side'].upper()} {pair}")
        print(f"  Entry: {entry_time} @ ${trade['entry_price']:.6f}")
        print(f"    → {entry_match}")
        print(f"  Exit:  {exit_time} @ ${trade['exit_price']:.6f} ({trade['exit_reason']})")
        print(f"    → {exit_match}")
        print(f"  PnL: ${trade['pnl_usd']:+.2f}")
    
    return verification_results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verify Backtest Trades")
    parser.add_argument("--pairs", type=int, default=19, help="Number of pairs")
    parser.add_argument("--days", type=int, default=1, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=100, help="Initial capital")
    parser.add_argument("--model-path", type=str, default="./models/v1_fresh")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--output", type=str, default="results/trades_verification.csv")
    parser.add_argument("--output-json", type=str, default="results/trades_verification.json")
    parser.add_argument("--min-conf", type=float, default=0.50)
    parser.add_argument("--min-timing", type=float, default=0.01)
    
    args = parser.parse_args()
    
    # Load model
    print("Loading V1 Fresh model...")
    model = V1FreshModel(args.model_path)
    model.load()
    print("✅ Model loaded")
    
    # Initialize feature engine
    mtf_engine = MTFFeatureEngine()
    
    # Get pairs to test
    pairs = V1_TOP_PAIRS[:args.pairs]
    print(f"\nTesting {len(pairs)} pairs for {args.days} days...")
    
    # Create backtester
    backtester = VerifyBacktester(
        initial_capital=args.capital,
        risk_per_trade=0.05,
        max_leverage=20.0
    )
    
    all_results = []
    
    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"Processing {pair}...")
        print('='*60)
        
        # Load data
        mtf_data = load_mtf_data(pair, args.data_dir)
        if mtf_data is None:
            print(f"❌ No data for {pair}")
            continue
            
        # Filter to test period
        mtf_data = filter_data_by_days(mtf_data, args.days)
        
        if len(mtf_data['5']) < 50:
            print(f"❌ Not enough data for {pair}")
            continue
            
        print(f"Data loaded: M1={len(mtf_data['1'])}, M5={len(mtf_data['5'])}, M15={len(mtf_data['15'])}")
        
        # Generate features
        features = mtf_engine.align_timeframes(
            mtf_data['1'], mtf_data['5'], mtf_data['15']
        )
        
        if len(features) == 0:
            print(f"❌ No features for {pair}")
            continue
            
        m5_df = mtf_data['5'].loc[mtf_data['5'].index.isin(features.index)]
        
        # Run backtest
        result = backtester.run(
            df=m5_df,
            features=features,
            model=model,
            pair=pair,
            min_confidence=args.min_conf,
            min_timing=args.min_timing
        )
        
        all_results.append({
            'pair': pair,
            **result
        })
        
        print(f"Result: {result['trades']} trades, PnL: ${result['pnl']:+.2f}")
    
    # Print all trades
    backtester.trade_collector.print_summary()
    
    # Export trades
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backtester.trade_collector.to_csv(args.output)
    backtester.trade_collector.to_json(args.output_json)
    
    # Verify against local data
    trades_df = backtester.trade_collector.to_dataframe()
    if len(trades_df) > 0:
        verify_with_mexc(trades_df, args.data_dir)
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print("="*100)
    
    total_trades = len(backtester.trade_collector.trades)
    wins = len([t for t in backtester.trade_collector.trades if t['pnl_usd'] > 0])
    total_pnl = sum(t['pnl_usd'] for t in backtester.trade_collector.trades)
    
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins} ({wins/total_trades*100:.1f}%)" if total_trades > 0 else "No trades")
    print(f"Total PnL: ${total_pnl:+.2f}")
    print(f"\nExported to:")
    print(f"  CSV:  {args.output}")
    print(f"  JSON: {args.output_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Compare two models on the same data

Usage:
    python scripts/compare_models.py --start 2025-11-01 --end 2025-11-30 --pairs 3
"""

import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from loguru import logger
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine


# ============================================================
# CONFIGURATION
# ============================================================

RISK_PCT = 0.05      # 5% риск на сделку
RR_RATIO = 2.0       # RR 1:2
SL_ATR_MULT = 1.5    # SL = 1.5 * ATR
MAX_LEVERAGE = 20.0  # Максимальное плечо
MAX_BARS = 50        # Максимум баров в сделке
FEE_PCT = 0.0002     # 0.02% комиссия


class BacktestModel:
    """Universal model wrapper for backtest."""
    
    def __init__(self, model_path: str, name: str = ""):
        self.model_path = Path(model_path)
        self.name = name or model_path
        self.direction_model = None
        self.scaler = None
        self.feature_columns = None
        self.entry_quality_long = None
        self.entry_quality_short = None
        
    def load(self):
        logger.info(f"Loading model: {self.name}")
        
        # Try .joblib first, then .pkl
        if (self.model_path / 'direction_model.joblib').exists():
            self.direction_model = joblib.load(self.model_path / 'direction_model.joblib')
            
            # Get feature names from model itself or from file
            if (self.model_path / 'feature_names.joblib').exists():
                self.feature_columns = joblib.load(self.model_path / 'feature_names.joblib')
            elif hasattr(self.direction_model, 'feature_names_in_'):
                self.feature_columns = list(self.direction_model.feature_names_in_)
            else:
                raise ValueError("Cannot find feature names")
            
            self.scaler = None
            
            # Try loading entry quality models
            eq_long_path = self.model_path / 'long_quality_model.joblib'
            eq_short_path = self.model_path / 'short_quality_model.joblib'
            
            if eq_long_path.exists() and eq_short_path.exists():
                self.entry_quality_long = joblib.load(eq_long_path)
                self.entry_quality_short = joblib.load(eq_short_path)
                logger.info(f"  Entry quality models loaded for {self.name}")
                
        elif (self.model_path / 'direction_model.pkl').exists():
            self.direction_model = joblib.load(self.model_path / 'direction_model.pkl')
            self.scaler = joblib.load(self.model_path / 'scaler.pkl')
            self.feature_columns = joblib.load(self.model_path / 'feature_columns.pkl')
            
            eq_long_path = self.model_path / 'entry_quality_long.pkl'
            eq_short_path = self.model_path / 'entry_quality_short.pkl'
            
            if eq_long_path.exists() and eq_short_path.exists():
                self.entry_quality_long = joblib.load(eq_long_path)
                self.entry_quality_short = joblib.load(eq_short_path)
        else:
            raise FileNotFoundError(f"No model found in {self.model_path}")
        
        logger.info(f"  {self.name}: {len(self.feature_columns)} features loaded")
        
    def predict(self, features: pd.DataFrame) -> Dict:
        available = [c for c in self.feature_columns if c in features.columns]
        X = features[available].iloc[-1:].copy()
        
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale if scaler exists
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Direction prediction
        direction_proba = self.direction_model.predict_proba(X_scaled)[0]
        direction_pred = self.direction_model.predict(X_scaled)[0]
        
        # Entry quality (if available)
        entry_quality = 0.6  # Default if no entry quality model
        if direction_pred == 1 and self.entry_quality_long:
            entry_quality = self.entry_quality_long.predict_proba(X_scaled)[0][1]
        elif direction_pred == 0 and self.entry_quality_short:
            entry_quality = self.entry_quality_short.predict_proba(X_scaled)[0][1]
        
        return {
            'direction': 'LONG' if direction_pred == 1 else 'SHORT',
            'direction_confidence': max(direction_proba),
            'entry_quality': entry_quality
        }


async def fetch_pair_data(exchange, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
    """Fetch data for a pair for the entire period."""
    try:
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        all_m1 = []
        current = since
        while current < until:
            ohlcv = await exchange.fetch_ohlcv(symbol, '1m', since=current, limit=1000)
            if not ohlcv:
                break
            all_m1.extend(ohlcv)
            current = ohlcv[-1][0] + 60000
            await asyncio.sleep(0.1)
        
        all_m5 = []
        current = since
        while current < until:
            ohlcv = await exchange.fetch_ohlcv(symbol, '5m', since=current, limit=1000)
            if not ohlcv:
                break
            all_m5.extend(ohlcv)
            current = ohlcv[-1][0] + 300000
            await asyncio.sleep(0.1)
        
        all_m15 = []
        current = since
        while current < until:
            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', since=current, limit=1000)
            if not ohlcv:
                break
            all_m15.extend(ohlcv)
            current = ohlcv[-1][0] + 900000
            await asyncio.sleep(0.1)
        
        if len(all_m1) < 100 or len(all_m5) < 50:
            return None
            
        def to_df(data):
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            return df.set_index('timestamp')
        
        return {
            'm1': to_df(all_m1),
            'm5': to_df(all_m5),
            'm15': to_df(all_m15)
        }
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def simulate_trade(df_m5: pd.DataFrame, entry_idx: int, direction: str, atr: float) -> Dict:
    """Simulate a trade on M5 data."""
    entry_price = df_m5.iloc[entry_idx]['close']
    
    sl_distance = atr * SL_ATR_MULT
    tp_distance = sl_distance * RR_RATIO
    
    sl_pct = sl_distance / entry_price
    leverage = min(RISK_PCT / sl_pct, MAX_LEVERAGE)
    
    if direction == 'LONG':
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
    
    outcome = 'time_exit'
    exit_idx = min(entry_idx + MAX_BARS, len(df_m5) - 1)
    exit_price = df_m5.iloc[exit_idx]['close']
    
    for i in range(entry_idx + 1, min(entry_idx + MAX_BARS + 1, len(df_m5))):
        bar = df_m5.iloc[i]
        
        if direction == 'LONG':
            if bar['low'] <= sl_price:
                outcome = 'stop_loss'
                exit_price = sl_price
                exit_idx = i
                break
            if bar['high'] >= tp_price:
                outcome = 'take_profit'
                exit_price = tp_price
                exit_idx = i
                break
        else:
            if bar['high'] >= sl_price:
                outcome = 'stop_loss'
                exit_price = sl_price
                exit_idx = i
                break
            if bar['low'] <= tp_price:
                outcome = 'take_profit'
                exit_price = tp_price
                exit_idx = i
                break
    
    if direction == 'LONG':
        pnl_pct = ((exit_price - entry_price) / entry_price) * leverage * 100
    else:
        pnl_pct = ((entry_price - exit_price) / entry_price) * leverage * 100
    
    fee_pct = FEE_PCT * leverage * 100 * 2
    net_pnl = pnl_pct - fee_pct
    
    return {
        'entry_idx': entry_idx,
        'exit_idx': exit_idx,
        'entry_time': df_m5.index[entry_idx],
        'exit_time': df_m5.index[exit_idx],
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'leverage': leverage,
        'outcome': outcome,
        'net_pnl_pct': net_pnl
    }


def run_model_backtest(
    model: BacktestModel,
    pairs_features: Dict,
    min_confidence: float = 0.55,
    min_entry_quality: float = 0.55,
    check_interval: int = 12
) -> List[Dict]:
    """Run backtest for a single model."""
    
    all_signals = []
    
    for symbol, pdata in pairs_features.items():
        features = pdata['features']
        m5 = pdata['m5']
        
        for i in range(100, len(features), check_interval):
            features_slice = features.iloc[:i+1]
            
            try:
                prediction = model.predict(features_slice)
            except Exception as e:
                continue
            
            dir_conf = prediction['direction_confidence']
            eq = prediction['entry_quality']
            
            if dir_conf >= min_confidence and eq >= min_entry_quality:
                signal_time = features.index[i]
                m5_idx = m5.index.get_indexer([signal_time], method='ffill')[0]
                
                if m5_idx >= 0 and m5_idx < len(m5) - MAX_BARS:
                    row = features.iloc[i]
                    atr = row.get('atr_14', row.get('M5_atr_14', row.get('m5_atr_14', 0)))
                    if atr <= 0:
                        atr = m5['high'].iloc[max(0,m5_idx-14):m5_idx].mean() - m5['low'].iloc[max(0,m5_idx-14):m5_idx].mean()
                    
                    all_signals.append({
                        'symbol': symbol,
                        'time': signal_time,
                        'm5_idx': m5_idx,
                        'direction': prediction['direction'],
                        'dir_conf': dir_conf,
                        'entry_quality': eq,
                        'atr': atr,
                        'm5': m5
                    })
    
    # Sort by time and execute one at a time
    all_signals.sort(key=lambda x: x['time'])
    
    trades = []
    position_end_time = None
    
    for signal in all_signals:
        if position_end_time and signal['time'] < position_end_time:
            continue
        
        trade = simulate_trade(
            signal['m5'],
            signal['m5_idx'],
            signal['direction'],
            signal['atr']
        )
        
        trade['pair'] = signal['symbol'].replace('/USDT:USDT', '')
        trade['dir_conf'] = signal['dir_conf']
        trade['entry_quality'] = signal['entry_quality']
        
        trades.append(trade)
        position_end_time = trade['exit_time']
    
    return trades


def print_comparison(results: Dict[str, List[Dict]]):
    """Print comparison of two models."""
    
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    
    # Headers
    print(f"\n{'Metric':<25}", end="")
    for name in results.keys():
        print(f"{name:>25}", end="")
    print()
    print("-" * 100)
    
    # Calculate stats for each model
    stats = {}
    for name, trades in results.items():
        if trades:
            wins = [t for t in trades if t['net_pnl_pct'] > 0]
            losses = [t for t in trades if t['net_pnl_pct'] <= 0]
            tp = [t for t in trades if t['outcome'] == 'take_profit']
            sl = [t for t in trades if t['outcome'] == 'stop_loss']
            
            stats[name] = {
                'trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'total_pnl': sum(t['net_pnl_pct'] for t in trades),
                'avg_win': np.mean([t['net_pnl_pct'] for t in wins]) if wins else 0,
                'avg_loss': np.mean([t['net_pnl_pct'] for t in losses]) if losses else 0,
                'tp_count': len(tp),
                'sl_count': len(sl),
                'profit_factor': abs(sum(t['net_pnl_pct'] for t in wins) / sum(t['net_pnl_pct'] for t in losses)) if losses and sum(t['net_pnl_pct'] for t in losses) != 0 else 0
            }
        else:
            stats[name] = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0, 'tp_count': 0, 'sl_count': 0, 'profit_factor': 0}
    
    # Print metrics
    metrics = [
        ('Total Trades', 'trades', ''),
        ('Wins', 'wins', ''),
        ('Losses', 'losses', ''),
        ('Win Rate (%)', 'win_rate', '.1f'),
        ('Total PnL (%)', 'total_pnl', '.2f'),
        ('Avg Win (%)', 'avg_win', '.2f'),
        ('Avg Loss (%)', 'avg_loss', '.2f'),
        ('Take Profits', 'tp_count', ''),
        ('Stop Losses', 'sl_count', ''),
        ('Profit Factor', 'profit_factor', '.2f'),
    ]
    
    for label, key, fmt in metrics:
        print(f"{label:<25}", end="")
        for name in results.keys():
            val = stats[name][key]
            if fmt:
                print(f"{val:>25{fmt}}", end="")
            else:
                print(f"{val:>25}", end="")
        print()
    
    print("=" * 100)
    
    # Winner
    pnls = {name: stats[name]['total_pnl'] for name in results.keys()}
    winner = max(pnls, key=pnls.get)
    print(f"\n🏆 WINNER: {winner} with {pnls[winner]:+.2f}% PnL")
    
    # Daily comparison
    print("\n" + "=" * 100)
    print("DAILY PnL COMPARISON")
    print("=" * 100)
    
    # Group by date
    daily = {name: {} for name in results.keys()}
    for name, trades in results.items():
        for t in trades:
            date = t['entry_time'].strftime('%Y-%m-%d')
            if date not in daily[name]:
                daily[name][date] = 0
            daily[name][date] += t['net_pnl_pct']
    
    # Get all dates
    all_dates = set()
    for name in results.keys():
        all_dates.update(daily[name].keys())
    
    cumulative = {name: 0 for name in results.keys()}
    
    print(f"\n{'Date':<15}", end="")
    for name in results.keys():
        print(f"{name + ' PnL':>20}", end="")
    print(f"{'Diff':>15}")
    print("-" * 100)
    
    for date in sorted(all_dates):
        print(f"{date:<15}", end="")
        pnls_day = []
        for name in results.keys():
            pnl = daily[name].get(date, 0)
            pnls_day.append(pnl)
            cumulative[name] += pnl
            print(f"{pnl:>+20.2f}", end="")
        diff = pnls_day[0] - pnls_day[1] if len(pnls_day) > 1 else 0
        print(f"{diff:>+15.2f}")
    
    print("-" * 100)
    print(f"{'TOTAL':<15}", end="")
    for name in results.keys():
        print(f"{cumulative[name]:>+20.2f}", end="")
    diff = list(cumulative.values())[0] - list(cumulative.values())[1] if len(cumulative) > 1 else 0
    print(f"{diff:>+15.2f}")
    print("=" * 100)


async def main():
    parser = argparse.ArgumentParser(description="Compare Models")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-entry-quality", type=float, default=0.55)
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("MODEL COMPARISON: v2_improved vs v1_fresh")
    print("=" * 100)
    print(f"Period:           {args.start} to {args.end}")
    print(f"Pairs:            {args.pairs}")
    print(f"Risk per trade:   {RISK_PCT*100:.0f}%")
    print(f"RR Ratio:         1:{RR_RATIO:.0f}")
    print("=" * 100)
    
    # Load models
    models = {
        'v2_improved': BacktestModel('./models/v2_improved', 'v2_improved'),
        'v1_fresh': BacktestModel('./models/v1_fresh', 'v1_fresh')
    }
    
    for model in models.values():
        model.load()
    
    # Load pairs
    import json
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_list.json'
    with open(pairs_file) as f:
        pairs_data = json.load(f)
    all_pairs = [p['symbol'] for p in pairs_data['pairs']]
    pairs = all_pairs[:args.pairs]
    
    # Fetch data
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d') + timedelta(days=1)
    
    logger.info(f"Fetching data from {args.start} to {args.end}")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    mtf_fe = MTFFeatureEngine()
    
    pairs_data_fetched = {}
    
    for symbol in pairs:
        logger.info(f"Fetching {symbol}...")
        data = await fetch_pair_data(exchange, symbol, start - timedelta(days=3), end)
        if data:
            pairs_data_fetched[symbol] = data
            logger.info(f"  {symbol}: {len(data['m1'])} m1, {len(data['m5'])} m5, {len(data['m15'])} m15 bars")
        await asyncio.sleep(0.2)
    
    await exchange.close()
    
    # Generate features
    logger.info("Generating features...")
    pairs_features = {}
    
    for symbol, data in pairs_data_fetched.items():
        try:
            features = mtf_fe.align_timeframes(data['m1'], data['m5'], data['m15'])
            features = features[(features.index >= start) & (features.index < end)]
            
            if len(features) > 50:
                pairs_features[symbol] = {
                    'features': features,
                    'm5': data['m5'][(data['m5'].index >= start) & (data['m5'].index < end)]
                }
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
    
    logger.info(f"Features ready for {len(pairs_features)} pairs")
    
    # Run backtest for each model
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nRunning backtest for {name}...")
        trades = run_model_backtest(
            model=model,
            pairs_features=pairs_features,
            min_confidence=args.min_confidence,
            min_entry_quality=args.min_entry_quality
        )
        results[name] = trades
        logger.info(f"  {name}: {len(trades)} trades")
    
    # Print comparison
    print_comparison(results)


if __name__ == '__main__':
    asyncio.run(main())

#!/usr/bin/env python3
"""
Backtest over a date range (e.g., November 2025)

Usage:
    python scripts/backtest_period.py --start 2025-11-01 --end 2025-11-30 --model-path models/v2_improved
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
RR_RATIO = 3.0       # RR 1:3
SL_ATR_MULT = 1.5    # SL = 1.5 * ATR
MAX_LEVERAGE = 20.0  # Максимальное плечо
MAX_BARS = 50        # Максимум баров в сделке
FEE_PCT = 0.0002     # 0.02% комиссия


class BacktestModel:
    """Model wrapper for backtest."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.direction_model = None
        self.scaler = None
        self.feature_columns = None
        self.entry_quality_long = None
        self.entry_quality_short = None
        
    def load(self):
        logger.info(f"Loading models from {self.model_path}")
        
        # Try different extensions
        if (self.model_path / 'direction_model.joblib').exists():
            self.direction_model = joblib.load(self.model_path / 'direction_model.joblib')
            self.feature_columns = joblib.load(self.model_path / 'feature_names.joblib')
            self.scaler = None  # V2 improved doesn't use scaler separately
            
            eq_long_path = self.model_path / 'long_quality_model.joblib'
            eq_short_path = self.model_path / 'short_quality_model.joblib'
        else:
            self.direction_model = joblib.load(self.model_path / 'direction_model.pkl')
            self.scaler = joblib.load(self.model_path / 'scaler.pkl')
            self.feature_columns = joblib.load(self.model_path / 'feature_columns.pkl')
            
            eq_long_path = self.model_path / 'entry_quality_long.pkl'
            eq_short_path = self.model_path / 'entry_quality_short.pkl'
        
        if eq_long_path.exists() and eq_short_path.exists():
            self.entry_quality_long = joblib.load(eq_long_path)
            self.entry_quality_short = joblib.load(eq_short_path)
            logger.info("Entry quality models loaded")
        
        logger.info("Models loaded successfully")
        
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
        
        # Entry quality
        entry_quality = 0.5
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
        # Fetch 1m, 5m, 15m data
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
            
        def to_df(data, tf):
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            return df.set_index('timestamp')
        
        return {
            'm1': to_df(all_m1, '1m'),
            'm5': to_df(all_m5, '5m'),
            'm15': to_df(all_m15, '15m')
        }
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def simulate_trade(df_m5: pd.DataFrame, entry_idx: int, direction: str, atr: float) -> Dict:
    """Simulate a trade on M5 data."""
    entry_price = df_m5.iloc[entry_idx]['close']
    
    # Calculate SL and TP
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
    
    # Simulate bar by bar
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
    
    # Calculate PnL
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
        'sl_price': sl_price,
        'tp_price': tp_price,
        'leverage': leverage,
        'outcome': outcome,
        'pnl_pct': pnl_pct,
        'net_pnl_pct': net_pnl
    }


async def run_period_backtest(
    model: BacktestModel,
    pairs: List[str],
    start_date: str,
    end_date: str,
    min_confidence: float = 0.55,
    min_entry_quality: float = 0.55,
    check_interval: int = 12  # Check every 12 M5 bars = 1 hour
):
    """Run backtest over a date range with ONE position at a time."""
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    # Fetch data for all pairs
    exchange = ccxt.binance({'enableRateLimit': True})
    fe = FeatureEngine()
    mtf_fe = MTFFeatureEngine()
    
    pairs_data = {}
    
    for symbol in pairs:
        logger.info(f"Fetching {symbol}...")
        data = await fetch_pair_data(exchange, symbol, start - timedelta(days=3), end)
        if data:
            pairs_data[symbol] = data
            logger.info(f"  {symbol}: {len(data['m1'])} m1, {len(data['m5'])} m5, {len(data['m15'])} m15 bars")
        await asyncio.sleep(0.2)
    
    await exchange.close()
    
    logger.info(f"\nLoaded data for {len(pairs_data)} pairs")
    
    # Generate features for all pairs
    logger.info("Generating features...")
    pairs_features = {}
    
    for symbol, data in pairs_data.items():
        try:
            features = mtf_fe.align_timeframes(data['m1'], data['m5'], data['m15'])
            
            # Filter to target period
            features = features[(features.index >= start) & (features.index < end)]
            
            if len(features) > 50:
                pairs_features[symbol] = {
                    'features': features,
                    'm5': data['m5'][(data['m5'].index >= start) & (data['m5'].index < end)]
                }
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
    
    logger.info(f"Features ready for {len(pairs_features)} pairs")
    
    # Collect all potential entry points
    all_signals = []
    
    for symbol, pdata in pairs_features.items():
        features = pdata['features']
        m5 = pdata['m5']
        
        # Check every check_interval bars
        for i in range(100, len(features), check_interval):
            row = features.iloc[i]
            features_slice = features.iloc[:i+1]
            
            prediction = model.predict(features_slice)
            
            dir_conf = prediction['direction_confidence']
            eq = prediction['entry_quality']
            
            if dir_conf >= min_confidence and eq >= min_entry_quality:
                # Find corresponding M5 index
                signal_time = features.index[i]
                m5_idx = m5.index.get_indexer([signal_time], method='ffill')[0]
                
                if m5_idx >= 0 and m5_idx < len(m5) - MAX_BARS:
                    # Get ATR
                    atr = row.get('atr_14', row.get('M5_atr_14', 0))
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
    
    logger.info(f"\nCollected {len(all_signals)} potential signals")
    
    # Sort by time and execute one at a time
    all_signals.sort(key=lambda x: x['time'])
    
    trades = []
    position_end_time = None
    
    for signal in all_signals:
        # Skip if we're in a position
        if position_end_time and signal['time'] < position_end_time:
            continue
        
        # Execute trade
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
        
        # Log trade
        emoji = "🟢" if trade['net_pnl_pct'] > 0 else "🔴"
        logger.info(f"{emoji} {trade['direction']} {trade['pair']} @ {trade['entry_time']} | "
                   f"Exit: {trade['outcome']} @ {trade['exit_time']} | "
                   f"Lev: {trade['leverage']:.0f}x | PnL: {trade['net_pnl_pct']:+.2f}%")
    
    return trades


def print_summary(trades: List[Dict]):
    """Print trading summary."""
    if not trades:
        print("\nNo trades executed")
        return
    
    print("\n" + "=" * 80)
    print("PERIOD BACKTEST SUMMARY")
    print("=" * 80)
    
    # Group by date
    trades_by_date = {}
    for t in trades:
        date = t['entry_time'].strftime('%Y-%m-%d')
        if date not in trades_by_date:
            trades_by_date[date] = []
        trades_by_date[date].append(t)
    
    print(f"\nTrades by date:")
    print("-" * 80)
    
    cumulative = 0
    for date in sorted(trades_by_date.keys()):
        day_trades = trades_by_date[date]
        day_pnl = sum(t['net_pnl_pct'] for t in day_trades)
        day_wins = sum(1 for t in day_trades if t['net_pnl_pct'] > 0)
        cumulative += day_pnl
        print(f"  {date}: {len(day_trades)} trades, {day_wins} wins, "
              f"PnL: {day_pnl:+.2f}%, Cumulative: {cumulative:+.2f}%")
    
    print("-" * 80)
    
    # Overall stats
    wins = [t for t in trades if t['net_pnl_pct'] > 0]
    losses = [t for t in trades if t['net_pnl_pct'] <= 0]
    
    total_pnl = sum(t['net_pnl_pct'] for t in trades)
    avg_win = np.mean([t['net_pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['net_pnl_pct'] for t in losses]) if losses else 0
    win_rate = len(wins) / len(trades) * 100
    
    tp_trades = [t for t in trades if t['outcome'] == 'take_profit']
    sl_trades = [t for t in trades if t['outcome'] == 'stop_loss']
    te_trades = [t for t in trades if t['outcome'] == 'time_exit']
    
    print(f"\nOVERALL STATISTICS")
    print("=" * 80)
    print(f"Total Trades:     {len(trades)}")
    print(f"Wins / Losses:    {len(wins)} / {len(losses)}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Total PnL:        {total_pnl:+.2f}%")
    print(f"Avg Win:          {avg_win:+.2f}%")
    print(f"Avg Loss:         {avg_loss:+.2f}%")
    
    print(f"\nBy Outcome:")
    if tp_trades:
        print(f"  Take Profit:    {len(tp_trades)} trades ({len(tp_trades)/len(trades)*100:.1f}%)")
    if sl_trades:
        print(f"  Stop Loss:      {len(sl_trades)} trades ({len(sl_trades)/len(trades)*100:.1f}%)")
    if te_trades:
        print(f"  Time Exit:      {len(te_trades)} trades ({len(te_trades)/len(trades)*100:.1f}%)")
    
    # By pair
    pairs_stats = {}
    for t in trades:
        pair = t['pair']
        if pair not in pairs_stats:
            pairs_stats[pair] = {'trades': 0, 'wins': 0, 'pnl': 0}
        pairs_stats[pair]['trades'] += 1
        pairs_stats[pair]['wins'] += 1 if t['net_pnl_pct'] > 0 else 0
        pairs_stats[pair]['pnl'] += t['net_pnl_pct']
    
    print(f"\nBy Pair (top 10 by trades):")
    sorted_pairs = sorted(pairs_stats.items(), key=lambda x: x[1]['trades'], reverse=True)[:10]
    for pair, stats in sorted_pairs:
        wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        print(f"  {pair:<8} {stats['trades']:>3} trades, {wr:>5.1f}% WR, PnL: {stats['pnl']:+.2f}%")
    
    print("=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Period Backtest")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=str, default="./models/v2_improved")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-entry-quality", type=float, default=0.55)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PERIOD BACKTEST - ONE POSITION AT A TIME")
    print("=" * 80)
    print(f"Period:           {args.start} to {args.end}")
    print(f"Model:            {args.model_path}")
    print(f"Risk per trade:   {RISK_PCT*100:.0f}%")
    print(f"RR Ratio:         1:{RR_RATIO:.0f}")
    print(f"Min Confidence:   {args.min_confidence*100:.0f}%")
    print(f"Min Entry Quality:{args.min_entry_quality*100:.0f}%")
    print("=" * 80)
    
    # Load pairs
    import json
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_list.json'
    with open(pairs_file) as f:
        pairs_data = json.load(f)
    all_pairs = [p['symbol'] for p in pairs_data['pairs']]
    pairs = all_pairs[:args.pairs]
    
    # Load model
    model = BacktestModel(args.model_path)
    model.load()
    
    # Run backtest
    trades = await run_period_backtest(
        model=model,
        pairs=pairs,
        start_date=args.start,
        end_date=args.end,
        min_confidence=args.min_confidence,
        min_entry_quality=args.min_entry_quality
    )
    
    print_summary(trades)


if __name__ == '__main__':
    asyncio.run(main())

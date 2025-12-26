#!/usr/bin/env python3
"""
Fetch fresh data from Binance and backtest on a specific date.

Usage:
    python scripts/fetch_and_backtest.py --date 2025-12-25 --model-path ./models/v1_best
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from loguru import logger
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine, TOP_20_PAIRS


# ============================================================
# V1 BEST MODEL WRAPPER
# ============================================================

class V1BestModel:
    """Wrapper for V1 Best model with entry quality."""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.direction_model = None
        self.timing_model = None
        self.strength_model = None
        self.volatility_model = None
        self.entry_quality_model = None
        self.long_quality_model = None
        self.short_quality_model = None
        self._is_trained = False
        
    def load(self):
        """Load all models."""
        logger.info(f"Loading V1 Best models from {self.model_dir}")
        
        self.direction_model = joblib.load(self.model_dir / 'direction_model.joblib')
        self.timing_model = joblib.load(self.model_dir / 'timing_model.joblib')
        self.strength_model = joblib.load(self.model_dir / 'strength_model.joblib')
        self.volatility_model = joblib.load(self.model_dir / 'volatility_model.joblib')
        
        # Entry quality models (optional)
        eq_path = self.model_dir / 'entry_quality_model.joblib'
        if eq_path.exists():
            self.entry_quality_model = joblib.load(eq_path)
            self.long_quality_model = joblib.load(self.model_dir / 'long_quality_model.joblib')
            self.short_quality_model = joblib.load(self.model_dir / 'short_quality_model.joblib')
            logger.info("Entry quality models loaded")
        
        self._is_trained = True
        logger.info("V1 Best models loaded successfully")
        
    def get_signal(self, X: pd.DataFrame) -> Dict:
        """Get trading signal with all predictions."""
        if not self._is_trained:
            raise RuntimeError("Models not loaded")
        
        direction_proba = self.direction_model.predict_proba(X)[0]
        p_down, p_sideways, p_up = direction_proba
        
        timing_proba = self.timing_model.predict_proba(X)[0][1]
        strength = self.strength_model.predict(X)[0]
        volatility = self.volatility_model.predict(X)[0]
        
        entry_quality = None
        long_quality = None
        short_quality = None
        
        if self.entry_quality_model is not None:
            entry_quality = self.entry_quality_model.predict_proba(X)[0][1]
            long_quality = self.long_quality_model.predict_proba(X)[0][1]
            short_quality = self.short_quality_model.predict_proba(X)[0][1]
        
        return {
            'p_down': p_down,
            'p_sideways': p_sideways,
            'p_up': p_up,
            'timing': timing_proba,
            'strength': strength,
            'volatility': volatility,
            'entry_quality': entry_quality,
            'long_quality': long_quality,
            'short_quality': short_quality
        }


# ============================================================
# BINANCE DATA FETCHER
# ============================================================

async def fetch_binance_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Binance Futures.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (1m, 5m, 15m, etc)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
    
    Returns:
        DataFrame with OHLCV data
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    try:
        await exchange.load_markets()
        
        # Convert symbol format
        # From 'XAUT/USDT:USDT' to 'XAUT/USDT'
        binance_symbol = symbol.split(':')[0] if ':' in symbol else symbol
        
        # Calculate timestamps
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        since = int(start_dt.timestamp() * 1000)
        until = int(end_dt.timestamp() * 1000)
        
        tf_ms = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '1h': 3600000,
        }
        
        limit = 1000
        all_ohlcv = []
        current_since = since
        
        while current_since < until:
            ohlcv = await exchange.fetch_ohlcv(
                binance_symbol,
                timeframe,
                since=current_since,
                limit=limit
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            last_ts = ohlcv[-1][0]
            if last_ts >= until:
                break
            
            if len(ohlcv) < limit:
                break
            
            current_since = last_ts + tf_ms.get(timeframe, 300000)
            await asyncio.sleep(0.1)
        
        if not all_ohlcv:
            return None
        
        df = pd.DataFrame(all_ohlcv, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Filter to date range
        df = df[(df.index >= start_dt) & (df.index < end_dt)]
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Binance: {e}")
        return None
    finally:
        await exchange.close()


async def fetch_pair_mtf(
    pair: str,
    start_date: str,
    end_date: str
) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch M1, M5, M15 data for a pair."""
    logger.info(f"  Fetching {pair} from Binance...")
    
    data = {}
    for tf in ['1m', '5m', '15m']:
        df = await fetch_binance_data(pair, tf, start_date, end_date)
        if df is None or len(df) < 100:
            logger.warning(f"    No {tf} data for {pair}")
            return None
        data[tf.replace('m', '')] = df
        logger.info(f"    {tf}: {len(df)} bars")
    
    return data


# ============================================================
# TRADE SIMULATION
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    max_bars: int = 50
) -> Dict:
    """Simulate a trade and return result."""
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        
        if side == 'long':
            if low <= stop_loss:
                pnl_pct = (stop_loss - entry_price) / entry_price
                return {'outcome': 'stop_loss', 'pnl_pct': pnl_pct, 'exit_price': stop_loss, 'bars_held': i - entry_idx}
            if high >= take_profit:
                pnl_pct = (take_profit - entry_price) / entry_price
                return {'outcome': 'take_profit', 'pnl_pct': pnl_pct, 'exit_price': take_profit, 'bars_held': i - entry_idx}
        else:
            if high >= stop_loss:
                pnl_pct = (entry_price - stop_loss) / entry_price
                return {'outcome': 'stop_loss', 'pnl_pct': pnl_pct, 'exit_price': stop_loss, 'bars_held': i - entry_idx}
            if low <= take_profit:
                pnl_pct = (entry_price - take_profit) / entry_price
                return {'outcome': 'take_profit', 'pnl_pct': pnl_pct, 'exit_price': take_profit, 'bars_held': i - entry_idx}
    
    exit_price = df['close'].iloc[min(entry_idx + max_bars, len(df) - 1)]
    if side == 'long':
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price
    
    return {'outcome': 'time_exit', 'pnl_pct': pnl_pct, 'exit_price': exit_price, 'bars_held': max_bars}


# ============================================================
# MAIN BACKTEST
# ============================================================

async def run_backtest(
    model: V1BestModel,
    pairs: List[str],
    target_date: str,
    min_confidence: float = 0.50,
    min_timing: float = 0.50,
    min_entry_quality: float = 0.50,
    sl_atr: float = 1.5,
    tp_rr: float = 3.0,  # RR 1:3
    risk_pct: float = 0.05  # 5% risk per trade
) -> List[Dict]:
    """Run backtest on specific date with fresh Binance data."""
    all_trades = []
    mtf_engine = MTFFeatureEngine()
    
    # Need extra days before for warmup
    target = datetime.strptime(target_date, '%Y-%m-%d')
    start_date = (target - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (target + timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    for pair in pairs:
        logger.info(f"\nProcessing {pair}...")
        
        # Fetch fresh data from Binance
        data = await fetch_pair_mtf(pair, start_date, end_date)
        if data is None:
            continue
        
        m1_df = data['1']
        m5_df = data['5']
        m15_df = data['15']
        
        if len(m5_df) < 200:
            logger.warning(f"Not enough data for {pair}")
            continue
        
        # Generate features
        try:
            features = mtf_engine.align_timeframes(m1_df, m5_df, m15_df)
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = pd.Categorical(features[col]).codes
            features = features.fillna(0)
        except Exception as e:
            logger.error(f"Feature generation failed for {pair}: {e}")
            continue
        
        # Calculate ATR
        atr = calculate_atr(m5_df)
        
        # Filter to target date only
        target_start = target
        target_end = target + timedelta(days=1)
        
        date_mask = (m5_df.index >= target_start) & (m5_df.index < target_end)
        target_indices = m5_df.index[date_mask]
        
        if len(target_indices) == 0:
            logger.warning(f"No data on {target_date} for {pair}")
            continue
        
        logger.info(f"  {len(target_indices)} bars on {target_date}")
        
        # Scan for signals every 15 bars
        position_open = False
        
        for idx, timestamp in enumerate(target_indices):
            if idx % 15 != 0:
                continue
            if position_open:
                continue
            if timestamp not in features.index:
                continue
            
            current_features = features.loc[[timestamp]]
            current_price = m5_df.loc[timestamp, 'close']
            current_atr = atr.loc[timestamp]
            
            signal = model.get_signal(current_features)
            
            side = None
            confidence = 0
            
            # Check for LONG
            if signal['p_up'] >= min_confidence and signal['p_up'] > signal['p_down']:
                if signal['timing'] >= min_timing:
                    if signal['long_quality'] is not None:
                        if signal['long_quality'] >= min_entry_quality:
                            side = 'long'
                            confidence = signal['p_up']
                    else:
                        side = 'long'
                        confidence = signal['p_up']
            
            # Check for SHORT
            elif signal['p_down'] >= min_confidence and signal['p_down'] > signal['p_up']:
                if signal['timing'] >= min_timing:
                    if signal['short_quality'] is not None:
                        if signal['short_quality'] >= min_entry_quality:
                            side = 'short'
                            confidence = signal['p_down']
                    else:
                        side = 'short'
                        confidence = signal['p_down']
            
            if side is None:
                continue
            
            # Calculate SL/TP
            sl_distance = current_atr * sl_atr
            tp_distance = sl_distance * tp_rr
            
            if side == 'long':
                stop_loss = current_price - sl_distance
                take_profit = current_price + tp_distance
            else:
                stop_loss = current_price + sl_distance
                take_profit = current_price - tp_distance
            
            try:
                entry_idx = m5_df.index.get_loc(timestamp)
            except:
                continue
            
            # Simulate trade
            result = simulate_trade(m5_df, entry_idx, side, current_price, stop_loss, take_profit, max_bars=50)
            
            # Calculate leveraged PnL with 5% risk
            # leverage = risk_pct / sl_pct
            # At SL: loss = sl_pct * leverage = risk_pct = 5%
            # At TP: profit = tp_pct * leverage = sl_pct * rr * (risk_pct/sl_pct) = risk_pct * rr = 15%
            sl_pct = abs(stop_loss - current_price) / current_price
            leverage = min(risk_pct / sl_pct, 20.0) if sl_pct > 0 else 1.0
            leverage = round(leverage)  # Round to integer
            leveraged_pnl = result['pnl_pct'] * leverage
            fee_pct = 0.0004 * leverage
            net_pnl = leveraged_pnl - fee_pct
            
            trade = {
                'pair': pair,
                'timestamp': timestamp,
                'side': side,
                'entry_price': current_price,
                'confidence': confidence,
                'timing': signal['timing'],
                'entry_quality': signal['entry_quality'],
                'outcome': result['outcome'],
                'pnl_pct': result['pnl_pct'] * 100,
                'leverage': leverage,
                'net_pnl_pct': net_pnl * 100
            }
            
            all_trades.append(trade)
            position_open = True
            
            logger.info(f"  {side.upper()} @ {timestamp.strftime('%H:%M')} | "
                       f"Conf: {confidence:.0%} | "
                       f"Result: {result['outcome']} | "
                       f"PnL: {net_pnl*100:+.2f}%")
    
    return all_trades


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance data and backtest")
    parser.add_argument("--date", type=str, default="2025-12-25",
                       help="Date to backtest (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=str, default="./models/v1_best",
                       help="Path to model")
    parser.add_argument("--pairs", type=int, default=20,
                       help="Number of pairs to test")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--min-timing", type=float, default=0.50)
    parser.add_argument("--min-entry-quality", type=float, default=0.50)
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"FETCH & BACKTEST ON {args.date}")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Pairs: {args.pairs}")
    print("="*70)
    
    # Load model
    model = V1BestModel(args.model_path)
    model.load()
    
    # Get pairs (convert format for Binance)
    pairs = TOP_20_PAIRS[:args.pairs]
    
    # Run backtest
    trades = asyncio.run(run_backtest(
        model=model,
        pairs=pairs,
        target_date=args.date,
        min_confidence=args.min_confidence,
        min_timing=args.min_timing,
        min_entry_quality=args.min_entry_quality
    ))
    
    if not trades:
        print("\n❌ No trades found")
        return 0
    
    # Summary
    df_trades = pd.DataFrame(trades)
    
    print("\n" + "="*70)
    print("TRADES SUMMARY")
    print("="*70)
    
    print(f"{'Pair':<12} {'Time':<6} {'Side':<6} {'Conf':<6} {'Outcome':<12} {'Net PnL':<10}")
    print("-" * 60)
    for _, t in df_trades.iterrows():
        print(f"{t['pair'].split('/')[0]:<12} "
              f"{t['timestamp'].strftime('%H:%M'):<6} "
              f"{t['side'].upper():<6} "
              f"{t['confidence']:.0%}   "
              f"{t['outcome']:<12} "
              f"{t['net_pnl_pct']:+.2f}%")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    total_trades = len(trades)
    wins = df_trades[df_trades['net_pnl_pct'] > 0]
    losses = df_trades[df_trades['net_pnl_pct'] <= 0]
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = df_trades['net_pnl_pct'].sum()
    
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:+.2f}%")
    
    if len(wins) > 0:
        print(f"Avg Win: {wins['net_pnl_pct'].mean():+.2f}%")
    if len(losses) > 0:
        print(f"Avg Loss: {losses['net_pnl_pct'].mean():+.2f}%")
    
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

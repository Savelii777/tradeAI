#!/usr/bin/env python3
"""
Train V8 TERMINATOR - Smart Logic & Advanced Features
"I'll be back... with profits."

Philosophy:
- User has ONE execution slot (can only hold 1 trade at a time).
- We cannot afford to waste time on low-probability or low-profit trades.
- We need High Win Rate (>60%) and High Expected Value.
- Target: ~10-15 trades per day across ALL 20 pairs (approx 0.5 - 1 trade per pair/day).

Changes from V7 Sniper:
1.  **Wick Features:** Model learns to recognize Pinbars/Shooting Stars.
2.  **BTC Veto:** Global trend filter. No Longs if BTC is dumping.
3.  **Smart Exit:** RSI-based early exit to secure profits before reversal.
4.  **Dynamic Trail:** Tighter trailing stop once in profit.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import ccxt
from loguru import logger
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine


# ============================================================
# CONFIG
# ============================================================
SL_ATR_MULT = 1.5
MAX_LEVERAGE = 20.0
RISK_PCT = 0.05      # 5% Risk per trade
FEE_PCT = 0.0002     # 0.02% Maker/Taker (MEXC Futures)
LOOKAHEAD = 12       # 1 hour on M5

# Smart Logic Thresholds (V8)
RSI_EXIT_LONG = 75
RSI_EXIT_SHORT = 25
WICK_REJECTION_RATIO = 2.0 # Wick must be 2x body to reject


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_binance_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    """Fetch candles from Binance (or available exchange) via CCXT."""
    exchange = ccxt.binance()
    
    # Convert symbol to CCXT format (e.g., BTC_USDT -> BTC/USDT)
    # Our pairs are like BTC/USDT already or BTC_USDT
    symbol = symbol.replace('_', '/')
    if '/' not in symbol:
        symbol = f"{symbol[:-4]}/{symbol[-4:]}" # AAVEUSDT -> AAVE/USDT
        
    since = int(start_date.timestamp() * 1000)
    limit = 1000
    all_candles = []
    
    try:
        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            # Stop if we passed end_date
            if candles[-1][0] > end_date.timestamp() * 1000:
                break
                
            time.sleep(0.1) # Rate limit
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter exact range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
        
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return pd.DataFrame()


# ============================================================
# FEATURES
# ============================================================
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume + Wick + Microstructure features (V8)."""
    df = df.copy()
    
    # 1. Volume Features (V3)
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # OBV
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # Volume momentum
    df['vol_momentum'] = df['volume'].pct_change(5)

    # 2. Wick Features (V4 Smart)
    # Upper Wick = High - max(Open, Close)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    # Lower Wick = min(Open, Close) - Low
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    # Body = abs(Close - Open)
    df['body'] = (df['close'] - df['open']).abs()
    
    # Ratios (Normalized)
    df['upper_wick_pct'] = df['upper_wick'] / df['close']
    df['lower_wick_pct'] = df['lower_wick'] / df['close']
    df['body_pct'] = df['body'] / df['close']
    
    # Wick-to-Body Ratio (for filtering)
    df['wick_ratio_up'] = df['upper_wick'] / (df['body'] + 1e-9)
    df['wick_ratio_down'] = df['lower_wick'] / (df['body'] + 1e-9)
    
    # 3. RSI (For Smart Exit)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


# ============================================================
# V1 TARGETS (The "Great Base")
# ============================================================
def create_targets_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Create V1-style targets."""
    df = df.copy()
    df['atr'] = calculate_atr(df)
    
    # 1. Direction (Adaptive Threshold)
    # V7: Set to 0.5% (Significant moves only). 
    # We don't want to learn noise.
    rolling_vol = df['close'].pct_change().rolling(window=100).std()
    threshold = np.maximum(rolling_vol, 0.005)  # Min 0.5%
    future_return = df['close'].pct_change(LOOKAHEAD).shift(-LOOKAHEAD)
    
    # 0=Down, 1=Sideways, 2=Up
    df['target_dir'] = np.where(
        future_return > threshold, 2,
        np.where(future_return < -threshold, 0, 1)
    )
    
    # 2. Timing (Entry Quality)
    future_lows = df['low'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
    future_highs = df['high'].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
    
    # For Long
    adv_long = (df['close'] - future_lows) / df['atr']
    fav_long = (future_highs - df['close']) / df['atr']
    # For Short
    adv_short = (future_highs - df['close']) / df['atr']
    fav_short = (df['close'] - future_lows) / df['atr']
    
    # Combined Timing Target (1 if good entry for the correct direction)
    is_good_long = (fav_long > adv_long) & (fav_long > 1.0)
    is_good_short = (fav_short > adv_short) & (fav_short > 1.0)
    
    df['target_timing'] = 0
    df.loc[(df['target_dir'] == 2) & is_good_long, 'target_timing'] = 1
    df.loc[(df['target_dir'] == 0) & is_good_short, 'target_timing'] = 1
    
    # 3. Strength (Potential Move in ATRs)
    # Used for Dynamic TP prediction
    move_long = (future_highs - df['close']) / df['atr']
    move_short = (df['close'] - future_lows) / df['atr']
    
    df['target_strength'] = np.where(df['target_dir'] == 2, move_long, 
                                   np.where(df['target_dir'] == 0, move_short, 0))
    df['target_strength'] = df['target_strength'].clip(0, 10)
    
    return df


# ============================================================
# TRAINING (V1 Params)
# ============================================================
def train_models(X_train, y_train, X_val, y_val):
    """Train models using V1 conservative parameters."""
    
    # 1. Direction Model (Multiclass)
    print("   Training Direction Model...")
    dir_model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        n_estimators=600, max_depth=5, num_leaves=20, # Slightly boosted for V7
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    dir_model.fit(X_train, y_train['target_dir'], 
                  eval_set=[(X_val, y_val['target_dir'])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # 2. Timing Model (Binary)
    print("   Training Timing Model...")
    timing_model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        n_estimators=400, max_depth=5, num_leaves=15,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    timing_model.fit(X_train, y_train['target_timing'],
                     eval_set=[(X_val, y_val['target_timing'])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # 3. Strength Model (Regression)
    print("   Training Strength Model...")
    strength_model = lgb.LGBMRegressor(
        objective='regression', metric='mse',
        n_estimators=400, max_depth=5, num_leaves=15,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    strength_model.fit(X_train, y_train['target_strength'],
                       eval_set=[(X_val, y_val['target_strength'])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
    
    return {
        'direction': dir_model,
        'timing': timing_model,
        'strength': strength_model
    }


# ============================================================
# PORTFOLIO BACKTEST (V8 Terminator)
# ============================================================
def generate_signals(df: pd.DataFrame, feature_cols: list, models: dict, pair_name: str,
                    min_conf: float = 0.55, min_timing: float = 0.60, min_strength: float = 2.0) -> list:
    """Generate all valid signals for a single pair."""
    signals = []
    
    # Predict in batches for speed
    X = df[feature_cols].values
    
    # 1. Direction
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    
    # 2. Timing
    timing_probs = models['timing'].predict_proba(X)[:, 1]
    
    # 3. Strength
    strength_preds = models['strength'].predict(X)
    
    # Iterate and filter
    for i in range(len(df)):
        if dir_preds[i] == 1: continue # Sideways
        
        if dir_confs[i] < min_conf: continue
        if timing_probs[i] < min_timing: continue
        if strength_preds[i] < min_strength: continue
        
        signals.append({
            'timestamp': df.index[i],
            'pair': pair_name,
            'direction': 'LONG' if dir_preds[i] == 2 else 'SHORT',
            'entry_price': df['close'].iloc[i],
            'atr': df['atr'].iloc[i],
            'score': dir_confs[i] * timing_probs[i], # Combined score for sorting if needed
            'timing_prob': timing_probs[i],
            'pred_strength': strength_preds[i],
            # V8 Features for Filtering
            'wick_ratio_up': df['wick_ratio_up'].iloc[i],
            'wick_ratio_down': df['wick_ratio_down'].iloc[i],
            'rsi': df['rsi'].iloc[i]
        })
        
    return signals


def simulate_trade(signal: dict, df: pd.DataFrame) -> dict:
    """Simulate a single trade on a specific pair dataframe (V8 Smart Exit)."""
    # Find start index
    try:
        start_idx = df.index.get_loc(signal['timestamp'])
    except KeyError:
        return None
        
    entry_price = signal['entry_price']
    atr = signal['atr']
    direction = signal['direction']
    
    sl_dist = atr * SL_ATR_MULT
    be_trigger_dist = atr * 1.2
    
    if direction == 'LONG':
        sl_price = entry_price - sl_dist
        be_trigger_price = entry_price + be_trigger_dist
    else:
        sl_price = entry_price + sl_dist
        be_trigger_price = entry_price - be_trigger_dist
        
    outcome = 'time_exit'
    exit_idx = min(start_idx + 150, len(df) - 1)
    exit_price = df['close'].iloc[exit_idx]
    exit_time = df.index[exit_idx]
    breakeven_active = False
    
    # Simulate bar by bar
    for j in range(start_idx + 1, min(start_idx + 150, len(df))):
        bar = df.iloc[j]
        
        if direction == 'LONG':
            # 1. Check SL
            if bar['low'] <= sl_price:
                outcome = 'stop_loss' if not breakeven_active else 'breakeven_stop'
                exit_price = sl_price
                exit_time = bar.name
                break
            
            # 2. Smart Exit (RSI) - V8 Feature
            if bar['rsi'] > RSI_EXIT_LONG:
                outcome = 'smart_exit_rsi'
                exit_price = bar['close']
                exit_time = bar.name
                break
            
            # 3. Breakeven Logic
            if not breakeven_active and bar['high'] >= be_trigger_price:
                breakeven_active = True
                sl_price = entry_price + (atr * 0.1)
            
            # 4. Trailing Stop
            if breakeven_active:
                new_sl = bar['high'] - (atr * 2.5)
                if new_sl > sl_price:
                    sl_price = new_sl
                    
        else: # SHORT
            # 1. Check SL
            if bar['high'] >= sl_price:
                outcome = 'stop_loss' if not breakeven_active else 'breakeven_stop'
                exit_price = sl_price
                exit_time = bar.name
                break
            
            # 2. Smart Exit (RSI) - V8 Feature
            if bar['rsi'] < RSI_EXIT_SHORT:
                outcome = 'smart_exit_rsi'
                exit_price = bar['close']
                exit_time = bar.name
                break
            
            # 3. Breakeven Logic
            if not breakeven_active and bar['low'] <= be_trigger_price:
                breakeven_active = True
                sl_price = entry_price - (atr * 0.1)
            
            # 4. Trailing Stop
            if breakeven_active:
                new_sl = bar['low'] + (atr * 2.5)
                if new_sl < sl_price:
                    sl_price = new_sl
                    
    return {
        'exit_time': exit_time,
        'exit_price': exit_price,
        'outcome': outcome,
        'sl_dist': sl_dist
    }


def run_portfolio_backtest(signals: list, pair_dfs: dict, btc_data=None, initial_balance: float = 10000.0) -> list:
    """
    Execute signals enforcing the 'Single Slot' constraint.
    Calculates Real Dollar PnL with 5% Risk.
    Includes V8 Veto Logic (BTC + Wicks).
    """
    # Sort by time. If times are equal, sort by score (descending)
    signals.sort(key=lambda x: (x['timestamp'], -x['score']))
    
    executed_trades = []
    last_exit_time = pd.Timestamp.min
    balance = initial_balance
    
    # BTC Data Lookup
    btc_dict = btc_data['close'].to_dict() if btc_data is not None else {}
    btc_open_dict = btc_data['open'].to_dict() if btc_data is not None else {}
    
    print(f"Processing {len(signals)} potential signals...")
    print(f"Initial Balance: ${balance:,.2f}")
    
    for signal in signals:
        # Constraint: Can only hold 1 position
        if signal['timestamp'] < last_exit_time:
            continue
            
        # -------------------------------------------------
        # V8 VETO LOGIC
        # -------------------------------------------------
        # A. BTC Veto
        if btc_data is not None:
            btc_close = btc_dict.get(signal['timestamp'])
            btc_open = btc_open_dict.get(signal['timestamp'])
            if btc_close and btc_open:
                is_btc_dumping = btc_close < btc_open
                is_btc_pumping = btc_close > btc_open
                if signal['direction'] == 'LONG' and is_btc_dumping:
                    continue # VETO
                if signal['direction'] == 'SHORT' and is_btc_pumping:
                    continue # VETO
        
        # B. Wick Veto
        if signal['direction'] == 'LONG' and signal['wick_ratio_up'] > WICK_REJECTION_RATIO:
            continue # VETO
        if signal['direction'] == 'SHORT' and signal['wick_ratio_down'] > WICK_REJECTION_RATIO:
            continue # VETO
            
        # Execute
        pair_df = pair_dfs[signal['pair']]
        result = simulate_trade(signal, pair_df)
        
        if result:
            # Calculate Position Size & PnL
            entry_price = signal['entry_price']
            exit_price = result['exit_price']
            sl_dist = result['sl_dist']
            
            # Risk Management
            risk_amount = balance * RISK_PCT # 5% of current balance
            sl_pct = sl_dist / entry_price
            
            # Position Size (Value in $)
            position_size = risk_amount / sl_pct
            
            # Leverage Check
            leverage = position_size / balance
            if leverage > MAX_LEVERAGE:
                position_size = balance * MAX_LEVERAGE
                leverage = MAX_LEVERAGE
            
            # PnL Calculation
            if signal['direction'] == 'LONG':
                raw_pnl_pct = (exit_price - entry_price) / entry_price
            else:
                raw_pnl_pct = (entry_price - exit_price) / entry_price
                
            gross_profit = position_size * raw_pnl_pct
            fees = position_size * FEE_PCT * 2 # Entry + Exit
            net_profit = gross_profit - fees
            
            balance += net_profit
            
            trade_record = signal.copy()
            trade_record.update(result)
            trade_record.update({
                'leverage': leverage,
                'position_size': position_size,
                'net_profit': net_profit,
                'balance_after': balance,
                'pnl_pct': (net_profit / (risk_amount / RISK_PCT)) * 100 # PnL % relative to account balance used (approx)
            })
            
            executed_trades.append(trade_record)
            last_exit_time = result['exit_time']
            
    return executed_trades, balance


def print_results(trades, final_balance, initial_balance=10000.0):
    if not trades:
        print("No trades.")
        return
        
    wins = [t for t in trades if t['net_profit'] > 0]
    losses = [t for t in trades if t['net_profit'] <= 0]
    
    total_pnl_dollar = final_balance - initial_balance
    total_pnl_pct = (total_pnl_dollar / initial_balance) * 100
    
    print("\n" + "="*50)
    print(f"PORTFOLIO RESULTS (Single Slot)")
    print("="*50)
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f}")
    print(f"Total PnL ($):   ${total_pnl_dollar:,.2f}")
    print(f"Total PnL (%):   {total_pnl_pct:.2f}%")
    print("-" * 30)
    print(f"Total Trades:    {len(trades)}")
    
    if len(trades) > 0:
        duration = (trades[-1]['timestamp'] - trades[0]['timestamp']).days
        if duration > 0:
            print(f"Trades per Day:  {len(trades)/duration:.1f}")
            
    win_rate = len(wins)/len(trades)*100
    print(f"Win Rate:        {win_rate:.1f}%")
    
    if losses:
        gross_win = sum(t['net_profit'] for t in wins)
        gross_loss = abs(sum(t['net_profit'] for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        print(f"Profit Factor:   {pf:.2f}")
    
    print("\nOutcomes:")
    for o in set(t['outcome'] for t in trades):
        count = len([t for t in trades if t['outcome'] == o])
        print(f"  {o}: {count}")
        
    print("="*50)

def print_trade_list(trades):
    """Print trades in the user-requested format for verification."""
    print("\n" + "="*50)
    print("DETAILED TRADE LIST (For Chart Verification)")
    print("="*50)
    
    # Sort by time
    trades.sort(key=lambda x: x['timestamp'])
    
    for t in trades:
        # Format: PIPPIN (LONG) 00:45 — Profit: +$1,138.92 (+11.3%)
        time_str = t['timestamp'].strftime("%H:%M")
        pair_clean = t['pair'].replace('_', '/').replace(':USDT', '')
        
        # Add emoji based on result
        emoji = "🚀" if t['pnl_pct'] > 20 else "✅" if t['net_profit'] > 0 else "❌"
        if t['net_profit'] > 0 and t['pnl_pct'] < 1: emoji = "🛡️" # Breakeven/Small profit
        
        print(f"{pair_clean} ({t['direction']}) {time_str} — Profit: ${t['net_profit']:+,.2f} ({t['pnl_pct']:+.1f}%) {emoji}")
        print(f"   Entry: {t['entry_price']:.5f} | Exit: {t['exit_price']:.5f} | Reason: {t['outcome']}")
        print("-" * 30)


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--test_days", type=int, default=14)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--output", type=str, default="./models/v8_terminator")
    parser.add_argument("--initial_balance", type=float, default=10000.0, help="Initial portfolio balance")
    parser.add_argument("--check-dec25", action="store_true", help="Fetch and test specifically for Dec 25, 2025")
    args = parser.parse_args()
    
    print("=" * 70)
    print("V8 TERMINATOR - SMART LOGIC & ADVANCED FEATURES")
    print("=" * 70)
    
    # Load pairs
    import json
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_list.json'
    with open(pairs_file) as f:
        pairs_data = json.load(f)
    pairs = [p['symbol'] for p in pairs_data['pairs'][:args.pairs]]
    print(f"Loaded {len(pairs)} pairs.")
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'candles'
    mtf_fe = MTFFeatureEngine()
    
    all_train = []
    test_dfs = {} 
    test_features = {} 
    
    # 1. LOAD TRAINING DATA (Local)
    print("\nLoading Training Data & 14-Day Test Data...")
    for pair in pairs:
        print(f"Processing {pair}...", end='\r')
        pair_name = pair.replace('/', '_').replace(':', '_')
        
        try:
            m1 = pd.read_csv(data_dir / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m5 = pd.read_csv(data_dir / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m15 = pd.read_csv(data_dir / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
        except FileNotFoundError:
            continue
        
        # Use all available data up to test_days ago for training
        test_start = datetime.now() - timedelta(days=args.test_days)
        train_start = test_start - timedelta(days=args.days)
        
        m1_train = m1[(m1.index >= train_start) & (m1.index < test_start)]
        m5_train = m5[(m5.index >= train_start) & (m5.index < test_start)]
        m15_train = m15[(m15.index >= train_start) & (m15.index < test_start)]
        
        if len(m5_train) < 500: continue
        
        ft_train = mtf_fe.align_timeframes(m1_train, m5_train, m15_train)
        ft_train = ft_train.join(m5_train[['open', 'high', 'low', 'close', 'volume']])
        ft_train = add_advanced_features(ft_train) # V8 Features
        ft_train = create_targets_v1(ft_train)
        ft_train['pair'] = pair
        all_train.append(ft_train)
        
        # Always load local test data for the 14-day stats
        m1_test = m1[m1.index >= test_start]
        m5_test = m5[m5.index >= test_start]
        m15_test = m15[m15.index >= test_start]
        
        ft_test = mtf_fe.align_timeframes(m1_test, m5_test, m15_test)
        ft_test = ft_test.join(m5_test[['open', 'high', 'low', 'close', 'volume']])
        ft_test = add_advanced_features(ft_test) # V8 Features
        ft_test = create_targets_v1(ft_test)
        ft_test['pair'] = pair
        test_features[pair] = ft_test
        test_dfs[pair] = ft_test

    print(f"\nData loaded. Training on {len(all_train)} pairs.")
    
    # Train
    train_df = pd.concat(all_train).dropna()
    exclude = ['pair', 'target_dir', 'target_timing', 'target_strength', 
               'open', 'high', 'low', 'close', 'volume', 'atr', 'price_change', 'obv', 'obv_sma',
               'upper_wick', 'lower_wick', 'body', 'future_change'] # Exclude raw prices
    features = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[features]
    y_train = {
        'target_dir': train_df['target_dir'],
        'target_timing': train_df['target_timing'],
        'target_strength': train_df['target_strength']
    }
    
    val_idx = int(len(X_train) * 0.9)
    X_t = X_train.iloc[:val_idx]
    X_v = X_train.iloc[val_idx:]
    y_t = {k: v.iloc[:val_idx] for k, v in y_train.items()}
    y_v = {k: v.iloc[val_idx:] for k, v in y_train.items()}
    
    models = train_models(X_t, y_t, X_v, y_v)
    
    # Save Models
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving models to {out}...")
    joblib.dump(models['direction'], out / 'model_direction.pkl')
    joblib.dump(models['timing'], out / 'model_timing.pkl')
    joblib.dump(models['strength'], out / 'model_strength.pkl')
    print("Models saved.")
    
    # ---------------------------------------------------------
    # FETCH BTC FOR VETO
    # ---------------------------------------------------------
    print("\nFetching BTC Data for Veto Logic...")
    btc_df = None
    try:
        # Try to load local BTC first
        btc_path = data_dir / "BTC_USDT_USDT_5m.csv"
        if btc_path.exists():
            btc_df = pd.read_csv(btc_path, parse_dates=['timestamp'], index_col='timestamp')
            print("Loaded BTC data from local file.")
        else:
            print("Local BTC not found.")
    except Exception as e:
        print(f"Warning: Could not load BTC data: {e}")

    # ---------------------------------------------------------
    # 1. STANDARD 14-DAY BACKTEST
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"RUNNING 14-DAY BACKTEST (Last {args.test_days} Days)")
    print("="*70)
    
    all_signals = []
    for pair, df in test_features.items():
        df_clean = df.dropna()
        if len(df_clean) == 0: continue
        sigs = generate_signals(df_clean, features, models, pair)
        all_signals.extend(sigs)
        
    trades, final_bal = run_portfolio_backtest(all_signals, test_dfs, btc_data=btc_df, initial_balance=args.initial_balance)
    print_trade_list(trades)
    print_results(trades, final_bal, initial_balance=args.initial_balance)
    
    # Save 14-day trades
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if trades:
        pd.DataFrame(trades).to_csv(out / 'backtest_trades_14d.csv', index=False)

    # ---------------------------------------------------------
    # 2. FETCH DEC 25 DATA IF REQUESTED
    # ---------------------------------------------------------
    if args.check_dec25:
        print("\n" + "="*70)
        print("RUNNING DEC 25 SPECIAL CHECK")
        print("="*70)
        print("Fetching Dec 25 Data from Binance...")
        
        # Fetch from Dec 23 to Dec 26 to ensure we have history for indicators
        fetch_start = datetime(2025, 12, 23)
        fetch_end = datetime(2025, 12, 26)
        
        dec25_features = {}
        dec25_dfs = {}
        
        for pair in pairs:
            print(f"Fetching {pair}...", end='\r')
            m1 = fetch_binance_data(pair, '1m', fetch_start, fetch_end)
            m5 = fetch_binance_data(pair, '5m', fetch_start, fetch_end)
            m15 = fetch_binance_data(pair, '15m', fetch_start, fetch_end)
            
            if len(m5) < 100:
                # print(f"Skipping {pair} (Insufficient data)")
                continue
                
            ft = mtf_fe.align_timeframes(m1, m5, m15)
            ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
            ft = add_advanced_features(ft) # V8 Features
            ft['atr'] = calculate_atr(ft) # Ensure ATR is present
            ft['pair'] = pair
            
            # Filter for Dec 25 ONLY for the backtest part
            dec25_mask = (ft.index >= datetime(2025, 12, 25)) & (ft.index < datetime(2025, 12, 26))
            ft_dec25 = ft[dec25_mask]
            
            if len(ft_dec25) > 0:
                dec25_features[pair] = ft_dec25
                dec25_dfs[pair] = ft_dec25
        print("\nFetch complete.")
        
        # Run Dec 25 Backtest
        dec25_signals = []
        for pair, df in dec25_features.items():
            df_clean = df.dropna()
            if len(df_clean) == 0: continue
            sigs = generate_signals(df_clean, features, models, pair)
            dec25_signals.extend(sigs)
            
        d25_trades, d25_bal = run_portfolio_backtest(dec25_signals, dec25_dfs, btc_data=btc_df, initial_balance=args.initial_balance)
        print_results(d25_trades, d25_bal, initial_balance=args.initial_balance)
        print_trade_list(d25_trades)
        
        if d25_trades:
            pd.DataFrame(d25_trades).to_csv(out / 'backtest_trades_dec25.csv', index=False)


if __name__ == '__main__':
    main()

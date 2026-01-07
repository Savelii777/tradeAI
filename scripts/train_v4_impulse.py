#!/usr/bin/env python3
"""
Train V4.1 IMPULSE RIDER - Ride the Full Wave
"Short Stop, Long Ride"

Philosophy:
- Hunt impulses with volume surges
- SHORT STOP (0.8 ATR) - quick exit if wrong
- NO FIXED TP - let impulses run!
- TRAILING STOP - ride the full move (5-10R possible)
- Enter INSIDE the impulse (current candle check)

Key Changes from V4:
1. REMOVED Fixed TP (was 2R) - now pure trailing
2. Trailing activates at 1R (not 1.5R)
3. Adaptive trailing based on R-multiple:
   - 1-2R: Wide trailing (2.5 ATR) - let it grow
   - 2-4R: Medium trailing (1.5 ATR) 
   - 4-6R: Tight trailing (1.0 ATR) - protect profit
   - 6R+: Very tight (0.7 ATR) - lock it in
4. Same short stop (0.8 ATR)
5. Target: Catch 5-10R moves on strong impulses
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
# CONFIG - V4.1 IMPULSE RIDER SETTINGS
# ============================================================
SL_ATR_MULT = 0.8       # SHORT stop - quick exit if wrong
TP_TARGET_R = None      # NO FIXED TP - pure trailing!
TRAILING_START_R = 1.0  # Activate trailing at 1R profit
MAX_LEVERAGE = 20.0
RISK_PCT = 0.05         # 5% Risk per trade
FEE_PCT = 0.0002        # 0.02% Maker/Taker
LOOKAHEAD = 10          # Shorter lookahead for impulses (50 min on M5)

# REALISTIC LIMITS
MAX_POSITION_SIZE = 50000.0  # Max $50K position
SLIPPAGE_PCT = 0.0001        # 0.01% slippage

# V4.1 IMPULSE RIDER FEATURES
USE_VOLUME_SURGE = True      # Main indicator
USE_MOMENTUM_ENTRY = True    # Enter on current momentum
USE_ADAPTIVE_TRAILING = True # Adaptive trailing based on R-multiple


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_binance_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    """Fetch candles from Binance via CCXT."""
    exchange = ccxt.binance()
    
    symbol = symbol.replace('_', '/')
    if '/' not in symbol:
        symbol = f"{symbol[:-4]}/{symbol[-4:]}"
        
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
            
            if candles[-1][0] > end_date.timestamp() * 1000:
                break
                
            time.sleep(0.1)
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
        
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return pd.DataFrame()


# ============================================================
# FEATURES - V4 IMPULSE DETECTION
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features + IMPULSE DETECTION features."""
    df = df.copy()
    
    # Basic volume features
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
    
    # === V4: IMPULSE DETECTION FEATURES ===
    
    # 1. VOLUME SURGE (Key Indicator!)
    df['volume_surge'] = df['volume'] / df['vol_sma_20']  # > 2.5 = impulse
    df['volume_acceleration'] = df['volume'].rolling(3).mean() / df['volume'].rolling(10).mean()
    df['institutional_volume'] = (df['volume'] > df['vol_sma_20'] * 5).astype(int)  # Whale activity
    
    # 2. PRICE VELOCITY
    df['momentum_1bar'] = df['close'].pct_change(1) * 100  # % change last bar
    df['momentum_3bar'] = df['close'].pct_change(3) * 100  # % change last 3 bars
    df['momentum_5bar'] = df['close'].pct_change(5) * 100  # % change last 5 bars
    
    # 3. CANDLE STRENGTH
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = df['candle_body'] / df['candle_range'].replace(0, 1)  # > 0.7 = strong
    
    # 4. BREAKOUT DETECTION
    df['donchian_high_10'] = df['high'].rolling(10).max()
    df['donchian_low_10'] = df['low'].rolling(10).min()
    df['breakout_up'] = (df['close'] > df['donchian_high_10'].shift(1)).astype(int)
    df['breakout_down'] = (df['close'] < df['donchian_low_10'].shift(1)).astype(int)
    
    # 5. MOMENTUM CONSISTENCY
    df['consecutive_up'] = (df['close'] > df['open']).astype(int).rolling(5).sum()
    df['consecutive_down'] = (df['close'] < df['open']).astype(int).rolling(5).sum()
    
    # 6. VOLATILITY EXPANSION
    df['atr'] = calculate_atr(df)
    df['atr_sma'] = df['atr'].rolling(20).mean()
    df['atr_expansion'] = df['atr'] / df['atr_sma']  # > 1.2 = expanding
    
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
# V4 TARGETS - IMPULSE DETECTION
# ============================================================
def create_targets_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create V4-style targets focused on FAST IMPULSES.
    
    Key Difference from V8:
    - V8: Looks for long trends (150 bars)
    - V4: Looks for SHORT sharp moves (3-10 bars)
    """
    df = df.copy()
    df['atr'] = calculate_atr(df)
    
    # 1. Direction (Same as V8, but with SHORTER lookahead)
    future_return = df['close'].pct_change(LOOKAHEAD).shift(-LOOKAHEAD)
    threshold = 0.003  # 0.3% minimum move (lower than V8's 0.5%)
    
    # 0=Down, 1=Sideways, 2=Up
    df['target_dir'] = np.where(
        future_return > threshold, 2,
        np.where(future_return < -threshold, 0, 1)
    )
    
    # 2. Timing (Same logic)
    future_lows = df['low'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
    future_highs = df['high'].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
    
    adv_long = (df['close'] - future_lows) / df['atr']
    fav_long = (future_highs - df['close']) / df['atr']
    adv_short = (future_highs - df['close']) / df['atr']
    fav_short = (df['close'] - future_lows) / df['atr']
    
    is_good_long = (fav_long > adv_long) & (fav_long > 1.0)
    is_good_short = (fav_short > adv_short) & (fav_short > 1.0)
    
    df['target_timing'] = 0
    df.loc[(df['target_dir'] == 2) & is_good_long, 'target_timing'] = 1
    df.loc[(df['target_dir'] == 0) & is_good_short, 'target_timing'] = 1
    
    # 3. Strength (Expected R-multiple)
    move_long = (future_highs - df['close']) / df['atr']
    move_short = (df['close'] - future_lows) / df['atr']
    
    df['target_strength'] = np.where(df['target_dir'] == 2, move_long, 
                                   np.where(df['target_dir'] == 0, move_short, 0))
    df['target_strength'] = df['target_strength'].clip(0, 10)
    
    # 4. ✨ V4 NEW: Impulse Detector (Binary: Will there be a fast 2R+ move?)
    # Look at MULTIPLE short periods (3, 5, 10 bars)
    df['impulse_detected'] = 0
    
    for period in [3, 5, 10]:
        future_high = df['high'].rolling(period).max().shift(-period)
        future_low = df['low'].rolling(period).min().shift(-period)
        
        # Movement in R (relative to current ATR)
        move_up_r = (future_high - df['close']) / df['atr']
        move_down_r = (df['close'] - future_low) / df['atr']
        
        # Volume confirmation
        volume_surge = df['volume_surge'] > 2.0
        
        # Impulse = 2R+ move in short period + volume
        impulse_up = (move_up_r >= 2.0) & volume_surge
        impulse_down = (move_down_r >= 2.0) & volume_surge
        
        # Mark if ANY period shows impulse
        df['impulse_detected'] |= (impulse_up | impulse_down).astype(int)
    
    return df


# ============================================================
# TRAINING (V4 - Add Impulse Model)
# ============================================================
def train_models(X_train, y_train, X_val, y_val):
    """Train 4 models (Direction, Timing, Strength, Impulse)."""
    
    # 1. Direction Model (Multiclass)
    print("   Training Direction Model...")
    dir_model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        n_estimators=600, max_depth=5, num_leaves=20,
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
    
    # 4. ✨ V4 NEW: Impulse Model (Binary)
    print("   Training Impulse Detector Model...")
    impulse_model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        n_estimators=400, max_depth=5, num_leaves=15,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.5,
        random_state=42, verbosity=-1
    )
    impulse_model.fit(X_train, y_train['impulse_detected'],
                      eval_set=[(X_val, y_val['impulse_detected'])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    
    return {
        'direction': dir_model,
        'timing': timing_model,
        'strength': strength_model,
        'impulse': impulse_model  # NEW!
    }


# ============================================================
# V4 SIGNAL GENERATION - SOFTER FILTERS
# ============================================================
def generate_signals(df: pd.DataFrame, feature_cols: list, models: dict, pair_name: str,
                    min_conf: float = 0.45, min_timing: float = 0.50, 
                    min_strength: float = 1.0, min_impulse: float = 0.60) -> list:
    """
    Generate signals with V4 SOFTER filters (more signals).
    
    Key Changes:
    - Lower thresholds (0.45 vs 0.50 conf)
    - New impulse filter
    - Volume surge check
    """
    signals = []
    
    X = df[feature_cols].values
    
    # Predictions
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    
    timing_probs = models['timing'].predict_proba(X)[:, 1]
    strength_preds = models['strength'].predict(X)
    impulse_probs = models['impulse'].predict_proba(X)[:, 1]  # NEW!
    
    # Iterate and filter
    for i in range(len(df)):
        if dir_preds[i] == 1: continue  # Sideways
        
        # V4 SOFTER filters
        if dir_confs[i] < min_conf: continue
        if timing_probs[i] < min_timing: continue
        if strength_preds[i] < min_strength: continue
        if impulse_probs[i] < min_impulse: continue  # NEW!
        
        # V4 CRITICAL: Volume surge check (main indicator!)
        if USE_VOLUME_SURGE:
            volume_surge = df['volume_surge'].iloc[i]
            if volume_surge < 2.0:  # Skip if no volume surge
                continue
        
        signals.append({
            'timestamp': df.index[i],
            'pair': pair_name,
            'direction': 'LONG' if dir_preds[i] == 2 else 'SHORT',
            'entry_price': df['close'].iloc[i],
            'atr': df['atr'].iloc[i],
            'score': dir_confs[i] * timing_probs[i],
            'timing_prob': timing_probs[i],
            'pred_strength': strength_preds[i],
            'impulse_prob': impulse_probs[i],  # NEW!
            'volume_surge': df['volume_surge'].iloc[i]  # Track volume
        })
        
    return signals


# ============================================================
# V4.1 TRADE SIMULATION - SHORT STOP, PURE TRAILING
# ============================================================
def simulate_trade(signal: dict, df: pd.DataFrame) -> dict:
    """
    Simulate a single trade with V4.1 IMPULSE RIDER LOGIC:
    - Short stop (0.8 ATR) - wrong = quick exit
    - NO Fixed TP - let impulses run!
    - Adaptive trailing - tighter as profit grows
    - Target: 5-10R on strong impulses
    """
    try:
        start_idx = df.index.get_loc(signal['timestamp'])
    except KeyError:
        return None
        
    entry_price = signal['entry_price']
    atr = signal['atr']
    direction = signal['direction']
    
    # V4.1: SHORT STOP (0.8 ATR)
    sl_mult = SL_ATR_MULT  # 0.8 ATR
    sl_dist = atr * sl_mult
    
    # V4.1: NO FIXED TP - pure trailing!
    
    if direction == 'LONG':
        sl_price = entry_price - sl_dist
    else:
        sl_price = entry_price + sl_dist
        
    outcome = 'time_exit'
    exit_idx = min(start_idx + 60, len(df) - 1)  # Max 60 bars (5 hours) - give impulses time
    exit_price = df['close'].iloc[exit_idx]
    exit_time = df.index[exit_idx]
    trailing_active = False
    max_r_reached = 0.0
    
    # Simulate bar by bar
    for j in range(start_idx + 1, min(start_idx + 60, len(df))):
        bar = df.iloc[j]
        
        if direction == 'LONG':
            # Check SL first
            if bar['low'] <= sl_price:
                outcome = 'stop_loss' if not trailing_active else 'trailing_stop'
                exit_price = sl_price
                exit_time = bar.name
                break
            
            # Track progress
            current_profit = bar['high'] - entry_price
            r_multiple = current_profit / sl_dist
            max_r_reached = max(max_r_reached, r_multiple)
            
            # V4.1: ADAPTIVE TRAILING (no fixed TP!)
            if r_multiple >= TRAILING_START_R:  # 1R profit
                trailing_active = True
                
                # Adaptive trailing based on R-multiple
                if USE_ADAPTIVE_TRAILING:
                    if r_multiple >= 6.0:
                        # MEGA PROFIT: Very tight trailing (lock it in!)
                        trail_mult = 0.7
                    elif r_multiple >= 4.0:
                        # BIG PROFIT: Tight trailing (protect most of it)
                        trail_mult = 1.0
                    elif r_multiple >= 2.0:
                        # MEDIUM PROFIT: Medium trailing (let it run a bit)
                        trail_mult = 1.5
                    else:
                        # SMALL PROFIT (1-2R): Wide trailing (give room to grow)
                        trail_mult = 2.5
                else:
                    # Default trailing
                    trail_mult = 1.5
                
                new_sl = bar['high'] - (atr * trail_mult)
                if new_sl > sl_price:
                    sl_price = new_sl
                    
        else:  # SHORT
            # Check SL first
            if bar['high'] >= sl_price:
                outcome = 'stop_loss' if not trailing_active else 'trailing_stop'
                exit_price = sl_price
                exit_time = bar.name
                break
            
            # Track progress
            current_profit = entry_price - bar['low']
            r_multiple = current_profit / sl_dist
            max_r_reached = max(max_r_reached, r_multiple)
            
            # V4.1: ADAPTIVE TRAILING (no fixed TP!)
            if r_multiple >= TRAILING_START_R:  # 1R profit
                trailing_active = True
                
                # Adaptive trailing based on R-multiple
                if USE_ADAPTIVE_TRAILING:
                    if r_multiple >= 6.0:
                        trail_mult = 0.7  # Very tight
                    elif r_multiple >= 4.0:
                        trail_mult = 1.0  # Tight
                    elif r_multiple >= 2.0:
                        trail_mult = 1.5  # Medium
                    else:
                        trail_mult = 2.5  # Wide
                else:
                    trail_mult = 1.5
                
                new_sl = bar['low'] + (atr * trail_mult)
                if new_sl < sl_price:
                    sl_price = new_sl
        
        # V4.1: Time exit if no momentum after 15 bars (75 minutes)
        # But ONLY if profit < 0.5R (not moving)
        if j - start_idx >= 15 and max_r_reached < 0.5:
            outcome = 'no_momentum'
            exit_price = bar['close']
            exit_time = bar.name
            break
                    
    return {
        'exit_time': exit_time,
        'exit_price': exit_price,
        'outcome': outcome,
        'sl_dist': sl_dist,
        'sl_mult': sl_mult,
        'max_r': max_r_reached,
        'trailing_active': trailing_active
    }


# ============================================================
# PORTFOLIO BACKTEST - COMPOUND INTEREST
# ============================================================
def run_portfolio_backtest(signals: list, pair_dfs: dict, initial_balance: float = 20.0) -> tuple:
    """
    Execute signals with COMPOUND INTEREST (growing capital).
    V4: Faster cooldown for more frequent trades.
    """
    signals.sort(key=lambda x: (x['timestamp'], -x['score']))
    
    executed_trades = []
    last_exit_time = pd.Timestamp.min
    balance = initial_balance
    
    print(f"Processing {len(signals)} potential signals...")
    print(f"Initial Balance: ${balance:.2f}")
    
    # V4: Track cooldown per outcome
    cooldown_minutes = 0
    
    for signal in signals:
        # V4: Short cooldown (5-10 min depending on last trade result)
        required_cooldown = timedelta(minutes=cooldown_minutes)
        
        if signal['timestamp'] < last_exit_time + required_cooldown:
            continue
            
        pair_df = pair_dfs[signal['pair']]
        result = simulate_trade(signal, pair_df)
        
        if result:
            entry_price = signal['entry_price']
            exit_price = result['exit_price']
            sl_dist = result['sl_dist']
            
            # Position sizing
            risk_amount = balance * RISK_PCT
            sl_pct = sl_dist / entry_price
            position_size = risk_amount / sl_pct
            
            # Leverage check
            leverage = position_size / balance
            if leverage > MAX_LEVERAGE:
                position_size = balance * MAX_LEVERAGE
                leverage = MAX_LEVERAGE
            
            # Position size cap
            if position_size > MAX_POSITION_SIZE:
                position_size = MAX_POSITION_SIZE
                leverage = position_size / balance
            
            # PnL calculation with slippage
            if signal['direction'] == 'LONG':
                effective_entry = entry_price * (1 + SLIPPAGE_PCT)
                effective_exit = exit_price * (1 - SLIPPAGE_PCT)
                raw_pnl_pct = (effective_exit - effective_entry) / effective_entry
            else:
                effective_entry = entry_price * (1 - SLIPPAGE_PCT)
                effective_exit = exit_price * (1 + SLIPPAGE_PCT)
                raw_pnl_pct = (effective_entry - effective_exit) / effective_entry
                
            gross_profit = position_size * raw_pnl_pct
            fees = position_size * FEE_PCT * 2
            net_profit = gross_profit - fees
            
            # COMPOUND INTEREST: Update balance
            balance += net_profit
            
            # V4: Adaptive cooldown based on result
            if net_profit > 0:
                cooldown_minutes = 5  # Short cooldown after win
            else:
                cooldown_minutes = 10  # Longer cooldown after loss
            
            trade_record = signal.copy()
            trade_record.update(result)
            trade_record.update({
                'leverage': leverage,
                'position_size': position_size,
                'net_profit': net_profit,
                'balance_after': balance,
                'pnl_pct': (net_profit / (balance - net_profit)) * 100,  # % of balance before trade
                'roe': (net_profit / (position_size / leverage)) * 100
            })
            
            executed_trades.append(trade_record)
            last_exit_time = result['exit_time']
            
    return executed_trades, balance


# ============================================================
# RESULTS DISPLAY
# ============================================================
def print_results(trades, final_balance, initial_balance=20.0):
    if not trades:
        print("No trades executed.")
        return
        
    wins = [t for t in trades if t['net_profit'] > 0]
    losses = [t for t in trades if t['net_profit'] <= 0]
    
    total_pnl_dollar = final_balance - initial_balance
    total_pnl_pct = (total_pnl_dollar / initial_balance) * 100
    
    print("\n" + "="*70)
    print(f"V4 IMPULSE HUNTER RESULTS")
    print("="*70)
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${final_balance:.2f}")
    print(f"Total PnL ($):   ${total_pnl_dollar:,.2f}")
    print(f"Total PnL (%):   {total_pnl_pct:.2f}%")
    print("-" * 70)
    print(f"Total Trades:    {len(trades)}")
    
    if len(trades) > 0:
        duration = (trades[-1]['timestamp'] - trades[0]['timestamp']).total_seconds() / 3600
        if duration > 0:
            print(f"Duration:        {duration:.1f} hours ({duration/24:.1f} days)")
            print(f"Trades per Day:  {len(trades)/(duration/24):.1f}")
            
    win_rate = len(wins)/len(trades)*100 if trades else 0
    print(f"Win Rate:        {win_rate:.1f}%")
    
    if losses:
        gross_win = sum(t['net_profit'] for t in wins)
        gross_loss = abs(sum(t['net_profit'] for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        print(f"Profit Factor:   {pf:.2f}")
    
    # Average R-multiple
    avg_r = np.mean([t['max_r'] for t in trades])
    print(f"Avg Max R:       {avg_r:.2f}R")
    
    # Outcomes breakdown
    print("\nOutcomes:")
    outcomes = {}
    for t in trades:
        outcome = t['outcome']
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    
    for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
        pct = count/len(trades)*100
        print(f"  {outcome}: {count} ({pct:.1f}%)")
    
    # R-multiple distribution
    print("\nR-Multiple Distribution:")
    r_ranges = [(0, 1, "0-1R"), (1, 2, "1-2R"), (2, 4, "2-4R"), (4, 6, "4-6R"), (6, 100, "6R+")]
    for r_min, r_max, label in r_ranges:
        count = len([t for t in trades if r_min <= t['max_r'] < r_max])
        if count > 0:
            pct = count/len(trades)*100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
    print("="*70)


def print_trade_list(trades):
    """Print detailed trade list."""
    print("\n" + "="*70)
    print("DETAILED TRADE LIST")
    print("="*70)
    
    trades.sort(key=lambda x: x['timestamp'])
    
    for t in trades:
        time_str = t['timestamp'].strftime("%m/%d %H:%M")
        pair_clean = t['pair'].replace('_', '/').replace(':USDT', '')
        
        emoji = "🚀" if t['pnl_pct'] > 5 else "✅" if t['net_profit'] > 0 else "❌"
        
        print(f"{pair_clean} ({t['direction']}) {time_str} — "
              f"${t['net_profit']:+.2f} ({t['pnl_pct']:+.1f}%) "
              f"MaxR: {t['max_r']:.1f}R | Vol: {t.get('volume_surge', 0):.1f}x | "
              f"{t['outcome']} {emoji}")


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="V4 Impulse Hunter - Volume Scalper")
    parser.add_argument("--days", type=int, default=90, help="Training period (days)")
    parser.add_argument("--test_days", type=int, default=5, help="Test period (days)")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs")
    parser.add_argument("--output", type=str, default="./models/v4_impulse", help="Output directory")
    parser.add_argument("--initial_balance", type=float, default=20.0, help="Initial capital")
    parser.add_argument("--reverse", action="store_true", 
                       help="Train on recent 90d, test on 5d BEFORE (recommended)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("V4.1 IMPULSE RIDER - RIDE THE FULL WAVE")
    print("=" * 70)
    print(f"Strategy: Catch impulses with SHORT stop + TRAILING (no TP)")
    print(f"Target: Ride full impulses for 5-10R (not just 2R)")
    print(f"Stop: {SL_ATR_MULT} ATR | Trailing starts at: {TRAILING_START_R}R")
    print("=" * 70)
    
    # Load pairs
    pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_20.json'
    if not pairs_file.exists():
        pairs_file = Path(__file__).parent.parent / 'config' / 'pairs_list.json'
        
    with open(pairs_file) as f:
        pairs_data = json.load(f)
    pairs = [p['symbol'] for p in pairs_data['pairs'][:args.pairs]]
    print(f"Loaded {len(pairs)} pairs from {pairs_file.name}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'candles'
    mtf_fe = MTFFeatureEngine()
    
    all_train = []
    test_dfs = {}
    test_features = {}
    
    print(f"\nLoading Data (Reverse={args.reverse})...")
    print(f"Training: Last {args.days} days")
    print(f"Testing: {args.test_days} days BEFORE training period")
    
    for pair in pairs:
        print(f"Processing {pair}...", end='\r')
        pair_name = pair.replace('/', '_').replace(':', '_')
        
        try:
            m1 = pd.read_csv(data_dir / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m5 = pd.read_csv(data_dir / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m15 = pd.read_csv(data_dir / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
        except FileNotFoundError:
            continue
        
        # REVERSE SPLIT: Train on Recent, Test on Previous
        now = datetime.now()
        if args.reverse:
            train_end = now
            train_start = now - timedelta(days=args.days)
            test_end = train_start
            test_start = test_end - timedelta(days=args.test_days)
        else:
            test_start = now - timedelta(days=args.test_days)
            train_start = test_start - timedelta(days=args.days)
            train_end = test_start
            test_end = now
        
        # Train data
        m1_train = m1[(m1.index >= train_start) & (m1.index < train_end)]
        m5_train = m5[(m5.index >= train_start) & (m5.index < train_end)]
        m15_train = m15[(m15.index >= train_start) & (m15.index < train_end)]
        
        if len(m5_train) < 500: continue
        
        ft_train = mtf_fe.align_timeframes(m1_train, m5_train, m15_train)
        ft_train = ft_train.join(m5_train[['open', 'high', 'low', 'close', 'volume']])
        ft_train = add_volume_features(ft_train)  # Includes V4 impulse features
        ft_train = create_targets_v4(ft_train)    # V4 targets
        ft_train['pair'] = pair
        all_train.append(ft_train)
        
        # Test data
        m1_test = m1[(m1.index >= test_start) & (m1.index < test_end)]
        m5_test = m5[(m5.index >= test_start) & (m5.index < test_end)]
        m15_test = m15[(m15.index >= test_start) & (m15.index < test_end)]
        
        ft_test = mtf_fe.align_timeframes(m1_test, m5_test, m15_test)
        ft_test = ft_test.join(m5_test[['open', 'high', 'low', 'close', 'volume']])
        ft_test = add_volume_features(ft_test)
        ft_test = create_targets_v4(ft_test)
        ft_test['pair'] = pair
        test_features[pair] = ft_test
        test_dfs[pair] = ft_test

    print(f"\nData loaded. Training on {len(all_train)} pairs.")
    
    # Prepare training data
    train_df = pd.concat(all_train).dropna()
    exclude = ['pair', 'target_dir', 'target_timing', 'target_strength', 'impulse_detected',
               'open', 'high', 'low', 'close', 'volume', 'atr', 'price_change', 'obv', 'obv_sma']
    features = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[features]
    y_train = {
        'target_dir': train_df['target_dir'],
        'target_timing': train_df['target_timing'],
        'target_strength': train_df['target_strength'],
        'impulse_detected': train_df['impulse_detected']  # NEW!
    }
    
    # Train/Val split
    val_idx = int(len(X_train) * 0.9)
    X_t = X_train.iloc[:val_idx]
    X_v = X_train.iloc[val_idx:]
    y_t = {k: v.iloc[:val_idx] for k, v in y_train.items()}
    y_v = {k: v.iloc[val_idx:] for k, v in y_train.items()}
    
    print("\n" + "="*70)
    print("TRAINING V4 MODELS (4 models)")
    print("="*70)
    models = train_models(X_t, y_t, X_v, y_v)
    
    # Backtest
    print("\n" + "="*70)
    print(f"RUNNING BACKTEST ({args.test_days} days BEFORE training)")
    print("="*70)
    
    all_signals = []
    for pair, df in test_features.items():
        df_clean = df.dropna()
        if len(df_clean) == 0: continue
        sigs = generate_signals(df_clean, features, models, pair)
        all_signals.extend(sigs)
    
    print(f"Generated {len(all_signals)} signals")
    
    trades, final_bal = run_portfolio_backtest(all_signals, test_dfs, 
                                               initial_balance=args.initial_balance)
    
    print_results(trades, final_bal, initial_balance=args.initial_balance)
    print_trade_list(trades)
    
    # Save models
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving models to {out}...")
    joblib.dump(models['direction'], out / 'direction_model.joblib')
    joblib.dump(models['timing'], out / 'timing_model.joblib')
    joblib.dump(models['strength'], out / 'strength_model.joblib')
    joblib.dump(models['impulse'], out / 'impulse_model.joblib')  # NEW!
    joblib.dump(features, out / 'feature_names.joblib')
    print("✅ Models saved!")
    
    # Save trades
    if trades:
        pd.DataFrame(trades).to_csv(out / f'backtest_trades_{args.test_days}d.csv', index=False)
        print(f"✅ Trades saved to {out / f'backtest_trades_{args.test_days}d.csv'}")
    
    # Save config
    config = {
        'version': 'v4.1_impulse_rider',
        'sl_atr_mult': SL_ATR_MULT,
        'tp_target_r': 'None (Pure Trailing)',
        'trailing_start_r': TRAILING_START_R,
        'max_leverage': MAX_LEVERAGE,
        'risk_pct': RISK_PCT,
        'lookahead': LOOKAHEAD,
        'filters': {
            'min_conf': 0.45,
            'min_timing': 0.50,
            'min_strength': 1.0,
            'min_impulse': 0.60,
            'min_volume_surge': 2.0
        },
        'trailing_logic': {
            '1-2R': '2.5 ATR (wide - let it grow)',
            '2-4R': '1.5 ATR (medium)',
            '4-6R': '1.0 ATR (tight - protect)',
            '6R+': '0.7 ATR (very tight - lock in)'
        }
    }
    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to {out / 'config.json'}")


if __name__ == '__main__':
    main()


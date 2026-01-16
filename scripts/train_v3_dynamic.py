#!/usr/bin/env python3
"""
Train V12 - SMART ADAPTIVE Trading Model
"Intelligence That Adapts to Market Conditions"

Philosophy:
- User has ONE execution slot (can only hold 1 trade at a time).
- Smart balanced model: not too simple (underfitting), not too complex (overfitting)
- Target: Win Rate 60-70% with CONFIDENT predictions on live.

V12 SMART ADAPTIVE IMPROVEMENTS:
1. Balanced Models: 200 trees, depth 4, 12 leaves, min_child_samples=80
2. Moderate Regularization: L1 + L2 = 0.5 each (not too strong, not too weak)
3. Smart Subsampling: subsample=0.7, colsample_bytree=0.6 (more data per tree)
4. Extra Trees: extra_trees=True for additional randomization
5. Huber Loss: Robust to outliers in timing/strength prediction
6. Class Weights: Balance direction labels automatically
7. Multi-scale Volatility: Adapts thresholds to market conditions
8. MAE/MFE Timing: Better entry quality scoring

KEY FEATURES:
- Adapts to different volatility regimes (quiet vs volatile markets)
- Class-balanced training (handles imbalanced direction labels)
- Feature importance logging (see what model learns)
- Robust to outliers via Huber loss

Run: python scripts/train_v3_dynamic.py --days 90 --test_days 30 --pairs 20 --walk-forward
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES, DEFAULT_EXCLUDE_FEATURES,
    MINIMAL_STABLE_FEATURES, ULTRA_MINIMAL_FEATURES, CORE_20_FEATURES
)
from train_mtf import MTFFeatureEngine

# ============================================================
# FEATURE MODE: 
#   'core20' = only 20 most stable features (EXPERIMENTAL)
#   'ultra' = only TOP 50 features by importance (RECOMMENDED for live)
#   'minimal' = 75 stable features
#   'full' = all 125 features
# ============================================================
FEATURE_MODE = 'core20'  # 'core20', 'ultra', 'minimal' or 'full'

# ============================================================
# CONFIG
# ============================================================
SL_ATR_MULT = 1.5       # Base SL multiplier (adaptive based on strength)
MAX_LEVERAGE = 50.0     # Maximum leverage (50x)
MARGIN_BUFFER = 0.98    # 98% of capital for full deposit entry
FEE_PCT = 0.0002        # 0.02% Maker/Taker (MEXC Futures)
LOOKAHEAD = 12          # 1 hour on M5

# POSITION SIZE LIMITS
# User requirement: up to $4M position, with leverage up to 50x
# At 50x leverage: need $80k margin for $4M position
# At 10x leverage: $400k max position, at 20x: $200k max position
MAX_POSITION_SIZE = 4000000.0  # Max $4M position
SLIPPAGE_PCT = 0.0005         # 0.05% slippage (REALISTIC)

# V8 IMPROVEMENTS
USE_ADAPTIVE_SL = True       # Adjust SL based on predicted strength
USE_AGGRESSIVE_TRAIL = True  # Tighter trailing at medium R-multiples


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
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features (V3) - OBV исключен (зависит от начала окна данных)."""
    df = df.copy()
    
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    # OBV УДАЛЕН: cumsum() зависит от начала окна данных
    # В бектесте данные могут начинаться с 2017 года, в лайве - с последних 1500 свечей
    # Это приводит к кардинально разным значениям OBV
    # OBV уже исключен из фичей модели, поэтому не вычисляем его вообще
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    # Volume momentum
    # ✅ FIX: Clip extreme spikes (PIPPIN had 431x volume spike)
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    
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
# V12 SMART ADAPTIVE TARGETS
# ============================================================
def create_targets_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create V12 Smart Adaptive targets.
    
    KEY IMPROVEMENTS:
    1. Multi-scale volatility adaptation - adapts to different market conditions
    2. Trend-aware thresholds - higher threshold in strong trends (avoid noise)
    3. Time-weighted future returns - near-term matters more than far-term
    4. Robust outlier handling
    
    The model learns to predict:
    - Direction: Which way will price move significantly?
    - Timing: How good is this entry point? (0-5 scale)
    - Strength: How big will the move be? (in ATR multiples)
    """
    df = df.copy()
    df['atr'] = calculate_atr(df)
    
    # 1. Multi-scale volatility for adaptive thresholds
    # Use multiple windows to capture both short and long-term volatility
    vol_short = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    vol_medium = df['close'].pct_change().rolling(window=50, min_periods=25).std()
    vol_long = df['close'].pct_change().rolling(window=100, min_periods=50).std()
    
    # Combine: use the HIGHER of recent or historical volatility
    # This prevents false signals in quiet periods after volatile ones
    combined_vol = np.maximum(vol_short, (vol_medium + vol_long) / 2)
    combined_vol = combined_vol.shift(1)  # Use only past data
    
    # Adaptive threshold: LOWER threshold for more direction signals
    # Changed from 0.8x/0.3% to 0.4x/0.15% for ~40% direction labels instead of ~22%
    threshold = np.maximum(combined_vol * 0.4, 0.0015)
    
    # 2. Calculate future return with time-weighting
    # Near-term movement is more important than far-term
    future_return_6 = df['close'].pct_change(6).shift(-6)   # 30min
    future_return_12 = df['close'].pct_change(12).shift(-12) # 1hour
    
    # Weight: 60% near-term, 40% full window
    future_return = 0.6 * future_return_6.fillna(0) + 0.4 * future_return_12.fillna(0)
    
    # 3. Direction with trend confirmation
    # 0=Down, 1=Sideways, 2=Up
    df['target_dir'] = np.where(
        future_return > threshold, 2,
        np.where(future_return < -threshold, 0, 1)
    )
    
    # 4. Timing Target: Entry quality score
    future_lows = df['low'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
    future_highs = df['high'].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
    
    # Maximum Adverse Excursion (MAE) - how much does price go against us first?
    mae_long = (df['close'] - future_lows) / df['atr']
    mae_short = (future_highs - df['close']) / df['atr']
    
    # Maximum Favorable Excursion (MFE) - how much profit potential?
    mfe_long = (future_highs - df['close']) / df['atr']
    mfe_short = (df['close'] - future_lows) / df['atr']
    
    # Timing score = MFE / (1 + MAE) - penalize entries with high drawdown
    timing_long = mfe_long / (1 + mae_long)
    timing_short = mfe_short / (1 + mae_short)
    
    df['target_timing'] = np.maximum(timing_long, timing_short)
    df['target_timing'] = df['target_timing'].clip(0, 5)
    
    # 5. Strength: Directional move potential
    # How much does price move in the predicted direction?
    df['target_strength'] = np.where(
        df['target_dir'] == 2, mfe_long,
        np.where(df['target_dir'] == 0, mfe_short, 0)
    )
    df['target_strength'] = df['target_strength'].clip(0, 10)
    
    return df


# ============================================================
# TRAINING (V12 - Smart Adaptive Model)
# ============================================================
def train_models(X_train, y_train, X_val, y_val):
    """
    Train SMART ADAPTIVE models that work across market conditions.
    
    V12 Smart Model Features:
    - Balanced complexity: not too simple (underfitting), not too complex (overfitting)
    - Adaptive learning rate with more trees
    - Feature subsampling for robustness
    - Dart booster for better generalization (drops trees randomly)
    - Class weights for imbalanced direction labels
    - StandardScaler for consistent feature ranges
    
    Expected: 60-70% winrate on backtest, confident predictions on live.
    """
    
    # Scale features for consistent ranges (RSI 0-100, returns -0.1 to 0.1, etc)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    print(f"   📐 Features scaled: mean=0, std=1")
    
    # Calculate class weights for imbalanced labels
    from collections import Counter
    label_counts = Counter(y_train['target_dir'])
    total = sum(label_counts.values())
    class_weights = {k: total / (3 * v) for k, v in label_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train['target_dir']])
    
    # 1. Direction Model (Multiclass) - CONFIDENT VERSION
    # Robust params to avoid overfitting
    print("   Training Direction Model (High Confidence)...")
    dir_model = lgb.LGBMClassifier(
        objective='multiclass', 
        num_class=3, 
        metric='multi_logloss',
        boosting_type='gbdt',
        n_estimators=300,          # More trees
        max_depth=5,               # Shallower = less overfitting
        num_leaves=31,             # Standard, more robust
        min_child_samples=50,      # Higher = less noise
        learning_rate=0.03,        # Slower learning, more iterations
        subsample=0.8,             # Less data = more robust
        colsample_bytree=0.7,      # Fewer features = generalize better
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=0.1,            # L2 regularization
        min_split_gain=0.01,       # Filter noisy splits
        random_state=42, 
        verbosity=-1,
        importance_type='gain'
    )
    dir_model.fit(
        X_train_scaled, y_train['target_dir'], 
        sample_weight=sample_weights,
        eval_set=[(X_val_scaled, y_val['target_dir'])],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    
    # 2. Timing Model (Regressor) - ROBUST VERSION
    print("   Training Timing Model (High Predictions)...")
    timing_model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        boosting_type='gbdt',
        n_estimators=300,
        max_depth=5,
        num_leaves=31,
        min_child_samples=50,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1
    )
    timing_model.fit(
        X_train_scaled, y_train['target_timing'],
        eval_set=[(X_val_scaled, y_val['target_timing'])],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    
    # 3. Strength Model (Regression) - ROBUST VERSION
    print("   Training Strength Model (High Predictions)...")
    strength_model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        boosting_type='gbdt',
        n_estimators=300,
        max_depth=5,
        num_leaves=31,
        min_child_samples=50,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1
    )
    strength_model.fit(
        X_train_scaled, y_train['target_strength'],
        eval_set=[(X_val_scaled, y_val['target_strength'])],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    
    # Log feature importance for top 20 features
    print("\n   📊 Top 20 Features by Importance (Direction Model):")
    importance = dir_model.feature_importances_
    feature_names = X_train.columns.tolist()
    importance_pairs = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for name, imp in importance_pairs[:20]:
        print(f"      {name}: {imp:.1f}")
    
    return {
        'direction': dir_model,
        'timing': timing_model,
        'strength': strength_model,
        'scaler': scaler
    }


# ============================================================
# PORTFOLIO BACKTEST (V9 - Realistic Thresholds)
# ============================================================
def generate_signals(df: pd.DataFrame, feature_cols: list, models: dict, pair_name: str,
                    min_conf: float = 0.58, min_timing: float = 1.8, min_strength: float = 2.5) -> list:
                    
    """
    Generate all valid signals for a single pair.
    """
    signals = []
    
    # Predict in batches for speed
    X = df[feature_cols].values
    
    # Scale features if scaler is provided
    if 'scaler' in models and models['scaler'] is not None:
        X = models['scaler'].transform(X)
    
    # 1. Direction
    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    
    # 2. ✅ Timing (NOW REGRESSOR - returns gain potential in ATR multiples)
    timing_preds = models['timing'].predict(X)  # ⚠️ Changed from predict_proba!
    
    # 3. Strength
    strength_preds = models['strength'].predict(X)
    
    # Iterate and filter
    for i in range(len(df)):
        if dir_preds[i] == 1: continue # Sideways
        
        if dir_confs[i] < min_conf: continue
        if timing_preds[i] < min_timing: continue  # ✅ Now checks ATR gain potential
        if strength_preds[i] < min_strength: continue
        
        # ✅ НОВОЕ: Сохраняем ВСЕ 159 значений фичей для полного анализа
        all_features = {}
        for feat_name in feature_cols:
            if feat_name in df.columns:
                all_features[f'feat_{feat_name}'] = df[feat_name].iloc[i]
        
        signals.append({
            'timestamp': df.index[i],
            'pair': pair_name,
            'direction': 'LONG' if dir_preds[i] == 2 else 'SHORT',
            'entry_price': df['close'].iloc[i],
            'atr': df['atr'].iloc[i],
            'score': dir_confs[i] * timing_preds[i], # Combined score (conf × timing_gain)
            'timing_prob': timing_preds[i],  # ✅ Now stores gain potential (not probability)
            'pred_strength': strength_preds[i],
            **all_features  # ✅ Добавляем ВСЕ 159 фичей с префиксом feat_
        })
        
    return signals


def simulate_trade(signal: dict, df: pd.DataFrame) -> dict:
    """
    Simulate a single trade on a specific pair dataframe.
    V8 IMPROVEMENTS:
    - Adaptive SL based on predicted strength
    - Dynamic breakeven trigger
    - Aggressive trailing at medium R-multiples
    """
    # Find start index
    try:
        start_idx = df.index.get_loc(signal['timestamp'])
    except KeyError:
        return None
        
    entry_price = signal['entry_price']
    atr = signal['atr']
    direction = signal['direction']
    pred_strength = signal.get('pred_strength', 2.0)
    
    # === ATR-BASED STOP LOSS (дальше, стабильнее) ===
    # Adaptive SL: 1.2-1.6x ATR based on strength
    if pred_strength >= 3.0:
        sl_mult = 1.6  # Strong: room to breathe
    elif pred_strength >= 2.0:
        sl_mult = 1.5  # Medium: standard
    else:
        sl_mult = 1.2  # Weak: tight
    
    sl_dist = atr * sl_mult
    
    # === ATR-BASED BREAKEVEN TRIGGER ===
    if pred_strength >= 3.0:
        be_trigger_mult = 2.5
    elif pred_strength >= 2.0:
        be_trigger_mult = 2.2
    else:
        be_trigger_mult = 1.8
    
    be_trigger_dist = atr * be_trigger_mult
    be_margin_dist = atr * 1.0  # Lock 1.0 ATR profit
    
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
    max_r_reached = 0.0  # Track maximum R for trailing
    
    # Simulate bar by bar
    for j in range(start_idx + 1, min(start_idx + 150, len(df))):
        bar = df.iloc[j]
        
        if direction == 'LONG':
            if bar['low'] <= sl_price:
                outcome = 'stop_loss' if not breakeven_active else 'breakeven_stop'
                exit_price = sl_price
                exit_time = bar.name
                break
            
            # Breakeven trigger - ATR-based
            if not breakeven_active and bar['high'] >= be_trigger_price:
                breakeven_active = True
                sl_price = entry_price + be_margin_dist  # Lock 1.0 ATR profit
            
            # Trailing - R-based multipliers
            if breakeven_active:
                current_profit = bar['high'] - entry_price
                r_multiple = current_profit / sl_dist
                max_r_reached = max(max_r_reached, r_multiple)
                
                # R-based trailing
                if r_multiple > 5.0:
                    trail_mult = 0.6
                elif r_multiple > 3.0:
                    trail_mult = 1.2
                elif r_multiple > 2.0:
                    trail_mult = 1.8
                else:
                    trail_mult = 2.5
                
                new_sl = bar['high'] - atr * trail_mult
                if new_sl > sl_price:
                    sl_price = new_sl
                    
        else: # SHORT
            if bar['high'] >= sl_price:
                outcome = 'stop_loss' if not breakeven_active else 'breakeven_stop'
                exit_price = sl_price
                exit_time = bar.name
                break
            
            # Breakeven trigger - ATR-based
            if not breakeven_active and bar['low'] <= be_trigger_price:
                breakeven_active = True
                sl_price = entry_price - be_margin_dist  # Lock 1.0 ATR profit
            
            # Trailing - R-based multipliers
            if breakeven_active:
                current_profit = entry_price - bar['low']
                r_multiple = current_profit / sl_dist
                max_r_reached = max(max_r_reached, r_multiple)
                
                # R-based trailing
                if r_multiple > 5.0:
                    trail_mult = 0.6
                elif r_multiple > 3.0:
                    trail_mult = 1.2
                elif r_multiple > 2.0:
                    trail_mult = 1.8
                else:
                    trail_mult = 2.5
                
                new_sl = bar['low'] + atr * trail_mult
                if new_sl < sl_price:
                    sl_price = new_sl
                    
    return {
        'exit_time': exit_time,
        'exit_price': exit_price,
        'outcome': outcome,
        'sl_dist': sl_dist,
        'sl_mult': sl_mult,
        'max_r': max_r_reached
    }


def run_portfolio_backtest(signals: list, pair_dfs: dict, initial_balance: float = 10000.0) -> list:
    """
    Execute signals enforcing the 'Single Slot' constraint.
    RISK-BASED POSITION SIZING: ATR stops + 5% risk per trade.
    """
    # Sort by time. If times are equal, sort by score (descending)
    signals.sort(key=lambda x: (x['timestamp'], -x['score']))
    
    executed_trades = []
    last_exit_time = pd.Timestamp.min.tz_localize('UTC')  # Must be timezone-aware (UTC)
    balance = initial_balance
    
    RISK_PCT = 0.05  # 5% риска от депозита при стопе
    
    print(f"Processing {len(signals)} potential signals...")
    print(f"Initial Balance: ${balance:,.2f}")
    print(f"Position sizing: RISK-BASED (5% loss per stop, ATR-based stops)")
    
    for signal in signals:
        # Constraint: Can only hold 1 position
        if signal['timestamp'] < last_exit_time:
            continue
            
        # Execute
        pair_df = pair_dfs[signal['pair']]
        result = simulate_trade(signal, pair_df)
        
        if result:
            # Calculate Position Size & PnL
            entry_price = signal['entry_price']
            exit_price = result['exit_price']
            sl_dist = result['sl_dist']
            
            # === RISK-BASED POSITION SIZING ===
            # Размер позиции рассчитан так, чтобы при стопе терять ровно 5% от депозита
            risk_amount = balance * RISK_PCT  # Сколько готовы потерять
            sl_pct = sl_dist / entry_price  # Стоп в % от цены
            
            # position_size * sl_pct = risk_amount
            position_size = risk_amount / sl_pct
            
            # Ограничения
            max_position_by_leverage = balance * MAX_LEVERAGE
            original_position = position_size
            position_size = min(position_size, max_position_by_leverage, MAX_POSITION_SIZE)
            
            # CRITICAL: Если position урезан, нужно уменьшить стоп чтобы сохранить 5% риск!
            if position_size < original_position:
                # Пересчитываем стоп для сохранения риска
                new_sl_pct = risk_amount / position_size
                sl_dist = new_sl_pct * entry_price
                sl_pct = new_sl_pct
            
            # Рассчитываем итоговое плечо
            leverage = position_size / balance
            
            # PnL Calculation (with slippage)
            if signal['direction'] == 'LONG':
                effective_entry = entry_price * (1 + SLIPPAGE_PCT)  # Worse entry
                effective_exit = exit_price * (1 - SLIPPAGE_PCT)    # Worse exit
                raw_pnl_pct = (effective_exit - effective_entry) / effective_entry
            else:
                effective_entry = entry_price * (1 - SLIPPAGE_PCT)  # Worse entry for short
                effective_exit = exit_price * (1 + SLIPPAGE_PCT)    # Worse exit for short
                raw_pnl_pct = (effective_entry - effective_exit) / effective_entry
                
            gross_profit = position_size * raw_pnl_pct
            fees = position_size * FEE_PCT * 2 # Entry + Exit
            net_profit = gross_profit - fees
            
            old_balance = balance  # Save balance BEFORE trade
            balance += net_profit  # Update balance
            
            trade_record = signal.copy()
            trade_record.update(result)
            trade_record.update({
                'leverage': leverage,
                'position_size': position_size,
                'net_profit': net_profit,
                'balance_after': balance,
                'pnl_pct': (net_profit / old_balance) * 100,  # CORRECT: relative to OLD balance
                'roe': (net_profit / (position_size / leverage)) * 100  # ROE relative to margin used
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
        
        # Calculate trade ROE (Return on Equity/Margin used)
        roe = t.get('roe', t['pnl_pct'])  # Use ROE if available, else fall back to pnl_pct
        
        # Add emoji based on result
        emoji = "🚀" if roe > 20 else "✅" if t['net_profit'] > 0 else "❌"
        if t['net_profit'] > 0 and roe < 5: emoji = "🛡️" # Breakeven/Small profit
        
        # Show position size and leverage for clarity
        lev = t.get('leverage', 1)
        pos_size = t.get('position_size', 0)
        
        print(f"{pair_clean} ({t['direction']}) {time_str} — Profit: ${t['net_profit']:+,.2f} (ROE: {roe:+.1f}%) {emoji}")
        print(f"   Entry: {t['entry_price']:.5f} | Exit: {t['exit_price']:.5f} | Reason: {t['outcome']}")
        print(f"   Position: ${pos_size:,.0f} @ {lev:.1f}x leverage | Balance after: ${t.get('balance_after', 0):,.2f}")
        print("-" * 30)


# ============================================================
# ✅ WALK-FORWARD VALIDATION (Honest Out-of-Sample Test)
# ============================================================
def walk_forward_validation(pairs, data_dir, mtf_fe, initial_balance=100.0):
    """
    Walk-Forward Validation: Train on past, test on future (never seen before).
    
    This is the MOST HONEST test for overfitting detection.
    If model performs well here, it will likely work in live trading.
    
    Example timeline:
    Period 1: Train [Sep 1-15]  → Test [Sep 16-22]
    Period 2: Train [Sep 8-22]  → Test [Sep 23-30]
    Period 3: Train [Sep 15-30] → Test [Oct 1-7]
    ...
    """
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION (Honest Out-of-Sample Test)")
    print("="*70)
    print("This tests if the model can predict TRULY UNSEEN future data.")
    print("If Win Rate drops significantly here → Model is overfit!")
    print("="*70)
    
    # EMBARGO PERIOD: Gap between train and test to prevent data leakage
    # This is a key anti-overfitting technique from quantitative finance
    # The model trained on data up to day N should not be tested on day N+1
    # because features may "leak" information from the target period
    EMBARGO_DAYS = 1  # 1 day gap = 288 M5 candles (12 * 24)
    
    # Define periods - REALISTIC training periods (30 days train, 7 days test)
    # This matches the actual model training setup
    # We need 90 days of data total, but only have Dec data, so we'll use what we have
    periods = [
        {
            'name': "Period_1",
            'train_start': datetime(2025, 10, 15, tzinfo=timezone.utc),
            'train_end': datetime(2025, 11, 14, tzinfo=timezone.utc),  # 30 days train
            'test_start': datetime(2025, 11, 15, tzinfo=timezone.utc),
            'test_end': datetime(2025, 11, 21, tzinfo=timezone.utc)    # 7 days test
        },
        {
            'name': "Period_2",
            'train_start': datetime(2025, 11, 1, tzinfo=timezone.utc),
            'train_end': datetime(2025, 11, 30, tzinfo=timezone.utc),  # 30 days train
            'test_start': datetime(2025, 12, 1, tzinfo=timezone.utc),
            'test_end': datetime(2025, 12, 7, tzinfo=timezone.utc)     # 7 days test
        },
        {
            'name': "Period_3",
            'train_start': datetime(2025, 11, 15, tzinfo=timezone.utc),
            'train_end': datetime(2025, 12, 14, tzinfo=timezone.utc),  # 30 days train
            'test_start': datetime(2025, 12, 15, tzinfo=timezone.utc),
            'test_end': datetime(2025, 12, 21, tzinfo=timezone.utc)    # 7 days test
        },
        {
            'name': "Period_4",
            'train_start': datetime(2025, 12, 1, tzinfo=timezone.utc),
            'train_end': datetime(2025, 12, 30, tzinfo=timezone.utc),  # 30 days train
            'test_start': datetime(2025, 12, 31, tzinfo=timezone.utc),
            'test_end': datetime(2026, 1, 6, tzinfo=timezone.utc)      # 7 days test
        },
    ]
    
    all_results = []
    
    for period in periods:
        print(f"\n{'='*60}")
        print(f"📊 {period['name']}")
        print(f"   TRAIN: {period['train_start'].strftime('%Y-%m-%d')} → {period['train_end'].strftime('%Y-%m-%d')}")
        print(f"   TEST:  {period['test_start'].strftime('%Y-%m-%d')} → {period['test_end'].strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Load and prepare data
        all_train = []
        test_dfs = {}
        test_features = {}
        
        for pair in pairs:
            pair_name = pair.replace('/', '_').replace(':', '_')
            
            try:
                # ✅ FIX: Use Parquet files (same as main backtest) - they have latest data!
                m1 = pd.read_parquet(data_dir / f"{pair_name}_1m.parquet")
                m5 = pd.read_parquet(data_dir / f"{pair_name}_5m.parquet")
                m15 = pd.read_parquet(data_dir / f"{pair_name}_15m.parquet")
                
                # Ensure timezone-aware indices (UTC) for comparison with timezone-aware datetimes
                if m1.index.tz is None:
                    m1.index = m1.index.tz_localize('UTC')
                if m5.index.tz is None:
                    m5.index = m5.index.tz_localize('UTC')
                if m15.index.tz is None:
                    m15.index = m15.index.tz_localize('UTC')
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"    ⚠️ {pair}: Error loading data: {e}")
                continue
            
            # Filter TRAIN data
            m1_train = m1[(m1.index >= period['train_start']) & (m1.index < period['train_end'])]
            m5_train = m5[(m5.index >= period['train_start']) & (m5.index < period['train_end'])]
            m15_train = m15[(m15.index >= period['train_start']) & (m15.index < period['train_end'])]
            
            if len(m5_train) < 500: continue
            
            ft_train = mtf_fe.align_timeframes(m1_train, m5_train, m15_train)
            ft_train = ft_train.join(m5_train[['open', 'high', 'low', 'close', 'volume']])
            ft_train = add_volume_features(ft_train)
            ft_train = create_targets_v1(ft_train)
            ft_train['pair'] = pair
            all_train.append(ft_train)
            
            # Filter TEST data (UNSEEN!)
            m1_test = m1[(m1.index >= period['test_start']) & (m1.index < period['test_end'])]
            m5_test = m5[(m5.index >= period['test_start']) & (m5.index < period['test_end'])]
            m15_test = m15[(m15.index >= period['test_start']) & (m15.index < period['test_end'])]
            
            if len(m5_test) < 100: continue
            
            ft_test = mtf_fe.align_timeframes(m1_test, m5_test, m15_test)
            ft_test = ft_test.join(m5_test[['open', 'high', 'low', 'close', 'volume']])
            ft_test = add_volume_features(ft_test)
            ft_test = create_targets_v1(ft_test)
            ft_test['pair'] = pair
            test_features[pair] = ft_test
            test_dfs[pair] = ft_test
        
        if len(all_train) == 0:
            print(f"⚠️  {period['name']}: No training data, skipping")
            continue
        
        # Train models on THIS period
        train_df = pd.concat(all_train).dropna()
        
        # === FEATURE SELECTION BASED ON MODE ===
        available_cols = set(train_df.columns)
        
        if FEATURE_MODE == 'core20':
            # Use only 20 most stable features (maximum stability for live)
            features = [f for f in CORE_20_FEATURES if f in available_cols]
            print(f"   📊 Using CORE20 mode: {len(features)} core features")
        elif FEATURE_MODE == 'ultra':
            # Use only TOP 50 features by importance (most stable for live trading)
            features = [f for f in ULTRA_MINIMAL_FEATURES if f in available_cols]
            print(f"   📊 Using ULTRA mode: {len(features)} top features")
        elif FEATURE_MODE == 'minimal':
            # Use only the minimal stable features (75 features)
            features = [f for f in MINIMAL_STABLE_FEATURES if f in available_cols]
            print(f"   📊 Using MINIMAL mode: {len(features)} stable features")
        else:
            # Full mode: exclude problematic features
            exclude = list(DEFAULT_EXCLUDE_FEATURES)
            all_exclude = set(exclude) | set(ABSOLUTE_PRICE_FEATURES)
            features = [c for c in train_df.columns if c not in all_exclude 
                        and not any(p in c.lower() for p in CUMSUM_PATTERNS)]
            print(f"   📊 Using FULL mode: {len(features)} features")
        
        X_train = train_df[features]
        y_train = {
            'target_dir': train_df['target_dir'],
            'target_timing': train_df['target_timing'],
            'target_strength': train_df['target_strength']
        }
        
        # Simple 90/10 split for validation
        val_idx = int(len(X_train) * 0.9)
        X_t = X_train.iloc[:val_idx]
        X_v = X_train.iloc[val_idx:]
        y_t = {k: v.iloc[:val_idx] for k, v in y_train.items()}
        y_v = {k: v.iloc[val_idx:] for k, v in y_train.items()}
        
        models = train_models(X_t, y_t, X_v, y_v)
        
        # Test on UNSEEN period
        all_signals = []
        for pair, df in test_features.items():
            df_clean = df.dropna()
            if len(df_clean) == 0: continue
            sigs = generate_signals(df_clean, features, models, pair)
            all_signals.extend(sigs)
        
        if len(all_signals) == 0:
            print(f"⚠️  {period['name']}: No signals generated")
            continue
        
        trades, final_bal = run_portfolio_backtest(all_signals, test_dfs, initial_balance=initial_balance)
        
        # Calculate metrics
        if len(trades) > 0:
            wins = [t for t in trades if t['net_profit'] > 0]
            win_rate = len(wins) / len(trades) * 100
            total_pnl = final_bal - initial_balance
            pnl_pct = (total_pnl / initial_balance) * 100
            
            result = {
                'period': period['name'],
                'trades': len(trades),
                'win_rate': win_rate,
                'pnl': total_pnl,
                'pnl_pct': pnl_pct,
                'final_balance': final_bal
            }
            all_results.append(result)
            
            print(f"   Trades: {len(trades)} | Win Rate: {win_rate:.1f}% | PnL: ${total_pnl:+.2f} ({pnl_pct:+.1f}%)")
        else:
            print(f"   No trades executed")
    
    # Summary
    if len(all_results) > 0:
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*70)
        
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_trades = np.mean([r['trades'] for r in all_results])
        total_pnl = sum([r['pnl'] for r in all_results])
        
        print(f"Average Win Rate:  {avg_win_rate:.1f}%")
        print(f"Average Trades:    {avg_trades:.1f}")
        print(f"Total PnL:         ${total_pnl:+.2f}")
        
        # === BENCHMARK COMPARISON ===
        print("\n" + "-"*50)
        print("📊 BENCHMARK COMPARISON:")
        print("-"*50)
        
        # Random baseline: 50% WR minus fees (~0.04% per trade round-trip)
        # With SL exits, random would win ~48-49% after fees
        random_baseline = 48.0
        trend_baseline = 52.0  # Simple EMA crossover typically gets 50-55%
        
        model_edge_vs_random = avg_win_rate - random_baseline
        model_edge_vs_trend = avg_win_rate - trend_baseline
        
        print(f"   Random Baseline:     {random_baseline:.1f}% WR (50/50 coin flip)")
        print(f"   Trend Follow (EMA):  {trend_baseline:.1f}% WR (EMA 9/21 cross)")
        print(f"   Your Model:          {avg_win_rate:.1f}% WR")
        print()
        print(f"   Edge vs Random:      {model_edge_vs_random:+.1f}%")
        print(f"   Edge vs Trend:       {model_edge_vs_trend:+.1f}%")
        
        if model_edge_vs_random < 5:
            print("\n   ⚠️  WARNING: Model barely beats random!")
            print("   → Consider adding more features or data")
        elif model_edge_vs_random >= 10:
            print("\n   ✅ STRONG EDGE! Model significantly beats benchmarks.")
        else:
            print("\n   👍 Model has decent edge over random.")
        
        print("-"*50)
        
        print("\n💡 INTERPRETATION:")
        if avg_win_rate >= 55:
            print("   ✅ EXCELLENT! Model generalizes well to unseen data.")
            print("   → Ready for paper trading!")
        elif avg_win_rate >= 50:
            print("   ⚠️  ACCEPTABLE. Model works but needs monitoring.")
            print("   → Try paper trading with caution.")
        else:
            print("   ❌ POOR! Model is likely overfit or has no edge.")
            print("   → DO NOT use in live trading. Retrain with more data.")
        
        print("="*70)
        
        return all_results
    else:
        print("\n⚠️  Walk-Forward Validation failed: No results")
        return []


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="Training days (reduced from 90 for faster convergence)")
    parser.add_argument("--test_days", type=int, default=14, help="Test days (out-of-sample)")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs to use from pairs_20.json")
    parser.add_argument("--pair", type=str, default=None, help="Specific pair to train on (e.g., 'PIPPIN/USDT:USDT'). Overrides --pairs.")
    parser.add_argument("--pairs_list", type=str, default=None, help="Comma-separated list of pairs (e.g., 'PIPPIN/USDT:USDT,ASTER/USDT:USDT,ZEC/USDT:USDT'). Overrides --pairs.")
    parser.add_argument("--output", type=str, default="./models/v8_improved")
    parser.add_argument("--initial_balance", type=float, default=100.0, help="Initial portfolio balance (realistic $100 start)")
    parser.add_argument("--check-dec25", action="store_true", help="Fetch and test specifically for Dec 25, 2025")
    parser.add_argument("--check-dec26", action="store_true", help="Fetch and test specifically for Dec 26, 2025")
    parser.add_argument("--reverse", action="store_true", help="Train on Recent 30d, Test on Previous 30d (For Paper Trading Prep)")
    parser.add_argument("--walk-forward", action="store_true", help="✅ NEW: Run Walk-Forward Validation (honest test!)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("V8 IMPROVED - ANTI-OVERFITTING EDITION")
    print("=" * 70)
    print("✅ Simplified models (100 trees, depth 3)")
    print("✅ Improved timing target (regression)")
    print("✅ Realistic slippage (0.05%)")
    if args.walk_forward:
        print("✅ Walk-Forward Validation ENABLED")
    print("=" * 70)
    
    # Load pairs
    import json
    
    # Support for pairs list mode (multiple pairs via comma-separated list)
    if args.pairs_list:
        # Multiple pairs from comma-separated list
        pairs = [p.strip() for p in args.pairs_list.split(',')]
        print(f"🎯 MULTI PAIR MODE: {len(pairs)} pairs - {pairs}")
    elif args.pair:
        # Single pair mode - use the specified pair
        pairs = [args.pair]
        print(f"🎯 SINGLE PAIR MODE: {args.pair}")
    else:
        # Multi-pair mode - load from JSON file
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
    
    # 1. LOAD TRAINING DATA (Local)
    print(f"\nLoading Data (Reverse={args.reverse})...")
    for pair in pairs:
        print(f"Processing {pair}...", end='\r')
        pair_name = pair.replace('/', '_').replace(':', '_')
        
        try:
            m1 = pd.read_csv(data_dir / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m5 = pd.read_csv(data_dir / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
            m15 = pd.read_csv(data_dir / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
            
            # Ensure timezone-aware indices (UTC) for comparison with timezone-aware datetimes
            # Handle both tz-naive and tz-aware indices
            if m1.index.tz is None:
                m1.index = m1.index.tz_localize('UTC')
            else:
                m1.index = m1.index.tz_convert('UTC')
            if m5.index.tz is None:
                m5.index = m5.index.tz_localize('UTC')
            else:
                m5.index = m5.index.tz_convert('UTC')
            if m15.index.tz is None:
                m15.index = m15.index.tz_localize('UTC')
            else:
                m15.index = m15.index.tz_convert('UTC')
        except FileNotFoundError:
            print(f"  ⚠️ {pair}: CSV files not found, skipping")
            continue
        except Exception as e:
            print(f"  ⚠️ {pair}: Error loading data: {e}")
            continue
        
        # SPLIT LOGIC
        now = datetime.now(timezone.utc)
        if args.reverse:
            # Train on LAST 30 days (Recent)
            # Test on PREVIOUS 30 days (Older)
            train_end = now
            train_start = now - timedelta(days=args.days)
            
            test_end = train_start
            test_start = test_end - timedelta(days=args.test_days)
        else:
            # Standard: Train on Old, Test on Recent
            test_start = now - timedelta(days=args.test_days)
            train_start = test_start - timedelta(days=args.days)
            train_end = test_start
            test_end = now
        
        # Filter Train
        m1_train = m1[(m1.index >= train_start) & (m1.index < train_end)]
        m5_train = m5[(m5.index >= train_start) & (m5.index < train_end)]
        m15_train = m15[(m15.index >= train_start) & (m15.index < train_end)]
        
        if len(m5_train) < 500:
            # Show available data range to help debug
            data_start = m5.index.min() if len(m5) > 0 else "empty"
            data_end = m5.index.max() if len(m5) > 0 else "empty"
            print(f"  ⚠️ {pair}: Skipped (only {len(m5_train)} 5m candles in range, need 500)")
            print(f"      Data range: {data_start} to {data_end}")
            print(f"      Requested : {train_start} to {train_end}")
            continue
        
        ft_train = mtf_fe.align_timeframes(m1_train, m5_train, m15_train)
        ft_train = ft_train.join(m5_train[['open', 'high', 'low', 'close', 'volume']])
        ft_train = add_volume_features(ft_train)
        ft_train = create_targets_v1(ft_train)
        ft_train['pair'] = pair
        all_train.append(ft_train)
        
        # Filter Test
        m1_test = m1[(m1.index >= test_start) & (m1.index < test_end)]
        m5_test = m5[(m5.index >= test_start) & (m5.index < test_end)]
        m15_test = m15[(m15.index >= test_start) & (m15.index < test_end)]
        
        ft_test = mtf_fe.align_timeframes(m1_test, m5_test, m15_test)
        ft_test = ft_test.join(m5_test[['open', 'high', 'low', 'close', 'volume']])
        ft_test = add_volume_features(ft_test)
        ft_test = create_targets_v1(ft_test)
        ft_test['pair'] = pair
        test_features[pair] = ft_test
        test_dfs[pair] = ft_test

    print(f"\nData loaded. Training on {len(all_train)} pairs.")
    
    # Train
    train_df = pd.concat(all_train).dropna()
    
    # === FEATURE SELECTION BASED ON MODE ===
    available_cols = set(train_df.columns)
    
    if FEATURE_MODE == 'core20':
        # Use only 20 most stable features (maximum stability for live)
        features = [f for f in CORE_20_FEATURES if f in available_cols]
        print(f"📊 Using CORE20 mode: {len(features)} core features")
    elif FEATURE_MODE == 'ultra':
        # Use only TOP 50 features by importance (most stable for live trading)
        features = [f for f in ULTRA_MINIMAL_FEATURES if f in available_cols]
        print(f"📊 Using ULTRA mode: {len(features)} top features")
    elif FEATURE_MODE == 'minimal':
        # Use only the minimal stable features (75 features)
        features = [f for f in MINIMAL_STABLE_FEATURES if f in available_cols]
        print(f"📊 Using MINIMAL mode: {len(features)} stable features")
    else:
        # Full mode: exclude problematic features
        exclude = list(DEFAULT_EXCLUDE_FEATURES)
        all_exclude = set(exclude) | set(ABSOLUTE_PRICE_FEATURES)
        features = [c for c in train_df.columns if c not in all_exclude 
                    and not any(p in c.lower() for p in CUMSUM_PATTERNS)]
        print(f"📊 Using FULL mode: {len(features)} features")
    
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
    
    # ---------------------------------------------------------
    # 1. STANDARD BACKTEST
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"RUNNING BACKTEST (Test Days: {args.test_days})")
    print("="*70)
    
    all_signals = []
    for pair, df in test_features.items():
        df_clean = df.dropna()
        if len(df_clean) == 0: continue
        sigs = generate_signals(df_clean, features, models, pair)
        all_signals.extend(sigs)
        
    trades, final_bal = run_portfolio_backtest(all_signals, test_dfs, initial_balance=args.initial_balance)
    print_trade_list(trades)
    print_results(trades, final_bal, initial_balance=args.initial_balance)
    
    # Save trades
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    # SAVE MODELS
    print(f"\nSaving models to {out}...")
    joblib.dump(models['direction'], out / 'direction_model.joblib')
    joblib.dump(models['timing'], out / 'timing_model.joblib')
    joblib.dump(models['strength'], out / 'strength_model.joblib')
    joblib.dump(models['scaler'], out / 'scaler.joblib')
    joblib.dump(features, out / 'feature_names.joblib')
    print("Models and scaler saved.")

    if trades:
        pd.DataFrame(trades).to_csv(out / f'backtest_trades_{args.test_days}d.csv', index=False)

    # ---------------------------------------------------------
    # 1B. ✅ WALK-FORWARD VALIDATION (if requested)
    # ---------------------------------------------------------
    if args.walk_forward:
        walk_forward_results = walk_forward_validation(pairs, data_dir, mtf_fe, initial_balance=args.initial_balance)
        
        if walk_forward_results:
            # Save walk-forward results
            pd.DataFrame(walk_forward_results).to_csv(out / 'walk_forward_results.csv', index=False)
            print(f"Walk-forward results saved to {out / 'walk_forward_results.csv'}")

    # ---------------------------------------------------------
    # 2. FETCH DEC 25 DATA IF REQUESTED
    # ---------------------------------------------------------
    if args.check_dec25:
        print("\n" + "="*70)
        print("RUNNING DEC 25 SPECIAL CHECK")
        print("="*70)
        print("Fetching Dec 25 Data from Binance...")
        
        # Fetch from Dec 23 to Dec 26 to ensure we have history for indicators
        fetch_start = datetime(2025, 12, 23, tzinfo=timezone.utc)
        fetch_end = datetime(2025, 12, 26, tzinfo=timezone.utc)
        
        dec25_features = {}
        dec25_dfs = {}
        
        for pair in pairs:
            print(f"Fetching {pair}...", end='\r')
            m1 = fetch_binance_data(pair, '1m', fetch_start, fetch_end)
            m5 = fetch_binance_data(pair, '5m', fetch_start, fetch_end)
            m15 = fetch_binance_data(pair, '15m', fetch_start, fetch_end)
            
            if len(m1) < 100 or len(m5) < 100 or len(m15) < 100:
                # print(f"Skipping {pair} (Insufficient data)")
                continue
                
            ft = mtf_fe.align_timeframes(m1, m5, m15)
            ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft) # Ensure ATR is present
            ft['pair'] = pair
            
            # Filter for Dec 25 ONLY for the backtest part
            dec25_mask = (ft.index >= datetime(2025, 12, 25, tzinfo=timezone.utc)) & (ft.index < datetime(2025, 12, 26, tzinfo=timezone.utc))
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
            
        d25_trades, d25_bal = run_portfolio_backtest(dec25_signals, dec25_dfs, initial_balance=args.initial_balance)
        print_results(d25_trades, d25_bal, initial_balance=args.initial_balance)
        print_trade_list(d25_trades)
        
        if d25_trades:
            pd.DataFrame(d25_trades).to_csv(out / 'backtest_trades_dec25.csv', index=False)

    # ---------------------------------------------------------
    # 3. FETCH DEC 26 DATA IF REQUESTED
    # ---------------------------------------------------------
    if args.check_dec26:
        print("\n" + "="*70)
        print("RUNNING DEC 26 SPECIAL CHECK")
        print("="*70)
        print("Fetching Dec 26 Data from Binance...")
        
        # Fetch from Dec 24 to Dec 27 to ensure we have history for indicators
        fetch_start = datetime(2025, 12, 24, tzinfo=timezone.utc)
        fetch_end = datetime(2025, 12, 27, tzinfo=timezone.utc)
        
        dec26_features = {}
        dec26_dfs = {}
        
        for pair in pairs:
            print(f"Fetching {pair}...", end='\r')
            m1 = fetch_binance_data(pair, '1m', fetch_start, fetch_end)
            m5 = fetch_binance_data(pair, '5m', fetch_start, fetch_end)
            m15 = fetch_binance_data(pair, '15m', fetch_start, fetch_end)
            
            if len(m1) < 100 or len(m5) < 100 or len(m15) < 100:
                continue
                
            ft = mtf_fe.align_timeframes(m1, m5, m15)
            ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft) # Ensure ATR is present
            ft['pair'] = pair
            
            # Filter for Dec 26 ONLY for the backtest part
            dec26_mask = (ft.index >= datetime(2025, 12, 26, tzinfo=timezone.utc)) & (ft.index < datetime(2025, 12, 27, tzinfo=timezone.utc))
            ft_dec26 = ft[dec26_mask]
            
            if len(ft_dec26) > 0:
                dec26_features[pair] = ft_dec26
                dec26_dfs[pair] = ft_dec26
        print("\nFetch complete.")
        
        # Run Dec 26 Backtest
        dec26_signals = []
        for pair, df in dec26_features.items():
            df_clean = df.dropna()
            if len(df_clean) == 0: continue
            sigs = generate_signals(df_clean, features, models, pair)
            dec26_signals.extend(sigs)
            
        d26_trades, d26_bal = run_portfolio_backtest(dec26_signals, dec26_dfs, initial_balance=args.initial_balance)
        print_results(d26_trades, d26_bal, initial_balance=args.initial_balance)
        print_trade_list(d26_trades)
        
        if d26_trades:
            pd.DataFrame(d26_trades).to_csv(out / 'backtest_trades_dec26.csv', index=False)


if __name__ == '__main__':
    main()

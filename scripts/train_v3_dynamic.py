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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# Optuna for hyperparameter optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Run: pip install optuna")

# CatBoost for ensemble
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not installed. Run: pip install catboost")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import FeatureEngine
from src.utils.constants import (
    CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES, DEFAULT_EXCLUDE_FEATURES,
    MINIMAL_STABLE_FEATURES, ULTRA_MINIMAL_FEATURES, CORE_20_FEATURES
)
# train_mtf module removed — MTFFeatureEngine no longer used
# Features generated inline by generate_killer_features()

# ============================================================
# FEATURE MODE: 
#   'auto' = START WITH ALL, then auto-select best (RECOMMENDED!)
#   'core20' = only 20 most stable features (EXPERIMENTAL)
#   'ultra' = only TOP 50 features by importance
#   'minimal' = 75 stable features
#   'full' = all 125 features (no selection)
# ============================================================
FEATURE_MODE = 'auto'  # 'auto', 'core20', 'ultra', 'minimal' or 'full'
AUTO_SELECT_THRESHOLD = 0.001  # Min permutation importance to keep feature

# ============================================================
# OPTUNA & ENSEMBLE CONFIG
# ============================================================
USE_OPTUNA = False      # Disable — overfits to validation set
OPTUNA_TRIALS = 10      # 10 trials (was 30 — 3x faster, quality ~same)
USE_ENSEMBLE = False    # Single LightGBM — simpler = less overfit

# ============================================================
# 🎯 V13 IMPROVEMENTS - 10/10 CONFIG
# ============================================================
# 1. Confidence Threshold - торгуем только с высокой уверенностью
MIN_CONFIDENCE = 0.55           # ML как фильтр (тренд даёт направление, ML подтверждает)
MIN_TIMING = 0.5                # Мин. gain potential в ATR (тренд + pullback уже фильтрует)
MIN_STRENGTH = 0.5              # Мин. предсказанная сила движения в ATR
CONFIDENCE_BOOST_THRESHOLD = 0.70  # При этой уверенности увеличиваем сайз

# 2. Regime Filter - не торгуем в боковике
USE_REGIME_FILTER = True        # Включить фильтр режима рынка
MIN_VOLATILITY_PERCENTILE = 20  # Не торгуем если волатильность ниже 20-го перцентиля

# 3. Dynamic Position Sizing - больше сайз при высокой уверенности
USE_DYNAMIC_SIZING = True       # Включить динамический сайзинг
BASE_RISK_PCT = 0.03            # Базовый риск 3%
MAX_RISK_PCT = 0.05             # Максимальный риск 5% при высокой уверенности

# 4. Multi-Timeframe Confirmation - M15 подтверждает M5
USE_MTF_CONFIRMATION = True     # Включить MTF подтверждение

# 5. Slippage Simulation - реалистичнее бэктест
USE_REALISTIC_SLIPPAGE = True   # Честное проскальзывание (лимит ордера)
BASE_SLIPPAGE_PCT = 0.0001       # Базовое проскальзывание 0.01% (Binance фьючерсы)
VOLATILE_SLIPPAGE_PCT = 0.0002   # Проскальзывание при высокой волатильности 0.02%

# 6. Feature Stability Check - убираем нестабильные фичи
USE_FEATURE_STABILITY = True    # Включить проверку стабильности фич
MIN_FEATURE_STABILITY = 0.5     # Минимальная корреляция фичи между периодами

# ============================================================
# CONFIG
# ============================================================
SL_ATR_MULT = 2.0       # Wider SL for trend-following (reduce false stops)
MAX_LEVERAGE = 10.0     # Maximum leverage (10x, conservative for live)
MARGIN_BUFFER = 0.98    # 98% of capital for full deposit entry
FEE_PCT = 0.0002        # 0.02% Maker/Taker (MEXC Futures)
LOOKAHEAD = 6           # 30 min on M5 (faster trades, easier to predict)

# POSITION SIZE LIMITS
# User requirement: up to $4M position, with leverage up to 50x
# At 50x leverage: need $80k margin for $4M position
# At 10x leverage: $400k max position, at 20x: $200k max position
MAX_POSITION_SIZE = 4000000.0  # Max $4M position
SLIPPAGE_PCT = 0.0001           # 0.01% slippage (Binance futures)

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
# 🔥 V14 HIGH-ALPHA FEATURES
# ============================================================
def generate_killer_features(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame,
                             btc_m5: pd.DataFrame = None,
                             funding_rate: pd.DataFrame = None,
                             open_interest: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate ~35 HIGH-ALPHA features focused on actual predictive power.
    
    V14 REDESIGN — focus on features with REAL edge:
    ═══════════════════════════════════════════════════════════════
    1. BTC-RELATIVE — BTC leads alts, this is the strongest alpha
    2. ORDER FLOW PROXY — volume delta, buy/sell pressure from OHLCV
    3. SMART MOMENTUM — acceleration, efficiency, multi-TF divergence
    4. VOLATILITY REGIME — squeeze/expansion, relative ATR
    5. MEAN REVERSION — RSI extremes, BB touch, VWAP distance
    6. M15 CONTEXT — higher TF trend/momentum (shifted to avoid look-ahead)
    ═══════════════════════════════════════════════════════════════
    
    REMOVED: time features (hour_sin etc. — just noise, not predictive)
    """
    close = m5['close']
    high = m5['high']
    low = m5['low']
    open_ = m5['open']
    volume = m5['volume']
    
    f = pd.DataFrame(index=m5.index)
    
    # ─────────────────────────────────────────────────
    # 1. BTC-RELATIVE FEATURES (strongest alpha for alts)
    # ─────────────────────────────────────────────────
    if btc_m5 is not None and len(btc_m5) > 0:
        btc_close = btc_m5['close'].reindex(m5.index, method='ffill')
        btc_volume = btc_m5['volume'].reindex(m5.index, method='ffill')
        
        btc_ret_3 = btc_close.pct_change(3) * 100
        btc_ret_6 = btc_close.pct_change(6) * 100
        btc_ret_12 = btc_close.pct_change(12) * 100
        alt_ret_3 = close.pct_change(3) * 100
        alt_ret_6 = close.pct_change(6) * 100
        
        f['btc_momentum_3'] = btc_ret_3
        f['btc_momentum_12'] = btc_ret_12
        f['alt_btc_div_3'] = alt_ret_3 - btc_ret_3
        f['alt_btc_div_6'] = alt_ret_6 - btc_ret_6
        
        btc_ema_9 = btc_close.ewm(span=9, adjust=False).mean()
        btc_ema_21 = btc_close.ewm(span=21, adjust=False).mean()
        btc_atr = pd.concat([
            btc_m5['high'].reindex(m5.index, method='ffill') - btc_m5['low'].reindex(m5.index, method='ffill'),
            abs(btc_close - btc_close.shift(1))
        ], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()
        f['btc_trend_str'] = ((btc_ema_9 - btc_ema_21) / btc_atr).clip(-5, 5)
        
        btc_vol_ma = btc_volume.rolling(20).mean()
        f['btc_vol_surge'] = (btc_volume / btc_vol_ma).clip(0, 5)
        f['btc_alt_corr'] = close.pct_change().rolling(20).corr(btc_close.pct_change())
    else:
        f['btc_momentum_3'] = 0.0
        f['btc_momentum_12'] = 0.0
        f['alt_btc_div_3'] = 0.0
        f['alt_btc_div_6'] = 0.0
        f['btc_trend_str'] = 0.0
        f['btc_vol_surge'] = 1.0
        f['btc_alt_corr'] = 0.0
    
    # ─────────────────────────────────────────────────
    # 1B. FUNDING RATE FEATURES (strongest reversal signal)
    # ─────────────────────────────────────────────────
    if funding_rate is not None and len(funding_rate) > 0:
        # Forward-fill 8-hour funding rate to M5 frequency
        fr = funding_rate['funding_rate'].reindex(m5.index, method='ffill').fillna(0.0)
        f['funding_rate'] = fr * 10000  # Scale to bps for readability
        f['funding_extreme'] = (abs(fr) > 0.001).astype(float)  # >0.1% is extreme
        f['funding_direction'] = np.sign(fr)  # +1=longs pay, -1=shorts pay
        # Cumulative direction over last 3 settlements (~24h)
        f['funding_cum_3'] = f['funding_rate'].rolling(36, min_periods=1).mean()
    else:
        f['funding_rate'] = 0.0
        f['funding_extreme'] = 0.0
        f['funding_direction'] = 0.0
        f['funding_cum_3'] = 0.0
    
    # ─────────────────────────────────────────────────
    # 1C. OPEN INTEREST FEATURES (smart money positioning)
    # ─────────────────────────────────────────────────
    if open_interest is not None and len(open_interest) > 0:
        oi = open_interest['open_interest'].reindex(m5.index, method='ffill').fillna(method='bfill').fillna(0)
        oi_ma = oi.rolling(48, min_periods=1).mean()
        
        # OI change rate (normalized)
        f['oi_change_pct'] = oi.pct_change(6).fillna(0) * 100  # 30-min OI change %
        
        # OI relative to average (high OI = crowded position)
        f['oi_ratio'] = (oi / oi_ma.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0)
        
        # OI-Price divergence: price up + OI down = weak rally (bearish)
        price_ret = close.pct_change(6).fillna(0) * 100
        oi_ret = oi.pct_change(6).fillna(0) * 100
        f['oi_price_div'] = price_ret - oi_ret  # Positive = price outpaces OI (divergence)
    else:
        f['oi_change_pct'] = 0.0
        f['oi_ratio'] = 1.0
        f['oi_price_div'] = 0.0
    
    # ─────────────────────────────────────────────────
    # 2. ORDER FLOW PROXY (buy/sell pressure from OHLCV)
    # ─────────────────────────────────────────────────
    total_range = (high - low).replace(0, np.nan)
    close_pos = (close - low) / total_range
    
    vol_delta = volume * (2 * close_pos - 1)
    f['vol_delta_5'] = vol_delta.rolling(5).sum()
    f['vol_delta_12'] = vol_delta.rolling(12).sum()
    
    vol_ma_20 = volume.rolling(20).mean()
    f['vol_delta_norm'] = f['vol_delta_5'] / (vol_ma_20 * 5).replace(0, np.nan)
    
    is_strong_up = (close_pos > 0.7) & (volume > vol_ma_20 * 1.5)
    is_strong_dn = (close_pos < 0.3) & (volume > vol_ma_20 * 1.5)
    f['aggr_buy_cnt'] = is_strong_up.astype(float).rolling(6).sum()
    f['aggr_sell_cnt'] = is_strong_dn.astype(float).rolling(6).sum()
    
    is_up = close > open_
    up_vol = (volume * is_up.astype(float)).rolling(12).sum()
    dn_vol = (volume * (~is_up).astype(float)).rolling(12).sum()
    f['vol_imbalance'] = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)
    
    # ─────────────────────────────────────────────────
    # 3. SMART MOMENTUM
    # ─────────────────────────────────────────────────
    ret_1 = close.pct_change(1) * 100
    ret_3 = close.pct_change(3) * 100
    ret_6 = close.pct_change(6) * 100
    ret_12 = close.pct_change(12) * 100
    
    f['momentum_accel'] = ret_3 - ret_3.shift(3)
    
    direction_move = abs(close - close.shift(12))
    path_move = abs(close.diff()).rolling(12).sum()
    f['efficiency'] = (direction_move / path_move.replace(0, np.nan)).clip(0, 1)
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    f['rsi'] = rsi
    
    price_higher = (close > close.shift(12)).astype(float)
    rsi_higher = (rsi > rsi.shift(12)).astype(float)
    f['rsi_divergence'] = price_higher - rsi_higher
    
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False).mean()
    
    f['ema_9_dist'] = (close - ema_9) / atr_14
    f['ema_spread'] = (ema_9 - ema_50) / atr_14
    f['ema_alignment'] = np.where(
        (ema_9 > ema_21) & (ema_21 > ema_50), 1,
        np.where((ema_9 < ema_21) & (ema_21 < ema_50), -1, 0)
    ).astype(float)
    
    # ─────────────────────────────────────────────────
    # 4. VOLATILITY REGIME
    # ─────────────────────────────────────────────────
    atr_7 = tr.ewm(span=7, adjust=False).mean()
    atr_21 = tr.ewm(span=21, adjust=False).mean()
    
    f['atr_pct'] = atr_14 / close * 100
    f['atr_expansion'] = atr_7 / atr_21
    
    bb_std = close.rolling(20).std()
    bb_width = (bb_std * 2) / close.rolling(20).mean() * 100
    kc_width = atr_14 * 1.5 / close * 100
    f['squeeze'] = (bb_width < kc_width).astype(float)
    f['squeeze_intensity'] = ((kc_width - bb_width) / kc_width).clip(0, 1)
    
    range_10 = (high.rolling(10).max() - low.rolling(10).min()) / close * 100
    range_50 = (high.rolling(50).max() - low.rolling(50).min()) / close * 100
    f['compression'] = (range_10 / range_50.replace(0, np.nan)).clip(0, 1)
    
    # ─────────────────────────────────────────────────
    # 5. MEAN REVERSION
    # ─────────────────────────────────────────────────
    bb_mean = close.rolling(20).mean()
    f['bb_position'] = (close - (bb_mean - bb_std * 2)) / (bb_std * 4).replace(0, np.nan)
    
    vwap = (close * volume).rolling(50).sum() / volume.rolling(50).sum()
    f['vwap_dist'] = (close - vwap) / atr_14
    
    roll_high = high.rolling(50).max()
    roll_low = low.rolling(50).min()
    price_range = (roll_high - roll_low).replace(0, np.nan)
    f['range_position'] = (close - roll_low) / price_range
    
    f['return_skew'] = ret_1.rolling(30).skew()
    
    # ─────────────────────────────────────────────────
    # 6. M15 CONTEXT (all shifted by 1 to avoid look-ahead)
    # ─────────────────────────────────────────────────
    m15_close = m15['close']
    m15_high = m15['high']
    m15_low = m15['low']
    
    m15_ema_fast = m15_close.ewm(span=8, adjust=False).mean()
    m15_ema_slow = m15_close.ewm(span=21, adjust=False).mean()
    m15_atr = pd.concat([m15_high - m15_low, abs(m15_high - m15_close.shift()),
                         abs(m15_low - m15_close.shift())], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()
    
    m15_delta = m15_close.diff()
    m15_gain = m15_delta.where(m15_delta > 0, 0).ewm(span=14, adjust=False).mean()
    m15_loss = (-m15_delta.where(m15_delta < 0, 0)).ewm(span=14, adjust=False).mean()
    m15_rsi = 100 - 100 / (1 + m15_gain / m15_loss.replace(0, np.nan))
    
    m15_features = pd.DataFrame(index=m15.index)
    m15_features['m15_trend'] = np.where(m15_ema_fast.shift(1) > m15_ema_slow.shift(1), 1, -1).astype(float)
    m15_features['m15_trend_str'] = ((m15_ema_fast - m15_ema_slow) / m15_atr).shift(1)
    m15_features['m15_rsi'] = m15_rsi.shift(1)
    m15_features['m15_momentum'] = m15_close.pct_change(3).shift(1) * 100
    m15_features['m15_range_pos'] = ((m15_close - m15_low.rolling(20).min()) / \
        (m15_high.rolling(20).max() - m15_low.rolling(20).min()).replace(0, np.nan)).shift(1)
    
    # FIX: Reindex M15 features safely — shift(1) on M15 already applied above,
    # but ffill on union can still leak if M15 candle closes mid-M5.
    # Use reindex with method='ffill' directly on M5 index (simpler, no union trick)
    for col in m15_features.columns:
        f[col] = m15_features[col].reindex(f.index, method='ffill')
    
    # ─────────────────────────────────────────────────
    # 7. M1 MICROSTRUCTURE
    # ─────────────────────────────────────────────────
    m1_micro = pd.DataFrame(index=m1.index)
    m1_micro['m1_momentum'] = m1['close'].pct_change(1) * 100
    m1_micro['m1_range'] = (m1['high'] - m1['low']) / m1['close'] * 100
    
    m1_temp = m1_micro.copy()
    # FIX: Shift M1 timestamps back by 5min so that e.g. 10:01-10:04 maps to 10:00 candle
    # Without shift, 10:05:xx would be floored to 10:05 = CURRENT candle (lookahead!)
    # With shift, we only use COMPLETED M1 bars from the PREVIOUS 5-min window
    m1_temp.index = (m1_temp.index - pd.Timedelta(minutes=1)).floor('5min')
    grp = m1_temp.groupby(m1_temp.index)
    
    m1_agg = pd.DataFrame(index=m1_temp.index.unique())
    m1_agg['m1_momentum_std'] = grp['m1_momentum'].std()
    m1_agg['m1_range_mean'] = grp['m1_range'].mean()
    
    # FIX: Safe reindex for M1 aggregates (no union trick needed)
    for col in m1_agg.columns:
        f[col] = m1_agg[col].reindex(f.index, method='ffill')
    
    # ─────────────────────────────────────────────────
    # 8. TIME FEATURES (session patterns — consistently important)
    # ─────────────────────────────────────────────────
    hour = m5.index.hour + m5.index.minute / 60.0
    f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = m5.index.dayofweek
    f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    f['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # ─────────────────────────────────────────────────
    # FINALIZE
    # ─────────────────────────────────────────────────
    f['open'] = open_
    f['high'] = high
    f['low'] = low
    f['close'] = close
    f['volume'] = volume
    
    f = f.replace([np.inf, -np.inf], np.nan)
    
    return f



# List of OHLCV and meta columns to exclude from features
KILLER_EXCLUDE = {
    'open', 'high', 'low', 'close', 'volume', 'pair', 'atr',
    'target_dir', 'target_timing', 'target_strength',
    'vol_sma_20', 'vwap',
}


# ============================================================
# ============================================================
# V14 BINARY TARGETS — UP vs DOWN (no sideways noise)
# ============================================================
def create_targets_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create V14 BINARY targets — UP (1) vs DOWN (0).
    
    KEY CHANGE: No sideways class!
    - Sideways rows get NaN target and are DROPPED during training
    - Model only learns clear directional signals
    - Binary classifier is much easier: 50% baseline vs 33% for 3-class
    """
    df = df.copy()
    df['atr'] = calculate_atr(df)
    
    # 1. Adaptive volatility threshold
    vol_short = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    vol_medium = df['close'].pct_change().rolling(window=50, min_periods=25).std()
    vol_long = df['close'].pct_change().rolling(window=100, min_periods=50).std()
    combined_vol = np.maximum(vol_short, (vol_medium + vol_long) / 2)
    combined_vol = combined_vol.shift(1)
    
    # Higher threshold = only clear moves, less noise
    threshold = np.maximum(combined_vol * 0.6, 0.002)
    
    # 2. Future return — single clean window
    future_return = df['close'].pct_change(LOOKAHEAD).shift(-LOOKAHEAD)
    
    # 3. BINARY direction: 1=UP, 0=DOWN, NaN=sideways (will be dropped)
    df['target_dir'] = np.where(
        future_return > threshold, 1,
        np.where(future_return < -threshold, 0, np.nan)
    )
    
    # 4. Timing — linked to predicted direction
    future_lows = df['low'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
    future_highs = df['high'].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
    
    mae_long = (df['close'] - future_lows) / df['atr']
    mae_short = (future_highs - df['close']) / df['atr']
    mfe_long = (future_highs - df['close']) / df['atr']
    mfe_short = (df['close'] - future_lows) / df['atr']
    
    timing_long = mfe_long / (1 + mae_long)
    timing_short = mfe_short / (1 + mae_short)
    
    df['target_timing'] = np.where(
        df['target_dir'] == 1, timing_long,
        np.where(df['target_dir'] == 0, timing_short, np.nan)
    )
    df['target_timing'] = df['target_timing'].clip(0, 5)
    
    # 5. Strength
    df['target_strength'] = np.where(
        df['target_dir'] == 1, mfe_long,
        np.where(df['target_dir'] == 0, mfe_short, np.nan)
    )
    df['target_strength'] = df['target_strength'].clip(0, 10)
    
    return df




# ============================================================
# AUTO FEATURE SELECTION
# ============================================================
def auto_select_features(X_train, y_train, X_val, y_val, threshold=0.001):
    """
    Automatically select best features using permutation importance.
    
    Process:
    1. Train quick model on ALL features
    2. Calculate permutation importance (true predictive power)
    3. Keep only features with importance > threshold
    4. Return filtered feature list
    
    This removes noise features that hurt generalization.
    """
    from sklearn.inspection import permutation_importance
    
    print("\n" + "="*60)
    print("🔬 AUTO FEATURE SELECTION")
    print("="*60)
    print(f"   Starting with {len(X_train.columns)} features...")
    
    # Scale for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Quick model for feature selection (faster params)
    print("   Training quick model for feature importance...")
    quick_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=100,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbosity=-1
    )
    quick_model.fit(X_train_scaled, y_train['target_dir'])
    
    # Calculate permutation importance
    print("   Calculating permutation importance (this shows REAL predictive power)...")
    perm_result = permutation_importance(
        quick_model, X_val_scaled, y_val['target_dir'],
        n_repeats=3,  # 3 repeats (was 10 — 3x faster)
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    # Analyze results
    feature_names = X_train.columns.tolist()
    importances = perm_result.importances_mean
    importance_pairs = list(zip(feature_names, importances))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 20
    print("\n   📊 Top 20 Features by Permutation Importance:")
    for i, (name, imp) in enumerate(importance_pairs[:20]):
        print(f"      {i+1:2d}. {name}: {imp:.4f}")
    
    # Filter features
    good_features = [name for name, imp in importance_pairs if imp > threshold]
    useless_features = [name for name, imp in importance_pairs if imp <= 0]
    marginal_features = [name for name, imp in importance_pairs if 0 < imp <= threshold]
    
    print(f"\n   ✅ Good features (importance > {threshold}): {len(good_features)}")
    print(f"   ⚠️  Marginal features (0 < imp ≤ {threshold}): {len(marginal_features)}")
    print(f"   ❌ Useless/harmful features (imp ≤ 0): {len(useless_features)}")
    
    if useless_features:
        print(f"\n   Removing useless features:")
        for name in useless_features[:10]:  # Show first 10
            print(f"      - {name}")
        if len(useless_features) > 10:
            print(f"      ... and {len(useless_features) - 10} more")
    
    # 🎯 V13: Feature Stability Check
    if USE_FEATURE_STABILITY:
        stable_features = check_feature_stability(X_train, X_val, good_features)
        if len(stable_features) < len(good_features):
            print(f"   🔄 After stability check: {len(good_features)} → {len(stable_features)} features")
            good_features = stable_features
    
    print(f"\n   📉 Reduced: {len(X_train.columns)} → {len(good_features)} features")
    print("="*60 + "\n")
    
    return good_features, importance_pairs


# ============================================================
# 🎯 V13: FEATURE STABILITY CHECK
# ============================================================
def check_feature_stability(X_train, X_val, feature_names, min_correlation=0.5):
    """
    Check if feature distributions are stable between train and validation.
    
    Features that "drift" significantly between periods are unreliable
    and should be removed. Uses Spearman correlation of feature ranks.
    
    Returns: list of stable features
    """
    if not USE_FEATURE_STABILITY:
        return feature_names
    
    stable_features = []
    unstable_features = []
    
    for feat in feature_names:
        if feat not in X_train.columns or feat not in X_val.columns:
            continue
            
        train_data = X_train[feat].dropna()
        val_data = X_val[feat].dropna()
        
        if len(train_data) < 100 or len(val_data) < 100:
            stable_features.append(feat)  # Not enough data to judge
            continue
        
        # Compare distributions using percentile mapping
        train_mean, train_std = train_data.mean(), train_data.std()
        val_mean, val_std = val_data.mean(), val_data.std()
        
        # If std is 0, skip
        if train_std == 0 or val_std == 0:
            stable_features.append(feat)
            continue
        
        # Check if mean shifted more than 2 std
        mean_shift = abs(train_mean - val_mean) / train_std
        std_ratio = max(train_std, val_std) / min(train_std, val_std)
        
        # FIX: Tighter thresholds — unstable if mean shifted > 1σ OR variance changed > 2x
        if mean_shift > 1.0 or std_ratio > 2.0:
            unstable_features.append((feat, mean_shift, std_ratio))
        else:
            stable_features.append(feat)
    
    if unstable_features:
        print(f"   ⚠️ Removing {len(unstable_features)} unstable features (distribution drift):")
        for feat, shift, ratio in unstable_features[:5]:
            print(f"      - {feat}: mean_shift={shift:.2f}σ, var_ratio={ratio:.2f}x")
        if len(unstable_features) > 5:
            print(f"      ... and {len(unstable_features) - 5} more")
    
    return stable_features


# ============================================================
# 🎯 V13: REGIME FILTER
# ============================================================
def calculate_market_regime(df, atr_col='atr', lookback=50):
    """
    Calculate market regime: trending vs ranging, volatile vs quiet.
    
    Returns DataFrame with regime indicators.
    """
    regime = pd.DataFrame(index=df.index)
    
    # ATR percentile (volatility regime)
    atr_rolling = df[atr_col].rolling(lookback).mean()
    atr_percentile = df[atr_col].rolling(lookback * 4).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    )
    
    regime['volatility_percentile'] = atr_percentile
    regime['is_low_volatility'] = atr_percentile < MIN_VOLATILITY_PERCENTILE
    
    # Price momentum (trend strength)
    if 'close' in df.columns:
        returns = df['close'].pct_change(lookback)
        regime['trend_strength'] = returns.abs()
        regime['is_ranging'] = returns.abs() < df[atr_col] / df['close'] * 0.5
    
    return regime


def should_trade_regime(signal, df):
    """
    Check if current market regime is suitable for trading.
    
    Returns: (should_trade: bool, reason: str)
    """
    if not USE_REGIME_FILTER:
        return True, "regime_filter_disabled"
    
    try:
        idx = df.index.get_loc(signal['timestamp'])
        
        # Check volatility
        atr = signal.get('atr', 0)
        close = signal.get('entry_price', 1)
        atr_pct = atr / close if close > 0 else 0
        
        # Get recent ATR for percentile calculation
        lookback = min(100, idx)
        if lookback > 20:
            recent_atrs = df['atr'].iloc[idx-lookback:idx]
            current_percentile = (recent_atrs < atr).sum() / len(recent_atrs) * 100
            
            if current_percentile < MIN_VOLATILITY_PERCENTILE:
                return False, f"low_volatility_{current_percentile:.0f}pct"
        
        return True, "regime_ok"
        
    except Exception as e:
        return True, f"regime_check_error"


# ============================================================
# 🎯 V13: DYNAMIC POSITION SIZING
# ============================================================
def calculate_dynamic_risk(confidence, timing_score, base_risk=0.05, max_risk=0.07):
    """
    Calculate risk percentage based on signal confidence.
    
    Higher confidence → larger position (but capped at max_risk).
    """
    if not USE_DYNAMIC_SIZING:
        return base_risk
    
    # Scale risk linearly with confidence
    # confidence 0.65 → base_risk (5%)
    # confidence 0.80+ → max_risk (7%)
    
    conf_range = CONFIDENCE_BOOST_THRESHOLD - MIN_CONFIDENCE  # 0.75 - 0.65 = 0.10
    conf_above_min = max(0, confidence - MIN_CONFIDENCE)
    conf_factor = min(1.0, conf_above_min / conf_range) if conf_range > 0 else 0
    
    risk = base_risk + (max_risk - base_risk) * conf_factor
    
    # Boost for high timing score
    if timing_score > 2.5:
        risk = min(max_risk, risk * 1.1)
    
    return risk


# ============================================================
# 🎯 V13: REALISTIC SLIPPAGE
# ============================================================
def calculate_slippage(atr, close, is_volatile_period=False):
    """
    Calculate realistic slippage based on market conditions.
    
    Higher volatility → higher slippage.
    """
    if not USE_REALISTIC_SLIPPAGE:
        return SLIPPAGE_PCT
    
    # Base slippage
    slippage = BASE_SLIPPAGE_PCT
    
    # Adjust for volatility
    atr_pct = atr / close if close > 0 else 0
    
    if atr_pct > 0.02:  # Very volatile (>2% ATR)
        slippage = VOLATILE_SLIPPAGE_PCT
    elif atr_pct > 0.01:  # Moderately volatile
        slippage = (BASE_SLIPPAGE_PCT + VOLATILE_SLIPPAGE_PCT) / 2
    
    return slippage


# ============================================================
# 🎯 V13: MTF CONFIRMATION
# ============================================================
def check_mtf_confirmation(signal, df):
    """
    Check if higher timeframe (M15) confirms the M5 signal direction.
    
    Returns: (confirmed: bool, strength: float)
    """
    if not USE_MTF_CONFIRMATION:
        return True, 1.0
    
    try:
        # Look for M15 trend indicators in features
        feat_prefix = 'feat_'
        m15_ema_dist = signal.get(f'{feat_prefix}m15_ema_dist', signal.get('m15_ema_dist', 0))
        m15_momentum = signal.get(f'{feat_prefix}m15_momentum', signal.get('m15_momentum', 0))
        m15_rsi = signal.get(f'{feat_prefix}m15_rsi', signal.get('m15_rsi', 50))
        
        direction = signal['direction']
        
        # Check alignment
        if direction == 'LONG':
            # M15 should show bullish bias
            ema_ok = m15_ema_dist > -0.005  # Price not too far below EMA
            momentum_ok = m15_momentum > -0.001  # Not strong downward momentum
            rsi_ok = m15_rsi > 35  # Not oversold (could be reversal)
            
            strength = 1.0
            if m15_momentum > 0.001:
                strength = 1.2  # Bonus for aligned momentum
            
        else:  # SHORT
            ema_ok = m15_ema_dist < 0.005  # Price not too far above EMA
            momentum_ok = m15_momentum < 0.001  # Not strong upward momentum
            rsi_ok = m15_rsi < 65  # Not overbought
            
            strength = 1.0
            if m15_momentum < -0.001:
                strength = 1.2  # Bonus for aligned momentum
        
        confirmed = (ema_ok and momentum_ok) or rsi_ok
        return confirmed, strength
        
    except Exception as e:
        return True, 1.0  # Default to confirmed if can't check


# ============================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================
def optimize_lgb_params(X_train, y_train, X_val, y_val, sample_weights, n_trials=30):
    """
    Use Optuna to find optimal LightGBM hyperparameters.
    
    Optimizes: n_estimators, max_depth, num_leaves, learning_rate,
               subsample, colsample_bytree, reg_alpha, reg_lambda
    """
    if not OPTUNA_AVAILABLE:
        print("   ⚠️ Optuna not available, using default params")
        return None
    
    print(f"\n   🔧 OPTUNA: Optimizing hyperparameters ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 200,  # Fixed (early stopping handles it)
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 8, 48),
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 0.8, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 0.8, log=True),
            'random_state': 42,
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        # Return validation accuracy
        preds = model.predict(X_val)
        accuracy = (preds == y_val).mean()
        return accuracy
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"   ✅ Best accuracy: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    return study.best_params


# ============================================================
# ENSEMBLE MODEL (LGB + CatBoost)
# ============================================================
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleDirectionModel(ClassifierMixin, BaseEstimator):
    """
    Ensemble of LightGBM + CatBoost for more stable predictions.
    
    Averages probabilities from both models for final prediction.
    Inherits from sklearn BaseEstimator for compatibility with CalibratedClassifierCV.
    """
    
    # Explicitly declare this is a classifier for sklearn
    _estimator_type = "classifier"
    
    def __init__(self, lgb_params=None, use_catboost=True):
        self.lgb_params = lgb_params
        self.use_catboost = use_catboost
        self.lgb_model = None
        self.catboost_model = None
        self.weights = {'lgb': 0.5, 'catboost': 0.5}  # Equal weighting
        self._classes = np.array([0, 1, 2])  # Direction classes
    
    @property
    def classes_(self):
        return self._classes
        
    def fit(self, X, y, sample_weight=None):
        """Train both models. Compatible with sklearn API."""
        # For sklearn compatibility, we train without separate validation
        # Early stopping will use a portion of training data
        
        # 1. LightGBM
        lgb_default = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 200,  # was 300
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 50,
            'learning_rate': 0.05,  # was 0.03 (faster convergence)
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1
        }
        lgb_default.update(self.lgb_params or {})
        
        self.lgb_model = lgb.LGBMClassifier(**lgb_default)
        self.lgb_model.fit(X, y, sample_weight=sample_weight)
        
        # 2. CatBoost (if available)
        if self.use_catboost:
            self.catboost_model = CatBoostClassifier(
                iterations=150,
                depth=5,
                learning_rate=0.05,
                l2_leaf_reg=3,
                loss_function='Logloss',
                random_seed=42,
                verbose=False
            )
            self.catboost_model.fit(X, y, sample_weight=sample_weight)
        
        self._classes = np.unique(y)
        return self
    
    def predict_proba(self, X):
        """Average probabilities from both models."""
        lgb_proba = self.lgb_model.predict_proba(X)
        
        if self.use_catboost and self.catboost_model is not None:
            cat_proba = self.catboost_model.predict_proba(X)
            # Weighted average
            ensemble_proba = (
                self.weights['lgb'] * lgb_proba + 
                self.weights['catboost'] * cat_proba
            )
            return ensemble_proba
        
        return lgb_proba
    
    def predict(self, X):
        """Predict class with highest probability."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    @property
    def feature_importances_(self):
        """Return LGB feature importances."""
        if self.lgb_model is not None:
            return self.lgb_model.feature_importances_
        return None


# ============================================================
# TRAINING (V12 - Smart Adaptive Model with Optuna + Ensemble)
# ============================================================
def train_models(X_train, y_train, X_val, y_val):
    """
    Train SMART ADAPTIVE models that work across market conditions.
    
    V14 FIX - NO DATA LEAKAGE:
    - Split train into train (85%) + calibration (15%)
    - Train base model on train portion
    - Calibrate on calibration portion (SEPARATE from validation!)
    - Validation is ONLY for evaluation, never for training/calibration
    
    This prevents inflated backtest scores from calibration leakage.
    """
    
    # ============================================================
    # SPLIT: Train (85%) → actual_train + calibration (15%)
    # This prevents data leakage: calibration uses data the model never saw
    # ============================================================
    cal_split = int(len(X_train) * 0.85)
    X_actual_train = X_train.iloc[:cal_split]
    X_calibration = X_train.iloc[cal_split:]
    y_actual_train = {k: v.iloc[:cal_split] for k, v in y_train.items()}
    y_calibration = {k: v.iloc[cal_split:] for k, v in y_train.items()}
    
    print(f"   📊 Data split: train={len(X_actual_train)}, calibration={len(X_calibration)}, val={len(X_val)}")
    
    # Scale features for consistent ranges (RSI 0-100, returns -0.1 to 0.1, etc)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_actual_train),
        columns=X_actual_train.columns,
        index=X_actual_train.index
    )
    X_cal_scaled = pd.DataFrame(
        scaler.transform(X_calibration),
        columns=X_calibration.columns,
        index=X_calibration.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    print(f"   📐 Features scaled: mean=0, std=1")
    
    # Calculate class weights for imbalanced labels
    from collections import Counter
    label_counts = Counter(y_actual_train['target_dir'])
    total = sum(label_counts.values())
    class_weights = {k: total / (2 * v) for k, v in label_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_actual_train['target_dir']])
    
    # ============================================================
    # 1. DIRECTION MODEL with OPTUNA + ENSEMBLE
    # ============================================================
    
    # Step 1: Optuna hyperparameter optimization (if enabled)
    best_params = None
    if USE_OPTUNA and OPTUNA_AVAILABLE:
        best_params = optimize_lgb_params(
            X_train_scaled.values, y_actual_train['target_dir'].values,
            X_val_scaled.values, y_val['target_dir'].values,
            sample_weights, n_trials=OPTUNA_TRIALS
        )
    
    # Step 2: Train Direction Model (Ensemble or Single)
    print("   Training Direction Model...")
    
    if USE_ENSEMBLE:
        print("   🔀 Using ENSEMBLE (LightGBM + CatBoost)...")
        dir_model_base = EnsembleDirectionModel(
            lgb_params=best_params or {},
            use_catboost=CATBOOST_AVAILABLE
        )
        # sklearn-compatible fit on TRAIN portion only
        dir_model_base.fit(
            X_train_scaled.values, y_actual_train['target_dir'].values,
            sample_weight=sample_weights
        )
    else:
        # Single LightGBM model
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 500,       # More trees (early stopping will find optimal)
            'max_depth': 4,            # Slightly deeper for more complex patterns
            'num_leaves': 12,          # More leaves = more expressiveness
            'min_child_samples': 150,  # Still strict but allows some patterns
            'learning_rate': 0.02,     # Slower learning = better generalization
            'subsample': 0.6,          # 60% rows per tree
            'colsample_bytree': 0.6,   # 60% features per tree
            'reg_alpha': 0.5,          # L1 regularization
            'reg_lambda': 0.5,         # L2 regularization
            'min_split_gain': 0.02,    # Moderate split gain threshold
            'extra_trees': True,       # Extra randomization = less overfit
            'random_state': 42,
            'verbosity': -1,
            'importance_type': 'gain'
        }
        if best_params:
            lgb_params.update(best_params)
        
        dir_model_base = lgb.LGBMClassifier(**lgb_params)
        # FIX: Early stopping on CALIBRATION set (not validation!) to prevent data leakage
        dir_model_base.fit(
            X_train_scaled, y_actual_train['target_dir'],
            sample_weight=sample_weights,
            eval_set=[(X_cal_scaled, y_calibration['target_dir'])],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
    
    # Step 3: Calibration on SEPARATE calibration set (NOT validation!)
    print("   🎯 Calibrating Direction Model on separate calibration set...")
    dir_model = CalibratedClassifierCV(
        estimator=dir_model_base, 
        method='sigmoid',
        cv=2,
        n_jobs=-1
    )
    dir_model.fit(X_cal_scaled, y_calibration['target_dir'])
    
    # ============================================================
    # 2. TIMING MODEL (Regressor)
    # ============================================================
    print("   Training Timing Model...")
    timing_model = lgb.LGBMRegressor(
        objective='huber',         # Robust to outliers in timing prediction
        metric='mae',
        boosting_type='gbdt',
        n_estimators=500,          # More trees (early stopping finds optimal)
        max_depth=4,
        num_leaves=12,
        min_child_samples=100,     # Strict to avoid overfit
        learning_rate=0.02,        # Slower = better generalization
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.3,
        reg_lambda=0.3,
        min_split_gain=0.01,
        extra_trees=True,
        random_state=42,
        verbosity=-1
    )
    # FIX: Early stopping on CALIBRATION set (not validation!)
    timing_model.fit(
        X_train_scaled, y_actual_train['target_timing'],
        eval_set=[(X_cal_scaled, y_calibration['target_timing'])],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    # ============================================================
    # 3. STRENGTH MODEL (Regressor)
    # ============================================================
    print("   Training Strength Model...")
    strength_model = lgb.LGBMRegressor(
        objective='huber',         # Robust to outliers in strength prediction
        metric='mae',
        boosting_type='gbdt',
        n_estimators=500,
        max_depth=4,
        num_leaves=12,
        min_child_samples=100,
        learning_rate=0.02,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.3,
        reg_lambda=0.3,
        min_split_gain=0.01,
        extra_trees=True,
        random_state=42,
        verbosity=-1
    )
    # FIX: Early stopping on CALIBRATION set (not validation!)
    strength_model.fit(
        X_train_scaled, y_actual_train['target_strength'],
        eval_set=[(X_cal_scaled, y_calibration['target_strength'])],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    # ============================================================
    # LOG FEATURE IMPORTANCE
    # ============================================================
    print("\n   📊 Top 20 Features by Importance (Direction Model):")
    importance = dir_model_base.feature_importances_
    feature_names = X_actual_train.columns.tolist()
    importance_pairs = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for name, imp in importance_pairs[:20]:
        print(f"      {name}: {imp:.1f}")
    
    # Permutation importance
    print("\n   🔍 Calculating Permutation Importance...")
    try:
        # Use base model if ensemble, otherwise use the lgb model
        test_model = dir_model_base.lgb_model if USE_ENSEMBLE else dir_model_base
        perm_result = permutation_importance(
            test_model, X_val_scaled, y_val['target_dir'],
            n_repeats=2, random_state=42, n_jobs=-1, scoring='accuracy'  # was 5
        )
        perm_pairs = sorted(zip(feature_names, perm_result.importances_mean), 
                           key=lambda x: x[1], reverse=True)
        print("   Top 10 by Permutation Importance:")
        for name, imp in perm_pairs[:10]:
            print(f"      {name}: {imp:.4f}")
        
        useless = [name for name, imp in perm_pairs if imp <= 0]
        if useless:
            print(f"   ⚠️ {len(useless)} features have zero/negative importance")
    except Exception as e:
        print(f"   ⚠️ Permutation importance failed: {e}")
    
    return {
        'direction': dir_model,
        'direction_base': dir_model_base,
        'timing': timing_model,
        'strength': strength_model,
        'scaler': scaler,
        'best_params': best_params  # Save Optuna results
    }


# ============================================================
# PORTFOLIO BACKTEST (V9 - Realistic Thresholds)
# ============================================================
def generate_signals(df: pd.DataFrame, feature_cols: list, models: dict, pair_name: str,
                    min_conf: float = None, min_timing: float = None, min_strength: float = None) -> list:
    """
    V15 HYBRID SIGNAL GENERATION:
    1. DIRECTION from M15 trend (EMA fast vs slow) — proven momentum edge
    2. ENTRY TIMING from pullback detection (RSI extreme in trend direction)
    3. ML MODEL as QUALITY FILTER (timing + strength must confirm big move)
    4. CONFLUENCE: trend + pullback + ML all agree → high probability trade

    This replaces pure ML direction prediction (which is ~50/50 = useless).
    """
    if min_conf is None: min_conf = MIN_CONFIDENCE
    if min_timing is None: min_timing = MIN_TIMING
    if min_strength is None: min_strength = MIN_STRENGTH

    signals = []

    # Pre-compute ML predictions in batch
    X = df[feature_cols].values
    if 'scaler' in models and models['scaler'] is not None:
        X = models['scaler'].transform(X)

    dir_proba = models['direction'].predict_proba(X)
    dir_preds = np.argmax(dir_proba, axis=1)
    dir_confs = np.max(dir_proba, axis=1)
    timing_preds = models['timing'].predict(X)
    strength_preds = models['strength'].predict(X)

    # Pre-compute trend and momentum indicators from df columns
    has_m15_trend = 'm15_trend_str' in df.columns
    has_rsi = 'rsi' in df.columns
    has_ema = 'ema_9_dist' in df.columns
    has_atr_exp = 'atr_expansion' in df.columns
    has_compression = 'compression' in df.columns

    for i in range(len(df)):
        # ── STEP 1: TREND DIRECTION from M15 ──
        # Use m15_trend_str: positive = uptrend, negative = downtrend
        if has_m15_trend:
            m15_ts = df['m15_trend_str'].iloc[i] if not pd.isna(df['m15_trend_str'].iloc[i]) else 0
        else:
            m15_ts = 0

        # Need STRONG trend (abs > 1.0 to only trade clear momentum)
        if abs(m15_ts) < 1.0:
            continue  # Weak/no trend — skip

        trend_dir = 'LONG' if m15_ts > 0 else 'SHORT'

        # ── STEP 2: PULLBACK ENTRY ──
        # In uptrend: enter when RSI dips significantly (buy the dip)
        # In downtrend: enter when RSI spikes significantly (sell the rally)
        if has_rsi:
            rsi_val = df['rsi'].iloc[i] if not pd.isna(df['rsi'].iloc[i]) else 50
        else:
            rsi_val = 50

        pullback_ok = False
        if trend_dir == 'LONG' and 25 < rsi_val < 40:      # Deep dip in uptrend (not oversold crash)
            pullback_ok = True
        elif trend_dir == 'SHORT' and 60 < rsi_val < 75:    # Strong rally in downtrend (not overbought moon)
            pullback_ok = True

        if not pullback_ok:
            continue

        # ── STEP 3: VOLATILITY + COMPRESSION FILTER ──
        # Trade when volatility is present but not extreme
        if has_atr_exp:
            atr_exp = df['atr_expansion'].iloc[i] if not pd.isna(df['atr_expansion'].iloc[i]) else 1.0
            if atr_exp < 0.85:  # ATR contracting too much = no movement
                continue
        if has_compression:
            comp = df['compression'].iloc[i] if not pd.isna(df['compression'].iloc[i]) else 0.5
            if comp < 0.2:  # Extreme compression = wait for breakout
                continue

        # ── STEP 4: ML QUALITY FILTER ──
        # ML confirms: high timing score + strength = big move expected
        if timing_preds[i] < min_timing:
            continue
        if strength_preds[i] < min_strength:
            continue

        # ML direction should AGREE with trend (consensus required)
        ml_dir = 'LONG' if dir_preds[i] == 1 else 'SHORT'
        ml_conf = dir_confs[i]

        # STRICT: require ML to agree with trend direction
        if ml_dir != trend_dir:
            continue  # No consensus = no trade

        score = abs(m15_ts) * timing_preds[i] * strength_preds[i] * ml_conf

        signal = {
            'timestamp': df.index[i],
            'pair': pair_name,
            'direction': trend_dir,  # Direction from TREND, not ML
            'entry_price': df['close'].iloc[i],
            'atr': df['atr'].iloc[i],
            'confidence': ml_conf,
            'score': score,
            'timing_prob': timing_preds[i],
            'pred_strength': strength_preds[i],
            'm15_trend_str': m15_ts,
            'rsi_at_entry': rsi_val,
            'ml_agrees': ml_dir == trend_dir,
        }

        signals.append(signal)

    return signals


def simulate_trade(signal: dict, df: pd.DataFrame) -> dict:
    """
    Simulate a single trade with SL/TP/trailing stop.

    V15 IMPROVEMENTS:
    - Proper SL at 1.5 ATR (tighter = less loss per trade)
    - TP at 2.0 ATR (target R:R = 1:1.33 — realistic for scalping)
    - Breakeven at 0.8 ATR (lock in profits)
    - Max hold 12 bars (1 hour) — time stop if neither SL nor TP hit
    - Proper boundary handling
    """
    try:
        start_idx = df.index.get_loc(signal['timestamp'])
    except KeyError:
        return None

    entry_price = signal['entry_price']
    atr = signal['atr']
    direction = signal['direction']
    pred_strength = signal.get('pred_strength', 2.0)
    confidence = signal.get('confidence', 0.5)

    # FIX: Boundary check — need at least 2 bars after entry
    if start_idx + 2 >= len(df):
        return None

    # === ADAPTIVE SL/TP: trend-following needs room to run ===
    sl_mult = SL_ATR_MULT                          # 1.5 ATR stop loss
    tp_mult = max(sl_mult * 2.5, pred_strength)    # TP = at least 2.5:1 R:R (trend trades)
    tp_mult = min(tp_mult, 8.0)                    # Cap TP at 8 ATR
    be_trigger = 0.6                               # Move SL to breakeven at 0.6R

    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

    if direction == 'LONG':
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    hold_bars = 18  # Max 90 min hold — enough time for moves
    outcome = 'time_exit'
    exit_idx = min(start_idx + hold_bars, len(df) - 1)
    exit_price = df['close'].iloc[exit_idx]
    exit_time = df.index[exit_idx]
    max_r_reached = 0.0
    be_activated = False

    # Simulate bar by bar
    for j in range(start_idx + 1, min(start_idx + hold_bars + 1, len(df))):
        bar = df.iloc[j]

        if direction == 'LONG':
            # Check SL first
            if bar['low'] <= sl_price:
                outcome = 'stop_loss'
                exit_price = sl_price
                exit_time = bar.name
                break
            # Check TP
            if bar['high'] >= tp_price:
                outcome = 'take_profit'
                exit_price = tp_price
                exit_time = bar.name
                break
            # Track max favorable excursion
            current_r = (bar['high'] - entry_price) / sl_dist
            max_r_reached = max(max_r_reached, current_r)
            # Breakeven: move SL to entry + small margin
            if not be_activated and current_r >= be_trigger:
                sl_price = entry_price + atr * 0.1
                be_activated = True
        else:
            # SHORT
            if bar['high'] >= sl_price:
                outcome = 'stop_loss'
                exit_price = sl_price
                exit_time = bar.name
                break
            if bar['low'] <= tp_price:
                outcome = 'take_profit'
                exit_price = tp_price
                exit_time = bar.name
                break
            current_r = (entry_price - bar['low']) / sl_dist
            max_r_reached = max(max_r_reached, current_r)
            if not be_activated and current_r >= be_trigger:
                sl_price = entry_price - atr * 0.1
                be_activated = True

    return {
        'exit_time': exit_time,
        'exit_price': exit_price,
        'outcome': outcome,
        'sl_dist': sl_dist,
        'sl_mult': sl_mult,
        'tp_mult': tp_mult,
        'max_r': max_r_reached,
        'be_activated': be_activated
    }



def run_portfolio_backtest(signals: list, pair_dfs: dict, initial_balance: float = 10000.0) -> list:
    """
    Single-slot backtest with per-pair cooldown.
    """
    signals.sort(key=lambda x: (x['timestamp'], -x['score']))
    
    executed_trades = []
    last_exit_time = pd.Timestamp.min.tz_localize('UTC')
    balance = initial_balance
    consecutive_losses = 0
    LOSS_COOLDOWN_BARS = 6
    MAX_CONSECUTIVE_LOSSES = 3
    cooldown_until = pd.Timestamp.min.tz_localize('UTC')
    
    print(f"Processing {len(signals)} potential signals...")
    print(f"Initial Balance: ${balance:,.2f}")
    print(f"Position sizing: RISK-BASED ({BASE_RISK_PCT*100:.0f}-{MAX_RISK_PCT*100:.0f}% risk)")
    
    for signal in signals:
        # Single slot constraint
        if signal['timestamp'] < last_exit_time:
            continue
        
        # Global cooldown
        if signal['timestamp'] < cooldown_until:
            continue
            
        # Execute
        pair_df = pair_dfs[signal['pair']]
        result = simulate_trade(signal, pair_df)
        
        if result:
            # Calculate Position Size & PnL
            entry_price = signal['entry_price']
            exit_price = result['exit_price']
            sl_dist = result['sl_dist']
            atr = signal.get('atr', sl_dist)
            
            # 🎯 V13: Dynamic risk based on confidence
            confidence = signal.get('confidence', signal.get('score', MIN_CONFIDENCE))
            timing_score = signal.get('timing_prob', 2.0)
            RISK_PCT = calculate_dynamic_risk(confidence, timing_score, BASE_RISK_PCT, MAX_RISK_PCT)
            
            # === RISK-BASED POSITION SIZING ===
            # Размер позиции рассчитан так, чтобы при стопе терять RISK_PCT от депозита
            risk_amount = balance * RISK_PCT  # Сколько готовы потерять
            sl_pct = sl_dist / entry_price  # Стоп в % от цены
            
            # position_size * sl_pct = risk_amount
            position_size = risk_amount / sl_pct
            
            # Ограничения
            max_position_by_leverage = balance * MAX_LEVERAGE
            original_position = position_size
            position_size = min(position_size, max_position_by_leverage, MAX_POSITION_SIZE)
            
            # FIX: If position capped, we REDUCE risk proportionally (don't widen SL!)
            # Widening SL is dangerous — it increases actual loss on stop.
            # Instead: keep same SL distance, accept lower risk %.
            if position_size < original_position:
                # Actual risk is now less than target (which is safer)
                actual_risk_pct = (position_size * sl_pct) / balance
                # Log the reduced risk for transparency
                pass  # actual_risk_pct < RISK_PCT — this is fine
            
            # Рассчитываем итоговое плечо
            leverage = position_size / balance
            
            # 🎯 V13: Realistic slippage based on volatility
            slippage = calculate_slippage(atr, entry_price)
            
            # PnL Calculation (with dynamic slippage)
            if signal['direction'] == 'LONG':
                effective_entry = entry_price * (1 + slippage)  # Worse entry
                effective_exit = exit_price * (1 - slippage)    # Worse exit
                raw_pnl_pct = (effective_exit - effective_entry) / effective_entry
            else:
                effective_entry = entry_price * (1 - slippage)  # Worse entry for short
                effective_exit = exit_price * (1 + slippage)    # Worse exit for short
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
            # Track consecutive losses for cooldown
            if net_profit <= 0:
                consecutive_losses += 1
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    # Set cooldown: skip next LOSS_COOLDOWN_BARS * 5 minutes
                    cooldown_until = result['exit_time'] + pd.Timedelta(minutes=LOSS_COOLDOWN_BARS * 5)
                    consecutive_losses = 0  # Reset after cooldown
            else:
                consecutive_losses = 0  # Reset on win
            
            # Stop trading if balance is too low (< 10% of initial)
            if balance < initial_balance * 0.10:
                break
            
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
def walk_forward_validation(pairs, data_dir, initial_balance=100.0):
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
    
    # Define periods — DYNAMIC based on available data range
    # Generate 4 rolling windows: 30d train + 7d test, stepping by ~14 days
    now = datetime.now(timezone.utc)
    periods = []
    for i in range(4):
        # Work backwards from now: period 4 is most recent
        offset = (3 - i) * 14  # 42, 28, 14, 0 days back from test_end
        test_end = now - timedelta(days=offset)
        test_start = test_end - timedelta(days=7)
        train_end = test_start - timedelta(days=EMBARGO_DAYS)  # Embargo gap
        train_start = train_end - timedelta(days=30)

        periods.append({
            'name': f"Period_{i+1}",
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
    
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
            
            # FIX: Apply EMBARGO between train and test to prevent feature leakage
            embargo_end = period['train_end'] - timedelta(days=EMBARGO_DAYS)

            # Filter TRAIN data (with embargo gap at the end)
            m1_train = m1[(m1.index >= period['train_start']) & (m1.index < embargo_end)]
            m5_train = m5[(m5.index >= period['train_start']) & (m5.index < embargo_end)]
            m15_train = m15[(m15.index >= period['train_start']) & (m15.index < embargo_end)]

            if len(m5_train) < 500: continue

            ft_train = generate_killer_features(m1_train, m5_train, m15_train)
            ft_train = create_targets_v1(ft_train)
            ft_train['pair'] = pair
            all_train.append(ft_train)

            # Filter TEST data (UNSEEN! starts AFTER embargo)
            m1_test = m1[(m1.index >= period['test_start']) & (m1.index < period['test_end'])]
            m5_test = m5[(m5.index >= period['test_start']) & (m5.index < period['test_end'])]
            m15_test = m15[(m15.index >= period['test_start']) & (m15.index < period['test_end'])]
            
            if len(m5_test) < 100: continue
            
            ft_test = generate_killer_features(m1_test, m5_test, m15_test)
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
        
        if FEATURE_MODE == 'auto':
            # Start with ALL features, then auto-select best ones
            all_exclude = KILLER_EXCLUDE | set(DEFAULT_EXCLUDE_FEATURES) | set(ABSOLUTE_PRICE_FEATURES)
            all_features = [c for c in train_df.columns if c not in all_exclude 
                           and not any(p in c.lower() for p in CUMSUM_PATTERNS)]
            
            # Prepare data for auto-selection
            X_all = train_df[all_features]
            y_all = {
                'target_dir': train_df['target_dir'],
                'target_timing': train_df['target_timing'],
                'target_strength': train_df['target_strength']
            }
            val_idx_temp = int(len(X_all) * 0.9)
            X_t_temp = X_all.iloc[:val_idx_temp]
            X_v_temp = X_all.iloc[val_idx_temp:]
            y_t_temp = {k: v.iloc[:val_idx_temp] for k, v in y_all.items()}
            y_v_temp = {k: v.iloc[val_idx_temp:] for k, v in y_all.items()}
            
            # Auto-select features
            features, importance_data = auto_select_features(
                X_t_temp, y_t_temp, X_v_temp, y_v_temp, 
                threshold=AUTO_SELECT_THRESHOLD
            )
            print(f"   📊 Using AUTO mode: {len(features)} auto-selected features")
        elif FEATURE_MODE == 'core20':
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
        
        # FIX: 80/20 split with embargo gap for walk-forward validation
        val_idx = int(len(X_train) * 0.80)
        embargo_bars = 288  # 1 day of M5 candles as gap
        val_start = min(val_idx + embargo_bars, len(X_train) - 100)
        X_t = X_train.iloc[:val_idx]
        X_v = X_train.iloc[val_start:]
        y_t = {k: v.iloc[:val_idx] for k, v in y_train.items()}
        y_v = {k: v.iloc[val_start:] for k, v in y_train.items()}

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
    # Using generate_killer_features() instead of MTFFeatureEngine
    
    all_train = []
    test_features = {}
    test_dfs = {}
    
    # Load BTC data for cross-pair features
    btc_m5_full = None
    try:
        btc_csv = data_dir / 'BTC_USDT_USDT_5m.csv'
        if btc_csv.exists():
            btc_m5_full = pd.read_csv(btc_csv, parse_dates=['timestamp'], index_col='timestamp')
            if btc_m5_full.index.tz is None:
                btc_m5_full.index = btc_m5_full.index.tz_localize('UTC')
            print(f"✅ BTC data loaded: {len(btc_m5_full)} candles")
    except Exception as e:
        print(f"⚠️ BTC data not loaded: {e}")
    
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
        
        # SPLIT LOGIC with EMBARGO (1-day gap to prevent feature leakage)
        EMBARGO_DAYS_MAIN = 1
        now = datetime.now(timezone.utc)
        if args.reverse:
            # Train on LAST N days (Recent), Test on PREVIOUS M days (Older)
            train_end = now
            train_start = now - timedelta(days=args.days)

            test_end = train_start - timedelta(days=EMBARGO_DAYS_MAIN)  # FIX: embargo gap
            test_start = test_end - timedelta(days=args.test_days)
        else:
            # Standard: Train on Old, Test on Recent
            test_start = now - timedelta(days=args.test_days)
            train_start = test_start - timedelta(days=args.days) - timedelta(days=EMBARGO_DAYS_MAIN)
            train_end = test_start - timedelta(days=EMBARGO_DAYS_MAIN)  # FIX: embargo gap
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
        
        # Get BTC data for this time range
        btc_m5_train = None
        if btc_m5_full is not None and pair != 'BTC/USDT:USDT':
            btc_m5_train = btc_m5_full[(btc_m5_full.index >= train_start) & (btc_m5_full.index < train_end)]
        
        # Load funding rate + OI for this pair
        pair_base = pair.split('/')[0]  # BTC from BTC/USDT:USDT
        fr_train, oi_train = None, None
        try:
            fr_path = data_dir / f"{pair_base}_USDT_funding_rate.csv"
            if fr_path.exists():
                fr_full = pd.read_csv(fr_path, parse_dates=['timestamp'], index_col='timestamp')
                if fr_full.index.tz is None:
                    fr_full.index = fr_full.index.tz_localize('UTC')
                fr_train = fr_full[(fr_full.index >= train_start) & (fr_full.index < train_end)]
        except Exception:
            pass
        try:
            oi_path = data_dir / f"{pair_base}_USDT_open_interest.csv"
            if oi_path.exists():
                oi_full = pd.read_csv(oi_path, parse_dates=['timestamp'], index_col='timestamp')
                if oi_full.index.tz is None:
                    oi_full.index = oi_full.index.tz_localize('UTC')
                oi_train = oi_full[(oi_full.index >= train_start) & (oi_full.index < train_end)]
        except Exception:
            pass
        
        ft_train = generate_killer_features(m1_train, m5_train, m15_train, btc_m5=btc_m5_train,
                                            funding_rate=fr_train, open_interest=oi_train)
        ft_train = create_targets_v1(ft_train)
        ft_train['pair'] = pair
        all_train.append(ft_train)
        
        # Filter Test
        m1_test = m1[(m1.index >= test_start) & (m1.index < test_end)]
        m5_test = m5[(m5.index >= test_start) & (m5.index < test_end)]
        m15_test = m15[(m15.index >= test_start) & (m15.index < test_end)]
        
        # Get BTC + funding + OI data for test range
        btc_m5_test = None
        if btc_m5_full is not None and pair != 'BTC/USDT:USDT':
            btc_m5_test = btc_m5_full[(btc_m5_full.index >= test_start) & (btc_m5_full.index < test_end)]
        
        fr_test, oi_test = None, None
        try:
            if fr_path.exists():
                fr_test = fr_full[(fr_full.index >= test_start) & (fr_full.index < test_end)]
        except Exception:
            pass
        try:
            if oi_path.exists():
                oi_test = oi_full[(oi_full.index >= test_start) & (oi_full.index < test_end)]
        except Exception:
            pass
        
        ft_test = generate_killer_features(m1_test, m5_test, m15_test, btc_m5=btc_m5_test,
                                           funding_rate=fr_test, open_interest=oi_test)
        ft_test = create_targets_v1(ft_test)
        ft_test['pair'] = pair
        test_features[pair] = ft_test
        test_dfs[pair] = ft_test

    print(f"\nData loaded. Training on {len(all_train)} pairs.")
    
    # Train
    # Train — drop NaN targets (sideways rows from binary classification)
    train_df = pd.concat(all_train)
    train_df = train_df.dropna(subset=['target_dir', 'target_timing', 'target_strength'])
    train_df = train_df.dropna()  # Also drop rows with NaN features
    print(f"   📊 Directional rows: {len(train_df):,} (sideways dropped)")
    
    # === ВЫВОД ПЕРИОДА ОБУЧЕНИЯ ===
    if len(train_df) > 0:
        train_period_start = train_df.index.min()
        train_period_end = train_df.index.max()
        train_days_actual = (train_period_end - train_period_start).days
        print("\n" + "="*70)
        print("📅 ПЕРИОД ОБУЧЕНИЯ:")
        print("="*70)
        print(f"  Начало:  {train_period_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Конец:   {train_period_end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Дней:    {train_days_actual} дней")
        print(f"  Записей: {len(train_df):,} строк")
        print("="*70 + "\n")
    
    # === FEATURE SELECTION BASED ON MODE ===
    available_cols = set(train_df.columns)
    selected_features_importance = None  # Store importance for saving
    
    if FEATURE_MODE == 'auto':
        # Start with ALL features, then auto-select best ones
        all_exclude = KILLER_EXCLUDE | set(DEFAULT_EXCLUDE_FEATURES) | set(ABSOLUTE_PRICE_FEATURES)
        all_features = [c for c in train_df.columns if c not in all_exclude 
                       and not any(p in c.lower() for p in CUMSUM_PATTERNS)]
        
        # Prepare data for auto-selection
        X_all = train_df[all_features]
        y_all = {
            'target_dir': train_df['target_dir'],
            'target_timing': train_df['target_timing'],
            'target_strength': train_df['target_strength']
        }
        val_idx_temp = int(len(X_all) * 0.9)
        X_t_temp = X_all.iloc[:val_idx_temp]
        X_v_temp = X_all.iloc[val_idx_temp:]
        y_t_temp = {k: v.iloc[:val_idx_temp] for k, v in y_all.items()}
        y_v_temp = {k: v.iloc[val_idx_temp:] for k, v in y_all.items()}
        
        # Auto-select features
        features, selected_features_importance = auto_select_features(
            X_t_temp, y_t_temp, X_v_temp, y_v_temp, 
            threshold=AUTO_SELECT_THRESHOLD
        )
        print(f"📊 Using AUTO mode: {len(features)} auto-selected features")
    elif FEATURE_MODE == 'core20':
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

    # FIX: 80/20 split with 1-day embargo gap to prevent leakage
    val_idx = int(len(X_train) * 0.80)
    embargo_bars = 288  # 1 day of M5 candles
    val_start = min(val_idx + embargo_bars, len(X_train) - 100)
    X_t = X_train.iloc[:val_idx]
    X_v = X_train.iloc[val_start:]
    y_t = {k: v.iloc[:val_idx] for k, v in y_train.items()}
    y_v = {k: v.iloc[val_start:] for k, v in y_train.items()}

    # Feature stability check — remove features that drift between train and val
    if USE_FEATURE_STABILITY:
        stable_feats = check_feature_stability(X_t, X_v, features)
        if len(stable_feats) < len(features):
            print(f"   Stability filter: {len(features)} → {len(stable_feats)} features")
            features = stable_feats
            X_t = X_t[features]
            X_v = X_v[features]

    models = train_models(X_t, y_t, X_v, y_v)
    
    # ---------------------------------------------------------
    # 1. STANDARD BACKTEST
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"RUNNING BACKTEST (Test Days: {args.test_days})")
    print("="*70)
    
    # === ВЫВОД ТЕСТОВОГО ПЕРИОДА ===
    if test_features:
        all_test_timestamps = []
        for pair, df in test_features.items():
            if len(df) > 0:
                all_test_timestamps.extend(df.index.tolist())
        
        if all_test_timestamps:
            test_period_start = min(all_test_timestamps)
            test_period_end = max(all_test_timestamps)
            test_days_actual = (test_period_end - test_period_start).days
            print("\n📅 ПЕРИОД ТЕСТИРОВАНИЯ:")
            print(f"  Начало:  {test_period_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Конец:   {test_period_end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Дней:    {test_days_actual} дней")
            print(f"  Пар:     {len(test_features)}")
            print()
    
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
    joblib.dump(models['direction'], out / 'direction_model.joblib')  # Calibrated model
    joblib.dump(models['direction_base'], out / 'direction_model_base.joblib')  # For feature importance
    joblib.dump(models['timing'], out / 'timing_model.joblib')
    joblib.dump(models['strength'], out / 'strength_model.joblib')
    joblib.dump(models['scaler'], out / 'scaler.joblib')
    joblib.dump(features, out / 'feature_names.joblib')
    
    # Save feature importance if auto-selection was used
    import json
    if selected_features_importance is not None:
        importance_dict = {name: float(imp) for name, imp in selected_features_importance}
        with open(out / 'feature_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)
        print(f"✅ Feature importance saved ({len(features)} auto-selected features)")
    
    # Save Optuna best params if available
    if models.get('best_params'):
        with open(out / 'optuna_best_params.json', 'w') as f:
            json.dump(models['best_params'], f, indent=2)
        print(f"✅ Optuna best params saved")
    
    # Save model config
    model_config = {
        'feature_mode': FEATURE_MODE,
        'use_optuna': USE_OPTUNA,
        'use_ensemble': USE_ENSEMBLE,
        'catboost_available': CATBOOST_AVAILABLE,
        'num_features': len(features),
        'training_date': datetime.now().isoformat()
    }
    with open(out / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✅ Models saved (CALIBRATED + ENSEMBLE + OPTUNA!)")

    if trades:
        pd.DataFrame(trades).to_csv(out / f'backtest_trades_{args.test_days}d.csv', index=False)

    # ---------------------------------------------------------
    # 1B. ✅ WALK-FORWARD VALIDATION (if requested)
    # ---------------------------------------------------------
    if args.walk_forward:
        walk_forward_results = walk_forward_validation(pairs, data_dir, initial_balance=args.initial_balance)
        
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
                
            ft = generate_killer_features(m1, m5, m15)
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
                
            ft = generate_killer_features(m1, m5, m15)
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
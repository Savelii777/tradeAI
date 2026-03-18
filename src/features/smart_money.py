#!/usr/bin/env python3
"""
Smart Money Concepts Features
Based on ICT (Inner Circle Trader) methodology

Key concepts:
1. Market Structure - Swing Highs/Lows, BOS, CHoCH
2. Fair Value Gaps (FVG) - Imbalances/inefficiencies
3. Order Blocks - Last opposite candle before impulse
4. Liquidity - Stop hunt zones above highs/below lows
5. Premium/Discount zones - Where smart money operates

These features capture institutional behavior patterns.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def detect_swing_points(
    high: pd.Series,
    low: pd.Series,
    left_bars: int = 5,
    right_bars: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect swing highs and swing lows.
    
    ⚠️ FIX: Original used right_bars (future data) = LOOKAHEAD BIAS!
    Now we only use left_bars (past data) for real-time compatibility.
    
    Swing High: A high that is higher than N bars to the LEFT only
    Swing Low: A low that is lower than N bars to the LEFT only
    
    This is slightly less accurate but works in real-time trading.
    
    Returns:
        (swing_highs, swing_lows) - Series with NaN except at swing points
    """
    swing_highs = pd.Series(np.nan, index=high.index)
    swing_lows = pd.Series(np.nan, index=low.index)
    
    for i in range(left_bars, len(high)):
        # Check for swing high - only look at PAST bars
        if all(high.iloc[i] > high.iloc[i-j] for j in range(1, left_bars + 1)):
            swing_highs.iloc[i] = high.iloc[i]
        
        # Check for swing low - only look at PAST bars
        if all(low.iloc[i] < low.iloc[i-j] for j in range(1, left_bars + 1)):
            swing_lows.iloc[i] = low.iloc[i]
    
    return swing_highs, swing_lows


def calculate_market_structure(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    swing_length: int = 5
) -> pd.DataFrame:
    """
    Calculate market structure features.
    
    Features:
    - Higher Highs (HH), Higher Lows (HL) = Uptrend
    - Lower Highs (LH), Lower Lows (LL) = Downtrend
    - Break of Structure (BOS) - continuation
    - Change of Character (CHoCH) - reversal
    """
    df = pd.DataFrame(index=close.index)
    
    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(high, low, swing_length, swing_length)
    
    # Forward fill last swing high/low for comparison
    last_swing_high = swing_highs.ffill()
    last_swing_low = swing_lows.ffill()
    
    # Previous swing high/low (shifted by 1)
    prev_swing_high = last_swing_high.shift(1)
    prev_swing_low = last_swing_low.shift(1)
    
    # Structure breaks
    df['structure_hh'] = (last_swing_high > prev_swing_high).astype(int)  # Higher High
    df['structure_hl'] = (last_swing_low > prev_swing_low).astype(int)    # Higher Low
    df['structure_lh'] = (last_swing_high < prev_swing_high).astype(int)  # Lower High
    df['structure_ll'] = (last_swing_low < prev_swing_low).astype(int)    # Lower Low
    
    # Trend detection
    # Uptrend: HH + HL, Downtrend: LH + LL
    df['structure_uptrend'] = (df['structure_hh'] & df['structure_hl']).astype(int)
    df['structure_downtrend'] = (df['structure_lh'] & df['structure_ll']).astype(int)
    
    # Break of Structure (BOS) - price breaks recent swing in trend direction
    df['structure_bos_bull'] = ((close > last_swing_high) & df['structure_uptrend']).astype(int)
    df['structure_bos_bear'] = ((close < last_swing_low) & df['structure_downtrend']).astype(int)
    
    # Change of Character (CHoCH) - price breaks recent swing against trend
    df['structure_choch_bull'] = ((close > last_swing_high) & df['structure_downtrend']).astype(int)
    df['structure_choch_bear'] = ((close < last_swing_low) & df['structure_uptrend']).astype(int)
    
    # Distance to swing points (in %)
    df['dist_to_swing_high'] = ((last_swing_high - close) / close * 100).fillna(0)
    df['dist_to_swing_low'] = ((close - last_swing_low) / close * 100).fillna(0)
    
    return df


def detect_fair_value_gaps(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    min_gap_pct: float = 0.1
) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG) - price imbalances.
    
    ⚠️ FIX: Original used shift(-1) = LOOKAHEAD BIAS!
    Now we detect FVG looking backwards only.
    
    FVG is detected on the MIDDLE candle of a 3-candle pattern.
    We shift the result by 2 bars so it's only available after the pattern completes.
    
    Bullish FVG (at candle i): candle[i-2].high < candle[i].low (gap up)
    Bearish FVG (at candle i): candle[i-2].low > candle[i].high (gap down)
    """
    df = pd.DataFrame(index=close.index)
    
    # Bullish FVG: low[i] > high[i-2] (we're at the 3rd candle looking back)
    bullish_fvg = (low > high.shift(2))
    bullish_fvg_size = ((low - high.shift(2)) / close * 100).clip(lower=0)
    
    # Bearish FVG: high[i] < low[i-2]
    bearish_fvg = (high < low.shift(2))
    bearish_fvg_size = ((low.shift(2) - high) / close * 100).clip(lower=0)
    
    # Only count gaps above minimum size
    df['fvg_bullish'] = (bullish_fvg & (bullish_fvg_size >= min_gap_pct)).astype(int)
    df['fvg_bearish'] = (bearish_fvg & (bearish_fvg_size >= min_gap_pct)).astype(int)
    df['fvg_bullish_size'] = bullish_fvg_size.fillna(0)
    df['fvg_bearish_size'] = bearish_fvg_size.fillna(0)
    
    # Recent FVG count (last 20 bars)
    df['fvg_bullish_recent'] = df['fvg_bullish'].rolling(20).sum()
    df['fvg_bearish_recent'] = df['fvg_bearish'].rolling(20).sum()
    
    # Unfilled FVG zones (price hasn't returned yet)
    # Simplified: assume FVG filled if price crosses back
    df['fvg_unfilled_bull'] = df['fvg_bullish'].copy()
    df['fvg_unfilled_bear'] = df['fvg_bearish'].copy()
    
    return df


def detect_order_blocks(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Detect Order Blocks - last opposite candle before strong impulse.
    
    Bullish Order Block: Last bearish candle before bullish impulse
    Bearish Order Block: Last bullish candle before bearish impulse
    
    These zones represent where institutions placed orders.
    """
    df = pd.DataFrame(index=close.index)
    
    # Candle direction
    is_bullish = (close > open_).astype(int)
    is_bearish = (close < open_).astype(int)
    
    # Impulse: strong move with volume
    body_size = abs(close - open_) / open_ * 100
    vol_avg = volume.rolling(20).mean()
    volume_spike = (volume > vol_avg * 1.5).astype(int)
    
    is_impulse_up = (is_bullish & (body_size > 1.0) & volume_spike).fillna(0).astype(int)
    is_impulse_down = (is_bearish & (body_size > 1.0) & volume_spike).fillna(0).astype(int)
    
    # Order block = last opposite candle before impulse
    # Simplified detection
    df['ob_bullish'] = (is_bearish.shift(1).fillna(0).astype(int) & is_impulse_up).astype(int)
    df['ob_bearish'] = (is_bullish.shift(1).fillna(0).astype(int) & is_impulse_down).astype(int)
    
    # Order block strength (based on body size)
    df['ob_bullish_strength'] = (body_size.shift(1) * df['ob_bullish']).fillna(0)
    df['ob_bearish_strength'] = (body_size.shift(1) * df['ob_bearish']).fillna(0)
    
    # Distance to nearest order block
    ob_bull_price = (high.shift(1) * df['ob_bullish']).replace(0, np.nan).ffill()
    ob_bear_price = (low.shift(1) * df['ob_bearish']).replace(0, np.nan).ffill()
    
    df['dist_to_ob_bull'] = ((close - ob_bull_price) / close * 100).fillna(0)
    df['dist_to_ob_bear'] = ((ob_bear_price - close) / close * 100).fillna(0)
    
    return df


def calculate_liquidity_zones(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Calculate liquidity zones - areas where stops are likely placed.
    
    Liquidity above swing highs (buy stops)
    Liquidity below swing lows (sell stops)
    
    Smart money hunts these stops before reversing.
    """
    df = pd.DataFrame(index=close.index)
    
    # Recent highs/lows where liquidity sits
    liquidity_high = high.rolling(lookback).max()
    liquidity_low = low.rolling(lookback).min()
    
    # Distance to liquidity zones
    df['dist_to_liquidity_high'] = ((liquidity_high - close) / close * 100).fillna(0)
    df['dist_to_liquidity_low'] = ((close - liquidity_low) / close * 100).fillna(0)
    
    # Equal highs/lows (liquidity pools)
    # Equal high: multiple candles at same high level
    equal_high_threshold = close * 0.001  # 0.1%
    equal_highs = (abs(high - high.shift(1)) < equal_high_threshold).astype(int)
    
    equal_low_threshold = close * 0.001
    equal_lows = (abs(low - low.shift(1)) < equal_low_threshold).astype(int)
    
    df['liquidity_equal_highs'] = equal_highs.rolling(5).sum()
    df['liquidity_equal_lows'] = equal_lows.rolling(5).sum()
    
    # Liquidity sweep: price briefly exceeds high/low then reverses
    sweep_high = ((high > liquidity_high.shift(1)) & (close < liquidity_high.shift(1))).astype(int)
    sweep_low = ((low < liquidity_low.shift(1)) & (close > liquidity_low.shift(1))).astype(int)
    
    df['liquidity_sweep_high'] = sweep_high
    df['liquidity_sweep_low'] = sweep_low
    df['liquidity_sweep_high_recent'] = sweep_high.rolling(10).sum()
    df['liquidity_sweep_low_recent'] = sweep_low.rolling(10).sum()
    
    return df


def calculate_premium_discount(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 50
) -> pd.DataFrame:
    """
    Calculate Premium/Discount zones.
    
    Smart money buys in discount (lower 50% of range)
    Smart money sells in premium (upper 50% of range)
    """
    df = pd.DataFrame(index=close.index)
    
    # Range highs/lows
    range_high = high.rolling(lookback).max()
    range_low = low.rolling(lookback).min()
    range_mid = (range_high + range_low) / 2
    
    # Position in range (0 = at low, 100 = at high)
    range_size = range_high - range_low
    range_position = np.where(
        range_size > 0,
        ((close - range_low) / range_size * 100),
        50
    )
    
    df['premium_discount_position'] = range_position
    
    # Zone classification
    df['in_premium'] = (range_position > 60).astype(int)  # Upper 40%
    df['in_equilibrium'] = ((range_position >= 40) & (range_position <= 60)).astype(int)
    df['in_discount'] = (range_position < 40).astype(int)  # Lower 40%
    
    # Extreme zones
    df['in_extreme_premium'] = (range_position > 80).astype(int)
    df['in_extreme_discount'] = (range_position < 20).astype(int)
    
    return df


def generate_smart_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all Smart Money Concept features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with SMC features
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Market Structure
    structure = calculate_market_structure(
        df['high'], df['low'], df['close'], swing_length=5
    )
    for col in structure.columns:
        features[f'smc_{col}'] = structure[col]
    
    # 2. Fair Value Gaps
    fvg = detect_fair_value_gaps(
        df['open'], df['high'], df['low'], df['close'], min_gap_pct=0.1
    )
    for col in fvg.columns:
        features[f'smc_{col}'] = fvg[col]
    
    # 3. Order Blocks
    ob = detect_order_blocks(
        df['open'], df['high'], df['low'], df['close'], df['volume'], lookback=20
    )
    for col in ob.columns:
        features[f'smc_{col}'] = ob[col]
    
    # 4. Liquidity Zones
    liquidity = calculate_liquidity_zones(
        df['high'], df['low'], df['close'], lookback=20
    )
    for col in liquidity.columns:
        features[f'smc_{col}'] = liquidity[col]
    
    # 5. Premium/Discount
    premium = calculate_premium_discount(
        df['high'], df['low'], df['close'], lookback=50
    )
    for col in premium.columns:
        features[f'smc_{col}'] = premium[col]
    
    return features

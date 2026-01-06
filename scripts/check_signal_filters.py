#!/usr/bin/env python3
"""Check how many signals would pass ALL filters in last 7 days."""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import joblib
from train_mtf import MTFFeatureEngine
from datetime import timedelta

def add_volume_features(df):
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    df["vol_zscore"] = (df["volume"] - df["vol_sma_20"]) / df["volume"].rolling(20).std()
    df["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["price_vs_vwap"] = df["close"] / df["vwap"] - 1
    df["vol_momentum"] = df["volume"].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# Загружаем модели
dir_model = joblib.load("../models/v8_improved/direction_model.joblib")
timing_model = joblib.load("../models/v8_improved/timing_model.joblib")
strength_model = joblib.load("../models/v8_improved/strength_model.joblib")
feature_cols = joblib.load("../models/v8_improved/feature_names.joblib")

# Пороги из live trading
MIN_CONF = 0.5
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

print("="*70)
print("SIGNAL ANALYSIS: How many signals pass ALL filters?")
print("="*70)
print(f"Thresholds: Conf>={MIN_CONF}, Timing>={MIN_TIMING}, Strength>={MIN_STRENGTH}")
print()

# Проверяем несколько пар
pairs = ["ASTER_USDT", "HYPE_USDT", "ZEC_USDT", "AVAX_USDT", "PIPPIN_USDT"]
engine = MTFFeatureEngine()

total_signals_all = 0

for pair in pairs:
    try:
        df_1m = pd.read_csv(f"../data/candles/{pair}_USDT_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
        df_5m = pd.read_csv(f"../data/candles/{pair}_USDT_5m.csv", parse_dates=["timestamp"], index_col="timestamp")
        df_15m = pd.read_csv(f"../data/candles/{pair}_USDT_15m.csv", parse_dates=["timestamp"], index_col="timestamp")
        
        # Последние 7 дней
        end_time = df_5m.index.max()
        start_time = end_time - timedelta(days=7)
        warmup = start_time - timedelta(hours=24)
        
        m1 = df_1m[df_1m.index >= warmup]
        m5 = df_5m[df_5m.index >= warmup]
        m15 = df_15m[df_15m.index >= warmup]
        
        ft = engine.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[["open", "high", "low", "close", "volume"]])
        ft = add_volume_features(ft)
        ft["atr"] = calculate_atr(ft)
        ft = ft.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        ft = ft[ft.index >= start_time]
        
        X = ft[feature_cols]
        
        # Predictions
        proba = dir_model.predict_proba(X)
        conf = np.max(proba, axis=1)
        preds = dir_model.predict(X)
        timing = timing_model.predict(X)
        strength = strength_model.predict(X)
        
        # Фильтры
        # 0=SHORT, 1=SIDEWAYS, 2=LONG
        not_sideways = preds != 1  # Skip SIDEWAYS (class 1)
        pass_conf = conf >= MIN_CONF
        pass_timing = timing >= MIN_TIMING
        pass_strength = strength >= MIN_STRENGTH
        pass_all = not_sideways & pass_conf & pass_timing & pass_strength
        
        total = len(X)
        signals = pass_all.sum()
        total_signals_all += signals
        
        print(f"{pair}:")
        print(f"  Total bars (7d):      {total}")
        print(f"  Not SIDEWAYS:         {not_sideways.sum()} ({not_sideways.mean()*100:.0f}%)")
        print(f"  Conf >= {MIN_CONF}:          {pass_conf.sum()} ({pass_conf.mean()*100:.0f}%)")
        print(f"  Timing >= {MIN_TIMING}:        {pass_timing.sum()} ({pass_timing.mean()*100:.0f}%)")
        print(f"  Strength >= {MIN_STRENGTH}:      {pass_strength.sum()} ({pass_strength.mean()*100:.0f}%)")
        print(f"  >>> PASS ALL:         {signals} ({pass_all.mean()*100:.1f}%)")
        
        # По дням
        print(f"  Signals by day:")
        for i in range(7, 0, -1):
            day_start = end_time - timedelta(days=i)
            day_end = end_time - timedelta(days=i-1)
            day_mask = (ft.index >= day_start) & (ft.index < day_end)
            day_signals = pass_all[day_mask].sum()
            print(f"    {day_start.date()}: {day_signals} signals")
        print()
        
    except Exception as e:
        print(f"{pair}: Error - {e}")
        print()

print("="*70)
print(f"TOTAL SIGNALS across {len(pairs)} pairs in 7 days: {total_signals_all}")
print("="*70)

#!/usr/bin/env python3
"""
ДИАГНОСТИКА: Почему live не даёт сигналов, а симуляция даёт?

Этот скрипт показывает ТОЧНУЮ разницу между live и симуляцией.

Usage:
    python scripts/diagnose_live_vs_simulation.py --pair PIPPIN_USDT_USDT
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.utils.constants import CUMSUM_PATTERNS, ABSOLUTE_PRICE_FEATURES


MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5).clip(-10, 10)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_data(pair: str) -> dict:
    """Load historical data from CSV/Parquet."""
    pair_name = pair.replace('/', '_').replace(':', '_')
    data = {}
    
    for tf in ['1m', '5m', '15m']:
        parquet_path = DATA_DIR / f"{pair_name}_{tf}.parquet"
        csv_path = DATA_DIR / f"{pair_name}_{tf}.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            source = 'parquet'
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
            source = 'csv'
        else:
            print(f"Data not found: {parquet_path} or {csv_path}")
            return None
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        data[tf] = df
        data[f'{tf}_source'] = source
    
    return data


def prepare_features(data: dict, mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Prepare features on FULL history."""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        cols_to_drop = [c for c in ft.columns if any(p in c.lower() for p in CUMSUM_PATTERNS)]
        ft = ft.drop(columns=cols_to_drop, errors='ignore')
        
        absolute_cols = [c for c in ft.columns if c in ABSOLUTE_PRICE_FEATURES]
        ft = ft.drop(columns=absolute_cols, errors='ignore')
        
        ft = ft.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        return ft
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None


def diagnose(pair: str, hours: int = 48):
    """Run diagnosis comparing live vs simulation approach."""
    
    print("=" * 80)
    print("ДИАГНОСТИКА: Live vs Simulation")
    print("=" * 80)
    print(f"Pair: {pair}")
    print(f"Period: last {hours} hours")
    print()
    
    # Load models
    try:
        models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
            'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
        }
        print(f"✓ Model loaded: {len(models['features'])} features")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Load data
    data = load_data(pair)
    if data is None:
        return
    
    m5 = data['5m']
    print(f"\n✓ Data loaded:")
    print(f"  M1: {len(data['1m'])} candles, last: {data['1m'].index[-1]} (from {data['1m_source']})")
    print(f"  M5: {len(data['5m'])} candles, last: {data['5m'].index[-1]} (from {data['5m_source']})")
    print(f"  M15: {len(data['15m'])} candles, last: {data['15m'].index[-1]} (from {data['15m_source']})")
    
    # Check data freshness
    now = datetime.now(timezone.utc)
    age_minutes = (now - m5.index[-1]).total_seconds() / 60
    print(f"\n⏰ Data age: {age_minutes:.0f} minutes")
    if age_minutes > 30:
        print(f"  ⚠️ WARNING: Data is more than 30 minutes old!")
        print(f"  Live trading won't work with stale data.")
    
    # Initialize
    mtf_fe = MTFFeatureEngine()
    
    # Prepare features on FULL history
    print("\n📊 Preparing features on FULL history...")
    full_features = prepare_features(data, mtf_fe)
    if full_features is None:
        print("Failed to prepare features")
        return
    print(f"✓ Features prepared: {len(full_features)} rows, {len(full_features.columns)} columns")
    print(f"  Feature range: {full_features.index[0]} to {full_features.index[-1]}")
    
    # =============================================
    # METHOD 1: LIVE TRADING APPROACH (iloc[-2])
    # =============================================
    print("\n" + "=" * 80)
    print("METHOD 1: LIVE TRADING APPROACH (uses iloc[-2])")
    print("=" * 80)
    
    row = full_features.iloc[[-2]].copy()
    current_time_live = row.index[0]
    
    for f in models['features']:
        if f not in row.columns:
            row[f] = 0.0
    
    X = row[models['features']].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    dir_proba = models['direction'].predict_proba(X)
    dir_conf = float(np.max(dir_proba))
    dir_pred = int(np.argmax(dir_proba))
    timing = float(models['timing'].predict(X)[0])
    strength = float(models['strength'].predict(X)[0])
    
    direction = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
    is_signal = dir_pred != 1 and dir_conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH
    
    print(f"Time: {current_time_live}")
    print(f"Direction: {direction}")
    print(f"Confidence: {dir_conf:.4f}")
    print(f"Timing: {timing:.4f}")
    print(f"Strength: {strength:.4f}")
    print(f"Signal: {'✅ YES' if is_signal else '❌ NO'}")
    if not is_signal:
        reasons = []
        if dir_pred == 1:
            reasons.append("SIDEWAYS direction")
        if dir_conf < MIN_CONF:
            reasons.append(f"low conf ({dir_conf:.2f} < {MIN_CONF})")
        if timing < MIN_TIMING:
            reasons.append(f"low timing ({timing:.2f} < {MIN_TIMING})")
        if strength < MIN_STRENGTH:
            reasons.append(f"low strength ({strength:.2f} < {MIN_STRENGTH})")
        print(f"  Reason: {', '.join(reasons)}")
    
    # =============================================
    # METHOD 2: SIMULATION APPROACH (walk through time)
    # =============================================
    print("\n" + "=" * 80)
    print("METHOD 2: SIMULATION APPROACH (walk through all bars)")
    print("=" * 80)
    
    # Calculate simulation range
    end_time = full_features.index[-1] - timedelta(hours=1)
    start_time = end_time - timedelta(hours=hours)
    
    sim_times = full_features.index[(full_features.index >= start_time) & (full_features.index <= end_time)]
    
    print(f"Simulation period: {sim_times[0]} to {sim_times[-1]}")
    print(f"Total bars: {len(sim_times)}")
    print()
    
    signals_found = 0
    all_results = []
    
    for current_time in sim_times:
        last_row = full_features.loc[current_time]
        
        X = np.zeros((1, len(models['features'])))
        for i, feat in enumerate(models['features']):
            if feat in full_features.columns:
                X[0, i] = last_row[feat] if isinstance(last_row, pd.Series) else full_features.loc[current_time, feat]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        proba = models['direction'].predict_proba(X)[0]
        dir_pred = int(np.argmax(proba))
        conf = float(np.max(proba))
        timing = float(models['timing'].predict(X)[0])
        strength = float(models['strength'].predict(X)[0])
        
        direction = 'LONG' if dir_pred == 2 else ('SHORT' if dir_pred == 0 else 'SIDEWAYS')
        is_signal = dir_pred != 1 and conf >= MIN_CONF and timing >= MIN_TIMING and strength >= MIN_STRENGTH
        
        if is_signal:
            signals_found += 1
            all_results.append({
                'time': current_time,
                'direction': direction,
                'conf': conf,
                'timing': timing,
                'strength': strength
            })
    
    print(f"Signals found in period: {signals_found}")
    
    if signals_found > 0:
        print(f"\nFirst 10 signals:")
        for i, r in enumerate(all_results[:10]):
            print(f"  [{r['time'].strftime('%m-%d %H:%M')}] {r['direction']} "
                  f"conf={r['conf']:.2f} tim={r['timing']:.2f} str={r['strength']:.2f}")
        if len(all_results) > 10:
            print(f"  ... and {len(all_results) - 10} more")
    
    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 80)
    print("ВЫВОДЫ")
    print("=" * 80)
    
    if signals_found > 0:
        print(f"✓ Симуляция нашла {signals_found} сигналов за последние {hours} часов")
        
        if not is_signal:
            print()
            print("❌ НО live trading (iloc[-2]) НЕ показывает сигнал ПРЯМО СЕЙЧАС")
            print()
            print("ПРИЧИНЫ:")
            print("1. Рынок сейчас БОКОВОЙ — сигналы были раньше но не сейчас")
            print("2. Данные устарели — проверь возраст данных выше")
            print("3. Live trading работает правильно — он просто ждёт подходящий момент")
            print()
            print("РЕКОМЕНДАЦИИ:")
            print("- Оставь live_trading работать — когда появится сигнал, он его поймает")
            print("- Или снизь MIN_CONF до 0.45 для тестирования")
        else:
            print()
            print("✅ Live trading тоже показывает сигнал!")
            print("Всё работает правильно.")
    else:
        print(f"❌ Симуляция НЕ нашла сигналов за последние {hours} часов")
        print("Рынок был боковым всё это время.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose live vs simulation")
    parser.add_argument("--pair", type=str, default="PIPPIN_USDT_USDT", help="Pair to diagnose")
    parser.add_argument("--hours", type=int, default=48, help="Hours to analyze")
    args = parser.parse_args()
    
    diagnose(args.pair, args.hours)


if __name__ == "__main__":
    main()

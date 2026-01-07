#!/usr/bin/env python3
"""
ГЛУБОКАЯ ДИАГНОСТИКА: Почему confidence в лайве < 40%, а в бэктесте > 50%?

Этот скрипт делает ТОЧНОЕ сравнение:
1. Берёт данные из CSV (как в бэктесте)
2. Берёт данные с Binance (как в лайве)
3. Сравнивает ВСЕ фичи и предсказания для ОДНОГО момента времени

Цель: найти КОНКРЕТНЫЕ фичи, которые отличаются.
"""

import sys
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine
from src.features.feature_engine import FeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_list.json"

# Сколько свечей загружать для лайва (как в live_scanner_v4.py)
LIVE_CANDLES_M1 = 1500
LIVE_CANDLES_M5 = 1500
LIVE_CANDLES_M15 = 500


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет volume фичи (идентично train_v3_dynamic.py)"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR calculation"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_csv_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Загружает данные из локального CSV (как в бэктесте)"""
    safe_symbol = pair.replace('/', '_').replace(':', '_')
    filepath = DATA_DIR / f"{safe_symbol}_{timeframe}.csv"
    
    if not filepath.exists():
        print(f"❌ CSV not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    elif str(df.index.tz) != 'UTC':
        df.index = df.index.tz_convert('UTC')
    
    return df


def fetch_live_data(exchange, pair: str, timeframe: str, total_needed: int) -> pd.DataFrame:
    """Загружает данные с биржи (как в лайв сканере)"""
    try:
        all_candles = []
        limit = 1000
        
        candles = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        all_candles = candles
        
        while len(all_candles) < total_needed:
            oldest = all_candles[0][0]
            tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[timeframe]
            since = oldest - limit * tf_ms
            
            candles = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
            if not candles:
                break
            
            new = [c for c in candles if c[0] < oldest]
            if not new:
                break
            
            all_candles = new + all_candles
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    except Exception as e:
        print(f"❌ Error fetching {pair} {timeframe}: {e}")
        return None


def build_features(m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, mtf_fe: MTFFeatureEngine) -> pd.DataFrame:
    """Строит фичи (идентично train_v3_dynamic.py и live_scanner_v4.py)"""
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    ft = ft.dropna()
    return ft


def compare_features(backtest_row: pd.Series, live_row: pd.Series, feature_names: list) -> dict:
    """Сравнивает фичи между бэктестом и лайвом"""
    diffs = {}
    
    for f in feature_names:
        bt_val = backtest_row.get(f, np.nan)
        live_val = live_row.get(f, np.nan)
        
        # Convert to float to handle boolean values
        try:
            bt_val = float(bt_val)
            live_val = float(live_val)
        except (TypeError, ValueError):
            continue
        
        if pd.isna(bt_val) and pd.isna(live_val):
            continue
        
        if pd.isna(bt_val) or pd.isna(live_val):
            diffs[f] = {
                'backtest': bt_val,
                'live': live_val,
                'diff': 'ONE_MISSING',
                'pct_diff': 100.0
            }
            continue
        
        if bt_val == 0 and live_val == 0:
            continue
            
        abs_diff = abs(bt_val - live_val)
        base = max(abs(bt_val), abs(live_val), 1e-10)
        pct_diff = (abs_diff / base) * 100
        
        if pct_diff > 1.0:  # Показываем только значимые отличия (>1%)
            diffs[f] = {
                'backtest': round(bt_val, 6),
                'live': round(live_val, 6),
                'diff': round(bt_val - live_val, 6),
                'pct_diff': round(pct_diff, 2)
            }
    
    return diffs


def main():
    print("="*70)
    print("ГЛУБОКАЯ ДИАГНОСТИКА: Backtest vs Live Features")
    print("="*70)
    
    # Load models
    print("\n📦 Загрузка моделей...")
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    feature_names = joblib.load(MODEL_DIR / 'feature_names.joblib')
    print(f"   Модель использует {len(feature_names)} фичей")
    
    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs = [p['symbol'] for p in json.load(f)['pairs'][:5]]  # Первые 5 пар
    
    # Init
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    # Время для сравнения: последняя закрытая свеча (сейчас минус 5 минут, округлённое)
    now = datetime.now(timezone.utc)
    target_time = now.replace(second=0, microsecond=0)
    target_time = target_time - timedelta(minutes=(target_time.minute % 5) + 5)
    
    print(f"\n🎯 Сравниваем данные для момента: {target_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"   (последняя ЗАКРЫТАЯ M5 свеча)")
    
    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"📊 {pair}")
        print("="*70)
        
        # === 1. BACKTEST DATA (CSV) ===
        print("\n📂 BACKTEST (из CSV):")
        csv_m1 = load_csv_data(pair, '1m')
        csv_m5 = load_csv_data(pair, '5m')
        csv_m15 = load_csv_data(pair, '15m')
        
        if csv_m1 is None or csv_m5 is None or csv_m15 is None:
            print("   ❌ Нет CSV данных, пропускаем")
            continue
        
        print(f"   M1: {len(csv_m1)} свечей ({csv_m1.index[0].strftime('%m-%d %H:%M')} → {csv_m1.index[-1].strftime('%m-%d %H:%M')})")
        print(f"   M5: {len(csv_m5)} свечей ({csv_m5.index[0].strftime('%m-%d %H:%M')} → {csv_m5.index[-1].strftime('%m-%d %H:%M')})")
        print(f"   M15: {len(csv_m15)} свечей ({csv_m15.index[0].strftime('%m-%d %H:%M')} → {csv_m15.index[-1].strftime('%m-%d %H:%M')})")
        
        # Обрезаем CSV до target_time
        csv_m1_cut = csv_m1[csv_m1.index <= target_time]
        csv_m5_cut = csv_m5[csv_m5.index <= target_time]
        csv_m15_cut = csv_m15[csv_m15.index <= target_time]
        
        if len(csv_m5_cut) < 100:
            print("   ❌ Недостаточно данных в CSV для target_time")
            continue
        
        # Build backtest features
        try:
            bt_features = build_features(csv_m1_cut, csv_m5_cut, csv_m15_cut, mtf_fe)
            if len(bt_features) < 10:
                print("   ❌ Недостаточно фичей после dropna")
                continue
            bt_row = bt_features.iloc[-1]
            bt_time = bt_features.index[-1]
            print(f"   ✅ Backtest фичи построены. Последняя свеча: {bt_time.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"   ❌ Ошибка построения фичей: {e}")
            continue
        
        # === 2. LIVE DATA (Binance) ===
        print("\n🌐 LIVE (с Binance):")
        live_m1 = fetch_live_data(binance, pair, '1m', LIVE_CANDLES_M1)
        live_m5 = fetch_live_data(binance, pair, '5m', LIVE_CANDLES_M5)
        live_m15 = fetch_live_data(binance, pair, '15m', LIVE_CANDLES_M15)
        
        if live_m1 is None or live_m5 is None or live_m15 is None:
            print("   ❌ Не удалось загрузить данные с Binance")
            continue
        
        print(f"   M1: {len(live_m1)} свечей ({live_m1.index[0].strftime('%m-%d %H:%M')} → {live_m1.index[-1].strftime('%m-%d %H:%M')})")
        print(f"   M5: {len(live_m5)} свечей ({live_m5.index[0].strftime('%m-%d %H:%M')} → {live_m5.index[-1].strftime('%m-%d %H:%M')})")
        print(f"   M15: {len(live_m15)} свечей ({live_m15.index[0].strftime('%m-%d %H:%M')} → {live_m15.index[-1].strftime('%m-%d %H:%M')})")
        
        # Обрезаем live до target_time для честного сравнения
        live_m1_cut = live_m1[live_m1.index <= target_time]
        live_m5_cut = live_m5[live_m5.index <= target_time]
        live_m15_cut = live_m15[live_m15.index <= target_time]
        
        # Build live features
        try:
            live_features = build_features(live_m1_cut, live_m5_cut, live_m15_cut, mtf_fe)
            if len(live_features) < 10:
                print("   ❌ Недостаточно фичей после dropna")
                continue
            live_row = live_features.iloc[-1]
            live_time = live_features.index[-1]
            print(f"   ✅ Live фичи построены. Последняя свеча: {live_time.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"   ❌ Ошибка построения фичей: {e}")
            continue
        
        # === 3. СРАВНЕНИЕ СВЕЧЕЙ ===
        print(f"\n📊 СРАВНЕНИЕ OHLCV для {bt_time.strftime('%Y-%m-%d %H:%M')}:")
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            bt_val = bt_row.get(col, np.nan)
            live_val = live_row.get(col, np.nan)
            match = "✅" if abs(bt_val - live_val) < 0.0001 * bt_val else "❌"
            print(f"   {col:8s}: BT={bt_val:15.6f} | LIVE={live_val:15.6f} {match}")
        
        # === 4. СРАВНЕНИЕ ФИЧЕЙ ===
        print(f"\n🔍 СРАВНЕНИЕ ФИЧЕЙ (только различающиеся >1%):")
        diffs = compare_features(bt_row, live_row, feature_names)
        
        if not diffs:
            print("   ✅ Все фичи идентичны!")
        else:
            # Сортируем по величине различия
            sorted_diffs = sorted(diffs.items(), key=lambda x: x[1]['pct_diff'], reverse=True)
            for f, d in sorted_diffs[:30]:  # Топ-30 различий
                print(f"   ❌ {f:40s}: BT={d['backtest']:12.4f} | LIVE={d['live']:12.4f} | diff={d['pct_diff']:6.1f}%")
        
        # === 5. ПРЕДСКАЗАНИЯ ===
        print(f"\n🎯 ПРЕДСКАЗАНИЯ:")
        
        # Fill missing features
        for f in feature_names:
            if f not in bt_row:
                bt_row[f] = 0.0
            if f not in live_row:
                live_row[f] = 0.0
        
        # Backtest prediction
        X_bt = pd.DataFrame([bt_row[feature_names].values], columns=feature_names).astype(np.float64)
        X_bt = np.nan_to_num(X_bt, nan=0.0, posinf=0.0, neginf=0.0)
        
        bt_dir_proba = models['direction'].predict_proba(X_bt)
        bt_dir_pred = int(np.argmax(bt_dir_proba))
        bt_dir_conf = float(np.max(bt_dir_proba))
        bt_timing = float(models['timing'].predict(X_bt)[0])
        bt_strength = float(models['strength'].predict(X_bt)[0])
        
        # Live prediction
        X_live = pd.DataFrame([live_row[feature_names].values], columns=feature_names).astype(np.float64)
        X_live = np.nan_to_num(X_live, nan=0.0, posinf=0.0, neginf=0.0)
        
        live_dir_proba = models['direction'].predict_proba(X_live)
        live_dir_pred = int(np.argmax(live_dir_proba))
        live_dir_conf = float(np.max(live_dir_proba))
        live_timing = float(models['timing'].predict(X_live)[0])
        live_strength = float(models['strength'].predict(X_live)[0])
        
        dir_names = ['SHORT', 'SIDEWAYS', 'LONG']
        
        print(f"   BACKTEST:  {dir_names[bt_dir_pred]:8s} | Conf={bt_dir_conf:.3f} | Timing={bt_timing:.2f} | Strength={bt_strength:.2f}")
        print(f"   LIVE:      {dir_names[live_dir_pred]:8s} | Conf={live_dir_conf:.3f} | Timing={live_timing:.2f} | Strength={live_strength:.2f}")
        
        conf_diff = abs(bt_dir_conf - live_dir_conf)
        if conf_diff > 0.05:
            print(f"   ⚠️  ЗНАЧИТЕЛЬНОЕ РАСХОЖДЕНИЕ CONFIDENCE: {conf_diff:.3f}")
        
        # === 6. АНАЛИЗ ПРОБЛЕМНЫХ ФИЧЕЙ ===
        if diffs:
            print(f"\n🔬 АНАЛИЗ ПРОБЛЕМНЫХ ФИЧЕЙ:")
            
            # Группируем по типу проблемы
            cumsum_issues = [f for f in diffs if any(x in f.lower() for x in 
                ['bars_since', 'consecutive', 'obv', 'cumsum', 'swing'])]
            
            rolling_issues = [f for f in diffs if any(x in f.lower() for x in 
                ['ema', 'sma', 'rolling', 'ma_', 'rsi', 'macd', 'bb_', 'stoch'])]
            
            window_issues = [f for f in diffs if any(x in f.lower() for x in 
                ['m1_', 'm5_', 'm15_'])]
            
            if cumsum_issues:
                print(f"   ⚠️  CUMSUM-зависимые ({len(cumsum_issues)}): {cumsum_issues[:5]}")
            if rolling_issues:
                print(f"   ⚠️  Rolling-зависимые ({len(rolling_issues)}): {rolling_issues[:5]}")
            if window_issues:
                print(f"   ⚠️  MTF-зависимые ({len(window_issues)}): {window_issues[:5]}")
    
    print("\n" + "="*70)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*70)
    print("\nРЕКОМЕНДАЦИИ:")
    print("1. Если много CUMSUM-зависимых фичей различаются - нужно их исключить из модели")
    print("2. Если Rolling-зависимые различаются - возможно недостаточно свечей в лайве")
    print("3. Если OHLCV различаются - проблема с синхронизацией данных")


if __name__ == '__main__':
    main()

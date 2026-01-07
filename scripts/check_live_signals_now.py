#!/usr/bin/env python3
"""
БЫСТРАЯ ПРОВЕРКА: Почему нет сигналов с conf > 50%?

Проверяем топовые пары из бэктеста:
- ASTER, PIPPIN, ZEC, HYPE, NEAR, AVAX

И смотрим РЕАЛЬНЫЕ предсказания модели СЕЙЧАС.
"""

import sys
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"

# Топовые пары из бэктеста
TOP_PAIRS = [
    'ASTER/USDT:USDT',
    'PIPPIN/USDT:USDT', 
    'ZEC/USDT:USDT',
    'HYPE/USDT:USDT',
    'NEAR/USDT:USDT',
    'AVAX/USDT:USDT',
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
]


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def fetch_candles(exchange, pair: str, timeframe: str, total_needed: int) -> pd.DataFrame:
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


def main():
    print("="*70)
    print("БЫСТРАЯ ПРОВЕРКА ТОП ПАРЫ - СЕЙЧАС")
    print("="*70)
    
    # Load models
    print("\n📦 Загрузка моделей...")
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    feature_names = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    now = datetime.now(timezone.utc)
    print(f"📅 Время: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"📊 Проверяем {len(TOP_PAIRS)} топовых пар\n")
    
    results = []
    
    for pair in TOP_PAIRS:
        try:
            # Загружаем данные
            m1 = fetch_candles(binance, pair, '1m', 1500)
            m5 = fetch_candles(binance, pair, '5m', 1500)
            m15 = fetch_candles(binance, pair, '15m', 500)
            
            # Build features
            ft = mtf_fe.align_timeframes(m1, m5, m15)
            ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
            
            if len(ft) < 10:
                print(f"❌ {pair}: недостаточно данных")
                continue
            
            # Fill missing features
            for f in feature_names:
                if f not in ft.columns:
                    ft[f] = 0.0
            
            # Predict on last 10 CLOSED candles (-2 to -11)
            print(f"\n📊 {pair}")
            print("-" * 60)
            
            for i in range(-2, -12, -1):
                row = ft.iloc[i]
                X = np.array([row[feature_names].values]).astype(np.float64)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
                dir_proba = models['direction'].predict_proba(X)
                dir_pred = int(np.argmax(dir_proba))
                dir_conf = float(np.max(dir_proba))
                timing = float(models['timing'].predict(X)[0])
                strength = float(models['strength'].predict(X)[0])
                
                ts = ft.index[i]
                dir_names = ['SHORT', 'SIDEWAYS', 'LONG']
                close_price = row['close']
                
                # Check if passes thresholds
                passes = (dir_pred != 1 and dir_conf >= 0.50 and timing >= 0.8 and strength >= 1.4)
                
                if passes:
                    emoji = "✅"
                elif dir_pred != 1 and dir_conf >= 0.40:
                    emoji = "🟡"
                else:
                    emoji = "⬜"
                
                print(f"{emoji} {ts.strftime('%H:%M')} | {dir_names[dir_pred]:8s} | "
                      f"Conf={dir_conf:.3f} | T={timing:.2f} | S={strength:.2f} | "
                      f"Close={close_price:.4f}")
                
                if passes:
                    results.append({
                        'pair': pair,
                        'time': ts,
                        'direction': dir_names[dir_pred],
                        'conf': dir_conf,
                        'timing': timing,
                        'strength': strength
                    })
                    
        except Exception as e:
            print(f"❌ {pair}: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("ИТОГИ")
    print("="*70)
    
    if results:
        print(f"\n✅ Найдено {len(results)} ВАЛИДНЫХ сигналов:")
        for r in results:
            print(f"   {r['pair']} {r['direction']} @ {r['time'].strftime('%H:%M')} "
                  f"| Conf={r['conf']:.3f} T={r['timing']:.2f} S={r['strength']:.2f}")
    else:
        print("\n❌ НЕТ валидных сигналов в последние 50 минут!")
        print("\nВозможные причины:")
        print("1. Рынок в боковике (SIDEWAYS) - модель правильно это детектит")
        print("2. Нет волатильности - нет opportunity")
        print("3. Это НОРМАЛЬНО - модель ждёт хороших условий")
        print("\n💡 В бэктесте было ~14 сделок в ДЕНЬ, не в час!")
        print("   При 20 парах это ~0.7 сделки на пару в день")
        print("   = 1 сделка каждые ~34 часа на пару")


if __name__ == '__main__':
    main()

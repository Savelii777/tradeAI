#!/usr/bin/env python3
"""
Детальная проверка: почему фичи отличаются между CSV и Live
"""

import sys
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")
DATA_DIR = Path("data/candles")


def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def main():
    features = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    pair = 'PIPPIN/USDT:USDT'
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    print("="*80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ M1 ДАННЫХ: CSV vs LIVE")
    print("="*80)
    
    # ============================================================
    # 1. Загрузка CSV
    # ============================================================
    print("\n[1] Загрузка CSV...")
    m1_csv = pd.read_csv(DATA_DIR / f'{pair_name}_1m.csv', index_col=0, parse_dates=True)
    m5_csv = pd.read_csv(DATA_DIR / f'{pair_name}_5m.csv', index_col=0, parse_dates=True)
    m15_csv = pd.read_csv(DATA_DIR / f'{pair_name}_15m.csv', index_col=0, parse_dates=True)
    
    m1_csv.index = m1_csv.index.tz_localize('UTC')
    m5_csv.index = m5_csv.index.tz_localize('UTC')
    m15_csv.index = m15_csv.index.tz_localize('UTC')
    
    # ============================================================
    # 2. Загрузка LIVE
    # ============================================================
    print("[2] Загрузка LIVE...")
    m1_live = pd.DataFrame(binance.fetch_ohlcv(pair, '1m', limit=500), 
                           columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    m1_live['timestamp'] = pd.to_datetime(m1_live['timestamp'], unit='ms', utc=True)
    m1_live.set_index('timestamp', inplace=True)
    
    m5_live = pd.DataFrame(binance.fetch_ohlcv(pair, '5m', limit=500),
                           columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    m5_live['timestamp'] = pd.to_datetime(m5_live['timestamp'], unit='ms', utc=True)
    m5_live.set_index('timestamp', inplace=True)
    
    m15_live = pd.DataFrame(binance.fetch_ohlcv(pair, '15m', limit=200),
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    m15_live['timestamp'] = pd.to_datetime(m15_live['timestamp'], unit='ms', utc=True)
    m15_live.set_index('timestamp', inplace=True)
    
    # ============================================================
    # 3. Найти общий период
    # ============================================================
    print("\n[3] Поиск общего периода...")
    common_m1 = m1_csv.index.intersection(m1_live.index)
    print(f"   Общих M1 свечей: {len(common_m1)}")
    
    if len(common_m1) == 0:
        print("   НЕТ ПЕРЕСЕЧЕНИЯ!")
        return
    
    # Берём конкретный 5-минутный период
    target_m5_time = common_m1[-10]  # Одна из последних общих
    target_m5_time = target_m5_time.floor('5min')
    print(f"   Анализируем период: {target_m5_time}")
    
    # ============================================================
    # 4. Сравнение M1 свечей в этом периоде
    # ============================================================
    print("\n[4] Сравнение M1 свечей в 5-минутном окне...")
    
    m1_start = target_m5_time
    m1_end = target_m5_time + pd.Timedelta(minutes=5)
    
    m1_csv_window = m1_csv[(m1_csv.index >= m1_start) & (m1_csv.index < m1_end)]
    m1_live_window = m1_live[(m1_live.index >= m1_start) & (m1_live.index < m1_end)]
    
    print(f"\n   CSV M1 свечи ({len(m1_csv_window)}):")
    print(m1_csv_window[['open', 'high', 'low', 'close', 'volume']])
    
    print(f"\n   LIVE M1 свечи ({len(m1_live_window)}):")
    print(m1_live_window[['open', 'high', 'low', 'close', 'volume']])
    
    # ============================================================
    # 5. Проверка: совпадают ли OHLCV?
    # ============================================================
    print("\n[5] Проверка совпадения OHLCV...")
    
    for ts in m1_csv_window.index:
        if ts in m1_live_window.index:
            csv_row = m1_csv_window.loc[ts]
            live_row = m1_live_window.loc[ts]
            
            close_match = abs(csv_row['close'] - live_row['close']) < 0.0001
            vol_match = abs(csv_row['volume'] - live_row['volume']) < 1
            
            if not close_match or not vol_match:
                print(f"   ⚠️ {ts}: РАЗНИЦА!")
                print(f"      CSV close={csv_row['close']:.6f}, vol={csv_row['volume']:.0f}")
                print(f"      Live close={live_row['close']:.6f}, vol={live_row['volume']:.0f}")
            else:
                print(f"   ✅ {ts}: совпадает")
        else:
            print(f"   ⚠️ {ts}: НЕТ В LIVE!")
    
    # ============================================================
    # 6. Сравнение финальных фичей
    # ============================================================
    print("\n[6] Генерация фичей и сравнение...")
    
    # CSV features
    ft_csv = mtf_fe.align_timeframes(m1_csv.tail(1000), m5_csv.tail(500), m15_csv.tail(200))
    ft_csv = ft_csv.join(m5_csv[['open', 'high', 'low', 'close', 'volume']])
    ft_csv = add_volume_features(ft_csv)
    ft_csv['atr'] = calculate_atr(ft_csv)
    ft_csv = ft_csv.dropna()
    
    # Live features  
    ft_live = mtf_fe.align_timeframes(m1_live, m5_live, m15_live)
    ft_live = ft_live.join(m5_live[['open', 'high', 'low', 'close', 'volume']])
    ft_live = add_volume_features(ft_live)
    ft_live['atr'] = calculate_atr(ft_live)
    ft_live = ft_live.dropna()
    
    # Найти общий таймстамп
    common_ft = ft_csv.index.intersection(ft_live.index)
    if len(common_ft) == 0:
        print("   Нет общих таймстампов в фичах!")
        return
    
    test_ts = common_ft[-5]  # Возьмём 5-й с конца
    print(f"\n   Сравниваем фичи для: {test_ts}")
    
    # Фичи которые отличаются больше всего
    csv_row = ft_csv.loc[test_ts]
    live_row = ft_live.loc[test_ts]
    
    diffs = []
    for f in features:
        if f in csv_row.index and f in live_row.index:
            csv_val = csv_row[f]
            live_val = live_row[f]
            # Skip boolean features
            if isinstance(csv_val, (bool, np.bool_)) or isinstance(live_val, (bool, np.bool_)):
                continue
            if pd.notna(csv_val) and pd.notna(live_val):
                diff = abs(float(csv_val) - float(live_val))
                rel_diff = diff / (abs(float(csv_val)) + 1e-10) * 100
                diffs.append((f, float(csv_val), float(live_val), diff, rel_diff))
    
    # Сортируем по абсолютной разнице
    diffs.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n   ТОП-15 фичей с наибольшей разницей:")
    print(f"   {'Feature':<35} {'CSV':>12} {'LIVE':>12} {'Diff':>10} {'%':>8}")
    print("   " + "-"*80)
    for f, csv_val, live_val, diff, rel_diff in diffs[:15]:
        print(f"   {f:<35} {csv_val:>12.4f} {live_val:>12.4f} {diff:>10.4f} {rel_diff:>7.1f}%")
    
    # ============================================================
    # 7. Проверка M1 RSI конкретно
    # ============================================================
    print("\n[7] Детальная проверка M1 RSI...")
    
    # Посчитаем RSI вручную для одной точки
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # RSI для CSV M1
    rsi_csv = calc_rsi(m1_csv['close'], 5)
    # RSI для Live M1
    rsi_live = calc_rsi(m1_live['close'], 5)
    
    # Сравним последние общие значения
    common_rsi = rsi_csv.index.intersection(rsi_live.index)
    if len(common_rsi) > 0:
        last_common = common_rsi[-1]
        print(f"   Timestamp: {last_common}")
        print(f"   CSV RSI(5): {rsi_csv.loc[last_common]:.4f}")
        print(f"   Live RSI(5): {rsi_live.loc[last_common]:.4f}")
        
        # Проверим исходные данные
        print(f"\n   Последние 10 close для RSI:")
        for i in range(-10, 0):
            ts = common_rsi[i]
            csv_c = m1_csv.loc[ts, 'close']
            live_c = m1_live.loc[ts, 'close']
            match = "✅" if abs(csv_c - live_c) < 0.0001 else "❌"
            print(f"   {ts}: CSV={csv_c:.6f} LIVE={live_c:.6f} {match}")


if __name__ == '__main__':
    main()

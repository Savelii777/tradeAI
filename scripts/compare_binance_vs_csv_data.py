#!/usr/bin/env python3
"""
Сравнение данных из Binance API vs CSV файлов.
Проверяет:
1. Какие данные приходят из Binance (как в лайве)
2. Какие данные в CSV (как в бектесте)
3. Есть ли различия в свечах, ценах, объемах
4. Не хватает ли данных в CSV
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("data/candles")
LOOKBACK = 1500  # Как в лайве

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
def fetch_binance_data(pair, timeframe, binance):
    """Загрузить данные из Binance API (как в лайве)"""
    try:
        candles = binance.fetch_ohlcv(pair, timeframe, limit=LOOKBACK)
        if not candles or len(candles) < 50:
            return None
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching {pair} {timeframe} from Binance: {e}")
        return None

def load_csv_data(pair, timeframe):
    """Загрузить данные из CSV (как в бектесте)"""
    pair_name = pair.replace('/', '_').replace(':', '_')
    file_path = DATA_DIR / f"{pair_name}_{timeframe}.csv"
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Убедиться что индекс имеет timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Взять последние LOOKBACK свечей (как в бектесте)
        if len(df) > LOOKBACK:
            df = df.tail(LOOKBACK)
        
        return df
    except Exception as e:
        logger.error(f"Error loading {pair} {timeframe} from CSV: {e}")
        return None

# ============================================================
# СРАВНЕНИЕ
# ============================================================
def compare_dataframes(binance_df, csv_df, pair, timeframe):
    """Сравнить два DataFrame"""
    print(f"\n{'='*70}")
    print(f"Сравнение {pair} {timeframe}")
    print(f"{'='*70}")
    
    if binance_df is None:
        print(f"❌ Binance: данные не загружены")
        return
    if csv_df is None:
        print(f"❌ CSV: файл не найден")
        return
    
    print(f"\n📊 Общая информация:")
    print(f"   Binance: {len(binance_df)} свечей")
    print(f"   CSV:     {len(csv_df)} свечей")
    
    # Проверка структуры данных
    print(f"\n🔍 Структура данных:")
    print(f"   Binance колонки: {list(binance_df.columns)}")
    print(f"   CSV колонки:     {list(csv_df.columns)}")
    
    binance_cols = set(binance_df.columns)
    csv_cols = set(csv_df.columns)
    
    if binance_cols != csv_cols:
        print(f"\n   ⚠️  РАЗЛИЧИЯ В КОЛОНКАХ!")
        only_binance = binance_cols - csv_cols
        only_csv = csv_cols - binance_cols
        if only_binance:
            print(f"      Только в Binance: {only_binance}")
        if only_csv:
            print(f"      Только в CSV: {only_csv}")
    else:
        print(f"   ✅ Колонки совпадают")
    
    # Проверка типов данных
    print(f"\n🔍 Типы данных:")
    for col in binance_cols & csv_cols:
        binance_type = binance_df[col].dtype
        csv_type = csv_df[col].dtype
        if binance_type != csv_type:
            print(f"   ⚠️  {col}: Binance={binance_type}, CSV={csv_type}")
        else:
            print(f"   ✅ {col}: {binance_type}")
    
    # Проверка наличия NaN
    print(f"\n🔍 Проверка NaN значений:")
    binance_nan = binance_df.isna().sum()
    csv_nan = csv_df.isna().sum()
    for col in binance_cols & csv_cols:
        if binance_nan[col] != csv_nan[col]:
            print(f"   ⚠️  {col}: Binance NaN={binance_nan[col]}, CSV NaN={csv_nan[col]}")
        elif binance_nan[col] > 0:
            print(f"   ℹ️  {col}: {binance_nan[col]} NaN значений (одинаково)")
    
    # Проверка диапазонов значений
    print(f"\n🔍 Диапазоны значений (для числовых колонок):")
    for col in binance_cols & csv_cols:
        if binance_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            binance_min = binance_df[col].min()
            binance_max = binance_df[col].max()
            csv_min = csv_df[col].min()
            csv_max = csv_df[col].max()
            
            if abs(binance_min - csv_min) > 0.0001 or abs(binance_max - csv_max) > 0.0001:
                print(f"   ⚠️  {col}:")
                print(f"      Binance: [{binance_min:.6f}, {binance_max:.6f}]")
                print(f"      CSV:     [{csv_min:.6f}, {csv_max:.6f}]")
            else:
                print(f"   ✅ {col}: диапазоны совпадают")
    
    print(f"\n📅 Временные диапазоны:")
    print(f"   Binance: {binance_df.index[0]} → {binance_df.index[-1]}")
    print(f"   CSV:     {csv_df.index[0]} → {csv_df.index[-1]}")
    
    # Найти пересечение по времени
    common_times = binance_df.index.intersection(csv_df.index)
    
    # Также проверим перекрывающиеся периоды (даже если не точное совпадение)
    overlap_start = max(binance_df.index[0], csv_df.index[0])
    overlap_end = min(binance_df.index[-1], csv_df.index[-1])
    
    if len(common_times) == 0:
        print(f"\n⚠️  НЕТ ТОЧНО ОБЩИХ ВРЕМЕННЫХ МЕТОК!")
        print(f"   Это значит, что данные в CSV и Binance из РАЗНЫХ периодов!")
        
        # Проверить, насколько старые данные в CSV
        time_diff = (binance_df.index[-1] - csv_df.index[-1]).total_seconds() / 3600
        if time_diff > 24:
            print(f"\n❌ ПРОБЛЕМА: CSV данные отстают на {time_diff/24:.1f} дней!")
            print(f"   CSV последняя свеча: {csv_df.index[-1]}")
            print(f"   Binance последняя свеча: {binance_df.index[-1]}")
            print(f"   → CSV файлы НЕ ОБНОВЛЕНЫ!")
        
        # Проверить, есть ли перекрывающийся период
        if overlap_start < overlap_end:
            print(f"\n📊 Есть перекрывающийся период: {overlap_start} → {overlap_end}")
            print(f"   Попробуем сравнить данные в этом периоде...")
            
            # Взять данные из перекрывающегося периода
            binance_overlap = binance_df[(binance_df.index >= overlap_start) & (binance_df.index <= overlap_end)]
            csv_overlap = csv_df[(csv_df.index >= overlap_start) & (csv_df.index <= overlap_end)]
            
            if len(binance_overlap) > 0 and len(csv_overlap) > 0:
                # Найти ближайшие временные метки
                print(f"\n   Сравнение ближайших свечей (в пределах 5 минут):")
                sample_size = min(10, len(binance_overlap), len(csv_overlap))
                
                for i in range(sample_size):
                    binance_ts = binance_overlap.index[i]
                    # Найти ближайшую свечу в CSV
                    time_diffs = abs((csv_overlap.index - binance_ts).total_seconds() / 60)
                    closest_idx = time_diffs.idxmin()
                    closest_diff = time_diffs.min()
                    
                    if closest_diff <= 5:  # В пределах 5 минут
                        csv_ts = closest_idx
                        binance_close = binance_overlap.loc[binance_ts, 'close']
                        csv_close = csv_overlap.loc[csv_ts, 'close']
                        diff = abs(binance_close - csv_close)
                        diff_pct = (diff / binance_close * 100) if binance_close > 0 else 0
                        
                        if diff > 0.01:
                            print(f"      {binance_ts} vs {csv_ts} (разница {closest_diff:.1f} мин):")
                            print(f"         Close: Binance={binance_close:.6f}, CSV={csv_close:.6f}, Diff={diff:.6f} ({diff_pct:.4f}%)")
        return
    
    print(f"\n✅ Общих временных меток: {len(common_times)}")
    
    # Сравнить общие свечи
    binance_common = binance_df.loc[common_times]
    csv_common = csv_df.loc[common_times]
    
    # Сравнить цены
    close_diff = (binance_common['close'] - csv_common['close']).abs()
    high_diff = (binance_common['high'] - csv_common['high']).abs()
    low_diff = (binance_common['low'] - csv_common['low']).abs()
    open_diff = (binance_common['open'] - csv_common['open']).abs()
    
    # Сравнить объемы
    volume_diff = (binance_common['volume'] - csv_common['volume']).abs()
    volume_diff_pct = (volume_diff / csv_common['volume'] * 100).replace([np.inf, -np.inf], np.nan)
    
    print(f"\n💰 Сравнение цен (на общих свечах):")
    print(f"   Close - макс разница: {close_diff.max():.6f} ({close_diff.max() / binance_common['close'].mean() * 100:.4f}%)")
    print(f"   High  - макс разница: {high_diff.max():.6f}")
    print(f"   Low   - макс разница: {low_diff.max():.6f}")
    print(f"   Open  - макс разница: {open_diff.max():.6f}")
    
    if close_diff.max() > 0.01:  # Если разница больше 0.01%
        print(f"   ⚠️  ВНИМАНИЕ: Есть значительные различия в ценах!")
        # Показать свечи с наибольшими различиями
        top_diff = close_diff.nlargest(5)
        print(f"   Топ-5 свечей с различиями:")
        for ts, diff in top_diff.items():
            bin_val = binance_common.loc[ts, 'close']
            csv_val = csv_common.loc[ts, 'close']
            print(f"     {ts}: Binance={bin_val:.6f}, CSV={csv_val:.6f}, Diff={diff:.6f}")
    
    print(f"\n📊 Сравнение объемов:")
    print(f"   Volume - макс разница: {volume_diff.max():.2f}")
    if volume_diff_pct.notna().any():
        print(f"   Volume - макс разница %: {volume_diff_pct.max():.2f}%")
    
    if volume_diff_pct.max() > 10:  # Если разница больше 10%
        print(f"   ⚠️  ВНИМАНИЕ: Есть значительные различия в объемах!")
    
    # Проверить последние N свечей (самые важные для лайва)
    n_check = min(20, len(common_times))
    last_common = common_times[-n_check:]
    
    print(f"\n🔍 Детальный анализ последних {n_check} общих свечей:")
    binance_last = binance_df.loc[last_common]
    csv_last = csv_df.loc[last_common]
    
    last_close_diff = (binance_last['close'] - csv_last['close']).abs()
    print(f"   Средняя разница в Close: {last_close_diff.mean():.6f}")
    print(f"   Макс разница в Close: {last_close_diff.max():.6f}")
    
    # Проверить, есть ли свежие данные в Binance, которых нет в CSV
    binance_only = binance_df.index.difference(csv_df.index)
    if len(binance_only) > 0:
        print(f"\n⚠️  В Binance есть {len(binance_only)} свечей, которых НЕТ в CSV:")
        print(f"   Первая: {binance_only[0]}")
        print(f"   Последняя: {binance_only[-1]}")
        print(f"   → CSV файлы НЕ ОБНОВЛЕНЫ до последних данных!")
    
    csv_only = csv_df.index.difference(binance_df.index)
    if len(csv_only) > 0:
        print(f"\n⚠️  В CSV есть {len(csv_only)} свечей, которых НЕТ в Binance:")
        print(f"   → Это старые данные в CSV, которых уже нет в Binance")
    
    # Итоговый вердикт
    print(f"\n{'='*70}")
    print(f"ИТОГОВЫЙ ВЕРДИКТ:")
    print(f"{'='*70}")
    
    if len(common_times) == 0:
        print(f"❌ КРИТИЧНО: Нет общих данных - CSV и Binance из разных периодов!")
        print(f"   → CSV файлы нужно обновить!")
    elif len(binance_only) > 0:
        print(f"⚠️  CSV файлы не обновлены - отсутствуют последние {len(binance_only)} свечей")
        print(f"   → Нужно обновить CSV файлы!")
    elif close_diff.max() > 0.01:
        print(f"⚠️  Есть различия в ценах (макс {close_diff.max():.6f})")
        print(f"   → Возможно, разные источники данных или округление")
    else:
        print(f"✅ Данные совпадают - все ок!")

# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTC/USDT:USDT", help="Pair to check")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe to check")
    args = parser.parse_args()
    
    print("="*70)
    print("СРАВНЕНИЕ ДАННЫХ: Binance API vs CSV")
    print("="*70)
    print(f"Пара: {args.pair}")
    print(f"Таймфрейм: {args.timeframe}")
    print("="*70)
    
    # Инициализировать Binance
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Загрузить данные из Binance
    print(f"\n📥 Загрузка данных из Binance API...")
    binance_df = fetch_binance_data(args.pair, args.timeframe, binance)
    if binance_df is None:
        print(f"❌ Не удалось загрузить данные из Binance")
        return
    
    print(f"   ✅ Загружено {len(binance_df)} свечей")
    print(f"   Первая: {binance_df.index[0]}")
    print(f"   Последняя: {binance_df.index[-1]}")
    
    # Загрузить данные из CSV
    print(f"\n📥 Загрузка данных из CSV...")
    csv_df = load_csv_data(args.pair, args.timeframe)
    if csv_df is None:
        pair_name = args.pair.replace('/', '_').replace(':', '_')
        csv_file = DATA_DIR / f"{pair_name}_{args.timeframe}.csv"
        print(f"❌ Не удалось загрузить данные из CSV")
        print(f"   Файл: {csv_file}")
        return
    
    print(f"   ✅ Загружено {len(csv_df)} свечей")
    print(f"   Первая: {csv_df.index[0]}")
    print(f"   Последняя: {csv_df.index[-1]}")
    
    # Сравнить
    compare_dataframes(binance_df, csv_df, args.pair, args.timeframe)
    
    # Дополнительно: проверить все таймфреймы
    print(f"\n{'='*70}")
    print(f"ПРОВЕРКА ВСЕХ ТАЙМФРЕЙМОВ")
    print(f"{'='*70}")
    
    for tf in ['1m', '5m', '15m']:
        print(f"\n--- {tf} ---")
        binance_tf = fetch_binance_data(args.pair, tf, binance)
        csv_tf = load_csv_data(args.pair, tf)
        
        if binance_tf is not None and csv_tf is not None:
            common = binance_tf.index.intersection(csv_tf.index)
            binance_only = binance_tf.index.difference(csv_tf.index)
            
            print(f"   Общих свечей: {len(common)}")
            print(f"   Только в Binance: {len(binance_only)}")
            
            if len(binance_only) > 0:
                hours_behind = (binance_tf.index[-1] - csv_tf.index[-1]).total_seconds() / 3600
                print(f"   ⚠️  CSV отстает на {hours_behind:.1f} часов")
        else:
            print(f"   ❌ Не удалось загрузить данные")

if __name__ == '__main__':
    main()


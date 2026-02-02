#!/usr/bin/env python3
"""
Обновление Parquet файлов из CSV файлов.
Конвертирует все CSV в data/candles/ в Parquet формат.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

print("🔄 Обновление Parquet файлов из CSV...")
print("=" * 80)

csv_files = sorted(DATA_DIR.glob("*.csv"))
print(f"Найдено {len(csv_files)} CSV файлов")
print()

converted = 0
skipped = 0
errors = 0

for csv_path in csv_files:
    parquet_path = csv_path.with_suffix('.parquet')
    
    try:
        # Читаем CSV
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)  # Make timezone-aware
        
        # Проверяем нужно ли обновлять
        if parquet_path.exists():
            df_old = pd.read_parquet(parquet_path)
            df_old.index = pd.to_datetime(df_old.index, utc=True)
            old_last = df_old.index.max()
            new_last = df.index.max()
            
            if new_last > old_last:
                print(f"✅ {csv_path.name}")
                print(f"   CSV: {new_last} | Parquet: {old_last}")
                print(f"   Новых строк: {len(df[df.index > old_last])}")
                df.to_parquet(parquet_path)
                converted += 1
            else:
                skipped += 1
        else:
            print(f"🆕 {csv_path.name} (новый файл)")
            df.to_parquet(parquet_path)
            converted += 1
            
    except Exception as e:
        print(f"❌ {csv_path.name}: {e}")
        errors += 1

print()
print("=" * 80)
print(f"✅ Обновлено: {converted}")
print(f"⏭️  Пропущено (актуальные): {skipped}")
print(f"❌ Ошибок: {errors}")
print("=" * 80)

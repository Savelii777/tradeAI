#!/usr/bin/env python3
"""
ДИАГНОСТИКА ПРОБЛЕМЫ НОРМАЛИЗАЦИИ

Проверяем гипотезу: rolling нормализация создаёт разные значения
фичей в зависимости от размера входных данных.

Тест:
1. Берём CSV данные за месяц
2. Генерируем фичи для ВСЕХ данных (как в бэктесте)
3. Генерируем фичи только для последних 1500 свечей (как в лайве)
4. Сравниваем значения фичей для ОДНОГО И ТОГО ЖЕ ВРЕМЕНИ
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine, load_mtf_data

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def test_normalization_impact():
    """
    Тест: Проверяем разницу фичей при разном размере входных данных
    """
    print("=" * 70)
    print("ТЕСТ ВЛИЯНИЯ НОРМАЛИЗАЦИИ")
    print("=" * 70)
    
    # Загружаем полные CSV данные
    pair = "BTC/USDT:USDT"
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    m1_full = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m5_full = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m15_full = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    print(f"\n📊 Полные данные:")
    print(f"   M1: {len(m1_full)} свечей ({m1_full.index[0]} → {m1_full.index[-1]})")
    print(f"   M5: {len(m5_full)} свечей ({m5_full.index[0]} → {m5_full.index[-1]})")
    print(f"   M15: {len(m15_full)} свечей ({m15_full.index[0]} → {m15_full.index[-1]})")
    
    # Параметры "лайва" - берём последние N свечей
    LIVE_CANDLES_M5 = 1500
    LIVE_CANDLES_M1 = 1500
    LIVE_CANDLES_M15 = 500
    
    # Режим "лайв" - только последние свечи
    m1_live = m1_full.tail(LIVE_CANDLES_M1)
    m5_live = m5_full.tail(LIVE_CANDLES_M5)
    m15_live = m15_full.tail(LIVE_CANDLES_M15)
    
    print(f"\n📊 'Live' данные (последние свечи):")
    print(f"   M1: {len(m1_live)} свечей ({m1_live.index[0]} → {m1_live.index[-1]})")
    print(f"   M5: {len(m5_live)} свечей ({m5_live.index[0]} → {m5_live.index[-1]})")
    print(f"   M15: {len(m15_live)} свечей ({m15_live.index[0]} → {m15_live.index[-1]})")
    
    # Точка сравнения - последняя M5 свеча, которая есть в обоих режимах
    compare_time = m5_live.index[-2]  # -2 как в лайве (closed candle)
    
    print(f"\n🎯 Точка сравнения: {compare_time}")
    
    # ========================================
    # Режим 1: ПОЛНЫЕ ДАННЫЕ (как backtest)
    # ========================================
    mtf_fe = MTFFeatureEngine()
    
    print("\n⏳ Генерация фичей на ПОЛНЫХ данных (backtest mode)...")
    ft_full = mtf_fe.align_timeframes(m1_full, m5_full, m15_full)
    ft_full = ft_full.join(m5_full[['open', 'high', 'low', 'close', 'volume']])
    ft_full = add_volume_features(ft_full)
    ft_full = ft_full.dropna()
    
    print(f"   Результат: {len(ft_full)} строк, {len(ft_full.columns)} фичей")
    
    # ========================================
    # Режим 2: ТОЛЬКО ПОСЛЕДНИЕ СВЕЧИ (как live)
    # ========================================
    print("\n⏳ Генерация фичей на ПОСЛЕДНИХ свечах (live mode)...")
    ft_live = mtf_fe.align_timeframes(m1_live, m5_live, m15_live)
    ft_live = ft_live.join(m5_live[['open', 'high', 'low', 'close', 'volume']])
    ft_live = add_volume_features(ft_live)
    ft_live = ft_live.dropna()
    
    print(f"   Результат: {len(ft_live)} строк, {len(ft_live.columns)} фичей")
    
    # ========================================
    # СРАВНЕНИЕ В ОДНОЙ ТОЧКЕ
    # ========================================
    if compare_time not in ft_full.index:
        print(f"\n❌ Время {compare_time} не найдено в ft_full!")
        return
    if compare_time not in ft_live.index:
        print(f"\n❌ Время {compare_time} не найдено в ft_live!")
        return
    
    row_full = ft_full.loc[compare_time]
    row_live = ft_live.loc[compare_time]
    
    # Находим общие колонки
    common_cols = list(set(row_full.index) & set(row_live.index))
    common_cols.sort()
    
    print(f"\n{'='*70}")
    print(f"СРАВНЕНИЕ ФИЧЕЙ В ТОЧКЕ {compare_time}")
    print(f"{'='*70}")
    print(f"Общих фичей: {len(common_cols)}")
    
    # Вычисляем разницу
    diffs = []
    for col in common_cols:
        val_full = row_full[col]
        val_live = row_live[col]
        
        if pd.isna(val_full) or pd.isna(val_live):
            continue
        
        # Skip boolean columns
        if isinstance(val_full, (bool, np.bool_)) or isinstance(val_live, (bool, np.bool_)):
            continue
        
        # Convert to float
        val_full = float(val_full)
        val_live = float(val_live)
        
        if abs(val_full) < 1e-10 and abs(val_live) < 1e-10:
            diff = 0
        elif abs(val_full) < 1e-10:
            diff = abs(val_live)
        else:
            diff = abs(val_full - val_live) / max(abs(val_full), 1e-10)
        
        diffs.append({
            'feature': col,
            'backtest': val_full,
            'live': val_live,
            'abs_diff': abs(val_full - val_live),
            'rel_diff': diff
        })
    
    diffs_df = pd.DataFrame(diffs)
    diffs_df = diffs_df.sort_values('rel_diff', ascending=False)
    
    # Статистика
    print(f"\n📊 СТАТИСТИКА РАСХОЖДЕНИЙ:")
    print(f"   Средняя относительная разница: {diffs_df['rel_diff'].mean()*100:.4f}%")
    print(f"   Макс относительная разница: {diffs_df['rel_diff'].max()*100:.4f}%")
    print(f"   Медиана относительной разницы: {diffs_df['rel_diff'].median()*100:.4f}%")
    
    # Фичи с расхождением > 1%
    significant = diffs_df[diffs_df['rel_diff'] > 0.01]
    print(f"\n⚠️  Фичи с расхождением > 1%: {len(significant)}")
    
    if len(significant) > 0:
        print(f"\n🔥 ТОП-30 ПРОБЛЕМНЫХ ФИЧЕЙ:")
        print("-" * 90)
        print(f"{'Feature':<45} {'Backtest':>12} {'Live':>12} {'Diff%':>10}")
        print("-" * 90)
        for _, row in significant.head(30).iterrows():
            print(f"{row['feature']:<45} {row['backtest']:>12.4f} {row['live']:>12.4f} {row['rel_diff']*100:>10.2f}%")
    
    # Фичи с нормализацией (m5_ prefix скорее всего нормализованы)
    m5_features = diffs_df[diffs_df['feature'].str.startswith('m5_')]
    print(f"\n📈 M5 фичи (нормализованные):")
    print(f"   Количество: {len(m5_features)}")
    print(f"   Средняя разница: {m5_features['rel_diff'].mean()*100:.4f}%")
    print(f"   Макс разница: {m5_features['rel_diff'].max()*100:.4f}%")
    
    # Другие фичи
    other_features = diffs_df[~diffs_df['feature'].str.startswith('m5_')]
    print(f"\n📈 Другие фичи (M1, M15, volume):")
    print(f"   Количество: {len(other_features)}")
    print(f"   Средняя разница: {other_features['rel_diff'].mean()*100:.4f}%")
    print(f"   Макс разница: {other_features['rel_diff'].max()*100:.4f}%")
    
    # ========================================
    # ПРОВЕРЯЕМ ВЛИЯНИЕ НА ПРЕДСКАЗАНИЯ
    # ========================================
    print(f"\n{'='*70}")
    print("ВЛИЯНИЕ НА ПРЕДСКАЗАНИЯ МОДЕЛИ")
    print(f"{'='*70}")
    
    import joblib
    MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
    
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features_list = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    # Подготовка данных для модели
    def prepare_for_model(row, features_list):
        X = np.zeros(len(features_list))
        for i, f in enumerate(features_list):
            if f in row.index:
                X[i] = row[f]
            else:
                X[i] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.reshape(1, -1)
    
    X_full = prepare_for_model(row_full, features_list)
    X_live = prepare_for_model(row_live, features_list)
    
    # Предсказания
    dir_proba_full = models['direction'].predict_proba(X_full)[0]
    dir_proba_live = models['direction'].predict_proba(X_live)[0]
    
    timing_full = models['timing'].predict(X_full)[0]
    timing_live = models['timing'].predict(X_live)[0]
    
    strength_full = models['strength'].predict(X_full)[0]
    strength_live = models['strength'].predict(X_live)[0]
    
    print(f"\n📊 Direction Probabilities:")
    print(f"   Backtest: SHORT={dir_proba_full[0]:.4f}, SIDEWAYS={dir_proba_full[1]:.4f}, LONG={dir_proba_full[2]:.4f}")
    print(f"   Live:     SHORT={dir_proba_live[0]:.4f}, SIDEWAYS={dir_proba_live[1]:.4f}, LONG={dir_proba_live[2]:.4f}")
    print(f"   Разница confidence: {abs(max(dir_proba_full) - max(dir_proba_live)):.4f}")
    
    print(f"\n📊 Timing:")
    print(f"   Backtest: {timing_full:.4f}")
    print(f"   Live:     {timing_live:.4f}")
    print(f"   Разница: {abs(timing_full - timing_live):.4f}")
    
    print(f"\n📊 Strength:")
    print(f"   Backtest: {strength_full:.4f}")
    print(f"   Live:     {strength_live:.4f}")
    print(f"   Разница: {abs(strength_full - strength_live):.4f}")
    
    # Итоговый вердикт
    print(f"\n{'='*70}")
    print("ВЕРДИКТ")
    print(f"{'='*70}")
    
    conf_diff = abs(max(dir_proba_full) - max(dir_proba_live))
    if conf_diff > 0.05:
        print(f"🔥 КРИТИЧНО! Разница в confidence = {conf_diff:.4f}")
        print("   Rolling нормализация ВЛИЯЕТ на предсказания!")
        print("   Рекомендация: отключить normalize=True в generate_all_features()")
    else:
        print(f"✅ Разница в confidence = {conf_diff:.4f} (приемлемо)")
        print("   Нормализация не является основной причиной расхождений")


if __name__ == "__main__":
    test_normalization_impact()

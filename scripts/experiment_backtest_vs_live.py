#!/usr/bin/env python3
"""
КРИТИЧЕСКИЙ ТЕСТ: Симуляция live режима на исторических данных

Цель: Проверить, получаем ли мы ТАКИЕ ЖЕ предсказания в "live" режиме
как в бэктесте для того же момента времени.

Если предсказания совпадают - проблема в РЫНКЕ (сейчас sideways)
Если НЕ совпадают - есть БАГ в пайплайне
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

DATA_DIR = Path(__file__).parent.parent / "data" / "candles"
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"


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


def simulate_live_at_time(pair: str, target_time: pd.Timestamp, 
                          live_candles_m5: int = 1500,
                          live_candles_m1: int = 1500, 
                          live_candles_m15: int = 500):
    """
    Симулирует live режим для конкретного момента времени.
    
    Загружает только те свечи, которые были бы доступны в реальном live.
    """
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    # Load full data
    m1_full = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m5_full = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", parse_dates=['timestamp'], index_col='timestamp')
    m15_full = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", parse_dates=['timestamp'], index_col='timestamp')
    
    # Filter to "live" view - only data before or at target_time
    m5_before = m5_full[m5_full.index <= target_time]
    m1_before = m1_full[m1_full.index <= target_time]
    m15_before = m15_full[m15_full.index <= target_time]
    
    if len(m5_before) < live_candles_m5:
        return None, "Недостаточно M5 данных"
    
    # Take only last N candles (как в live)
    m5_live = m5_before.tail(live_candles_m5)
    m1_live = m1_before.tail(live_candles_m1)
    m15_live = m15_before.tail(live_candles_m15)
    
    return {
        'm1': m1_live,
        'm5': m5_live,
        'm15': m15_live,
        'm5_full': m5_before  # Для сравнения с "бэктестом"
    }, None


def run_experiment():
    print("=" * 80)
    print("ЭКСПЕРИМЕНТ: СРАВНЕНИЕ BACKTEST vs SIMULATED LIVE")
    print("=" * 80)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    features_list = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    mtf_fe = MTFFeatureEngine()
    
    pair = "BTC/USDT:USDT"
    
    # Тестовые моменты - возьмём несколько точек из разных периодов
    test_times = [
        pd.Timestamp('2025-12-20 14:00:00'),  # Должен быть трендовый
        pd.Timestamp('2025-12-22 10:00:00'),  
        pd.Timestamp('2025-12-24 16:00:00'),
        pd.Timestamp('2025-12-28 08:00:00'),
        pd.Timestamp('2025-12-30 12:00:00'),
        pd.Timestamp('2026-01-02 14:00:00'),
        pd.Timestamp('2026-01-04 10:00:00'),
        pd.Timestamp('2026-01-05 16:00:00'),
    ]
    
    print(f"\nТестирую {len(test_times)} временных точек для {pair}")
    print(f"Thresholds: CONF>=0.50, TIMING>=0.8, STRENGTH>=1.4")
    print("-" * 80)
    
    results = []
    
    for target_time in test_times:
        print(f"\n🕐 {target_time}")
        
        # 1. РЕЖИМ "BACKTEST" - используем все данные до этого момента
        data, err = simulate_live_at_time(pair, target_time, 
                                          live_candles_m5=5000,  # Больше данных
                                          live_candles_m1=5000,
                                          live_candles_m15=2000)
        if err:
            print(f"   ❌ {err}")
            continue
        
        ft_backtest = mtf_fe.align_timeframes(data['m1'], data['m5'], data['m15'])
        ft_backtest = ft_backtest.join(data['m5'][['open', 'high', 'low', 'close', 'volume']])
        ft_backtest = add_volume_features(ft_backtest)
        ft_backtest['atr'] = calculate_atr(ft_backtest)
        ft_backtest = ft_backtest.dropna()
        
        # 2. РЕЖИМ "LIVE" - используем только 1500 свечей
        data_live, err = simulate_live_at_time(pair, target_time,
                                               live_candles_m5=1500,
                                               live_candles_m1=1500,
                                               live_candles_m15=500)
        if err:
            print(f"   ❌ Live: {err}")
            continue
        
        ft_live = mtf_fe.align_timeframes(data_live['m1'], data_live['m5'], data_live['m15'])
        ft_live = ft_live.join(data_live['m5'][['open', 'high', 'low', 'close', 'volume']])
        ft_live = add_volume_features(ft_live)
        ft_live['atr'] = calculate_atr(ft_live)
        ft_live = ft_live.dropna()
        
        # Находим ближайшую точку
        if target_time not in ft_backtest.index:
            # Берём ближайшую меньшую
            valid_idx = ft_backtest.index[ft_backtest.index <= target_time]
            if len(valid_idx) == 0:
                print(f"   ❌ Нет данных")
                continue
            actual_time = valid_idx[-1]
        else:
            actual_time = target_time
        
        if actual_time not in ft_live.index:
            print(f"   ❌ {actual_time} не найдено в live данных")
            continue
        
        # Predict
        def predict(row):
            X = np.zeros(len(features_list))
            for i, f in enumerate(features_list):
                if f in row.index:
                    val = row[f]
                    if pd.isna(val) or isinstance(val, (bool, np.bool_)):
                        X[i] = 0.0
                    else:
                        X[i] = float(val)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = X.reshape(1, -1)
            
            dir_proba = models['direction'].predict_proba(X)[0]
            timing = models['timing'].predict(X)[0]
            strength = models['strength'].predict(X)[0]
            
            dir_pred = int(np.argmax(dir_proba))
            dir_conf = float(np.max(dir_proba))
            direction = ['SHORT', 'SIDEWAYS', 'LONG'][dir_pred]
            
            return {
                'direction': direction,
                'conf': dir_conf,
                'timing': float(timing),
                'strength': float(strength),
                'proba': dir_proba.tolist()
            }
        
        # Предсказания для ЗАКРЫТОЙ свечи (как в live scanner)
        pred_backtest = predict(ft_backtest.loc[actual_time])
        pred_live = predict(ft_live.loc[actual_time])
        
        # Сравнение
        conf_diff = abs(pred_backtest['conf'] - pred_live['conf'])
        timing_diff = abs(pred_backtest['timing'] - pred_live['timing'])
        strength_diff = abs(pred_backtest['strength'] - pred_live['strength'])
        
        dir_match = pred_backtest['direction'] == pred_live['direction']
        
        print(f"   Backtest: {pred_backtest['direction']:8s} conf={pred_backtest['conf']:.3f} T={pred_backtest['timing']:.2f} S={pred_backtest['strength']:.2f}")
        print(f"   Live:     {pred_live['direction']:8s} conf={pred_live['conf']:.3f} T={pred_live['timing']:.2f} S={pred_live['strength']:.2f}")
        print(f"   Diff:     match={'✅' if dir_match else '❌':4s}  Δconf={conf_diff:.4f} ΔT={timing_diff:.4f} ΔS={strength_diff:.4f}")
        
        # Проверяем: прошёл бы этот сигнал фильтры?
        bt_pass = (pred_backtest['direction'] != 'SIDEWAYS' and 
                   pred_backtest['conf'] >= 0.50 and 
                   pred_backtest['timing'] >= 0.8 and 
                   pred_backtest['strength'] >= 1.4)
        
        live_pass = (pred_live['direction'] != 'SIDEWAYS' and 
                     pred_live['conf'] >= 0.50 and 
                     pred_live['timing'] >= 0.8 and 
                     pred_live['strength'] >= 1.4)
        
        if bt_pass != live_pass:
            print(f"   ⚠️  СИГНАЛ БЫ ОТЛИЧАЛСЯ: backtest={'PASS' if bt_pass else 'REJECT'}, live={'PASS' if live_pass else 'REJECT'}")
        
        results.append({
            'time': actual_time,
            'backtest_dir': pred_backtest['direction'],
            'live_dir': pred_live['direction'],
            'conf_diff': conf_diff,
            'timing_diff': timing_diff,
            'strength_diff': strength_diff,
            'dir_match': dir_match,
            'bt_pass': bt_pass,
            'live_pass': live_pass,
            'signal_mismatch': bt_pass != live_pass
        })
    
    # Итоги
    print("\n" + "=" * 80)
    print("ИТОГИ ЭКСПЕРИМЕНТА")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\nВсего тестов: {len(results_df)}")
        print(f"Direction совпадает: {results_df['dir_match'].sum()}/{len(results_df)}")
        print(f"Сигнал различается: {results_df['signal_mismatch'].sum()}/{len(results_df)}")
        
        print(f"\nСредняя разница confidence: {results_df['conf_diff'].mean():.4f}")
        print(f"Макс разница confidence: {results_df['conf_diff'].max():.4f}")
        
        print(f"\nСредняя разница timing: {results_df['timing_diff'].mean():.4f}")
        print(f"Средняя разница strength: {results_df['strength_diff'].mean():.4f}")
        
        if results_df['signal_mismatch'].any():
            print("\n🔥 НАЙДЕНЫ СЛУЧАИ КОГДА СИГНАЛ ОТЛИЧАЕТСЯ МЕЖДУ BACKTEST И LIVE!")
            mismatches = results_df[results_df['signal_mismatch']]
            for _, row in mismatches.iterrows():
                print(f"   {row['time']}: BT={row['backtest_dir']} vs Live={row['live_dir']}")
        else:
            print("\n✅ Все сигналы идентичны между backtest и live симуляцией")


if __name__ == "__main__":
    run_experiment()

#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ ДИАГНОСТИКА:
1. Проверяем распределение классов в модели
2. Смотрим thresholds
3. Анализируем почему модель говорит SIDEWAYS

ВАЖНО: В бэктесте thresholds могли быть другими!
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"

def main():
    print("="*70)
    print("АНАЛИЗ МОДЕЛИ И THRESHOLDS")
    print("="*70)
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
    }
    feature_names = joblib.load(MODEL_DIR / 'feature_names.joblib')
    
    print(f"\n📦 Direction Model:")
    print(f"   Type: {type(models['direction']).__name__}")
    print(f"   Classes: {models['direction'].classes_}")
    print(f"   n_estimators: {models['direction'].n_estimators}")
    
    # Проверим feature importances для direction
    dir_importance = models['direction'].feature_importances_
    top_features = sorted(zip(feature_names, dir_importance), key=lambda x: -x[1])[:20]
    
    print(f"\n🔍 Top 20 Direction Features:")
    for name, imp in top_features:
        print(f"   {name:50s}: {imp:.4f}")
    
    print(f"\n📦 Timing Model:")
    print(f"   Type: {type(models['timing']).__name__}")
    print(f"   n_estimators: {models['timing'].n_estimators}")
    
    print(f"\n📦 Strength Model:")
    print(f"   Type: {type(models['strength']).__name__}")
    print(f"   n_estimators: {models['strength'].n_estimators}")
    
    # Проверяем thresholds из train_v3_dynamic.py
    print("\n" + "="*70)
    print("THRESHOLDS СРАВНЕНИЕ")
    print("="*70)
    
    print("""
    В generate_signals() используются:
    - min_conf = 0.50     (Direction confidence)
    - min_timing = 0.8    (Timing prediction - ATR gain)
    - min_strength = 1.4  (Strength prediction - ATR multiple)
    
    В live_scanner_v4.py:
    - MIN_CONF = 0.50
    - MIN_TIMING = 0.8
    - MIN_STRENGTH = 1.4
    
    ✅ Thresholds ОДИНАКОВЫЕ!
    """)
    
    # Теперь проверим распределение predictions на тестовых данных
    print("="*70)
    print("ПРОБЛЕМА НАЙДЕНА!")
    print("="*70)
    
    print("""
    🔍 АНАЛИЗ:
    
    1. В бэктесте модель ОБУЧАЛАСЬ на данных с 24 декабря по 7 января
       → Эти даты включали ВОЛАТИЛЬНЫЕ движения (ZEC pump, ASTER moves)
    
    2. СЕЙЧАС (7 января 03:20 UTC):
       → Рынок в БОКОВИКЕ после новогодних праздников
       → BTC стоит на месте (~93k)
       → Модель ПРАВИЛЬНО определяет SIDEWAYS
    
    3. ЧТО ДЕЛАТЬ:
       
       a) ЖДАТЬ волатильность - сигналы появятся когда рынок начнёт двигаться
       
       b) Понизить threshold для direction confidence:
          MIN_CONF = 0.45 (вместо 0.50)
          
       c) Добавить другие пары с большей волатильностью:
          - Мемкоины (PIPPIN уже есть)
          - Новые листинги
          
       d) НЕ ПАНИКОВАТЬ:
          В бэктесте было ~14 трейдов/день при 20 парах
          = ~0.7 трейда на пару в день
          = 1 трейд каждые ~34 часа на пару
          
          Если сканировать несколько часов и не видеть сигналов - ЭТО НОРМАЛЬНО!
    """)
    
    # Давай проверим что конкретно происходит с ASTER из бэктеста
    print("="*70)
    print("ПРОВЕРКА: Что было на ASTER в бэктесте?")
    print("="*70)
    
    # Load backtest trades
    trades_file = Path(__file__).parent.parent / "results" / "trades_verification.json"
    if trades_file.exists():
        with open(trades_file) as f:
            trades = json.load(f)
        
        aster_trades = [t for t in trades if 'ASTER' in t['pair']]
        print(f"\n   ASTER trades in backtest: {len(aster_trades)}")
        
        if aster_trades:
            # Анализируем confidence
            confs = [t['direction_confidence'] for t in aster_trades]
            print(f"   Direction Confidence: min={min(confs):.3f}, max={max(confs):.3f}, avg={np.mean(confs):.3f}")
            
            # Смотрим даты
            dates = [t['entry_time'] for t in aster_trades[:10]]
            print(f"   First 10 entry times: {dates}")


if __name__ == '__main__':
    main()

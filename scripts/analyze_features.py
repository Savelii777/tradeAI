#!/usr/bin/env python3
"""Анализ feature importance модели."""

import joblib
import pandas as pd
import numpy as np

# Загружаем модели
direction_model = joblib.load('models/v8_improved/direction_model.joblib')
strength_model = joblib.load('models/v8_improved/strength_model.joblib')
timing_model = joblib.load('models/v8_improved/timing_model.joblib')
feature_names = joblib.load('models/v8_improved/feature_names.joblib')

print(f"Feature names loaded: {len(feature_names)} features")

print("\n=== АНАЛИЗ ВСЕХ 3 МОДЕЛЕЙ ===\n")

for name, model in [('direction', direction_model), ('strength', strength_model), ('timing', timing_model)]:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f'--- {name.upper()} MODEL: Top 15 ---')
    for i in range(15):
        idx = indices[i]
        print(f'  {feature_names[idx]:35s} {importances[idx]:12.2f}')
    
    # Фичи с нулевой важностью
    zero_features = [feature_names[i] for i in range(len(importances)) if importances[i] == 0]
    print(f'  Фичей с importance=0: {len(zero_features)}')
    if zero_features:
        print(f'  Нулевые: {zero_features}')
    print()

# Объединённый анализ - какие фичи важны во ВСЕХ моделях
print("=== ФИЧИ ВАЖНЫЕ ВО ВСЕХ 3 МОДЕЛЯХ (Top 20 в каждой) ===")
top_sets = []
for model in [direction_model, strength_model, timing_model]:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_sets.append(set(feature_names[i] for i in indices))

common_top = top_sets[0] & top_sets[1] & top_sets[2]
print(f"Общие топ-фичи: {len(common_top)}")
for f in sorted(common_top):
    print(f"  {f}")

# Рекомендация по минимальному набору фичей
print("\n=== РЕКОМЕНДУЕМЫЙ МИНИМАЛЬНЫЙ НАБОР ФИЧЕЙ ===")
all_importances = {}
for model in [direction_model, strength_model, timing_model]:
    importances = model.feature_importances_
    for i, name in enumerate(feature_names):
        if name not in all_importances:
            all_importances[name] = 0
        all_importances[name] += importances[i]

# Сортируем по суммарной важности
sorted_features = sorted(all_importances.items(), key=lambda x: x[1], reverse=True)
print("Top 40 фичей по суммарной важности:")
for i, (name, imp) in enumerate(sorted_features[:40]):
    print(f"  {i+1:2d}. {name:35s} {imp:12.2f}")

# Получаем feature importance
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    features = model.feature_names_in_
    
    # Сортируем по важности
    indices = np.argsort(importances)[::-1]
    
    print('=== TOP 30 ВАЖНЫХ ФИЧЕЙ ===')
    for i in range(min(30, len(indices))):
        idx = indices[i]
        print(f'{i+1:2d}. {features[idx]:40s} {importances[idx]:.4f}')
    
    print(f'\n=== ПОСЛЕДНИЕ 30 (наименее важные) ===')
    for i in range(max(0, len(indices)-30), len(indices)):
        idx = indices[i]
        print(f'{i+1:3d}. {features[idx]:40s} {importances[idx]:.6f}')
    
    # Статистика
    print(f'\nВсего фичей: {len(features)}')
    print(f'Фичей с importance > 0.01: {sum(importances > 0.01)}')
    print(f'Фичей с importance > 0.005: {sum(importances > 0.005)}')
    print(f'Фичей с importance < 0.001: {sum(importances < 0.001)}')
    
    # Сохраняем список важных фичей
    important_features = [features[indices[i]] for i in range(len(indices)) if importances[indices[i]] > 0.005]
    print(f'\n=== РЕКОМЕНДУЕМЫЕ ФИЧИ (importance > 0.005): {len(important_features)} ===')
    for f in important_features:
        print(f"    '{f}',")
else:
    print("Модель не имеет feature_importances_")

#!/usr/bin/env python3
"""
Сравнение подготовки данных в бектесте vs лайве
Находит реальную причину различий в предсказаниях
"""

import sys
import pandas as pd
import numpy as np
import joblib
import ccxt
from pathlib import Path
from datetime import datetime, timedelta, timezone
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

# Импортируем функции из train_v3_dynamic
import importlib.util
spec = importlib.util.spec_from_file_location("train_v3_dynamic", Path(__file__).parent / "train_v3_dynamic.py")
train_v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_v3)
add_volume_features = train_v3.add_volume_features
calculate_atr = train_v3.calculate_atr

# Config
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
DATA_DIR = Path("data/candles")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 1500

# Thresholds
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4


def load_models():
    """Load trained models"""
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    logger.info(f"✅ Loaded models: {len(models['features'])} features")
    return models


def prepare_features_backtest(m1, m5, m15, mtf_fe):
    """Подготовка фичей КАК В БЕКТЕСТЕ (из train_v3_dynamic.py)"""
    # Точно как в train_v3_dynamic.py строка 724-726
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    return ft


def prepare_features_live(data, mtf_fe):
    """Подготовка фичей КАК НА ЛАЙВЕ (из live_trading_mexc_v8.py)"""
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    if len(m1) < 50 or len(m5) < 50 or len(m15) < 50:
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    for df in [m1, m5, m15]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    
    try:
        ft = mtf_fe.align_timeframes(m1, m5, m15)
        if len(ft) == 0:
            return pd.DataFrame()
        
        ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
        ft = add_volume_features(ft)
        ft['atr'] = calculate_atr(ft)
        
        # Fill NaN (как на лайве)
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        ft = ft.ffill().bfill()
        
        if ft.isna().any().any():
            ft = ft.fillna(0)
        
        return ft
    except Exception as e:
        logger.error(f"Error: {e}")
        return pd.DataFrame()


def compare_features(ft_backtest, ft_live, feature_names, timestamp):
    """Сравнить фичи между бектестом и лайвом"""
    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 Сравнение фичей для timestamp: {timestamp}")
    logger.info(f"{'='*70}")
    
    # Найти строку в бектесте
    if timestamp not in ft_backtest.index:
        logger.warning(f"Timestamp {timestamp} не найден в бектесте!")
        return None
    
    row_backtest = ft_backtest.loc[[timestamp]]
    row_live = ft_live.loc[[timestamp]]
    
    if len(row_live) == 0:
        logger.warning(f"Timestamp {timestamp} не найден на лайве!")
        return None
    
    # Сравнить каждую фичу
    differences = []
    missing_in_live = []
    missing_in_backtest = []
    
    for feat in feature_names:
        if feat not in row_backtest.columns:
            missing_in_backtest.append(feat)
            continue
        
        if feat not in row_live.columns:
            missing_in_live.append(feat)
            continue
        
        val_backtest = row_backtest[feat].iloc[0]
        val_live = row_live[feat].iloc[0]
        
        if pd.isna(val_backtest) or pd.isna(val_live):
            if pd.isna(val_backtest) != pd.isna(val_live):
                differences.append({
                    'feature': feat,
                    'backtest': val_backtest,
                    'live': val_live,
                    'diff': 'NaN mismatch'
                })
            continue
        
        # Сравнить значения
        if isinstance(val_backtest, (int, float)) and isinstance(val_live, (int, float)):
            diff_pct = abs(val_backtest - val_live) / (abs(val_backtest) + 1e-10) * 100
            if diff_pct > 0.1 or abs(val_backtest - val_live) > 1e-6:  # Разница > 0.1% или > 1e-6
                differences.append({
                    'feature': feat,
                    'backtest': val_backtest,
                    'live': val_live,
                    'diff_pct': diff_pct,
                    'diff_abs': abs(val_backtest - val_live)
                })
        elif val_backtest != val_live:
            differences.append({
                'feature': feat,
                'backtest': val_backtest,
                'live': val_live,
                'diff': 'value mismatch'
            })
    
    # Вывести результаты
    if missing_in_backtest:
        logger.warning(f"⚠️  Фичи отсутствуют в бектесте: {missing_in_backtest[:10]}")
    
    if missing_in_live:
        logger.error(f"❌ Фичи отсутствуют на лайве: {missing_in_live[:10]}")
        return None
    
    if differences:
        logger.warning(f"⚠️  Найдено {len(differences)} различий в фичах!")
        logger.info(f"\nТоп-20 наибольших различий:")
        
        # Сортировать по абсолютной разнице
        sorted_diffs = sorted(differences, key=lambda x: x.get('diff_abs', 0) or x.get('diff_pct', 0), reverse=True)
        
        for i, diff in enumerate(sorted_diffs[:20], 1):
            if 'diff_pct' in diff:
                logger.info(f"  {i}. {diff['feature']}: "
                          f"backtest={diff['backtest']:.6f}, "
                          f"live={diff['live']:.6f}, "
                          f"diff={diff['diff_pct']:.2f}%")
            else:
                logger.info(f"  {i}. {diff['feature']}: "
                          f"backtest={diff['backtest']}, "
                          f"live={diff['live']}, "
                          f"diff={diff.get('diff', 'N/A')}")
        
        return sorted_diffs
    else:
        logger.info("✅ Все фичи совпадают!")
        return []


def compare_predictions(ft_backtest, ft_live, models, timestamp):
    """Сравнить предсказания на одинаковых данных"""
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 Сравнение предсказаний для timestamp: {timestamp}")
    logger.info(f"{'='*70}")
    
    # Получить строки
    if timestamp not in ft_backtest.index:
        return None
    
    row_backtest = ft_backtest.loc[[timestamp]]
    row_live = ft_live.loc[[timestamp]]
    
    if len(row_live) == 0:
        return None
    
    # Проверить наличие фичей
    missing_backtest = [f for f in models['features'] if f not in row_backtest.columns]
    missing_live = [f for f in models['features'] if f not in row_live.columns]
    
    if missing_backtest:
        logger.warning(f"⚠️  Отсутствуют фичи в бектесте: {missing_backtest[:5]}")
        # Добавить нули
        for f in missing_backtest:
            row_backtest[f] = 0
    
    if missing_live:
        logger.error(f"❌ Отсутствуют фичи на лайве: {missing_live[:5]}")
        return None
    
    # Предсказания на бектесте
    X_backtest = row_backtest[models['features']].values
    if pd.isna(X_backtest).any():
        logger.warning("⚠️  NaN в фичах бектеста!")
        X_backtest = np.nan_to_num(X_backtest, nan=0.0)
    
    dir_proba_backtest = models['direction'].predict_proba(X_backtest)
    dir_conf_backtest = float(np.max(dir_proba_backtest))
    dir_pred_backtest = int(np.argmax(dir_proba_backtest))
    timing_backtest = float(models['timing'].predict(X_backtest)[0])
    strength_backtest = float(models['strength'].predict(X_backtest)[0])
    
    # Предсказания на лайве
    X_live = row_live[models['features']].values
    if pd.isna(X_live).any():
        logger.warning("⚠️  NaN в фичах лайва!")
        X_live = np.nan_to_num(X_live, nan=0.0)
    
    dir_proba_live = models['direction'].predict_proba(X_live)
    dir_conf_live = float(np.max(dir_proba_live))
    dir_pred_live = int(np.argmax(dir_proba_live))
    timing_live = float(models['timing'].predict(X_live)[0])
    strength_live = float(models['strength'].predict(X_live)[0])
    
    # Вывести сравнение
    logger.info(f"\n📊 Предсказания:")
    logger.info(f"  Direction: backtest={dir_pred_backtest} (conf={dir_conf_backtest:.3f}), "
              f"live={dir_pred_live} (conf={dir_conf_live:.3f}), "
              f"diff={abs(dir_conf_backtest - dir_conf_live):.3f}")
    logger.info(f"  Timing: backtest={timing_backtest:.3f}, live={timing_live:.3f}, "
              f"diff={abs(timing_backtest - timing_live):.3f}")
    logger.info(f"  Strength: backtest={strength_backtest:.3f}, live={strength_live:.3f}, "
              f"diff={abs(strength_backtest - strength_live):.3f}")
    
    # Проверить, проходят ли фильтры
    def check_filters(conf, timing, strength, pred):
        passed = []
        if pred == 1:
            return False, ["SIDEWAYS"]
        if conf < MIN_CONF:
            passed.append(f"Conf({conf:.2f}<{MIN_CONF})")
        if timing < MIN_TIMING:
            passed.append(f"Timing({timing:.2f}<{MIN_TIMING})")
        if strength < MIN_STRENGTH:
            passed.append(f"Strength({strength:.1f}<{MIN_STRENGTH})")
        return len(passed) == 0, passed
    
    backtest_passed, backtest_reasons = check_filters(dir_conf_backtest, timing_backtest, strength_backtest, dir_pred_backtest)
    live_passed, live_reasons = check_filters(dir_conf_live, timing_live, strength_live, dir_pred_live)
    
    logger.info(f"\n✅ Фильтры:")
    logger.info(f"  Backtest: {'✅ PASSED' if backtest_passed else '❌ REJECTED'} {backtest_reasons}")
    logger.info(f"  Live: {'✅ PASSED' if live_passed else '❌ REJECTED'} {live_reasons}")
    
    return {
        'backtest': {
            'conf': dir_conf_backtest,
            'timing': timing_backtest,
            'strength': strength_backtest,
            'pred': dir_pred_backtest,
            'passed': backtest_passed
        },
        'live': {
            'conf': dir_conf_live,
            'timing': timing_live,
            'strength': strength_live,
            'pred': dir_pred_live,
            'passed': live_passed
        }
    }


def main():
    import json
    
    logger.info("=" * 70)
    logger.info("🔍 Сравнение бектеста и лайва - поиск причины различий")
    logger.info("=" * 70)
    
    # Загрузить модели
    models = load_models()
    
    # Загрузить пары
    with open(PAIRS_FILE) as f:
        pairs_data = json.load(f)
    pairs = [p['symbol'] for p in pairs_data['pairs'][:3]]  # Тестируем 3 пары
    
    mtf_fe = MTFFeatureEngine()
    
    # Инициализировать Binance
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Используем период из CSV (берем последние доступные данные)
    # Сначала проверим, какие данные есть в CSV
    logger.info("Проверка доступных данных в CSV...")
    
    # Найдем последнюю доступную дату в CSV
    test_pair_name = pairs[0].replace('/', '_').replace(':', '_')
    try:
        m5_test = pd.read_csv(DATA_DIR / f"{test_pair_name}_5m.csv", 
                             parse_dates=['timestamp'], index_col='timestamp')
        if m5_test.index.tz is None:
            m5_test.index = m5_test.index.tz_localize('UTC')
        
        if len(m5_test) > 0:
            # Используем последние 3 дня из CSV
            end_date = m5_test.index.max()
            start_date = end_date - timedelta(days=3)
            logger.info(f"Используем период из CSV: {start_date.date()} - {end_date.date()}")
        else:
            # Fallback: последние 7 дней
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)
            logger.info(f"CSV пуст, используем последние 7 дней: {start_date.date()} - {end_date.date()}")
    except FileNotFoundError:
        # Fallback: последние 7 дней
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        logger.info(f"CSV не найден, используем последние 7 дней: {start_date.date()} - {end_date.date()}")
    
    for pair in pairs:
        logger.info(f"\n{'='*70}")
        logger.info(f"Пара: {pair}")
        logger.info(f"{'='*70}")
        
        pair_name = pair.replace('/', '_').replace(':', '_')
        
        try:
            # 1. Загрузить данные из CSV (как в бектесте)
            logger.info("📂 Загрузка данных из CSV (бектест)...")
            try:
                m1_backtest = pd.read_csv(DATA_DIR / f"{pair_name}_1m.csv", 
                                         parse_dates=['timestamp'], index_col='timestamp')
                m5_backtest = pd.read_csv(DATA_DIR / f"{pair_name}_5m.csv", 
                                         parse_dates=['timestamp'], index_col='timestamp')
                m15_backtest = pd.read_csv(DATA_DIR / f"{pair_name}_15m.csv", 
                                          parse_dates=['timestamp'], index_col='timestamp')
                
                # Убедиться, что индексы имеют timezone (если CSV без timezone, добавить UTC)
                if m1_backtest.index.tz is None:
                    m1_backtest.index = m1_backtest.index.tz_localize('UTC')
                if m5_backtest.index.tz is None:
                    m5_backtest.index = m5_backtest.index.tz_localize('UTC')
                if m15_backtest.index.tz is None:
                    m15_backtest.index = m15_backtest.index.tz_localize('UTC')
                
                # Использовать последние доступные данные из CSV (не фильтровать по датам)
                # Берем последние N свечей, которые есть
                if len(m5_backtest) < 200:
                    logger.warning(f"Мало данных в CSV для {pair} (только {len(m5_backtest)} свечей)")
                    if len(m5_backtest) < 50:
                        continue
                
                # Берем последние 200 свечей из CSV для сравнения
                m5_backtest = m5_backtest.tail(200)
                # Найти соответствующие индексы для m1 и m15
                m5_start = m5_backtest.index[0]
                m5_end = m5_backtest.index[-1]
                m1_backtest = m1_backtest[(m1_backtest.index >= m5_start) & (m1_backtest.index <= m5_end)]
                m15_backtest = m15_backtest[(m15_backtest.index >= m5_start) & (m15_backtest.index <= m5_end)]
                
                # Обновить период для API запроса
                actual_start = m5_backtest.index[0]
                actual_end = m5_backtest.index[-1]
                logger.info(f"  Используем период из CSV: {actual_start} - {actual_end}")
                
                logger.info(f"  CSV: M1={len(m1_backtest)}, M5={len(m5_backtest)}, M15={len(m15_backtest)}")
                
            except FileNotFoundError:
                logger.warning(f"CSV файлы не найдены для {pair}, пропускаем")
                continue
            
            # 2. Загрузить данные через API (как на лайве) за тот же период
            logger.info("🌐 Загрузка данных через API (лайв)...")
            data_live = {}
            # Используем период из CSV
            actual_start = m5_backtest.index[0]
            actual_end = m5_backtest.index[-1]
            
            for tf in TIMEFRAMES:
                # Загрузить достаточно данных, чтобы покрыть период
                since = int((actual_start - timedelta(days=1)).timestamp() * 1000)
                candles = binance.fetch_ohlcv(pair, tf, since=since, limit=LOOKBACK)
                if not candles or len(candles) < 50:
                    logger.warning(f"Недостаточно данных {tf} для {pair}")
                    break
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                
                # Фильтровать по периоду из CSV
                df = df[(df.index >= actual_start) & (df.index <= actual_end)]
                data_live[tf] = df
            
            if len(data_live) < 3:
                continue
            
            logger.info(f"  API: M1={len(data_live['1m'])}, M5={len(data_live['5m'])}, M15={len(data_live['15m'])}")
            
            # 3. Подготовить фичи (бектест)
            logger.info("🔧 Подготовка фичей (бектест)...")
            ft_backtest = prepare_features_backtest(m1_backtest, m5_backtest, m15_backtest, mtf_fe)
            logger.info(f"  Фичи бектеста: {len(ft_backtest)} строк, {len(ft_backtest.columns)} колонок")
            
            # 4. Подготовить фичи (лайв)
            logger.info("🔧 Подготовка фичей (лайв)...")
            ft_live = prepare_features_live(data_live, mtf_fe)
            logger.info(f"  Фичи лайва: {len(ft_live)} строк, {len(ft_live.columns)} колонок")
            
            if len(ft_backtest) == 0 or len(ft_live) == 0:
                logger.warning("Не удалось подготовить фичи")
                continue
            
            # 5. Найти общие timestamps
            common_timestamps = ft_backtest.index.intersection(ft_live.index)
            if len(common_timestamps) == 0:
                logger.warning("Нет общих timestamps!")
                continue
            
            logger.info(f"  Общих timestamps: {len(common_timestamps)}")
            
            # 6. Сравнить несколько последних свечей
            test_timestamps = common_timestamps[-5:]  # Последние 5 свечей
            
            for ts in test_timestamps:
                # Сравнить фичи
                diff_features = compare_features(ft_backtest, ft_live, models['features'], ts)
                
                # Сравнить предсказания
                pred_comparison = compare_predictions(ft_backtest, ft_live, models, ts)
                
                if pred_comparison and not pred_comparison['backtest']['passed'] and pred_comparison['live']['passed']:
                    logger.error(f"🚨 НАЙДЕНА ПРОБЛЕМА! На {ts} бектест отклоняет, а лайв принимает!")
                elif pred_comparison and pred_comparison['backtest']['passed'] and not pred_comparison['live']['passed']:
                    logger.error(f"🚨 НАЙДЕНА ПРОБЛЕМА! На {ts} бектест принимает, а лайв отклоняет!")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {pair}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()


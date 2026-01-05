#!/usr/bin/env python3
"""
Полный анализ различий между лайвом и бектестом.
Сравнивает:
1. Загрузку данных (сколько свечей, временные метки)
2. Вычисление фичей (OBV, индикаторы)
3. Предсказания моделей
4. Фильтрацию сигналов
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import ccxt
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v8_improved")
PAIRS_FILE = Path("config/pairs_list.json")
DATA_DIR = Path("data/candles")
LOOKBACK = 1500  # Как в лайве

# Пороги как в лайве
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# ============================================================
# UTILS
# ============================================================
def add_volume_features(df):
    """Точно как в лайве и бектесте"""
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    
    df['vol_momentum'] = df['volume'].pct_change(5)
    
    return df

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe):
    """Точно как в лайве"""
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
        
        # Fill NaN
        critical_cols = ['close', 'atr']
        ft = ft.dropna(subset=critical_cols)
        ft = ft.ffill().bfill()
        
        if ft.isna().any().any():
            ft = ft.fillna(0)
        
        return ft
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()

# ============================================================
# DATA LOADING
# ============================================================
def load_backtest_data(pair, target_date):
    """Загрузить данные из CSV (как в бектесте)"""
    pair_name = pair.replace('/', '_').replace(':', '_')
    
    data = {}
    for tf in ['1m', '5m', '15m']:
        file_path = DATA_DIR / f"{pair_name}_{tf}.csv"
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Убедиться что индекс имеет timezone (если нет - добавить UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Фильтруем до target_date (как в бектесте - все данные до этой даты)
        df = df[df.index <= target_date]
        
        # Берем последние LOOKBACK свечей (как в лайве)
        if len(df) > LOOKBACK:
            df = df.tail(LOOKBACK)
        
        data[tf] = df
    
    return data

def fetch_live_data(pair, binance):
    """Загрузить данные через API (как в лайве)"""
    data = {}
    for tf in ['1m', '5m', '15m']:
        try:
            candles = binance.fetch_ohlcv(pair, tf, limit=LOOKBACK)
            if not candles or len(candles) < 50:
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        except Exception as e:
            logger.error(f"Error fetching {pair} {tf}: {e}")
            return None
    
    return data

# ============================================================
# ANALYSIS
# ============================================================
def compare_data_loading(pair, target_date):
    """Сравнить загрузку данных"""
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 СРАВНЕНИЕ ЗАГРУЗКИ ДАННЫХ: {pair}")
    logger.info(f"{'='*70}")
    
    # Backtest data
    backtest_data = load_backtest_data(pair, target_date)
    if backtest_data is None:
        logger.error(f"❌ Не удалось загрузить бектест данные для {pair}")
        return None
    
    # Live data
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    live_data = fetch_live_data(pair, binance)
    if live_data is None:
        logger.error(f"❌ Не удалось загрузить лайв данные для {pair}")
        return None
    
    comparison = {}
    
    for tf in ['1m', '5m', '15m']:
        bt_df = backtest_data[tf]
        lv_df = live_data[tf]
        
        logger.info(f"\n{tf}:")
        logger.info(f"  Бектест: {len(bt_df)} свечей | Первая: {bt_df.index[0]} | Последняя: {bt_df.index[-1]}")
        logger.info(f"  Лайв:    {len(lv_df)} свечей | Первая: {lv_df.index[0]} | Последняя: {lv_df.index[-1]}")
        
        # Сравнить последние N свечей
        n_compare = min(10, len(bt_df), len(lv_df))
        bt_last = bt_df.tail(n_compare)
        lv_last = lv_df.tail(n_compare)
        
        # Найти пересечение по времени
        common_times = bt_last.index.intersection(lv_last.index)
        
        if len(common_times) > 0:
            bt_common = bt_last.loc[common_times]
            lv_common = lv_last.loc[common_times]
            
            close_diff = (bt_common['close'] - lv_common['close']).abs()
            volume_diff = (bt_common['volume'] - lv_common['volume']).abs()
            
            logger.info(f"  ✅ Общих свечей: {len(common_times)}")
            logger.info(f"  Макс разница close: {close_diff.max():.6f} ({close_diff.max() / bt_common['close'].mean() * 100:.4f}%)")
            logger.info(f"  Макс разница volume: {volume_diff.max():.2f} ({volume_diff.max() / bt_common['volume'].mean() * 100:.2f}%)")
        else:
            logger.warning(f"  ⚠️ Нет общих временных меток!")
        
        comparison[tf] = {
            'backtest_count': len(bt_df),
            'live_count': len(lv_df),
            'backtest_first': bt_df.index[0],
            'backtest_last': bt_df.index[-1],
            'live_first': lv_df.index[0],
            'live_last': lv_df.index[-1],
            'common_times': len(common_times) if len(common_times) > 0 else 0
        }
    
    return {
        'pair': pair,
        'target_date': target_date,
        'data_comparison': comparison,
        'backtest_data': backtest_data,
        'live_data': live_data
    }

def compare_features(comparison_result):
    """Сравнить вычисление фичей"""
    logger.info(f"\n{'='*70}")
    logger.info(f"🔧 СРАВНЕНИЕ ФИЧЕЙ: {comparison_result['pair']}")
    logger.info(f"{'='*70}")
    
    mtf_fe = MTFFeatureEngine()
    
    # Backtest features
    bt_features = prepare_features(comparison_result['backtest_data'], mtf_fe)
    if bt_features.empty:
        logger.error("❌ Не удалось создать бектест фичи")
        return None
    
    # Live features
    lv_features = prepare_features(comparison_result['live_data'], mtf_fe)
    if lv_features.empty:
        logger.error("❌ Не удалось создать лайв фичи")
        return None
    
    logger.info(f"\nБектест фичи: {len(bt_features)} строк, {len(bt_features.columns)} колонок")
    logger.info(f"Лайв фичи:    {len(lv_features)} строк, {len(lv_features.columns)} колонок")
    
    # Сравнить последнюю свечу (как в лайве используется iloc[-2])
    bt_last = bt_features.iloc[[-2]] if len(bt_features) >= 2 else bt_features.iloc[[-1]]
    lv_last = lv_features.iloc[[-2]] if len(lv_features) >= 2 else lv_features.iloc[[-1]]
    
    logger.info(f"\nБектест последняя свеча: {bt_last.index[0]}")
    logger.info(f"Лайв последняя свеча:    {lv_last.index[0]}")
    
    # Сравнить общие фичи
    common_features = set(bt_features.columns) & set(lv_features.columns)
    logger.info(f"\nОбщих фичей: {len(common_features)}")
    
    # Сравнить OBV (критично!)
    if 'obv' in common_features:
        bt_obv = bt_features['obv'].iloc[-10:].values
        lv_obv = lv_features['obv'].iloc[-10:].values
        
        logger.info(f"\n📊 OBV сравнение (последние 10):")
        logger.info(f"  Бектест: {bt_obv}")
        logger.info(f"  Лайв:    {lv_obv}")
        logger.info(f"  Разница: {np.abs(bt_obv - lv_obv[:len(bt_obv)])}")
    
    # Сравнить значения фичей для последней свечи
    feature_diffs = {}
    for feat in common_features:
        if feat in bt_last.columns and feat in lv_last.columns:
            bt_val = bt_last[feat].iloc[0]
            lv_val = lv_last[feat].iloc[0]
            
            if pd.notna(bt_val) and pd.notna(lv_val):
                if isinstance(bt_val, (int, float)) and isinstance(lv_val, (int, float)):
                    diff = abs(bt_val - lv_val)
                    if diff > 1e-6:  # Значимая разница
                        feature_diffs[feat] = {
                            'backtest': bt_val,
                            'live': lv_val,
                            'diff': diff,
                            'diff_pct': (diff / abs(bt_val) * 100) if bt_val != 0 else 0
                        }
    
    if feature_diffs:
        logger.info(f"\n⚠️ Найдено {len(feature_diffs)} фичей с различиями:")
        for feat, diff_info in sorted(feature_diffs.items(), key=lambda x: x[1]['diff'], reverse=True)[:20]:
            logger.info(f"  {feat}: BT={diff_info['backtest']:.6f}, LV={diff_info['live']:.6f}, "
                       f"Diff={diff_info['diff']:.6f} ({diff_info['diff_pct']:.2f}%)")
    
    return {
        'backtest_features': bt_features,
        'live_features': lv_features,
        'backtest_last': bt_last,
        'live_last': lv_last,
        'feature_diffs': feature_diffs
    }

def compare_predictions(comparison_result, feature_comparison):
    """Сравнить предсказания моделей"""
    logger.info(f"\n{'='*70}")
    logger.info(f"🤖 СРАВНЕНИЕ ПРЕДСКАЗАНИЙ: {comparison_result['pair']}")
    logger.info(f"{'='*70}")
    
    # Load models
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
        'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    
    bt_features = feature_comparison['backtest_features']
    lv_features = feature_comparison['live_features']
    
    bt_last = feature_comparison['backtest_last']
    lv_last = feature_comparison['live_last']
    
    # Проверить наличие всех фичей
    missing_bt = [f for f in models['features'] if f not in bt_last.columns]
    missing_lv = [f for f in models['features'] if f not in lv_last.columns]
    
    if missing_bt:
        logger.warning(f"⚠️ Бектест: отсутствуют фичи: {missing_bt[:10]}")
    if missing_lv:
        logger.warning(f"⚠️ Лайв: отсутствуют фичи: {missing_lv[:10]}")
    
    # Предсказания для бектеста
    bt_X = bt_last[models['features']].values
    if pd.isna(bt_X).any():
        logger.warning("⚠️ Бектест: NaN в фичах, заполняю нулями")
        bt_X = np.nan_to_num(bt_X)
    
    bt_dir_proba = models['direction'].predict_proba(bt_X)
    bt_dir_conf = float(np.max(bt_dir_proba))
    bt_dir_pred = int(np.argmax(bt_dir_proba))
    bt_timing = float(models['timing'].predict(bt_X)[0])
    bt_strength = float(models['strength'].predict(bt_X)[0])
    
    # Предсказания для лайва
    lv_X = lv_last[models['features']].values
    if pd.isna(lv_X).any():
        logger.warning("⚠️ Лайв: NaN в фичах, заполняю нулями")
        lv_X = np.nan_to_num(lv_X)
    
    lv_dir_proba = models['direction'].predict_proba(lv_X)
    lv_dir_conf = float(np.max(lv_dir_proba))
    lv_dir_pred = int(np.argmax(lv_dir_proba))
    lv_timing = float(models['timing'].predict(lv_X)[0])
    lv_strength = float(models['strength'].predict(lv_X)[0])
    
    direction_map = {0: 'SHORT', 1: 'SIDEWAYS', 2: 'LONG'}
    
    logger.info(f"\n📊 БЕКТЕСТ ПРЕДСКАЗАНИЯ:")
    logger.info(f"  Direction: {direction_map[bt_dir_pred]} (conf: {bt_dir_conf:.3f})")
    logger.info(f"  Timing: {bt_timing:.3f} ATR")
    logger.info(f"  Strength: {bt_strength:.2f}")
    
    logger.info(f"\n📊 ЛАЙВ ПРЕДСКАЗАНИЯ:")
    logger.info(f"  Direction: {direction_map[lv_dir_pred]} (conf: {lv_dir_conf:.3f})")
    logger.info(f"  Timing: {lv_timing:.3f} ATR")
    logger.info(f"  Strength: {lv_strength:.2f}")
    
    logger.info(f"\n📊 РАЗНИЦЫ:")
    logger.info(f"  Direction: {'✅' if bt_dir_pred == lv_dir_pred else '❌'} "
               f"({direction_map[bt_dir_pred]} vs {direction_map[lv_dir_pred]})")
    logger.info(f"  Conf: {abs(bt_dir_conf - lv_dir_conf):.4f}")
    logger.info(f"  Timing: {abs(bt_timing - lv_timing):.4f}")
    logger.info(f"  Strength: {abs(bt_strength - lv_strength):.4f}")
    
    # Проверка фильтров
    bt_passes = (bt_dir_pred != 1 and 
                 bt_dir_conf >= MIN_CONF and 
                 bt_timing >= MIN_TIMING and 
                 bt_strength >= MIN_STRENGTH)
    
    lv_passes = (lv_dir_pred != 1 and 
                 lv_dir_conf >= MIN_CONF and 
                 lv_timing >= MIN_TIMING and 
                 lv_strength >= MIN_STRENGTH)
    
    logger.info(f"\n🎯 ФИЛЬТРЫ:")
    logger.info(f"  Бектест проходит: {'✅' if bt_passes else '❌'}")
    logger.info(f"  Лайв проходит:    {'✅' if lv_passes else '❌'}")
    
    if not bt_passes:
        reasons = []
        if bt_dir_pred == 1:
            reasons.append("SIDEWAYS")
        if bt_dir_conf < MIN_CONF:
            reasons.append(f"Conf({bt_dir_conf:.2f}<{MIN_CONF})")
        if bt_timing < MIN_TIMING:
            reasons.append(f"Timing({bt_timing:.2f}<{MIN_TIMING})")
        if bt_strength < MIN_STRENGTH:
            reasons.append(f"Strength({bt_strength:.2f}<{MIN_STRENGTH})")
        logger.info(f"  Бектест причины отклонения: {', '.join(reasons)}")
    
    if not lv_passes:
        reasons = []
        if lv_dir_pred == 1:
            reasons.append("SIDEWAYS")
        if lv_dir_conf < MIN_CONF:
            reasons.append(f"Conf({lv_dir_conf:.2f}<{MIN_CONF})")
        if lv_timing < MIN_TIMING:
            reasons.append(f"Timing({lv_timing:.2f}<{MIN_TIMING})")
        if lv_strength < MIN_STRENGTH:
            reasons.append(f"Strength({lv_strength:.2f}<{MIN_STRENGTH})")
        logger.info(f"  Лайв причины отклонения: {', '.join(reasons)}")
    
    return {
        'backtest': {
            'direction': bt_dir_pred,
            'direction_str': direction_map[bt_dir_pred],
            'conf': bt_dir_conf,
            'timing': bt_timing,
            'strength': bt_strength,
            'passes': bt_passes
        },
        'live': {
            'direction': lv_dir_pred,
            'direction_str': direction_map[lv_dir_pred],
            'conf': lv_dir_conf,
            'timing': lv_timing,
            'strength': lv_strength,
            'passes': lv_passes
        }
    }

# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="BTC/USDT:USDT", help="Pair to analyze")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), defaults to yesterday")
    args = parser.parse_args()
    
    # Default to yesterday
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    logger.info("="*70)
    logger.info("ПОЛНЫЙ АНАЛИЗ РАЗЛИЧИЙ ЛАЙВ vs БЕКТЕСТ")
    logger.info("="*70)
    logger.info(f"Пара: {args.pair}")
    logger.info(f"Целевая дата: {target_date}")
    logger.info("="*70)
    
    # 1. Сравнить загрузку данных
    comparison = compare_data_loading(args.pair, target_date)
    if comparison is None:
        logger.error("❌ Не удалось загрузить данные")
        return
    
    # 2. Сравнить фичи
    feature_comp = compare_features(comparison)
    if feature_comp is None:
        logger.error("❌ Не удалось сравнить фичи")
        return
    
    # 3. Сравнить предсказания
    pred_comp = compare_predictions(comparison, feature_comp)
    
    # 4. Итоговый отчет
    logger.info(f"\n{'='*70}")
    logger.info("📋 ИТОГОВЫЙ ОТЧЕТ")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n✅ Данные загружены:")
    logger.info(f"  Бектест: {comparison['data_comparison']['5m']['backtest_count']} свечей 5m")
    logger.info(f"  Лайв:    {comparison['data_comparison']['5m']['live_count']} свечей 5m")
    
    logger.info(f"\n✅ Фичи созданы:")
    logger.info(f"  Бектест: {len(feature_comp['backtest_features'])} строк")
    logger.info(f"  Лайв:    {len(feature_comp['live_features'])} строк")
    logger.info(f"  Различий в фичах: {len(feature_comp['feature_diffs'])}")
    
    logger.info(f"\n✅ Предсказания:")
    logger.info(f"  Бектест: {pred_comp['backtest']['direction_str']} "
               f"(conf={pred_comp['backtest']['conf']:.3f}, "
               f"timing={pred_comp['backtest']['timing']:.2f}, "
               f"strength={pred_comp['backtest']['strength']:.2f})")
    logger.info(f"  Лайв:    {pred_comp['live']['direction_str']} "
               f"(conf={pred_comp['live']['conf']:.3f}, "
               f"timing={pred_comp['live']['timing']:.2f}, "
               f"strength={pred_comp['live']['strength']:.2f})")
    
    logger.info(f"\n✅ Сигнал:")
    logger.info(f"  Бектест проходит фильтры: {'✅ ДА' if pred_comp['backtest']['passes'] else '❌ НЕТ'}")
    logger.info(f"  Лайв проходит фильтры:    {'✅ ДА' if pred_comp['live']['passes'] else '❌ НЕТ'}")
    
    if pred_comp['backtest']['passes'] and not pred_comp['live']['passes']:
        logger.error("\n❌ ПРОБЛЕМА: Бектест дает сигнал, а лайв - нет!")
        logger.error("   Это объясняет почему в лайве нет сделок.")
    elif not pred_comp['backtest']['passes'] and pred_comp['live']['passes']:
        logger.warning("\n⚠️  Лайв дает сигнал, а бектест - нет (необычно)")
    elif pred_comp['backtest']['passes'] and pred_comp['live']['passes']:
        logger.info("\n✅ Оба дают сигнал - все работает!")
    else:
        logger.info("\nℹ️  Оба не дают сигнал - нет сделок в обоих случаях")

if __name__ == '__main__':
    main()


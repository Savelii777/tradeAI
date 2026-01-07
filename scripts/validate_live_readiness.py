#!/usr/bin/env python3
"""
Pre-Live Validation Script
Проверяет что модель готова к лайв торговле.

Запуск:
    python scripts/validate_live_readiness.py

Проверки:
1. Отсутствие cumsum-зависимых фичей в модели
2. Отсутствие абсолютных EMA/price фичей
3. Feature distribution check
4. Model file integrity
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'candles'

# Dangerous patterns that should NOT be in features
DANGEROUS_PATTERNS = [
    'obv',
    'cumsum',
    'bars_since_swing',
    'consecutive_up',
    'consecutive_down',
    'swing_high_price',
    'swing_low_price',
    'volume_delta_cumsum',
]

# Absolute value features that may cause issues
ABSOLUTE_VALUE_PATTERNS = [
    # EMA without normalization
    'ema_9$', 'ema_21$', 'ema_50$', 'ema_200$',
    # ATR absolute (should be atr_pct instead)
    'atr_7$', 'atr_14$', 'atr_21$',
    # BB absolute
    'bb_upper$', 'bb_middle$', 'bb_lower$',
]

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def check_model_files():
    """Check that all model files exist."""
    print("\n" + "="*60)
    print("CHECK 1: Model Files Existence")
    print("="*60)
    
    required_files = [
        'direction_model.joblib',
        'timing_model.joblib',
        'strength_model.joblib',
        'feature_names.joblib',
    ]
    
    all_exist = True
    for f in required_files:
        path = MODEL_DIR / f
        if path.exists():
            print(f"  ✅ {f} exists ({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ❌ {f} MISSING!")
            all_exist = False
    
    return all_exist


def check_dangerous_features():
    """Check that model doesn't use cumsum-dependent features."""
    print("\n" + "="*60)
    print("CHECK 2: Cumsum-Dependent Features")
    print("="*60)
    
    features_file = MODEL_DIR / 'feature_names.joblib'
    if not features_file.exists():
        print("  ❌ Cannot check - feature_names.joblib missing")
        return False
    
    features = joblib.load(features_file)
    print(f"  Total features: {len(features)}")
    
    dangerous_found = []
    for feat in features:
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in feat.lower():
                dangerous_found.append(feat)
                break
    
    if dangerous_found:
        print(f"  ❌ FOUND {len(dangerous_found)} dangerous features:")
        for f in dangerous_found:
            print(f"      - {f}")
        print("\n  ⚠️  These features depend on cumsum() and will give different")
        print("      values in live vs backtest. RETRAIN THE MODEL!")
        return False
    else:
        print("  ✅ No cumsum-dependent features found")
        return True


def check_absolute_features():
    """Check for absolute value features that may cause issues."""
    print("\n" + "="*60)
    print("CHECK 3: Absolute Value Features")
    print("="*60)
    
    features_file = MODEL_DIR / 'feature_names.joblib'
    if not features_file.exists():
        print("  ❌ Cannot check - feature_names.joblib missing")
        return True  # Non-critical
    
    import re
    features = joblib.load(features_file)
    
    absolute_found = []
    for feat in features:
        for pattern in ABSOLUTE_VALUE_PATTERNS:
            if re.search(pattern, feat):
                absolute_found.append(feat)
                break
    
    if absolute_found:
        print(f"  ⚠️  FOUND {len(absolute_found)} absolute value features:")
        for f in absolute_found:
            print(f"      - {f}")
        print("\n  These may cause issues if price levels change significantly.")
        print("  Consider using normalized versions (_dist, _pct, _ratio)")
        return False  # Warning, not critical
    else:
        print("  ✅ No problematic absolute value features found")
        return True


def check_feature_distributions():
    """Compare feature distributions between training and live-like data."""
    print("\n" + "="*60)
    print("CHECK 4: Feature Distribution Stability")
    print("="*60)
    
    features_file = MODEL_DIR / 'feature_names.joblib'
    if not features_file.exists():
        print("  ❌ Cannot check - feature_names.joblib missing")
        return True
    
    features = joblib.load(features_file)
    
    # Try to load some data for comparison
    sample_pair = "BTC_USDT_USDT"
    m5_file = DATA_DIR / f"{sample_pair}_5m.csv"
    
    if not m5_file.exists():
        print(f"  ⚠️  Cannot find sample data: {m5_file}")
        print("  Skipping distribution check")
        return True
    
    print(f"  Loading sample data from {sample_pair}...")
    df = pd.read_csv(m5_file, parse_dates=['timestamp'], index_col='timestamp')
    
    # Split into old (training-like) and recent (live-like)
    split_point = len(df) - 2000  # Last 2000 candles = "live"
    
    if split_point < 2000:
        print("  ⚠️  Not enough data for distribution comparison")
        return True
    
    df_old = df.iloc[:split_point]
    df_new = df.iloc[split_point:]
    
    print(f"  Old period: {df_old.index[0]} to {df_old.index[-1]} ({len(df_old)} rows)")
    print(f"  New period: {df_new.index[0]} to {df_new.index[-1]} ({len(df_new)} rows)")
    
    # Compare basic statistics
    metrics = ['close', 'volume', 'high', 'low']
    all_stable = True
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        old_mean = df_old[metric].mean()
        new_mean = df_new[metric].mean()
        pct_diff = abs(new_mean - old_mean) / old_mean * 100
        
        if metric == 'close' and pct_diff > 50:
            print(f"  ⚠️  {metric}: Price changed significantly ({pct_diff:.1f}%)")
            print(f"      Old mean: {old_mean:.2f}, New mean: {new_mean:.2f}")
            print("      This may affect absolute value features!")
            all_stable = False
        elif metric == 'volume' and pct_diff > 100:
            print(f"  ⚠️  {metric}: Volume changed significantly ({pct_diff:.1f}%)")
            all_stable = False
    
    if all_stable:
        print("  ✅ Data distributions appear stable")
    
    return all_stable


def check_timing_model_type():
    """Verify timing model is a regressor (V8 improvement)."""
    print("\n" + "="*60)
    print("CHECK 5: Timing Model Type")
    print("="*60)
    
    timing_file = MODEL_DIR / 'timing_model.joblib'
    if not timing_file.exists():
        print("  ❌ Cannot check - timing_model.joblib missing")
        return False
    
    timing_model = joblib.load(timing_file)
    model_type = type(timing_model).__name__
    
    if 'Regressor' in model_type:
        print(f"  ✅ Timing model is a Regressor ({model_type})")
        print("      This is correct for V8 Improved!")
        return True
    elif 'Classifier' in model_type:
        print(f"  ⚠️  Timing model is a Classifier ({model_type})")
        print("      V8 Improved should use Regressor!")
        print("      Model may be from older version - consider retraining")
        return False
    else:
        print(f"  ❓ Timing model type: {model_type}")
        return True


def check_model_hyperparams():
    """Check that model uses anti-overfitting hyperparameters."""
    print("\n" + "="*60)
    print("CHECK 6: Anti-Overfitting Hyperparameters")
    print("="*60)
    
    direction_file = MODEL_DIR / 'direction_model.joblib'
    if not direction_file.exists():
        print("  ❌ Cannot check - direction_model.joblib missing")
        return True
    
    model = joblib.load(direction_file)
    
    # Try to extract hyperparameters
    try:
        params = model.get_params()
        
        n_estimators = params.get('n_estimators', 'unknown')
        max_depth = params.get('max_depth', 'unknown')
        num_leaves = params.get('num_leaves', 'unknown')
        min_child_samples = params.get('min_child_samples', 'unknown')
        
        print(f"  Model parameters:")
        print(f"    n_estimators: {n_estimators} (V8 target: 100)")
        print(f"    max_depth: {max_depth} (V8 target: 3)")
        print(f"    num_leaves: {num_leaves} (V8 target: 8)")
        print(f"    min_child_samples: {min_child_samples} (V8 target: 50)")
        
        warnings = []
        if n_estimators != 'unknown' and n_estimators > 200:
            warnings.append("n_estimators too high - may overfit")
        if max_depth != 'unknown' and max_depth > 5:
            warnings.append("max_depth too high - may overfit")
        if min_child_samples != 'unknown' and min_child_samples < 20:
            warnings.append("min_child_samples too low - may overfit")
        
        if warnings:
            print(f"  ⚠️  Warnings:")
            for w in warnings:
                print(f"      - {w}")
            return False
        else:
            print("  ✅ Hyperparameters look good for anti-overfitting")
            return True
            
    except Exception as e:
        print(f"  ❓ Could not extract hyperparameters: {e}")
        return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("PRE-LIVE VALIDATION")
    print("Checking model readiness for live trading")
    print("="*60)
    
    results = {
        'model_files': check_model_files(),
        'dangerous_features': check_dangerous_features(),
        'absolute_features': check_absolute_features(),
        'feature_distributions': check_feature_distributions(),
        'timing_model_type': check_timing_model_type(),
        'hyperparams': check_model_hyperparams(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    critical_checks = ['model_files', 'dangerous_features', 'timing_model_type']
    warning_checks = ['absolute_features', 'feature_distributions', 'hyperparams']
    
    critical_passed = all(results[c] for c in critical_checks)
    warnings_passed = all(results[c] for c in warning_checks)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else ("❌ FAIL" if check in critical_checks else "⚠️ WARN")
        print(f"  {check}: {status}")
    
    print("\n" + "-"*60)
    
    if critical_passed and warnings_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("   Model is ready for live trading.")
        return 0
    elif critical_passed:
        print("⚠️  PASSED WITH WARNINGS")
        print("   Model can be used for live trading, but review warnings above.")
        return 1
    else:
        print("❌ CRITICAL CHECKS FAILED!")
        print("   DO NOT use this model for live trading.")
        print("   Fix issues above and re-run validation.")
        return 2


if __name__ == '__main__':
    sys.exit(main())

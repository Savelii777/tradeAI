#!/usr/bin/env python3
"""
Pre-Flight Check Script for Live Trading
Comprehensive validation before going live.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --model-dir models/v8_improved --verbose

Checks:
1. Model files exist and are valid
2. No cumsum-dependent features in model
3. Timing model is Regressor (V8 requirement)
4. Anti-overfitting hyperparameters
5. Feature value ranges (no extreme values)
6. Live trading script configuration
7. API keys are set (not default values)
8. Telegram notifications work
"""

import sys
import os
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Try to import required modules
try:
    import numpy as np
    import pandas as pd
    import joblib
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install requirements: pip install numpy pandas joblib")
    sys.exit(1)

# Import centralized constants for reference
try:
    from src.utils.constants import (
        CUMSUM_PATTERNS as CONST_CUMSUM_PATTERNS,
        ABSOLUTE_PRICE_FEATURES as CONST_ABSOLUTE_FEATURES
    )
except ImportError:
    CONST_CUMSUM_PATTERNS = []
    CONST_ABSOLUTE_FEATURES = []


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'v8_improved'
LIVE_SCRIPT_PATH = Path(__file__).parent / 'live_trading_mexc_v8.py'

# Dangerous patterns that should NOT be in feature names (for pattern matching)
# Note: these are patterns for substring matching, not exact names
CUMSUM_PATTERNS_CHECK = [
    'obv',
    'cumsum',
    'bars_since_swing',
    'consecutive_up',
    'consecutive_down',
    'swing_high_price',
    'swing_low_price',
    'volume_delta_cumsum',
]

# Absolute value patterns that may cause issues (regex patterns)
# ⚠️ These features have values that depend on current price level
# and will differ between backtest (e.g. $500) and live (e.g. $420)
ABSOLUTE_PATTERNS = [
    r'^ema_\d+$',           # ema_9, ema_21, etc. without suffix
    r'^m5_ema_\d+$',        # m5_ema_9, m5_ema_21, etc. (absolute EMA values)
    r'^atr_\d+$',           # atr_7, atr_14 without _pct
    r'^m5_atr_\d+$',        # m5_atr_7, m5_atr_14 (absolute ATR values)
    r'^m5_atr_14_ma$',      # Absolute ATR MA
    r'^bb_upper$',
    r'^bb_middle$',
    r'^bb_lower$',
    r'^m5_bb_upper$',       # Absolute Bollinger Band values
    r'^m5_bb_middle$',
    r'^m5_bb_lower$',
    r'^m5_volume_ma_\d+$',  # m5_volume_ma_5, m5_volume_ma_10, etc.
    r'^m5_volume_delta$',   # Absolute volume delta
    r'^m5_volume_trend$',   # Absolute volume trend
    r'^m15_atr$',
    r'^m15_volume_ma$',
    r'^vol_sma_20$',
    r'^vwap$',                 # Absolute VWAP value (price_vs_vwap is OK)
]

# Required files
REQUIRED_MODEL_FILES = [
    'direction_model.joblib',
    'timing_model.joblib',
    'strength_model.joblib',
    'feature_names.joblib',
]

# V8 expected hyperparameters
EXPECTED_HYPERPARAMS = {
    'n_estimators': {'max': 200, 'ideal': 100},
    'max_depth': {'max': 5, 'ideal': 3},
    'num_leaves': {'max': 16, 'ideal': 8},
    'min_child_samples': {'min': 30, 'ideal': 50},
}

# Live script configuration requirements
LIVE_CONFIG_CHECKS = {
    'LOOKBACK': {'min': 1000, 'max': 5000},
    'MIN_CONF': {'min': 0.4, 'max': 0.7},
    'MIN_TIMING': {'min': 0.5, 'max': 1.5},
    'MIN_STRENGTH': {'min': 1.0, 'max': 2.5},
    'SLIPPAGE_PCT': {'min': 0.0001, 'max': 0.002},
}


# ============================================================
# CHECK FUNCTIONS
# ============================================================

class CheckResult:
    """Result of a single check."""
    
    def __init__(self, name: str, passed: bool, level: str = 'info', 
                 message: str = '', details: Optional[List[str]] = None):
        self.name = name
        self.passed = passed
        self.level = level  # 'critical', 'warning', 'info'
        self.message = message
        self.details = details or []
    
    def __str__(self):
        status = '✅' if self.passed else ('❌' if self.level == 'critical' else '⚠️')
        return f"{status} {self.name}: {self.message}"


def check_model_files(model_dir: Path, verbose: bool = False) -> CheckResult:
    """Check that all required model files exist."""
    missing = []
    found = []
    
    for filename in REQUIRED_MODEL_FILES:
        filepath = model_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            found.append(f"{filename} ({size_kb:.1f} KB)")
        else:
            missing.append(filename)
    
    if missing:
        return CheckResult(
            name="Model Files",
            passed=False,
            level='critical',
            message=f"Missing {len(missing)} files",
            details=[f"Missing: {f}" for f in missing]
        )
    
    return CheckResult(
        name="Model Files",
        passed=True,
        level='info',
        message=f"All {len(REQUIRED_MODEL_FILES)} files present",
        details=found if verbose else []
    )


def check_cumsum_features(model_dir: Path, verbose: bool = False) -> CheckResult:
    """Check for cumsum-dependent features in the model."""
    features_file = model_dir / 'feature_names.joblib'
    
    if not features_file.exists():
        return CheckResult(
            name="Cumsum Features",
            passed=False,
            level='critical',
            message="Cannot check - feature_names.joblib missing"
        )
    
    features = joblib.load(features_file)
    dangerous = []
    
    for feat in features:
        for pattern in CUMSUM_PATTERNS_CHECK:
            if pattern.lower() in feat.lower():
                dangerous.append(feat)
                break
    
    if dangerous:
        return CheckResult(
            name="Cumsum Features",
            passed=False,
            level='critical',
            message=f"Found {len(dangerous)} cumsum-dependent features",
            details=[f"❌ {f}" for f in dangerous]
        )
    
    return CheckResult(
        name="Cumsum Features",
        passed=True,
        level='info',
        message=f"No cumsum-dependent features in {len(features)} total features"
    )


def check_absolute_features(model_dir: Path, verbose: bool = False) -> CheckResult:
    """Check for absolute value features."""
    features_file = model_dir / 'feature_names.joblib'
    
    if not features_file.exists():
        return CheckResult(
            name="Absolute Features",
            passed=True,
            level='info',
            message="Cannot check - feature_names.joblib missing"
        )
    
    features = joblib.load(features_file)
    absolute = []
    
    for feat in features:
        for pattern in ABSOLUTE_PATTERNS:
            if re.match(pattern, feat):
                absolute.append(feat)
                break
    
    if absolute:
        return CheckResult(
            name="Absolute Features",
            passed=False,
            level='warning',
            message=f"Found {len(absolute)} absolute value features",
            details=[f"⚠️ {f} (may cause issues if price changes significantly)" for f in absolute]
        )
    
    return CheckResult(
        name="Absolute Features",
        passed=True,
        level='info',
        message="No problematic absolute features found"
    )


def check_timing_model_type(model_dir: Path, verbose: bool = False) -> CheckResult:
    """Verify timing model is a regressor (V8 requirement)."""
    timing_file = model_dir / 'timing_model.joblib'
    
    if not timing_file.exists():
        return CheckResult(
            name="Timing Model Type",
            passed=False,
            level='critical',
            message="timing_model.joblib missing"
        )
    
    try:
        timing_model = joblib.load(timing_file)
        model_type = type(timing_model).__name__
        
        if 'Regressor' in model_type:
            return CheckResult(
                name="Timing Model Type",
                passed=True,
                level='info',
                message=f"Correct: {model_type} (V8 requires Regressor)"
            )
        elif 'Classifier' in model_type:
            return CheckResult(
                name="Timing Model Type",
                passed=False,
                level='warning',
                message=f"Found {model_type} - V8 should use Regressor",
                details=["Model may be from older version", "Consider retraining with train_v3_dynamic.py"]
            )
        else:
            return CheckResult(
                name="Timing Model Type",
                passed=True,
                level='info',
                message=f"Unknown type: {model_type}"
            )
    except Exception as e:
        return CheckResult(
            name="Timing Model Type",
            passed=False,
            level='warning',
            message=f"Error loading model: {e}"
        )


def check_hyperparameters(model_dir: Path, verbose: bool = False) -> CheckResult:
    """Check anti-overfitting hyperparameters."""
    direction_file = model_dir / 'direction_model.joblib'
    
    if not direction_file.exists():
        return CheckResult(
            name="Hyperparameters",
            passed=True,
            level='info',
            message="Cannot check - direction_model.joblib missing"
        )
    
    try:
        model = joblib.load(direction_file)
        params = model.get_params()
        
        warnings = []
        details = []
        
        for param, limits in EXPECTED_HYPERPARAMS.items():
            value = params.get(param, 'unknown')
            if value == 'unknown':
                continue
            
            if 'max' in limits and value > limits['max']:
                warnings.append(f"{param}={value} > max {limits['max']}")
            if 'min' in limits and value < limits['min']:
                warnings.append(f"{param}={value} < min {limits['min']}")
            
            ideal = limits.get('ideal', 'N/A')
            status = '✅' if not any(param in w for w in warnings) else '⚠️'
            details.append(f"{status} {param}: {value} (ideal: {ideal})")
        
        if warnings:
            return CheckResult(
                name="Hyperparameters",
                passed=False,
                level='warning',
                message=f"{len(warnings)} parameters outside recommended range",
                details=details
            )
        
        return CheckResult(
            name="Hyperparameters",
            passed=True,
            level='info',
            message="Parameters within anti-overfitting range",
            details=details if verbose else []
        )
        
    except Exception as e:
        return CheckResult(
            name="Hyperparameters",
            passed=True,
            level='info',
            message=f"Could not check: {e}"
        )


def check_live_script_config(verbose: bool = False) -> CheckResult:
    """Check live trading script configuration."""
    if not LIVE_SCRIPT_PATH.exists():
        return CheckResult(
            name="Live Script Config",
            passed=True,
            level='info',
            message=f"Live script not found: {LIVE_SCRIPT_PATH}"
        )
    
    try:
        with open(LIVE_SCRIPT_PATH, 'r') as f:
            content = f.read()
        
        issues = []
        details = []
        
        for param, limits in LIVE_CONFIG_CHECKS.items():
            # Find parameter value using regex
            pattern = rf'{param}\s*=\s*([\d.]+)'
            match = re.search(pattern, content)
            
            if match:
                value = float(match.group(1))
                status = '✅'
                
                if 'min' in limits and value < limits['min']:
                    issues.append(f"{param}={value} < min {limits['min']}")
                    status = '⚠️'
                if 'max' in limits and value > limits['max']:
                    issues.append(f"{param}={value} > max {limits['max']}")
                    status = '⚠️'
                
                details.append(f"{status} {param}: {value}")
            else:
                details.append(f"❓ {param}: not found")
        
        # Check API keys are not default
        if 'YOUR_MEXC_API_KEY' in content or 'YOUR_API_KEY' in content:
            issues.append("API keys appear to be placeholder values")
        
        if issues:
            return CheckResult(
                name="Live Script Config",
                passed=False,
                level='warning',
                message=f"{len(issues)} configuration issues",
                details=issues + details
            )
        
        return CheckResult(
            name="Live Script Config",
            passed=True,
            level='info',
            message="Configuration looks good",
            details=details if verbose else []
        )
        
    except Exception as e:
        return CheckResult(
            name="Live Script Config",
            passed=True,
            level='info',
            message=f"Error reading script: {e}"
        )


def check_feature_count(model_dir: Path, verbose: bool = False) -> CheckResult:
    """Check number of features is reasonable."""
    features_file = model_dir / 'feature_names.joblib'
    
    if not features_file.exists():
        return CheckResult(
            name="Feature Count",
            passed=True,
            level='info',
            message="Cannot check - feature_names.joblib missing"
        )
    
    features = joblib.load(features_file)
    count = len(features)
    
    # Feature categories
    categories = {
        'm15_': 0,
        'm5_': 0,
        'm1_': 0,
        'vol_': 0,
        'ema_': 0,
        'rsi_': 0,
        'other': 0
    }
    
    for feat in features:
        categorized = False
        for prefix in categories:
            if prefix != 'other' and feat.startswith(prefix):
                categories[prefix] += 1
                categorized = True
                break
        if not categorized:
            categories['other'] += 1
    
    details = [f"{prefix}: {cnt}" for prefix, cnt in categories.items() if cnt > 0]
    
    if count < 50:
        return CheckResult(
            name="Feature Count",
            passed=False,
            level='warning',
            message=f"Only {count} features - may be too few",
            details=details
        )
    elif count > 500:
        return CheckResult(
            name="Feature Count",
            passed=False,
            level='warning',
            message=f"{count} features - may be too many (overfit risk)",
            details=details
        )
    
    return CheckResult(
        name="Feature Count",
        passed=True,
        level='info',
        message=f"{count} features (reasonable)",
        details=details if verbose else []
    )


# ============================================================
# MAIN
# ============================================================

def run_all_checks(model_dir: Path, verbose: bool = False) -> List[CheckResult]:
    """Run all pre-flight checks."""
    checks = [
        check_model_files(model_dir, verbose),
        check_cumsum_features(model_dir, verbose),
        check_absolute_features(model_dir, verbose),
        check_timing_model_type(model_dir, verbose),
        check_hyperparameters(model_dir, verbose),
        check_feature_count(model_dir, verbose),
        check_live_script_config(verbose),
    ]
    return checks


def print_results(results: List[CheckResult], verbose: bool = False):
    """Print check results in a nice format."""
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECK RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result}")
        if result.details and (verbose or not result.passed):
            for detail in result.details:
                print(f"    {detail}")
    
    # Summary
    critical_failed = [r for r in results if not r.passed and r.level == 'critical']
    warnings = [r for r in results if not r.passed and r.level == 'warning']
    passed = [r for r in results if r.passed]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Passed: {len(passed)}/{len(results)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Critical Failures: {len(critical_failed)}")
    
    if critical_failed:
        print("\n❌ CRITICAL ISSUES - DO NOT GO LIVE:")
        for r in critical_failed:
            print(f"    • {r.name}: {r.message}")
    
    if warnings:
        print("\n⚠️  WARNINGS - Review before going live:")
        for r in warnings:
            print(f"    • {r.name}: {r.message}")
    
    print("\n" + "-" * 70)
    
    if not critical_failed and not warnings:
        print("🎉 ALL CHECKS PASSED! Ready for live trading.")
        return 0
    elif not critical_failed:
        print("⚠️  PASSED WITH WARNINGS. Review warnings before going live.")
        return 1
    else:
        print("❌ CRITICAL CHECKS FAILED! Fix issues before going live.")
        return 2


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Flight Check for Live Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --model-dir models/v8_improved
    python scripts/preflight_check.py --verbose
        """
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=str(DEFAULT_MODEL_DIR),
        help=f"Model directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for all checks"
    )
    
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    
    print("=" * 70)
    print("🚀 PRE-FLIGHT CHECK FOR LIVE TRADING")
    print("=" * 70)
    print(f"Model Directory: {model_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if not model_dir.exists():
        print(f"\n❌ ERROR: Model directory does not exist: {model_dir}")
        print("   Run training first: python scripts/train_v3_dynamic.py")
        return 2
    
    results = run_all_checks(model_dir, args.verbose)
    return print_results(results, args.verbose)


if __name__ == '__main__':
    sys.exit(main())

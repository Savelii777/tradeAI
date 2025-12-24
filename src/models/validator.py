"""
AI Trading Bot - Model Validator
Validates model performance and detects overfitting.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd


class ModelValidator:
    """
    Validates ML models for overfitting and unrealistic performance.
    
    Key checks:
    - Train/validation metric gap (overfitting indicator)
    - Cross-instrument generalization
    - Realistic performance bounds
    - Data leakage detection
    
    Attributes:
        thresholds: Dictionary of validation thresholds.
        logger: Logger instance.
    """
    
    # Default thresholds for overfitting detection
    DEFAULT_THRESHOLDS = {
        'max_train_val_gap': 0.20,  # Max 20% difference
        'max_realistic_winrate': 0.70,  # 70% max
        'max_realistic_monthly_return': 2.0,  # 200% max
        'min_realistic_sharpe': 1.0,  # Sharpe should be >1 for high returns
        'max_realistic_drawdown': 0.30,  # 30% max
        'min_realistic_profit_factor': 1.2,
        'max_realistic_profit_factor': 5.0,
    }
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize ModelValidator.
        
        Args:
            thresholds: Custom validation thresholds.
            logger: Optional logger instance.
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_no_leakage(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Tuple[bool, List[str]]:
        """
        Verify that test data doesn't overlap with training data.
        
        Args:
            train_data: Training DataFrame with datetime index.
            test_data: Test DataFrame with datetime index.
            
        Returns:
            Tuple of (is_valid, list of issues).
        """
        issues = []
        
        # Get time ranges
        def get_range(df):
            if isinstance(df.index, pd.DatetimeIndex):
                return df.index.min(), df.index.max()
            elif 'timestamp' in df.columns:
                return df['timestamp'].min(), df['timestamp'].max()
            return None, None
        
        train_start, train_end = get_range(train_data)
        test_start, test_end = get_range(test_data)
        
        if train_end is None or test_start is None:
            issues.append("Cannot verify leakage: missing datetime info")
            return False, issues
        
        # Check for overlap
        if test_start <= train_end:
            issues.append(
                f"DATA LEAKAGE: Test data starts ({test_start}) before "
                f"train data ends ({train_end})"
            )
            return False, issues
        
        # Check for suspicious gap (might indicate data issues)
        gap = (test_start - train_end).total_seconds()
        if gap < 0:
            issues.append(f"Negative gap between train and test: {gap}s")
            return False, issues
        
        self.logger.info(
            f"✓ No leakage: train ends {train_end}, test starts {test_start}"
        )
        return True, issues
    
    def check_overfitting(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        Check if model shows signs of overfitting.
        
        Overfitting indicators:
        - Train accuracy >> Validation accuracy
        - Train loss << Validation loss
        - Perfect or near-perfect train metrics
        
        Args:
            train_metrics: Metrics on training data.
            val_metrics: Metrics on validation data.
            
        Returns:
            Tuple of (is_overfitting, list of warnings).
        """
        warnings = []
        is_overfitting = False
        max_gap = self.thresholds['max_train_val_gap']
        
        # Check accuracy gap
        train_acc = train_metrics.get('accuracy', train_metrics.get('direction_accuracy'))
        val_acc = val_metrics.get('accuracy', val_metrics.get('direction_accuracy'))
        
        if train_acc is not None and val_acc is not None:
            gap = train_acc - val_acc
            if gap > max_gap:
                warnings.append(
                    f"⚠️ OVERFITTING: Train accuracy ({train_acc:.3f}) is "
                    f"{gap:.1%} higher than validation ({val_acc:.3f})"
                )
                is_overfitting = True
            
            # Perfect training is suspicious
            if train_acc > 0.99:
                warnings.append(
                    f"⚠️ SUSPICIOUS: Perfect train accuracy ({train_acc:.3f}) "
                    "suggests memorization"
                )
                is_overfitting = True
        
        # Check F1 score gap
        train_f1 = train_metrics.get('f1', train_metrics.get('direction_f1'))
        val_f1 = val_metrics.get('f1', val_metrics.get('direction_f1'))
        
        if train_f1 is not None and val_f1 is not None:
            gap = train_f1 - val_f1
            if gap > max_gap:
                warnings.append(
                    f"⚠️ OVERFITTING: Train F1 ({train_f1:.3f}) is "
                    f"{gap:.1%} higher than validation ({val_f1:.3f})"
                )
                is_overfitting = True
        
        # Check AUC gap
        train_auc = train_metrics.get('auc', train_metrics.get('timing_auc'))
        val_auc = val_metrics.get('auc', val_metrics.get('timing_auc'))
        
        if train_auc is not None and val_auc is not None:
            gap = train_auc - val_auc
            if gap > max_gap:
                warnings.append(
                    f"⚠️ OVERFITTING: Train AUC ({train_auc:.3f}) is "
                    f"{gap:.1%} higher than validation ({val_auc:.3f})"
                )
                is_overfitting = True
        
        # Check MAE gap (for regression)
        train_mae = train_metrics.get('mae', train_metrics.get('strength_mae'))
        val_mae = val_metrics.get('mae', val_metrics.get('strength_mae'))
        
        if train_mae is not None and val_mae is not None:
            if val_mae > 0:
                ratio = train_mae / val_mae
                if ratio < (1 - max_gap):
                    warnings.append(
                        f"⚠️ OVERFITTING: Train MAE ({train_mae:.4f}) is much "
                        f"lower than validation ({val_mae:.4f})"
                    )
                    is_overfitting = True
        
        if not is_overfitting:
            self.logger.info("✓ No significant overfitting detected")
        else:
            for w in warnings:
                self.logger.warning(w)
        
        return is_overfitting, warnings
    
    def validate_results(
        self,
        backtest_results: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        Validate backtest results for realism.
        
        Unrealistic indicators:
        - Win rate > 70% (very rare in real trading)
        - Monthly return > 200% (unsustainable)
        - High returns with low Sharpe (contradictory)
        - High win rate with high drawdown (contradictory)
        
        Args:
            backtest_results: Dictionary of backtest metrics.
            
        Returns:
            Tuple of (is_realistic, list of warnings).
        """
        warnings = []
        is_realistic = True
        
        # Extract metrics
        win_rate = backtest_results.get('win_rate', 0)
        if win_rate > 1:
            win_rate = win_rate / 100  # Convert from percentage
        
        total_return = backtest_results.get('total_return', 0)
        if total_return > 100:
            total_return = total_return / 100  # Convert from percentage
        
        sharpe = backtest_results.get('sharpe_ratio', 0)
        max_dd = backtest_results.get('max_drawdown', 0)
        if max_dd > 1:
            max_dd = max_dd / 100  # Convert from percentage
        
        profit_factor = backtest_results.get('profit_factor', 1)
        num_trades = backtest_results.get('total_trades', 0)
        
        # Check win rate
        max_wr = self.thresholds['max_realistic_winrate']
        if win_rate > max_wr:
            warnings.append(
                f"⚠️ UNREALISTIC: Win rate {win_rate:.1%} exceeds "
                f"realistic maximum of {max_wr:.0%}"
            )
            is_realistic = False
        
        # Check monthly return (assume 30 days of data = 1 month)
        # This is simplified; adjust based on actual period
        max_monthly = self.thresholds['max_realistic_monthly_return']
        if total_return > max_monthly:
            warnings.append(
                f"⚠️ UNREALISTIC: Return {total_return:.1%} exceeds "
                f"realistic maximum of {max_monthly:.0%}"
            )
            is_realistic = False
        
        # Check Sharpe consistency
        min_sharpe = self.thresholds['min_realistic_sharpe']
        if total_return > 1.0 and sharpe < min_sharpe:  # >100% return
            warnings.append(
                f"⚠️ INCONSISTENT: High return ({total_return:.1%}) with "
                f"low Sharpe ({sharpe:.2f}) is contradictory"
            )
            is_realistic = False
        
        # Check drawdown vs win rate consistency
        max_dd_threshold = self.thresholds['max_realistic_drawdown']
        if win_rate > 0.8 and max_dd > max_dd_threshold:
            warnings.append(
                f"⚠️ INCONSISTENT: High win rate ({win_rate:.1%}) with "
                f"high drawdown ({max_dd:.1%}) is contradictory"
            )
            is_realistic = False
        
        # Check profit factor
        max_pf = self.thresholds['max_realistic_profit_factor']
        if profit_factor > max_pf:
            warnings.append(
                f"⚠️ UNREALISTIC: Profit factor {profit_factor:.2f} exceeds "
                f"realistic maximum of {max_pf:.1f}"
            )
            is_realistic = False
        
        # Check for too few trades (not statistically significant)
        if num_trades < 30:
            warnings.append(
                f"⚠️ INSUFFICIENT DATA: Only {num_trades} trades, "
                "need at least 30 for statistical significance"
            )
        
        if is_realistic:
            self.logger.info("✓ Backtest results appear realistic")
        else:
            self.logger.warning("❌ Backtest results show signs of overfitting")
            for w in warnings:
                self.logger.warning(w)
        
        return is_realistic, warnings
    
    async def cross_instrument_test(
        self,
        model: Any,
        feature_engine: Any,
        instruments: List[str],
        fetch_data_func: Any,
        days: int = 90,
    ) -> Dict[str, Dict[str, float]]:
        """
        Test model on multiple instruments to check generalization.
        
        A well-generalized model should show positive results on
        instruments it wasn't trained on.
        
        Args:
            model: Trained model with predict method.
            feature_engine: FeatureEngine instance.
            instruments: List of instrument symbols.
            fetch_data_func: Async function to fetch data.
            days: Days of data to test.
            
        Returns:
            Dictionary mapping instrument to its metrics.
        """
        results = {}
        
        for symbol in instruments:
            try:
                self.logger.info(f"Testing on {symbol}...")
                
                # Fetch data
                df = await fetch_data_func(symbol, '5m', days)
                if df is None or len(df) < 500:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Generate features
                features = feature_engine.generate_all_features(df, normalize=True)
                
                # Handle categorical columns
                for col in features.columns:
                    if features[col].dtype == 'object':
                        features[col] = pd.Categorical(features[col]).codes
                
                # Get predictions
                predictions = model.predict(features.dropna())
                
                # Calculate simple metrics
                if 'direction_proba_up' in predictions:
                    up_proba = predictions['direction_proba_up']
                    
                    # Calculate accuracy based on actual price movement
                    df_aligned = df.iloc[features.dropna().index]
                    actual_direction = (
                        df_aligned['close'].shift(-5) > df_aligned['close']
                    ).dropna()
                    
                    if len(actual_direction) > 0 and len(up_proba) > 0:
                        min_len = min(len(actual_direction), len(up_proba))
                        predicted_up = up_proba[:min_len] > 0.5
                        actual_up = actual_direction.values[:min_len]
                        
                        accuracy = (predicted_up == actual_up).mean()
                        results[symbol] = {
                            'accuracy': float(accuracy),
                            'samples': min_len,
                        }
                        self.logger.info(f"{symbol}: accuracy={accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error testing {symbol}: {e}")
                continue
        
        # Check generalization
        if len(results) >= 2:
            accuracies = [r['accuracy'] for r in results.values()]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            if mean_acc < 0.5:
                self.logger.warning(
                    "❌ Poor cross-instrument performance suggests overfitting"
                )
            elif std_acc > 0.15:
                self.logger.warning(
                    f"⚠️ High variance ({std_acc:.3f}) across instruments"
                )
            else:
                self.logger.info(
                    f"✓ Good generalization: mean accuracy {mean_acc:.3f} "
                    f"(std: {std_acc:.3f})"
                )
        
        return results
    
    def validate_fold_stability(
        self,
        fold_results: List[Dict[str, float]],
        max_cv: float = 0.30,  # Max coefficient of variation
    ) -> Tuple[bool, List[str]]:
        """
        Check if results are stable across walk-forward folds.
        
        High variance across folds indicates model is not robust.
        
        Args:
            fold_results: List of metrics from each fold.
            max_cv: Maximum acceptable coefficient of variation.
            
        Returns:
            Tuple of (is_stable, list of warnings).
        """
        warnings = []
        is_stable = True
        
        if len(fold_results) < 3:
            warnings.append("Too few folds to assess stability")
            return True, warnings
        
        metrics_to_check = ['accuracy', 'direction_accuracy', 'sharpe_ratio', 'win_rate']
        
        for metric in metrics_to_check:
            values = [r.get(metric) for r in fold_results if metric in r]
            if len(values) < 2:
                continue
            
            values = [v for v in values if v is not None and not np.isnan(v)]
            if not values:
                continue
            
            mean = np.mean(values)
            std = np.std(values)
            
            if mean > 0:
                cv = std / mean
                if cv > max_cv:
                    warnings.append(
                        f"⚠️ UNSTABLE: {metric} has high variance "
                        f"(CV={cv:.2f}, mean={mean:.3f}, std={std:.3f})"
                    )
                    is_stable = False
        
        if is_stable:
            self.logger.info("✓ Results are stable across folds")
        
        return is_stable, warnings
    
    def generate_validation_report(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        backtest_results: Optional[Dict[str, float]] = None,
        fold_results: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            test_metrics: Test metrics.
            backtest_results: Optional backtest results.
            fold_results: Optional walk-forward fold results.
            
        Returns:
            Validation report dictionary.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics,
            },
            'checks': {},
            'warnings': [],
            'is_valid': True,
        }
        
        # Check overfitting
        is_overfitting, overfit_warnings = self.check_overfitting(
            train_metrics, val_metrics
        )
        report['checks']['overfitting'] = not is_overfitting
        report['warnings'].extend(overfit_warnings)
        if is_overfitting:
            report['is_valid'] = False
        
        # Check backtest results
        if backtest_results:
            is_realistic, bt_warnings = self.validate_results(backtest_results)
            report['checks']['realistic_backtest'] = is_realistic
            report['warnings'].extend(bt_warnings)
            if not is_realistic:
                report['is_valid'] = False
        
        # Check fold stability
        if fold_results:
            is_stable, stability_warnings = self.validate_fold_stability(fold_results)
            report['checks']['fold_stability'] = is_stable
            report['warnings'].extend(stability_warnings)
            if not is_stable:
                report['is_valid'] = False
        
        # Summary
        if report['is_valid']:
            report['summary'] = "✓ Model passed all validation checks"
        else:
            report['summary'] = (
                f"❌ Model failed validation with {len(report['warnings'])} warnings"
            )
        
        return report

#!/usr/bin/env python3
"""
Feature Selection for MTF Trading Bot

Analyzes feature importance and performs recursive feature elimination
to reduce dimensionality and improve model generalization.

Usage:
    python scripts/feature_selection.py --target-features 50
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
import joblib
import lightgbm as lgb
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# TOP PAIRS
# ============================================================

TOP_20_PAIRS = [
    'XAUT/USDT:USDT', 'BTC/USDT:USDT', 'BNB/USDT:USDT', 'TONCOIN/USDT:USDT',
    'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
    'LTC/USDT:USDT', 'BCH/USDT:USDT', 'UNI/USDT:USDT', 'AAVE/USDT:USDT',
    'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT', 'OP/USDT:USDT',
]


# ============================================================
# FEATURE ANALYSIS
# ============================================================

class FeatureAnalyzer:
    """
    Analyzes and selects important features for the MTF model.
    """
    
    def __init__(
        self,
        model_path: str = './models/saved_mtf',
        data_dir: str = './data/candles'
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        
    def get_feature_importance_from_model(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from trained ensemble model."""
        from src.models.ensemble import EnsembleModel
        
        model = EnsembleModel()
        model.load(str(self.model_path))
        
        return model.get_feature_importance()
    
    def analyze_correlation(
        self, 
        features: pd.DataFrame, 
        threshold: float = 0.95
    ) -> List[str]:
        """
        Find highly correlated features to remove.
        
        Args:
            features: Feature DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to remove
        """
        logger.info(f"Analyzing correlation (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_remove = [
            column for column in upper.columns 
            if any(upper[column] > threshold)
        ]
        
        logger.info(f"Found {len(to_remove)} highly correlated features to remove")
        
        return to_remove
    
    def analyze_zero_importance(
        self, 
        importance_df: pd.DataFrame,
        min_importance: float = 0.0
    ) -> List[str]:
        """Find features with zero or very low importance."""
        zero_importance = importance_df[
            importance_df['importance'] <= min_importance
        ]['feature'].tolist()
        
        logger.info(f"Found {len(zero_importance)} features with zero/low importance")
        
        return zero_importance
    
    def run_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int = 50,
        step: int = 10,
        cv_folds: int = 5
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Run Recursive Feature Elimination with Cross-Validation.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            target_features: Minimum features to select
            step: Number of features to remove per iteration
            cv_folds: Number of CV folds
            
        Returns:
            Tuple of (selected features list, ranking DataFrame)
        """
        logger.info(f"Running RFE (target={target_features} features)...")
        
        # Create base estimator
        estimator = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            verbosity=-1,
            random_state=42
        )
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # RFE with CV
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=tscv,
            scoring='accuracy',
            min_features_to_select=target_features,
            n_jobs=-1
        )
        
        # Fit selector
        logger.info("Fitting RFE (this may take a while)...")
        selector.fit(X, y)
        
        # Get results
        selected_features = X.columns[selector.support_].tolist()
        
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('ranking')
        
        logger.info(f"RFE selected {len(selected_features)} features")
        logger.info(f"Optimal number of features: {selector.n_features_}")
        
        return selected_features, ranking_df
    
    def run_fast_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int = 50,
        step: int = 10
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Run fast RFE without cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            target_features: Number of features to select
            step: Number of features to remove per iteration
            
        Returns:
            Tuple of (selected features list, ranking DataFrame)
        """
        logger.info(f"Running fast RFE (target={target_features} features)...")
        
        # Create base estimator
        estimator = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            verbosity=-1,
            random_state=42
        )
        
        # RFE without CV
        selector = RFE(
            estimator=estimator,
            n_features_to_select=target_features,
            step=step
        )
        
        # Fit selector
        selector.fit(X, y)
        
        # Get results
        selected_features = X.columns[selector.support_].tolist()
        
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('ranking')
        
        logger.info(f"Fast RFE selected {len(selected_features)} features")
        
        return selected_features, ranking_df
    
    def select_top_features(
        self,
        importance_df: pd.DataFrame,
        n_features: int = 50,
        min_importance: float = 0.0
    ) -> List[str]:
        """
        Select top N features by importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            n_features: Number of features to select
            min_importance: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        # Filter by minimum importance
        filtered = importance_df[importance_df['importance'] > min_importance]
        
        # Sort and take top N
        top_features = filtered.nlargest(n_features, 'importance')['feature'].tolist()
        
        logger.info(f"Selected top {len(top_features)} features by importance")
        
        return top_features
    
    def generate_feature_report(
        self,
        importance_dict: Dict[str, pd.DataFrame],
        output_path: str = './reports/feature_importance.csv'
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature importance report.
        
        Args:
            importance_dict: Dict mapping model name to importance DataFrame
            output_path: Path to save report
            
        Returns:
            Combined importance DataFrame
        """
        # Combine importances from all models
        combined = pd.DataFrame()
        
        for model_name, imp_df in importance_dict.items():
            if combined.empty:
                combined['feature'] = imp_df['feature']
            combined[f'{model_name}_importance'] = imp_df.set_index('feature').reindex(
                combined['feature']
            )['importance'].values
        
        # Calculate average importance
        importance_cols = [c for c in combined.columns if 'importance' in c]
        combined['avg_importance'] = combined[importance_cols].mean(axis=1)
        combined['max_importance'] = combined[importance_cols].max(axis=1)
        
        # Sort by average importance
        combined = combined.sort_values('avg_importance', ascending=False)
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        
        logger.info(f"Feature report saved to {output_path}")
        
        return combined
    
    def get_selected_features_mask(
        self,
        all_features: List[str],
        selected_features: List[str]
    ) -> np.ndarray:
        """Create boolean mask for selected features."""
        return np.array([f in selected_features for f in all_features])


def save_selected_features(
    features: List[str],
    output_path: str = './config/selected_features.json'
) -> None:
    """Save selected features to JSON file."""
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'selected_features': features,
            'num_features': len(features)
        }, f, indent=2)
    
    logger.info(f"Saved {len(features)} selected features to {output_path}")


def load_selected_features(path: str = './config/selected_features.json') -> List[str]:
    """Load selected features from JSON file."""
    import json
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data['selected_features']


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Selection")
    parser.add_argument("--target-features", type=int, default=50, help="Target number of features")
    parser.add_argument("--method", type=str, default="importance", 
                        choices=["importance", "rfe", "fast_rfe", "correlation"],
                        help="Selection method")
    parser.add_argument("--model-path", type=str, default="./models/saved_mtf")
    parser.add_argument("--data-dir", type=str, default="./data/candles")
    parser.add_argument("--pairs", type=int, default=10, help="Number of pairs for RFE")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FEATURE SELECTION")
    logger.info("="*60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Target features: {args.target_features}")
    
    analyzer = FeatureAnalyzer(
        model_path=args.model_path,
        data_dir=args.data_dir
    )
    
    # Get importance from trained model
    importance_dict = analyzer.get_feature_importance_from_model()
    
    # Generate report
    report = analyzer.generate_feature_report(importance_dict)
    
    # Print top features from each model
    print("\nTop 20 Features by Model:")
    print("-"*60)
    
    for model_name, imp_df in importance_dict.items():
        print(f"\n{model_name.upper()}:")
        for i, row in imp_df.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Select features based on method
    if args.method == "importance":
        # Use average importance across models
        selected = report.nlargest(args.target_features, 'avg_importance')['feature'].tolist()
        
    elif args.method == "correlation":
        # First get top features, then remove correlated
        top_features = report.nlargest(args.target_features * 2, 'avg_importance')['feature'].tolist()
        
        # Load data for correlation analysis
        from train_mtf import prepare_mtf_data
        X, y, pairs_col = prepare_mtf_data(TOP_20_PAIRS[:args.pairs], args.data_dir)
        
        X_subset = X[top_features]
        correlated = analyzer.analyze_correlation(X_subset)
        
        selected = [f for f in top_features if f not in correlated][:args.target_features]
        
    elif args.method == "rfe":
        # Load data for RFE
        from train_mtf import prepare_mtf_data
        X, y, pairs_col = prepare_mtf_data(TOP_20_PAIRS[:args.pairs], args.data_dir)
        
        selected, ranking = analyzer.run_rfe(
            X, y['direction'],
            target_features=args.target_features
        )
        
        # Save ranking
        ranking.to_csv('./reports/rfe_ranking.csv', index=False)
        
    elif args.method == "fast_rfe":
        # Load data for fast RFE
        from train_mtf import prepare_mtf_data
        X, y, pairs_col = prepare_mtf_data(TOP_20_PAIRS[:args.pairs], args.data_dir)
        
        selected, ranking = analyzer.run_fast_rfe(
            X, y['direction'],
            target_features=args.target_features
        )
        
        # Save ranking
        ranking.to_csv('./reports/rfe_ranking.csv', index=False)
    
    # Save selected features
    save_selected_features(selected)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE SELECTION RESULTS")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Original features: {len(report)}")
    print(f"Selected features: {len(selected)}")
    print(f"Reduction: {(1 - len(selected)/len(report)) * 100:.1f}%")
    
    print("\nSelected Features:")
    print("-"*60)
    for i, f in enumerate(selected[:30], 1):
        print(f"  {i}. {f}")
    if len(selected) > 30:
        print(f"  ... and {len(selected) - 30} more")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

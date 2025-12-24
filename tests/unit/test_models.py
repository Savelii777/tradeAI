"""
AI Trading Bot - Unit Tests for Models
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import DirectionModel, StrengthModel, VolatilityModel, TimingModel, EnsembleModel


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    return features


@pytest.fixture
def sample_labels(sample_features):
    """Create sample labels for testing."""
    n_samples = len(sample_features)
    np.random.seed(42)
    
    return {
        'direction': pd.Series(np.random.choice([-1, 0, 1], n_samples)),
        'strength': pd.Series(np.random.uniform(0, 3, n_samples)),
        'volatility': pd.Series(np.random.uniform(0, 0.05, n_samples)),
        'timing': pd.Series(np.random.choice([0, 1], n_samples))
    }


class TestDirectionModel:
    """Tests for DirectionModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = DirectionModel()
        assert model.model_type == "lightgbm"
        assert not model._is_trained
        
    def test_model_training(self, sample_features, sample_labels):
        """Test model training."""
        model = DirectionModel()
        
        # Split data
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = sample_labels['direction'].iloc[:train_size]
        X_val = sample_features.iloc[train_size:]
        y_val = sample_labels['direction'].iloc[train_size:]
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        assert model._is_trained
        assert 'train_accuracy' in metrics
        assert metrics['train_accuracy'] > 0
        
    def test_model_prediction(self, sample_features, sample_labels):
        """Test model prediction."""
        model = DirectionModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = sample_labels['direction'].iloc[:train_size]
        
        model.train(X_train, y_train)
        
        predictions = model.predict(sample_features.iloc[train_size:])
        
        assert len(predictions) == len(sample_features) - train_size
        assert all(p in [-1, 0, 1] for p in predictions)
        
    def test_probability_prediction(self, sample_features, sample_labels):
        """Test probability prediction."""
        model = DirectionModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = sample_labels['direction'].iloc[:train_size]
        
        model.train(X_train, y_train)
        
        probas = model.predict_proba(sample_features.iloc[train_size:])
        
        assert probas.shape[1] == 3  # Three classes
        assert all(np.abs(probas.sum(axis=1) - 1) < 0.01)  # Probabilities sum to 1
        
    def test_feature_importance(self, sample_features, sample_labels):
        """Test feature importance extraction."""
        model = DirectionModel()
        
        model.train(sample_features, sample_labels['direction'])
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestStrengthModel:
    """Tests for StrengthModel."""
    
    def test_model_training(self, sample_features, sample_labels):
        """Test model training."""
        model = StrengthModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = sample_labels['strength'].iloc[:train_size]
        
        metrics = model.train(X_train, y_train)
        
        assert model._is_trained
        assert 'train_mae' in metrics
        
    def test_model_prediction(self, sample_features, sample_labels):
        """Test model prediction."""
        model = StrengthModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = sample_labels['strength'].iloc[:train_size]
        
        model.train(X_train, y_train)
        predictions = model.predict(sample_features.iloc[train_size:])
        
        assert len(predictions) == len(sample_features) - train_size


class TestEnsembleModel:
    """Tests for EnsembleModel."""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleModel()
        
        assert ensemble.direction_model is not None
        assert ensemble.strength_model is not None
        assert ensemble.volatility_model is not None
        assert ensemble.timing_model is not None
        
    def test_ensemble_training(self, sample_features, sample_labels):
        """Test ensemble training."""
        ensemble = EnsembleModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = {k: v.iloc[:train_size] for k, v in sample_labels.items()}
        X_val = sample_features.iloc[train_size:]
        y_val = {k: v.iloc[train_size:] for k, v in sample_labels.items()}
        
        metrics = ensemble.train_all(X_train, y_train, X_val, y_val)
        
        assert ensemble._is_trained
        assert 'direction' in metrics
        
    def test_ensemble_prediction(self, sample_features, sample_labels):
        """Test ensemble prediction."""
        ensemble = EnsembleModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = {k: v.iloc[:train_size] for k, v in sample_labels.items()}
        
        ensemble.train_all(X_train, y_train)
        
        predictions = ensemble.predict(sample_features.iloc[train_size:])
        
        assert 'direction' in predictions
        assert 'strength' in predictions
        
    def test_trading_signal_generation(self, sample_features, sample_labels):
        """Test trading signal generation."""
        ensemble = EnsembleModel()
        
        train_size = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:train_size]
        y_train = {k: v.iloc[:train_size] for k, v in sample_labels.items()}
        
        ensemble.train_all(X_train, y_train)
        
        # Test single sample
        signal = ensemble.get_trading_signal(sample_features.iloc[[train_size]])
        
        assert 'signal' in signal
        assert signal['signal'] in ['buy', 'sell', 'hold']
        assert 'confidence' in signal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

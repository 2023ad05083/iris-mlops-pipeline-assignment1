import pytest
import numpy as np
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel

@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 3, 100)
    return X, y

def test_logistic_regression_model(sample_data):
    """Test logistic regression model"""
    X, y = sample_data
    
    model = LogisticRegressionModel()
    model.train(X, y)
    
    assert model.is_trained
    assert model.model is not None
    
    # Test prediction
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert all(pred in [0, 1, 2] for pred in predictions)

def test_random_forest_model(sample_data):
    """Test random forest model"""
    X, y = sample_data
    
    model = RandomForestModel()
    model.train(X, y, n_estimators=10)  # Small number for testing
    
    assert model.is_trained
    assert model.model is not None
    
    # Test prediction
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    
    # Test probability prediction
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, 3)

def test_svm_model(sample_data):
    """Test SVM model"""
    X, y = sample_data
    
    model = SVMModel()
    model.train(X, y)
    
    assert model.is_trained
    assert model.model is not None
    
    # Test prediction
    predictions = model.predict(X[:10])
    assert len(predictions) == 10

def test_model_evaluation(sample_data):
    """Test model evaluation"""
    X, y = sample_data
    
    model = LogisticRegressionModel()
    model.train(X, y)
    
    metrics, predictions = model.evaluate(X, y)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert len(predictions) == len(y)
    assert 0 <= metrics['accuracy'] <= 1
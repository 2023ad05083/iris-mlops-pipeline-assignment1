import pytest
import pandas as pd
import numpy as np
from src.data.load_data import load_iris_data
from src.data.preprocess import DataPreprocessor

def test_load_iris_data():
    """Test iris data loading"""
    df = load_iris_data()
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 150  # Iris dataset has 150 samples
    assert df.shape[1] == 6    # 4 features + target + target_name
    assert 'target' in df.columns
    assert 'target_name' in df.columns

def test_data_preprocessor():
    """Test data preprocessing"""
    # Create sample data
    X = np.random.rand(10, 4)
    
    preprocessor = DataPreprocessor()
    
    # Test fit_transform
    X_scaled = preprocessor.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert preprocessor.is_fitted
    
    # Test transform
    X_new = np.random.rand(5, 4)
    X_new_scaled = preprocessor.transform(X_new)
    
    assert X_new_scaled.shape == X_new.shape

def test_preprocessor_not_fitted():
    """Test error when using unfitted preprocessor"""
    preprocessor = DataPreprocessor()
    X = np.random.rand(5, 4)
    
    with pytest.raises(ValueError):
        preprocessor.transform(X)
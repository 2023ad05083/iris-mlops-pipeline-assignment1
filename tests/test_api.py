from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock
import json


# We need to mock the model loading since we don't have trained models in tests
@patch("api.main.load_model_and_scaler")
@patch("api.main.model")
@patch("api.main.scaler")
@patch("api.main.model_info")
def test_api_endpoints(mock_model_info, mock_scaler, mock_model, mock_load):
    """Test API endpoints"""
    # Setup mocks
    mock_load.return_value = True
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_scaler.transform.return_value = np.array([[1, 2, 3, 4]])
    mock_model_info.get.return_value = {"accuracy": 0.95, "model_type": "test"}

    from api.main import app

    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

    # Test metrics endpoint - handle both JSON and Prometheus formats
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Check if response is JSON format (basic metrics)
    try:
        metrics_data = response.json()
        # Should have basic metrics structure
        assert any(key in metrics_data for key in ["total_predictions", "uptime_seconds", "model_accuracy"])
    except json.JSONDecodeError:
        # If not JSON, should be Prometheus format (text)
        response_text = response.text
        assert any(metric in response_text for metric in ["total_predictions", "uptime_seconds", "api_requests"])


@patch("api.main.model")
@patch("api.main.scaler")
@patch("api.main.model_info")
def test_prediction_endpoint(mock_model_info, mock_scaler, mock_model):
    """Test prediction endpoint"""
    # Setup mocks
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_scaler.transform.return_value = np.array([[1, 2, 3, 4]])
    mock_model_info.get.return_value = {"accuracy": 0.95, "model_type": "test"}

    from api.main import app

    client = TestClient(app)

    # Test data
    test_data = {
        "features": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ]
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200

    result = response.json()
    assert "predictions" in result
    assert "request_id" in result
    assert len(result["predictions"]) == 1
    
    # Verify prediction structure
    prediction = result["predictions"][0]
    assert "prediction" in prediction
    assert "prediction_name" in prediction
    assert "confidence" in prediction
    assert "probabilities" in prediction


def test_invalid_prediction_data():
    """Test prediction with invalid data"""
    from api.main import app

    client = TestClient(app)

    # Test with invalid data - negative values should be rejected by Pydantic
    invalid_data = {
        "features": [
            {
                "sepal_length": -1,  # Invalid negative value
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ]
    }

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_missing_model_scenario():
    """Test API behavior when model is not loaded"""
    from api.main import app

    # Create fresh client without mocked model
    client = TestClient(app)

    # Test prediction without loaded model should return 503
    test_data = {
        "features": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ]
    }

    response = client.post("/predict", json=test_data)
    # Should return 503 Service Unavailable when model not loaded
    assert response.status_code == 503


@patch("api.main.model")
@patch("api.main.scaler") 
@patch("api.main.model_info")
def test_batch_prediction(mock_model_info, mock_scaler, mock_model):
    """Test batch prediction with multiple samples"""
    # Setup mocks
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    mock_scaler.transform.side_effect = [np.array([[1, 2, 3, 4]]), np.array([[5, 6, 7, 8]])]
    mock_model_info.get.return_value = {"accuracy": 0.95, "model_type": "test"}

    from api.main import app

    client = TestClient(app)

    # Test data with multiple samples
    test_data = {
        "features": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
            {
                "sepal_length": 6.2,
                "sepal_width": 2.9,
                "petal_length": 4.3,
                "petal_width": 1.3,
            }
        ]
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200

    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 2  # Should handle batch prediction
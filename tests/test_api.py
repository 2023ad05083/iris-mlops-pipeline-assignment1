from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch


# We need to mock the model loading since we don't have trained models in tests
@patch("api.main.load_model_and_scaler")
@patch("api.main.model")
@patch("api.main.scaler")
def test_api_endpoints(mock_scaler, mock_model, mock_load):
    """Test API endpoints"""
    # Setup mocks
    mock_load.return_value = True
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_scaler.transform.return_value = np.array([[1, 2, 3, 4]])

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

    # Test metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_predictions" in response.json()


@patch("api.main.model")
@patch("api.main.scaler")
def test_prediction_endpoint(mock_scaler, mock_model):
    """Test prediction endpoint"""
    # Setup mocks
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_scaler.transform.return_value = np.array([[1, 2, 3, 4]])

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


def test_invalid_prediction_data():
    """Test prediction with invalid data"""
    from api.main import app

    client = TestClient(app)

    # Test with invalid data
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

import os
from pathlib import Path

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "iris_classification"
MODEL_REGISTRY_NAME = "iris_model"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)

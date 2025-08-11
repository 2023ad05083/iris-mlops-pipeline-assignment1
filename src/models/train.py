import pandas as pd
import joblib
import json
from pathlib import Path
import logging

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import models
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel

# Import MLflow utilities
from src.utils.mlflow_utils import setup_mlflow, log_model_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load preprocessed training and test data"""
    try:
        train_df = pd.read_csv("data/processed/train_scaled.csv")
        test_df = pd.read_csv("data/processed/test_scaled.csv")

        feature_columns = [col for col in train_df.columns if col != "target"]

        X_train = train_df[feature_columns].values
        y_train = train_df["target"].values
        X_test = test_df[feature_columns].values
        y_test = test_df["target"].values

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test, feature_columns
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_models():
    """Train multiple models and track experiments"""
    # Setup MLflow
    setup_mlflow()

    # Load data
    X_train, X_test, y_train, y_test, feature_columns = load_data()

    # Define models and their hyperparameters
    model_configs = {
        "logistic_regression": {
            "model_class": LogisticRegressionModel,
            "params": [
                {"C": 1.0, "solver": "liblinear"},
                {"C": 0.1, "solver": "liblinear"},
                {"C": 10.0, "solver": "liblinear"},
            ],
        },
        "random_forest": {
            "model_class": RandomForestModel,
            "params": [
                {"n_estimators": 100, "max_depth": None},
                {"n_estimators": 50, "max_depth": 5},
                {"n_estimators": 200, "max_depth": 10},
            ],
        },
        "svm": {
            "model_class": SVMModel,
            "params": [
                {"C": 1.0, "kernel": "rbf"},
                {"C": 0.1, "kernel": "rbf"},
                {"C": 1.0, "kernel": "linear"},
            ],
        },
    }

    best_model = None
    best_accuracy = 0
    best_model_info = {}

    # Train and evaluate models
    for model_type, config in model_configs.items():
        model_class = config["model_class"]

        for i, params in enumerate(config["params"]):
            logger.info(f"Training {model_type} with params: {params}")

            # Create and train model
            model = model_class()
            model.train(X_train, y_train, **params)

            # Evaluate model
            metrics, y_pred = model.evaluate(X_test, y_test)

            # Prepare MLflow logging
            run_name = f"{model_type}_run_{i+1}"
            mlflow_params = {"model_type": model_type, **params}

            # Log to MLflow
            run_id = log_model_artifacts(
                model=model.model,
                model_name=run_name,
                metrics=metrics,
                params=mlflow_params,
            )

            # Track best model
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_model = model
                best_model_info = {
                    "model_type": model_type,
                    "params": params,
                    "metrics": metrics,
                    "run_id": run_id,
                }

            logger.info(f"{run_name} - Accuracy: {metrics['accuracy']:.4f}")

    # Save best model
    if best_model:
        joblib.dump(best_model.model, "models/best_model.joblib")

        # Save model info
        with open("models/best_model_info.json", "w") as f:
            json.dump(best_model_info, f, indent=2)

        logger.info(
            f"Best model: {best_model_info['model_type']} with accuracy: {best_accuracy:.4f}"
        )

    return best_model_info


if __name__ == "__main__":
    best_model_info = train_models()
    print("Training completed!")
    print(f"Best model: {best_model_info}")

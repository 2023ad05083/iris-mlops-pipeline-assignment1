import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import logging
from configs.mlflow_config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Initialize MLflow tracking"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Created new experiment: {EXPERIMENT_NAME}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {EXPERIMENT_NAME}")

        mlflow.set_experiment(EXPERIMENT_NAME)
        return experiment_id
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        raise


def log_model_artifacts(model, model_name, metrics, params, artifacts=None):
    """Log model and artifacts to MLflow"""
    try:
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )

            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)

            logger.info(f"Model {model_name} logged to MLflow")
            return run.info.run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        raise


def get_best_model():
    """Get the best model from MLflow registry"""
    try:
        client = MlflowClient()
        latest_version = client.get_latest_versions(
            MODEL_REGISTRY_NAME, stages=["Production", "Staging"]
        )

        if not latest_version:
            # If no production/staging model, get latest version
            latest_version = client.get_latest_versions(MODEL_REGISTRY_NAME)

        if latest_version:
            model_version = latest_version[0]
            model_uri = f"models:/{MODEL_REGISTRY_NAME}/{model_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model version {model_version.version}")
            return model, model_version
        else:
            raise ValueError("No model found in registry")
    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        raise

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, X):
        """Fit scaler and transform features"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
            logger.info("Scaler fitted and data transformed")
            return X_scaled
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise

    def transform(self, X):
        """Transform features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")

        try:
            X_scaled = self.scaler.transform(X)
            logger.info("Data transformed using fitted scaler")
            return X_scaled
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            raise

    def save_scaler(self, filepath):
        """Save fitted scaler"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.scaler, filepath)
            logger.info(f"Scaler saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
            raise

    def load_scaler(self, filepath):
        """Load fitted scaler"""
        try:
            self.scaler = joblib.load(filepath)
            self.is_fitted = True
            logger.info(f"Scaler loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise


def preprocess_data():
    """Main preprocessing function"""
    # Load processed data
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Separate features and targets
    feature_columns = [col for col in train_df.columns if col != "target"]

    X_train = train_df[feature_columns]
    y_train = train_df["target"]
    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Fit and transform training data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Save preprocessor
    preprocessor.save_scaler("models/scaler.joblib")

    # Save scaled data
    train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=feature_columns, index=X_train.index
    )
    train_scaled_df["target"] = y_train

    test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=feature_columns, index=X_test.index
    )
    test_scaled_df["target"] = y_test

    train_scaled_df.to_csv("data/processed/train_scaled.csv", index=False)
    test_scaled_df.to_csv("data/processed/test_scaled.csv", index=False)

    logger.info("Preprocessing completed successfully")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocess_data()

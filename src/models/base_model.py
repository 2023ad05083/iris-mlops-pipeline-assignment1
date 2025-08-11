from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def create_model(self, **params):
        """Create model instance with parameters"""

    def train(self, X_train, y_train, **params):
        """Train the model"""
        try:
            self.model = self.create_model(**params)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info(f"{self.name} trained successfully")
        except Exception as e:
            logger.error(f"Error training {self.name}: {e}")
            raise

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"{self.name} not trained yet")

        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions with {self.name}: {e}")
            raise

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError(f"{self.name} not trained yet")

        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X)
                return probabilities
            else:
                raise AttributeError(
                    f"{self.name} doesn't support probability prediction"
                )
        except Exception as e:
            logger.error(f"Error getting probabilities from {self.name}: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            y_pred = self.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }

            logger.info(f"{self.name} evaluation completed")
            return metrics, y_pred
        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {e}")
            raise

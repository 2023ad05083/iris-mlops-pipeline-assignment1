from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("Logistic Regression")

    def create_model(self, **params):
        default_params = {"random_state": 42, "max_iter": 1000}
        default_params.update(params)
        return LogisticRegression(**default_params)

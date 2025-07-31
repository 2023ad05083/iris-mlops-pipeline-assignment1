from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")

    def create_model(self, **params):
        default_params = {"random_state": 42, "n_estimators": 100}
        default_params.update(params)
        return RandomForestClassifier(**default_params)

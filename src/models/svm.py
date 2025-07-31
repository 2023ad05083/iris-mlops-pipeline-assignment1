from sklearn.svm import SVC
from src.models.base_model import BaseModel


class SVMModel(BaseModel):
    def __init__(self):
        super().__init__("SVM")

    def create_model(self, **params):
        default_params = {
            "random_state": 42,
            "probability": True,  # Enable probability prediction
        }
        default_params.update(params)
        return SVC(**default_params)

from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }


class PredictionRequest(BaseModel):
    features: List[IrisFeatures]

    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    predictions: List[dict]
    model_info: dict
    request_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    timestamp: str


class MetricsResponse(BaseModel):
    total_predictions: int
    model_accuracy: Optional[float]
    uptime_seconds: float
    last_prediction_time: Optional[str]

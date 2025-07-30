# Iris MLOps Pipeline

A complete MLOps pipeline for Iris flower classification with experiment tracking, containerization, and monitoring.

## Architecture
- **Data**: Iris dataset with preprocessing
- **Models**: Logistic Regression, Random Forest, SVM
- **Tracking**: MLflow for experiments and model registry
- **API**: FastAPI for model serving
- **Deployment**: Docker containerization
- **CI/CD**: GitHub Actions pipeline
- **Monitoring**: Logging and metrics collection

## Quick Start
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/models/train.py`
4. Start API: `uvicorn api.main:app --reload`
5. View MLflow UI: `mlflow ui`

## API Endpoints
- `POST /predict`: Make predictions
- `GET /health`: Health check
- `GET /metrics`: Monitoring metrics
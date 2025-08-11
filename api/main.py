from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from datetime import datetime
import uuid
import os
import logging
import time
import sqlite3
import json

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
)

# Enhanced monitoring imports (add these if files exist, otherwise comment out)
try:
    from api.middleware import MonitoringMiddleware

    MONITORING_MIDDLEWARE_AVAILABLE = True
except ImportError:
    MONITORING_MIDDLEWARE_AVAILABLE = False
    print("MonitoringMiddleware not available - continuing without advanced middleware")

try:
    from src.utils.prometheus_metrics import metrics_collector

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus metrics not available - continuing without Prometheus")

try:
    from src.utils.monitoring_db import MonitoringDB

    MONITORING_DB_AVAILABLE = True
except ImportError:
    MONITORING_DB_AVAILABLE = False
    print("MonitoringDB not available - using basic SQLite logging")

try:
    from src.utils.logging_config import setup_logging

    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    print("Enhanced logging not available - using basic logging")

# Configure logging
if ENHANCED_LOGGING_AVAILABLE:
    setup_logging()
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
    )

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="ML API for Iris flower classification with MLOps best practices",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware if available
if MONITORING_MIDDLEWARE_AVAILABLE:
    app.add_middleware(MonitoringMiddleware)

# Mount static files (only if directory exists)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize monitoring database if available
monitoring_db = None
if MONITORING_DB_AVAILABLE:
    monitoring_db = MonitoringDB()

# Global variables
model = None
scaler = None
model_info = {}
start_time = time.time()
prediction_count = 0
last_prediction_time = None

# Class labels
class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}


def init_database():
    """Initialize SQLite database for logging predictions"""
    try:
        os.makedirs("logs", exist_ok=True)
        conn = sqlite3.connect("logs/predictions.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                features TEXT,
                prediction INTEGER,
                prediction_name TEXT,
                confidence REAL,
                response_time_ms REAL,
                model_version TEXT,
                user_agent TEXT,
                ip_address TEXT
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


def load_model_and_scaler():
    """Load the trained model and scaler"""
    global model, scaler, model_info

    try:
        # Load model
        model_path = "models/best_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("No model file found, using dummy model")
            return False

        # Load scaler
        scaler_path = "models/scaler.joblib"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("No scaler file found")
            return False

        # Load model info
        info_path = "models/best_model_info.json"
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                model_info = json.load(f)
            logger.info("Model info loaded successfully")

        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def log_prediction_enhanced(
    request_id: str,
    features: list,
    prediction: int,
    confidence: float,
    response_time: float,
    user_agent: str = "",
    ip_address: str = "",
):
    """Enhanced logging function with additional metadata"""
    try:
        conn = sqlite3.connect("logs/predictions.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions 
            (id, timestamp, features, prediction, prediction_name, confidence, 
             response_time_ms, model_version, user_agent, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                request_id,
                datetime.now().isoformat(),
                json.dumps(features),
                prediction,
                class_names[prediction],
                confidence,
                response_time,
                model_info.get("model_type", "unknown"),
                user_agent,
                ip_address,
            ),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")


def log_prediction(
    request_id: str,
    features: list,
    prediction: int,
    confidence: float,
    response_time: float,
):
    """Basic logging function for backward compatibility"""
    log_prediction_enhanced(request_id, features, prediction, confidence, response_time)


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    init_database()
    if not load_model_and_scaler():
        logger.warning("Failed to load model and scaler")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "dashboard": "/dashboard",
            "monitoring": "/monitoring/dashboard",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        version="1.0.0",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, background_tasks: BackgroundTasks, req: Request = None
):
    """Make predictions on iris features with enhanced monitoring"""
    global prediction_count, last_prediction_time

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time_pred = time.time()
    request_id = (
        getattr(req.state, "request_id", str(uuid.uuid4()))
        if req
        else str(uuid.uuid4())
    )

    # Get client info if request object is available
    user_agent = req.headers.get("user-agent", "") if req else ""
    ip_address = req.client.host if req else ""

    # Increment active connections if Prometheus is available
    if PROMETHEUS_AVAILABLE:
        metrics_collector.increment_connections()

    try:
        predictions = []

        for features in request.features:
            # Convert features to array
            feature_array = np.array(
                [
                    [
                        features.sepal_length,
                        features.sepal_width,
                        features.petal_length,
                        features.petal_width,
                    ]
                ]
            )

            # Scale features
            feature_scaled = scaler.transform(feature_array)

            # Make prediction with timing
            pred_start = time.time()
            prediction = model.predict(feature_scaled)[0]
            pred_duration = time.time() - pred_start

            # Get prediction probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(feature_scaled)[0]
                confidence = float(max(probabilities))
            else:
                probabilities = [0, 0, 0]
                confidence = 1.0

            prediction_result = {
                "prediction": int(prediction),
                "prediction_name": class_names[prediction],
                "confidence": confidence,
                "probabilities": {
                    class_names[i]: float(prob)
                    for i, prob in enumerate(
                        probabilities if hasattr(model, "predict_proba") else [0, 0, 0]
                    )
                },
            }

            predictions.append(prediction_result)

            # Record Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                metrics_collector.record_prediction(
                    model_type=model_info.get("model_type", "unknown"),
                    prediction_class=class_names[prediction],
                    duration=pred_duration,
                )

            # Enhanced logging with more details
            if MONITORING_DB_AVAILABLE and monitoring_db:
                try:
                    monitoring_db.log_prediction(
                        {
                            "id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "features": [
                                features.sepal_length,
                                features.sepal_width,
                                features.petal_length,
                                features.petal_width,
                            ],
                            "prediction": prediction,
                            "prediction_name": class_names[prediction],
                            "confidence": confidence,
                            "response_time_ms": (time.time() - start_time_pred) * 1000,
                            "model_version": model_info.get("model_type", "unknown"),
                            "user_agent": user_agent,
                            "ip_address": ip_address,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error with enhanced logging: {e}")
                    # Fall back to basic logging
                    background_tasks.add_task(
                        log_prediction_enhanced,
                        request_id,
                        [
                            features.sepal_length,
                            features.sepal_width,
                            features.petal_length,
                            features.petal_width,
                        ],
                        prediction,
                        confidence,
                        (time.time() - start_time_pred) * 1000,
                        user_agent,
                        ip_address,
                    )
            else:
                # Basic logging if enhanced monitoring not available
                background_tasks.add_task(
                    log_prediction_enhanced,
                    request_id,
                    [
                        features.sepal_length,
                        features.sepal_width,
                        features.petal_length,
                        features.petal_width,
                    ],
                    prediction,
                    confidence,
                    (time.time() - start_time_pred) * 1000,
                    user_agent,
                    ip_address,
                )

        # Update metrics
        prediction_count += len(request.features)
        last_prediction_time = datetime.now().isoformat()

        response = PredictionResponse(
            predictions=predictions,
            model_info={
                "model_type": model_info.get("model_type", "unknown"),
                "accuracy": model_info.get("metrics", {}).get("accuracy", None),
                "version": "1.0.0",
            },
            request_id=request_id,
        )

        logger.info(f"Prediction completed for request {request_id}")
        return response

    except Exception as e:
        logger.error(
            f"Error making prediction: {e}",
            extra={"request_id": request_id} if req else {},
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Decrement active connections if Prometheus is available
        if PROMETHEUS_AVAILABLE:
            metrics_collector.decrement_connections()


@app.get("/metrics")
async def get_metrics():
    """Get metrics - Prometheus format if available, otherwise basic metrics"""
    if PROMETHEUS_AVAILABLE:
        return Response(
            content=metrics_collector.get_metrics(),
            media_type=metrics_collector.get_content_type(),
        )
    else:
        # Return basic metrics in JSON format
        uptime = time.time() - start_time

        return {
            "total_predictions": prediction_count,
            "model_accuracy": model_info.get("metrics", {}).get("accuracy", None),
            "uptime_seconds": uptime,
            "last_prediction_time": last_prediction_time,
        }


@app.get("/metrics/json", response_model=MetricsResponse)
async def get_basic_metrics():
    """Get basic API metrics in JSON format"""
    uptime = time.time() - start_time

    return MetricsResponse(
        total_predictions=prediction_count,
        model_accuracy=model_info.get("metrics", {}).get("accuracy", None),
        uptime_seconds=uptime,
        last_prediction_time=last_prediction_time,
    )


@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if not model_info:
        raise HTTPException(status_code=404, detail="Model info not available")

    return {
        "model_info": model_info,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "monitoring_features": {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "monitoring_db_available": MONITORING_DB_AVAILABLE,
            "enhanced_logging_available": ENHANCED_LOGGING_AVAILABLE,
            "monitoring_middleware_available": MONITORING_MIDDLEWARE_AVAILABLE,
        },
    }


@app.get("/dashboard")
async def dashboard():
    """Serve monitoring dashboard"""
    if os.path.exists("static/dashboard.html"):
        return FileResponse("static/dashboard.html")
    else:
        return {"message": "Dashboard not available - static/dashboard.html not found"}


@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data with enhanced features"""
    try:
        # Enhanced monitoring if available
        if MONITORING_DB_AVAILABLE and monitoring_db:
            try:
                # Get prediction stats
                prediction_stats = monitoring_db.get_prediction_stats(hours=24)

                # Get API metrics
                api_metrics = monitoring_db.get_api_metrics(hours=24)

                # System info
                uptime = time.time() - start_time

                return {
                    "system": {
                        "uptime_seconds": uptime,
                        "total_predictions": prediction_count,
                        "last_prediction": last_prediction_time,
                    },
                    "predictions": prediction_stats,
                    "api_metrics": api_metrics,
                    "model_info": model_info,
                }
            except Exception as e:
                logger.error(f"Error with enhanced monitoring: {e}")
                # Fall back to basic monitoring

        # Basic monitoring fallback
        conn = sqlite3.connect("logs/predictions.db")
        cursor = conn.cursor()

        # Get total predictions in last 24 hours
        from datetime import datetime, timedelta

        since = (datetime.now() - timedelta(hours=24)).isoformat()

        cursor.execute(
            """
            SELECT COUNT(*) as total_predictions,
                   AVG(confidence) as avg_confidence,
                   AVG(response_time_ms) as avg_response_time
            FROM predictions 
            WHERE timestamp > ?
        """,
            (since,),
        )

        stats = cursor.fetchone()

        # Get prediction distribution
        cursor.execute(
            """
            SELECT prediction_name, COUNT(*) as count
            FROM predictions 
            WHERE timestamp > ?
            GROUP BY prediction_name
        """,
            (since,),
        )

        distribution = dict(cursor.fetchall())
        conn.close()

        # System info
        uptime = time.time() - start_time

        return {
            "system": {
                "uptime_seconds": uptime,
                "total_predictions": prediction_count,
                "last_prediction": last_prediction_time,
            },
            "predictions": {
                "total_predictions": stats[0] if stats[0] else 0,
                "avg_confidence": stats[1] if stats[1] else 0,
                "avg_response_time": stats[2] if stats[2] else 0,
                "prediction_distribution": distribution,
            },
            "api_metrics": [
                {
                    "endpoint": "/predict",
                    "request_count": prediction_count,
                    "avg_response_time": stats[2] if stats[2] else 0,
                    "error_count": 0,
                }
            ],
            "model_info": model_info,
            "monitoring_status": {
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "monitoring_db_available": MONITORING_DB_AVAILABLE,
                "enhanced_logging_available": ENHANCED_LOGGING_AVAILABLE,
            },
        }
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring data")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

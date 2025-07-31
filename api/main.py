from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import os
import logging
import time
from pathlib import Path
import sqlite3
import json

from api.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse, 
    MetricsResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="ML API for Iris flower classification with MLOps best practices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
model = None
scaler = None
model_info = {}
start_time = time.time()
prediction_count = 0
last_prediction_time = None

# Class labels
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

def init_database():
    """Initialize SQLite database for logging predictions"""
    try:
        os.makedirs('logs', exist_ok=True)
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                features TEXT,
                prediction INTEGER,
                prediction_name TEXT,
                confidence REAL,
                response_time_ms REAL
            )
        ''')
        
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
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            logger.info("Model info loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def log_prediction(request_id: str, features: list, prediction: int, confidence: float, response_time: float):
    """Log prediction to database"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (id, timestamp, features, prediction, prediction_name, confidence, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            request_id,
            datetime.now().isoformat(),
            json.dumps(features),
            prediction,
            class_names[prediction],
            confidence,
            response_time
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

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
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        version="1.0.0",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make predictions on iris features"""
    global prediction_count, last_prediction_time
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time_pred = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        predictions = []
        
        for features in request.features:
            # Convert features to array
            feature_array = np.array([[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]])
            
            # Scale features
            feature_scaled = scaler.transform(feature_array)
            
            # Make prediction
            prediction = model.predict(feature_scaled)[0]
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
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
                    for i, prob in enumerate(probabilities if hasattr(model, 'predict_proba') else [0, 0, 0])
                }
            }
            
            predictions.append(prediction_result)
            
            # Log prediction in background
            background_tasks.add_task(
                log_prediction,
                request_id,
                [features.sepal_length, features.sepal_width, features.petal_length, features.petal_width],
                prediction,
                confidence,
                (time.time() - start_time_pred) * 1000
            )
        
        # Update metrics
        prediction_count += len(request.features)
        last_prediction_time = datetime.now().isoformat()
        
        response = PredictionResponse(
            predictions=predictions,
            model_info={
                "model_type": model_info.get("model_type", "unknown"),
                "accuracy": model_info.get("metrics", {}).get("accuracy", None)
            },
            request_id=request_id
        )
        
        logger.info(f"Prediction completed for request {request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API metrics"""
    uptime = time.time() - start_time
    
    return MetricsResponse(
        total_predictions=prediction_count,
        model_accuracy=model_info.get("metrics", {}).get("accuracy", None),
        uptime_seconds=uptime,
        last_prediction_time=last_prediction_time
    )

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if not model_info:
        raise HTTPException(status_code=404, detail="Model info not available")
    
    return {
        "model_info": model_info,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/dashboard")
async def dashboard():
    """Serve monitoring dashboard"""
    return FileResponse("static/dashboard.html")

@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        # Get basic stats from database
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        
        # Get total predictions in last 24 hours
        from datetime import datetime, timedelta
        since = (datetime.now() - timedelta(hours=24)).isoformat()
        
        cursor.execute('''
            SELECT COUNT(*) as total_predictions,
                   AVG(confidence) as avg_confidence,
                   AVG(response_time_ms) as avg_response_time
            FROM predictions 
            WHERE timestamp > ?
        ''', (since,))
        
        stats = cursor.fetchone()
        
        # Get prediction distribution
        cursor.execute('''
            SELECT prediction_name, COUNT(*) as count
            FROM predictions 
            WHERE timestamp > ?
            GROUP BY prediction_name
        ''', (since,))
        
        distribution = dict(cursor.fetchall())
        
        conn.close()
        
        # System info
        uptime = time.time() - start_time
        
        return {
            "system": {
                "uptime_seconds": uptime,
                "total_predictions": prediction_count,
                "last_prediction": last_prediction_time
            },
            "predictions": {
                "total_predictions": stats[0] if stats[0] else 0,
                "avg_confidence": stats[1] if stats[1] else 0,
                "avg_response_time": stats[2] if stats[2] else 0,
                "prediction_distribution": distribution
            },
            "api_metrics": [
                {
                    "endpoint": "/predict",
                    "request_count": prediction_count,
                    "avg_response_time": stats[2] if stats[2] else 0,
                    "error_count": 0
                }
            ],
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
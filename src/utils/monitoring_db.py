import sqlite3
import json
import os
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MonitoringDB:
    def __init__(self, db_path="logs/monitoring.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    prediction_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    response_time_ms REAL NOT NULL,
                    model_version TEXT,
                    user_agent TEXT,
                    ip_address TEXT
                )
            """
            )

            # API metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    response_time_ms REAL NOT NULL,
                    user_agent TEXT,
                    ip_address TEXT
                )
            """
            )

            # Model performance table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    training_time_ms REAL,
                    dataset_size INTEGER
                )
            """
            )

            # System metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    active_connections INTEGER
                )
            """
            )

            conn.commit()
            logger.info("Monitoring database initialized")

    def log_prediction(self, prediction_data):
        """Log prediction to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions 
                (id, timestamp, features, prediction, prediction_name, 
                 confidence, response_time_ms, model_version, user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    prediction_data["id"],
                    prediction_data["timestamp"],
                    json.dumps(prediction_data["features"]),
                    prediction_data["prediction"],
                    prediction_data["prediction_name"],
                    prediction_data["confidence"],
                    prediction_data["response_time_ms"],
                    prediction_data.get("model_version"),
                    prediction_data.get("user_agent"),
                    prediction_data.get("ip_address"),
                ),
            )
            conn.commit()

    def log_api_metric(self, metric_data):
        """Log API metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO api_metrics 
                (timestamp, endpoint, method, status_code, response_time_ms, user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric_data["timestamp"],
                    metric_data["endpoint"],
                    metric_data["method"],
                    metric_data["status_code"],
                    metric_data["response_time_ms"],
                    metric_data.get("user_agent"),
                    metric_data.get("ip_address"),
                ),
            )
            conn.commit()

    def get_prediction_stats(self, hours=24):
        """Get prediction statistics for the last N hours"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            since = (datetime.now() - timedelta(hours=hours)).isoformat()

            # Total predictions
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

            stats = dict(cursor.fetchone())

            # Prediction distribution
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
            stats["prediction_distribution"] = distribution

            return stats

    def get_api_metrics(self, hours=24):
        """Get API metrics for the last N hours"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            since = (datetime.now() - timedelta(hours=hours)).isoformat()

            cursor.execute(
                """
                SELECT endpoint,
                       COUNT(*) as request_count,
                       AVG(response_time_ms) as avg_response_time,
                       COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
                FROM api_metrics 
                WHERE timestamp > ?
                GROUP BY endpoint
            """,
                (since,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_data(self, days=30):
        """Clean up old data to prevent database bloat"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Clean predictions
            cursor.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
            predictions_deleted = cursor.rowcount

            # Clean API metrics
            cursor.execute("DELETE FROM api_metrics WHERE timestamp < ?", (cutoff,))
            metrics_deleted = cursor.rowcount

            conn.commit()

            logger.info(
                f"Cleaned up {predictions_deleted} predictions and {metrics_deleted} API metrics"
            )

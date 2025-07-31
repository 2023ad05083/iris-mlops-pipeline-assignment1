from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time
import psutil
import logging

logger = logging.getLogger(__name__)

# Define Prometheus metrics
prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of ML predictions made",
    ["model_type", "prediction_class"],
)

prediction_latency = Histogram(
    "ml_prediction_duration_seconds",
    "Time spent on ML predictions",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
)

api_requests_counter = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

api_request_duration = Histogram(
    "api_request_duration_seconds", "Time spent on API requests", ["method", "endpoint"]
)

model_accuracy_gauge = Gauge(
    "ml_model_accuracy", "Current model accuracy", ["model_type"]
)

# System metrics
cpu_usage_gauge = Gauge("system_cpu_usage_percent", "Current CPU usage percentage")
memory_usage_gauge = Gauge(
    "system_memory_usage_percent", "Current memory usage percentage"
)
active_connections_gauge = Gauge(
    "api_active_connections", "Number of active API connections"
)


class MetricsCollector:
    """Collect and expose Prometheus metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.active_connections = 0

    def record_prediction(
        self, model_type: str, prediction_class: str, duration: float
    ):
        """Record a prediction metric"""
        prediction_counter.labels(
            model_type=model_type, prediction_class=prediction_class
        ).inc()

        prediction_latency.observe(duration)

    def record_api_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record an API request metric"""
        api_requests_counter.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

        api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def update_model_accuracy(self, model_type: str, accuracy: float):
        """Update model accuracy metric"""
        model_accuracy_gauge.labels(model_type=model_type).set(accuracy)

    def update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage_gauge.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_gauge.set(memory.percent)

            # Active connections
            active_connections_gauge.set(self.active_connections)

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def increment_connections(self):
        """Increment active connections counter"""
        self.active_connections += 1

    def decrement_connections(self):
        """Decrement active connections counter"""
        self.active_connections = max(0, self.active_connections - 1)

    def get_metrics(self):
        """Get Prometheus metrics in text format"""
        self.update_system_metrics()
        return generate_latest()

    def get_content_type(self):
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
metrics_collector = MetricsCollector()

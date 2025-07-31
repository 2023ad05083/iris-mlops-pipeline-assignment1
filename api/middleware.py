import time
import uuid
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from src.utils.monitoring_db import MonitoringDB

# Initialize monitoring database
monitoring_db = MonitoringDB()

# Create API access logger
api_logger = logging.getLogger("api_access")


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API requests and responses"""

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Get client info
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Log API access
        api_logger.info(
            "API Request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "endpoint": str(request.url.path),
                "status_code": response.status_code,
                "response_time_ms": response_time_ms,
                "client_ip": client_ip,
                "user_agent": user_agent,
            },
        )

        # Store metrics in database
        try:
            monitoring_db.log_api_metric(
                {
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": str(request.url.path),
                    "method": request.method,
                    "status_code": response.status_code,
                    "response_time_ms": response_time_ms,
                    "user_agent": user_agent,
                    "ip_address": client_ip,
                }
            )
        except Exception as e:
            api_logger.error(f"Failed to log API metric: {e}")

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

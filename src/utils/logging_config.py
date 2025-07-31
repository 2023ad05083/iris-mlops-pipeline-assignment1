import logging
import logging.handlers
import os
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if available
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """Setup comprehensive logging configuration"""

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # File handler for general logs
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "application.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(log_level)

    # File handler for API access logs
    api_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "api_access.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    api_handler.setFormatter(JSONFormatter())
    api_handler.setLevel(logging.INFO)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Create specific logger for API access
    api_logger = logging.getLogger("api_access")
    api_logger.addHandler(api_handler)
    api_logger.propagate = False

    logging.info("Logging configuration completed")

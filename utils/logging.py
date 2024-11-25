import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging settings with both console and file output."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamp-based log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"classify_{timestamp}.log")

    # Configure formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Configure file handler with rotation (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Configure root logger
    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    # Set PIL logging to INFO to suppress debug messages
    logging.getLogger("PIL").setLevel(logging.INFO)

    # Get our logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)

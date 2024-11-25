import torch
import logging
from typing import Optional, Union, List
import gc
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime


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


logger = setup_logging()


def get_device() -> torch.device:
    """Get the most suitable device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def clear_memory(device: Optional[torch.device] = None):
    """Clear unused memory on specified device."""
    gc.collect()
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
    elif device and device.type == "mps":
        torch.mps.empty_cache()


def move_to_device(
    data: Union[torch.Tensor, List[torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Safely move data to specified device."""
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device) if data is not None else None

import torch
from typing import Optional
from .logging import get_logger

logger = get_logger(__name__)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if device_str:
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif device_str == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        elif device_str == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            logger.warning(
                f"Requested device '{device_str}' not available, falling back to best available device"
            )
            return get_device()
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def clear_memory(device: Optional[torch.device] = None):
    """Clear unused memory on specified device."""
    import gc

    gc.collect()
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
    elif device and device.type == "mps":
        torch.mps.empty_cache()


def move_to_device(data, device: torch.device):
    """Safely move data to specified device."""
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device) if data is not None else None

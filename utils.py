import torch
import logging
from typing import Optional, Union, List
import gc


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging settings."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.INFO)
    return logging.getLogger(__name__)


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

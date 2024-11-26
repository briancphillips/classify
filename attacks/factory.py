import torch
from typing import Optional
from config.types import PoisonType
from config.dataclasses import PoisonConfig
from .base import PoisonAttack
from .pgd import PGDPoisonAttack
from .gradient_ascent import GradientAscentAttack
from .label_flip import LabelFlipAttack
from utils.logging import get_logger

logger = get_logger(__name__)


def create_poison_attack(
    config: PoisonConfig, device: Optional[torch.device] = None
) -> PoisonAttack:
    """Create appropriate poison attack based on config.

    Args:
        config: Attack configuration
        device: Optional device to use for the attack

    Returns:
        PoisonAttack: Configured attack instance

    Raises:
        ValueError: If attack type is not supported
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.poison_type == PoisonType.PGD:
        logger.info("Creating PGD attack")
        return PGDPoisonAttack(config, device)
    elif config.poison_type == PoisonType.GRADIENT_ASCENT:
        logger.info("Creating Gradient Ascent attack")
        return GradientAscentAttack(config, device)
    elif config.poison_type in [
        PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
        PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
        PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
    ]:
        logger.info("Creating Label Flip attack")
        return LabelFlipAttack(config, device)
    else:
        raise ValueError(f"Unknown poison type: {config.poison_type}")

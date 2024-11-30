import torch
from typing import Optional
from torch.utils.data import Dataset
from config.types import PoisonType
from config.dataclasses import PoisonConfig
from .base import PoisonAttack
from .pgd import PGDPoisonAttack
from .gradient_ascent import GradientAscentAttack
from .label_flip import LabelFlipAttack
from utils.logging import get_logger

logger = get_logger(__name__)


def create_poison_attack(
    config: PoisonConfig,
    device: Optional[torch.device] = None,
) -> PoisonAttack:
    """Create appropriate poison attack based on config.

    Args:
        config: Poison configuration
        device: Device to use for the attack
        
    Returns:
        PoisonAttack: The created attack instance
    """
    if isinstance(config, dict):
        config = PoisonConfig(**config)
    
    if config.poison_type == PoisonType.PGD:
        return PGDPoisonAttack(config=config, device=device)
    elif config.poison_type == PoisonType.GRADIENT_ASCENT:
        return GradientAscentAttack(config=config, device=device)
    elif config.poison_type in [PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM, 
                        PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
                        PoisonType.LABEL_FLIP_SOURCE_TO_TARGET]:
        return LabelFlipAttack(config=config, device=device)
    else:
        raise ValueError(f"Unknown poison type: {config.poison_type}")

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
    poison_type: PoisonType,
    model: torch.nn.Module,
    dataset: Dataset,
    **kwargs
) -> PoisonAttack:
    """Create appropriate poison attack based on config.

    Args:
        poison_type: Type of poisoning attack to create
        model: Model to use for the attack
        dataset: Dataset to poison
        **kwargs: Additional arguments to pass to the attack

    Returns:
        PoisonAttack: The created attack instance
    """
    device = kwargs.pop('device', None) or next(model.parameters()).device
    config = PoisonConfig(poison_type=poison_type, **kwargs)
    
    if poison_type == PoisonType.PGD:
        return PGDPoisonAttack(config=config, model=model, dataset=dataset, device=device)
    elif poison_type == PoisonType.GRADIENT_ASCENT:
        return GradientAscentAttack(config=config, model=model, dataset=dataset, device=device)
    elif poison_type in [PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM, 
                        PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
                        PoisonType.LABEL_FLIP_SOURCE_TO_TARGET]:
        return LabelFlipAttack(config=config, model=model, dataset=dataset, device=device)
    else:
        raise ValueError(f"Unknown poison type: {poison_type}")

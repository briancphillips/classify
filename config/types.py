from enum import Enum


class PoisonType(Enum):
    """Types of poisoning attacks"""

    PGD = "pgd"  # Projected Gradient Descent
    GRADIENT_ASCENT = "gradient_ascent"  # Gradient Ascent
    LABEL_FLIP_RANDOM_TO_RANDOM = "label_flip_random_random"
    LABEL_FLIP_RANDOM_TO_TARGET = "label_flip_random_target"
    LABEL_FLIP_SOURCE_TO_TARGET = "label_flip_source_target"

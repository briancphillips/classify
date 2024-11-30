from enum import Enum

class PoisonType(str, Enum):
    """Type of poisoning attack."""
    PGD = "pgd"
    GRADIENT_ASCENT = "ga"  # Map 'ga' to GRADIENT_ASCENT
    LABEL_FLIP_RANDOM_TO_RANDOM = "label_flip_random_to_random"
    LABEL_FLIP_RANDOM_TO_TARGET = "label_flip_random_to_target"
    LABEL_FLIP_SOURCE_TO_TARGET = "label_flip_source_to_target"

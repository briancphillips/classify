#!/usr/bin/env python3
"""
Entry point for running data poisoning experiments.
"""

import logging
import sys
import torch
import os
import json
import yaml
from typing import Dict, Any, Tuple
from config.types import PoisonType
from config.dataclasses import PoisonConfig
from experiment import PoisonExperiment
from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
import torch.optim as optim
from models.data import get_dataset
from models.architectures import get_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from attacks.pgd import PGDPoisonAttack
from attacks.gradient_ascent import GradientAscentAttack
from attacks.label_flip import LabelFlipPoisonAttack

logger = get_logger(__name__)
error_logger = get_error_logger()

def get_loaders(dataset: str):
    """Get train and test data loaders for a dataset.
    
    Args:
        dataset: Name of dataset ('cifar100', 'gtsrb', 'imagenette')
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
    """
    # Get datasets with default transforms
    train_dataset = get_dataset(dataset, train=True)
    test_dataset = get_dataset(dataset, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # Default batch size from config
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def run_poison_experiment(
    dataset: str,
    attack: str,
    output_dir: str,
    poison_ratio: float = 0.1,
    seed: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """Run a poisoning experiment with the given parameters.
    
    Args:
        dataset: Name of dataset to use
        attack: Type of poisoning attack
        output_dir: Directory to save results
        poison_ratio: Ratio of training data to poison
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to specific attacks
        
    Returns:
        Dict containing experiment results
    """
    try:
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Load dataset
        train_loader, test_loader = get_loaders(dataset)
        
        # Initialize model and optimizer
        model = get_model(dataset)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        
        # Run attack
        results = {}
        if attack == "pgd":
            results = run_pgd_attack(model, train_loader, test_loader, poison_ratio)
        elif attack == "gradient_ascent":
            results = run_gradient_ascent(model, train_loader, test_loader, poison_ratio)
        elif attack == "label_flip_random_random":
            results = run_label_flip(model, train_loader, test_loader, poison_ratio, mode="random")
        elif attack == "label_flip_random_target":
            results = run_label_flip(model, train_loader, test_loader, poison_ratio, mode="target")
        elif attack == "label_flip_source_target":
            results = run_label_flip(model, train_loader, test_loader, poison_ratio, mode="source_target")
        else:
            raise ValueError(f"Unknown attack type: {attack}")
            
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{dataset}_{attack}_{poison_ratio}.json")
        with open(results_file, "w") as f:
            json.dump(results, f)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in poison experiment: {str(e)}")
        raise

def run_pgd_attack(model, train_loader, test_loader, poison_ratio):
    """Run PGD poisoning attack."""
    config = PoisonConfig(
        poison_type=PoisonType.PGD,
        poison_ratio=poison_ratio,
        pgd_eps=0.3,
        pgd_alpha=0.1,
        pgd_steps=40
    )
    attack = PGDPoisonAttack(config)
    return attack.poison_dataset(train_loader.dataset, model)

def run_gradient_ascent(model, train_loader, test_loader, poison_ratio):
    """Run gradient ascent poisoning attack."""
    config = PoisonConfig(
        poison_type=PoisonType.GRADIENT_ASCENT,
        poison_ratio=poison_ratio,
        ga_epsilon=0.3,
        ga_learning_rate=0.1,
        ga_steps=40
    )
    attack = GradientAscentAttack(config)
    return attack.poison_dataset(train_loader.dataset, model)

def run_label_flip(model, train_loader, test_loader, poison_ratio, mode="random"):
    """Run label flip poisoning attack."""
    if mode == "random":
        poison_type = PoisonType.LABEL_FLIP_RANDOM_RANDOM
    elif mode == "target":
        poison_type = PoisonType.LABEL_FLIP_RANDOM_TARGET
    else:  # source_target
        poison_type = PoisonType.LABEL_FLIP_SOURCE_TARGET
        
    config = PoisonConfig(
        poison_type=poison_type,
        poison_ratio=poison_ratio,
        source_class=0,  # Only used for source_target mode
        target_class=1   # Only used for target modes
    )
    attack = LabelFlipPoisonAttack(config)
    return attack.poison_dataset(train_loader.dataset, model)

if __name__ == "__main__":
    logger.warning("This script is not meant to be run directly. Please use run_experiments.py instead.")
    sys.exit(1)

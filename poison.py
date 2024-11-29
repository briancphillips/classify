#!/usr/bin/env python3
"""
Entry point for running data poisoning experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
from models.data import get_dataset
from models.architectures import get_model
from attacks.pgd import PGDPoisonAttack
from attacks.gradient_ascent import GradientAscentAttack
from attacks.label_flip import LabelFlipAttack

# Initialize logging
setup_logging()
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

@dataclass
class PoisonResults:
    """Results from a poisoning attack."""
    poison_success_rate: float
    poisoned_indices: List[int]

def run_poison_experiment(
    dataset: str,
    attack: str,
    output_dir: str,
    poison_ratio: float = 0.1,
    subset_size: int = None,
    target_class: int = None,
    source_class: int = None,
    seed: int = 0
) -> Dict[str, Any]:
    """Run a poisoning experiment with the given parameters.
    
    Args:
        dataset: Name of dataset to use
        attack: Type of poisoning attack
        output_dir: Directory to save results
        poison_ratio: Ratio of training data to poison
        subset_size: Optional size of dataset subset to use
        target_class: Optional target class for targeted attacks
        source_class: Optional source class for targeted attacks
        seed: Random seed for reproducibility
        
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Get training parameters from config
        if dataset.lower() == "cifar100":
            from config.config import default_config
            config = default_config
            epochs = config.epochs
            lr = config.learning_rate
            momentum = config.momentum
            weight_decay = config.weight_decay
            lr_schedule = config.lr_schedule
            lr_factor = config.lr_factor
            
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule, gamma=lr_factor)
        else:  # gtsrb and imagenette - use their original settings
            epochs = 10
            lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = None
            
        criterion = nn.CrossEntropyLoss()
        
        # Train model on clean data first
        logger.info("Training model on clean data...")
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            if scheduler is not None:
                scheduler.step()
            
            # Always log training progress
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            accuracy = 100. * correct / total
            logger.info(f'Clean Training - Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.3f} Test Accuracy: {accuracy:.2f}%')
            model.train()
        
        # Evaluate original model accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        original_accuracy = 100. * correct / total
        logger.info(f'Original model accuracy: {original_accuracy:.2f}%')
        
        # Run attack
        if attack == "pgd":
            poisoned_dataset, poison_results = run_pgd_attack(model, train_loader, test_loader, poison_ratio)
        elif attack == "gradient_ascent":
            poisoned_dataset, poison_results = run_gradient_ascent(model, train_loader, test_loader, poison_ratio)
        elif attack == "label_flip":
            poisoned_dataset, poison_results = run_label_flip(model, train_loader, test_loader, poison_ratio, mode="random")
        elif attack == "label_flip_target":
            poisoned_dataset, poison_results = run_label_flip(model, train_loader, test_loader, poison_ratio, mode="target")
        elif attack == "label_flip_source":
            poisoned_dataset, poison_results = run_label_flip(model, train_loader, test_loader, poison_ratio, mode="source_target")
        else:
            raise ValueError(f"Unknown attack type: {attack}")
            
        # Create new data loader with poisoned dataset
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Reset model and optimizer for training on poisoned data
        model = get_model(dataset)
        model = model.to(device)
        
        if dataset.lower() == "cifar100":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule, gamma=lr_factor)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = None
        
        # Train model on poisoned data
        logger.info("Training model on poisoned data...")
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, targets) in enumerate(poisoned_loader):
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            if scheduler is not None:
                scheduler.step()
            
            # Always log training progress
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            accuracy = 100. * correct / total
            logger.info(f'Poisoned Training - Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(poisoned_loader):.3f} Test Accuracy: {accuracy:.2f}%')
            model.train()
        
        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        final_accuracy = 100. * correct / total
        logger.info(f'Final model accuracy: {final_accuracy:.2f}%')
        
        # Update results
        results = {
            "original_accuracy": original_accuracy,
            "poisoned_accuracy": final_accuracy,
            "poison_success_rate": poison_results.poison_success_rate,
            "poisoned_indices": poison_results.poisoned_indices
        }
            
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{dataset}_{attack}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f)
            
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = PGDPoisonAttack(config, device)
    return attack.poison_dataset(train_loader.dataset, model)

def run_gradient_ascent(model, train_loader, test_loader, poison_ratio):
    """Run gradient ascent poisoning attack."""
    config = PoisonConfig(
        poison_type=PoisonType.GRADIENT_ASCENT,
        poison_ratio=poison_ratio,
        ga_steps=40,
        ga_iterations=100,
        ga_lr=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = GradientAscentAttack(config, device)
    return attack.poison_dataset(train_loader.dataset, model)

def run_label_flip(model, train_loader, test_loader, poison_ratio, mode="random"):
    """Run label flip poisoning attack."""
    if mode == "random":
        poison_type = PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM
    elif mode == "target":
        poison_type = PoisonType.LABEL_FLIP_RANDOM_TO_TARGET
    else:  # source_target
        poison_type = PoisonType.LABEL_FLIP_SOURCE_TO_TARGET
        
    config = PoisonConfig(
        poison_type=poison_type,
        poison_ratio=poison_ratio,
        source_class=0,  # Only used for source_target mode
        target_class=1   # Only used for target modes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = LabelFlipAttack(config, device)
    return attack.poison_dataset(train_loader.dataset, model)

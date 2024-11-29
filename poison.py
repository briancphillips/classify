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
from typing import Dict, Any, List, Tuple

from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
from models.data import get_dataset
from models.architectures import get_model
from attacks.pgd import PGDPoisonAttack
from attacks.gradient_ascent import GradientAscentAttack
from attacks.label_flip import LabelFlipAttack
from config.dataclasses import PoisonConfig
from config.types import PoisonType

# Initialize logging
setup_logging()
logger = get_logger(__name__)
error_logger = get_error_logger()

def get_loaders(dataset: str) -> Tuple[DataLoader, DataLoader]:
    """Get train and test data loaders for a dataset.
    
    Args:
        dataset: Name of dataset ('cifar100', 'gtsrb', 'imagenette')
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
    """
    # Get datasets
    train_dataset = get_dataset(dataset, train=True)
    test_dataset = get_dataset(dataset, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
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

class PoisonResults:
    """Results from a poisoning attack."""
    def __init__(self, poison_success_rate: float, poisoned_indices: List[int]):
        self.poison_success_rate = poison_success_rate
        self.poisoned_indices = poisoned_indices

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
    # Set random seed
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_loaders(dataset)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(dataset).to(device)
    
    # Create attack config
    if attack.lower() == "pgd":
        poison_type = PoisonType.PGD
        config = PoisonConfig(
            poison_type=poison_type,
            poison_ratio=poison_ratio,
            pgd_eps=0.3,
            pgd_alpha=0.01,
            pgd_steps=40,
            target_class=target_class,
            source_class=source_class,
            random_seed=seed
        )
    elif attack.lower() == "gradient_ascent":
        poison_type = PoisonType.GRADIENT_ASCENT
        config = PoisonConfig(
            poison_type=poison_type,
            poison_ratio=poison_ratio,
            ga_steps=50,
            ga_iterations=100,
            ga_lr=0.1,
            target_class=target_class,
            source_class=source_class,
            random_seed=seed
        )
    else:  # Label flip attacks
        if target_class is not None and source_class is not None:
            poison_type = PoisonType.LABEL_FLIP_SOURCE_TO_TARGET
        elif target_class is not None:
            poison_type = PoisonType.LABEL_FLIP_RANDOM_TO_TARGET
        else:
            poison_type = PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM
            
        config = PoisonConfig(
            poison_type=poison_type,
            poison_ratio=poison_ratio,
            target_class=target_class,
            source_class=source_class,
            random_seed=seed
        )
    
    try:
        # Train clean model first
        logger.info("Training clean model...")
        clean_model = get_model(dataset).to(device)
        train_model(clean_model, train_loader, test_loader, device)
        clean_acc = evaluate_model(clean_model, test_loader, device)
        logger.info(f"Clean model accuracy: {clean_acc:.2f}%")
        
        # Run appropriate attack using the trained clean model
        if config.poison_type == PoisonType.PGD:
            poisoned_dataset, poison_results = run_pgd_attack(clean_model, train_loader, test_loader, config)
        elif config.poison_type == PoisonType.GRADIENT_ASCENT:
            poisoned_dataset, poison_results = run_gradient_ascent(clean_model, train_loader, test_loader, config)
        else:
            poisoned_dataset, poison_results = run_label_flip(clean_model, train_loader, test_loader, config)
            
        # Create poisoned data loader
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Train model on poisoned data
        logger.info("Training model on poisoned data...")
        poisoned_model = get_model(dataset).to(device)
        train_model(poisoned_model, poisoned_loader, test_loader, device)
        
        # Evaluate final model
        clean_acc = evaluate_model(poisoned_model, test_loader, device)
        poisoned_acc = evaluate_model(poisoned_model, poisoned_loader, device)
        
        # Save results
        results = {
            'dataset': dataset,
            'attack': attack,
            'poison_ratio': poison_ratio,
            'target_class': target_class,
            'source_class': source_class,
            'clean_accuracy': clean_acc,
            'poisoned_accuracy': poisoned_acc,
            'poison_success_rate': poison_results.poison_success_rate,
            'num_poisoned': len(poison_results.poisoned_indices)
        }
        
        results_file = os.path.join(output_dir, f"{dataset}_{attack}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Results saved to {results_file}")
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        error_logger.exception("Detailed error information:")
        raise

def run_pgd_attack(model, train_loader, test_loader, config):
    """Run PGD poisoning attack."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = PGDPoisonAttack(config, device)
    return attack.poison_dataset(train_loader.dataset, model)

def run_gradient_ascent(model, train_loader, test_loader, config):
    """Run gradient ascent poisoning attack."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = GradientAscentAttack(config, device)
    return attack.poison_dataset(train_loader.dataset, model)

def run_label_flip(model, train_loader, test_loader, config):
    """Run label flip poisoning attack."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = LabelFlipAttack(config, device)
    return attack.poison_dataset(train_loader.dataset, model)

def evaluate_model(model, data_loader, device):
    """Evaluate model on given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    return accuracy

def train_model(model, train_loader, test_loader, device):
    """Train model on given data loader."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        # Training metrics
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss/len(train_loader)
        train_acc = 100. * correct / total
        
        # Test metrics
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss/len(test_loader)
        test_acc = 100. * correct / total
        
        logger.info(f'Epoch [{epoch+1}/10] Train Loss: {train_loss:.3f} Train Acc: {train_acc:.2f}% Test Loss: {test_loss:.3f} Test Acc: {test_acc:.2f}%')

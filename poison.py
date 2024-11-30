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
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from IPython import display
import sys
import numpy as np

from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger
from models.data import get_dataset
from models.architectures import get_model
from attacks.pgd import PGDPoisonAttack
from attacks.gradient_ascent import GradientAscentAttack
from attacks.label_flip import LabelFlipAttack
from config.dataclasses import PoisonConfig
from config.types import PoisonType
from config.defaults import get_dataset_config

# Initialize logging
setup_logging()
logger = get_logger(__name__)
error_logger = get_error_logger()

def get_loaders(dataset: str, subset_size: Optional[int] = None, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Get train and test data loaders for a dataset.
    
    Args:
        dataset: Name of dataset ('cifar100', 'gtsrb', 'imagenette')
        subset_size: Optional size of dataset subset to use
        batch_size: Optional batch size override. If not provided, uses dataset config batch_size
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
    """
    # Get dataset config for default batch size
    config = get_dataset_config(dataset)
    if batch_size is None:
        batch_size = config['batch_size']
    
    # Get datasets
    train_dataset = get_dataset(dataset, train=True, subset_size=subset_size)
    test_dataset = get_dataset(dataset, train=False, subset_size=subset_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
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
    poison_config: PoisonConfig,
    subset_size: Optional[int] = None
) -> Dict[str, Any]:
    """Run a poisoning experiment with the given parameters."""
    # Set random seed for reproducibility
    torch.manual_seed(poison_config.random_seed)
    np.random.seed(poison_config.random_seed)
    
    logger.info(f"Starting poison experiment with params:")
    logger.info(f"  - dataset: {dataset}")
    logger.info(f"  - attack: {attack}")
    logger.info(f"  - poison_ratio: {poison_config.poison_ratio}")
    logger.info(f"  - batch_size: {poison_config.batch_size}")
    logger.info(f"  - subset_size: {subset_size}")
    if poison_config.target_class is not None:
        logger.info(f"  - target_class: {poison_config.target_class}")
    if poison_config.source_class is not None:
        logger.info(f"  - source_class: {poison_config.source_class}")
    logger.info(f"  - seed: {poison_config.random_seed}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_loaders(dataset, subset_size=subset_size, batch_size=poison_config.batch_size)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(dataset).to(device)
    
    # Train clean model
    logger.info("Training clean model...")
    model = train_clean_model(model, train_loader, test_loader, device)
    
    # Run attack based on type
    if attack.lower() == "pgd":
        poisoned_dataset, poison_results = run_pgd_attack(model, train_loader, test_loader, poison_config)
    elif attack.lower() == "gradient_ascent":
        poisoned_dataset, poison_results = run_gradient_ascent(model, train_loader, test_loader, poison_config)
    else:
        poisoned_dataset, poison_results = run_label_flip(model, train_loader, test_loader, poison_config)
    
    # Create poisoned data loader
    poisoned_loader = DataLoader(
        poisoned_dataset,
        batch_size=poison_config.batch_size,
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
        'poison_ratio': poison_config.poison_ratio,
        'target_class': poison_config.target_class,
        'source_class': poison_config.source_class,
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

def is_jupyter() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return False
    except NameError:  # Probably standard Python interpreter
        return False
    return False

def train_model(model, train_loader, test_loader, device):
    """Train model on given data loader."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD with correct parameters
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,  # Initial learning rate
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler with specified milestones
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2  # Decay factor
    )
    
    # Setup checkpoint directory - using model name instead of hardcoding dataset
    model_name = model.__class__.__name__.lower()
    checkpoint_dir = os.path.join('checkpoints', model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = model_name
    
    # Try to load checkpoint if exists
    start_epoch = 0
    best_acc = 0
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    # Initialize metrics tracking
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epochs_recorded = []
    
    if latest_checkpoint:
        checkpoint = load_checkpoint(latest_checkpoint, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Load previous metrics if available
        train_losses = checkpoint.get('train_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        test_losses = checkpoint.get('test_losses', [])
        test_accs = checkpoint.get('test_accs', [])
        epochs_recorded = checkpoint.get('epochs_recorded', list(range(len(train_losses))))
        
        # Verify checkpoint consistency
        assert len(train_losses) == len(test_losses) == len(train_accs) == len(test_accs) == len(epochs_recorded), \
            f"Checkpoint metric lengths don't match: {len(train_losses)}, {len(test_losses)}, {len(train_accs)}, {len(test_accs)}, {len(epochs_recorded)}"
        assert all(e < start_epoch for e in epochs_recorded), \
            f"Recorded epochs {epochs_recorded} should all be less than start_epoch {start_epoch}"
        
        logger.info(f"Loaded checkpoint - Metrics: {len(train_losses)}, Start epoch: {start_epoch}, Epochs recorded: {epochs_recorded}")
    else:
        epochs_recorded = []
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        logger.info(f"Starting fresh training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, 200):  # 200 epochs as specified
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
        
        # Verify metric consistency before update
        assert len(train_losses) == len(test_losses) == len(train_accs) == len(test_accs) == len(epochs_recorded), \
            f"Pre-update metric lengths don't match: {len(train_losses)}, {len(test_losses)}, {len(train_accs)}, {len(test_accs)}, {len(epochs_recorded)}"
        
        # Update metrics history and record this epoch
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        epochs_recorded.append(epoch)
        
        # Verify metric consistency after update
        assert len(train_losses) == len(test_losses) == len(train_accs) == len(test_accs) == len(epochs_recorded), \
            f"Post-update metric lengths don't match: {len(train_losses)}, {len(test_losses)}, {len(train_accs)}, {len(test_accs)}, {len(epochs_recorded)}"
        
        # Only plot if in Jupyter
        if is_jupyter():
            # Create new figure for this update
            plt.figure(figsize=(15, 5))
            
            # Plot losses
            plt.subplot(1, 2, 1)
            plt.plot(epochs_recorded, train_losses, label='Train Loss', color='#2ecc71', linewidth=2)
            plt.plot(epochs_recorded, test_losses, label='Test Loss', color='#e74c3c', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot accuracies
            plt.subplot(1, 2, 2)
            plt.plot(epochs_recorded, train_accs, label='Train Acc', color='#2ecc71', linewidth=2)
            plt.plot(epochs_recorded, test_accs, label='Test Acc', color='#e74c3c', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Test Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Update display
            display.clear_output(wait=True)
            plt.tight_layout()
            display.display(plt.gcf())
            plt.close('all')
        
        # Log with current learning rate
        logger.info(f'Epoch [{epoch+1}/200] LR: {scheduler.get_last_lr()[0]:.6f} Train Loss: {train_loss:.3f} Train Acc: {train_acc:.2f}% Test Loss: {test_loss:.3f} Test Acc: {test_acc:.2f}%')
        
        # Create checkpoint with current metrics
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'epochs_recorded': epochs_recorded  # Save the epoch numbers too
        }
        
        # Always save latest checkpoint with current metrics
        save_checkpoint(checkpoint, checkpoint_dir, checkpoint_name)
        
        # If this is the best model so far, save it as best
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint['is_best'] = True
            save_checkpoint(checkpoint, checkpoint_dir, checkpoint_name)

        # Step the scheduler
        scheduler.step()
        
    # Final plot cleanup
    plt.close('all')

def train_clean_model(model, train_loader, test_loader, device):
    """Train clean model."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD with correct parameters
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,  # Initial learning rate
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler with specified milestones
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2  # Decay factor
    )
    
    for epoch in range(200):  # 200 epochs as specified
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
        
        # Log with current learning rate
        logger.info(f'Epoch [{epoch+1}/200] LR: {scheduler.get_last_lr()[0]:.6f} Train Loss: {train_loss:.3f} Train Acc: {train_acc:.2f}% Test Loss: {test_loss:.3f} Test Acc: {test_acc:.2f}%')
        
        # Step the scheduler
        scheduler.step()
        
    return model

def get_latest_checkpoint(checkpoint_dir):
    """Get the path to the latest checkpoint."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getmtime)
    return latest_checkpoint

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def save_checkpoint(checkpoint, checkpoint_dir, checkpoint_name):
    """Save a checkpoint."""
    # Log checkpoint contents before saving
    logger.info(f"Saving checkpoint - Metrics lengths: train_losses={len(checkpoint['train_losses'])}, epochs_recorded={len(checkpoint.get('epochs_recorded', []))}")
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_latest.pt")
    torch.save(checkpoint, latest_path)
    
    # If this is the best model, save it as best
    if checkpoint.get('is_best', False):
        best_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_best.pt")
        torch.save(checkpoint, best_path)

# Hardware configuration
hardware_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
}

# Common settings
num_workers = min(os.cpu_count(), 8)

# Define checkpoint paths
from pathlib import Path
checkpoints = {
    'cifar100': Path('/workspace/classify/checkpoints/wideresnet/wideresnet_best.pt'),
    'gtsrb': Path('/workspace/classify/checkpoints/gtsrb/gtsrb_best.pt'),
    'imagenette': Path('/workspace/classify/checkpoints/resnet50/resnet50_best.pt')
}

# CIFAR-100 with WideResNet
config = {
    'dataset': 'cifar100',
    'attack': 'pgd',
    'poison_ratio': 0.1,
    'batch_size': 128,
    'target_class': 0,
    'source_class': 1,
    'random_seed': 42,
    'checkpoint_path': checkpoints['cifar100']
}

# GTSRB with ResNet-50
config = {
    'dataset': 'gtsrb',
    'attack': 'pgd',
    'poison_ratio': 0.1,
    'batch_size': 128,
    'target_class': 0,
    'source_class': 1,
    'random_seed': 42,
    'checkpoint_path': checkpoints['gtsrb']
}

# Imagenette with ResNet-50
config = {
    'dataset': 'imagenette',
    'attack': 'pgd',
    'poison_ratio': 0.1,
    'batch_size': 128,
    'target_class': 0,
    'source_class': 1,
    'random_seed': 42,
    'checkpoint_path': checkpoints['imagenette']
}

#!/usr/bin/env python3
"""
Enhanced test script for evaluating the Wide ResNet on CIFAR100 with advanced training techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import numpy as np
from torchvision import transforms
import random
from torch.optim.swa_utils import AveragedModel, SWALR
from contextlib import nullcontext
import math
import shutil
from pathlib import Path
import argparse

from models import get_model, get_dataset
from utils.device import get_device, clear_memory
from utils.logging import setup_logging, get_logger
from utils.checkpoints import save_checkpoint, load_checkpoint, cleanup_checkpoints

logger = get_logger(__name__)

class RandAugment:
    """RandAugment implementation."""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m  # magnitude of augmentation
        self.augment_list = [
            transforms.RandomAdjustSharpness(self.m/10),
            transforms.RandomEqualize(),
            transforms.RandomPosterize(bits=int(8-4*(self.m/10))),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1*(self.m/10), 0.1*(self.m/10))),
            transforms.ColorJitter(
                brightness=0.1*(self.m/10),
                contrast=0.1*(self.m/10),
                saturation=0.1*(self.m/10)
            ),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)
        return img

class Cutout:
    """Cutout augmentation."""
    def __init__(self, size=8):  # Using our optimized size of 8
        self.size = size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)
        mask[y:y + self.size, x:x + self.size] = 0
        mask = mask.unsqueeze(0).expand_as(img)
        img = img * mask
        return img

def mixup_data(x, y, alpha=1.0, device=None):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def evaluate_model(model, dataloader, device, criterion):
    """Evaluate model accuracy and loss."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

def get_lr(epoch):
    """Get learning rate according to schedule."""
    if epoch < 10:
        return 0.1 * (epoch + 1) / 10
    else:
        progress = (epoch - 10) / (200 - 10)
        return 0.1 * (1 + math.sin(math.pi * progress - math.pi/2)) * 1.5

def convert_tensors_to_python(obj):
    """Convert torch tensors to Python native types for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_python(item) for item in obj]
    return obj

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CIFAR100 Training')
    parser.add_argument('--force-restart', action='store_true',
                       help='Force restart training from scratch')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Configuration
    batch_size = 128
    epochs = 200
    base_lr = 0.1
    warmup_epochs = 10
    momentum = 0.9
    weight_decay = 1e-4
    grad_clip = 0.5
    mixup_alpha = 0.4
    label_smoothing = 0.15
    swa_start = 100
    swa_lr = 0.05
    force_restart = args.force_restart  # Use command line argument
    
    # Check if training was already completed
    checkpoint_path = Path("checkpoints/checkpoint.pth.tar")
    if checkpoint_path.exists() and not force_restart:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if checkpoint['epoch'] >= epochs:
            logger.info(f"Training was already completed (epoch {checkpoint['epoch']}/{epochs})")
            logger.info(f"Best accuracy achieved: {checkpoint['best_acc']:.2f}%")
            logger.info("To start a new training run, delete the checkpoints directory or set force_restart=True")
            return
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    use_amp = device.type == 'cuda'  # Only enable AMP for CUDA
    
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        autocast_fn = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast_fn = nullcontext
    
    # Create output directory
    output_dir = "cifar100_enhanced_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=3, m=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        Cutout(size=8)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Get datasets with custom transforms
    train_dataset = get_dataset("cifar100", train=True, transform=train_transform)
    test_dataset = get_dataset("cifar100", train=False, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = get_model("cifar100").to(device)
    
    # SWA model
    swa_model = AveragedModel(model)
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(classes=100, smoothing=label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    # Training history
    history = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "learning_rates": []
    }
    
    # Initialize variables
    start_epoch = 0
    best_acc = 0
    swa_n = 0

    # Try to load checkpoint
    checkpoint_data = load_checkpoint(model, optimizer, swa_model)
    if checkpoint_data is not None:
        if len(checkpoint_data) == 3:
            start_epoch, best_acc, swa_n = checkpoint_data
        else:
            start_epoch, best_acc = checkpoint_data
        logger.info(f"Resuming from epoch {start_epoch} with best accuracy {best_acc:.2f}%")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            # Set learning rate for current epoch
            current_lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                # Apply mixup with 50% probability
                do_mixup = random.random() < 0.5
                if do_mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device)
                
                if use_amp:
                    # Mixed precision training
                    with autocast_fn():
                        outputs = model(inputs)
                        if do_mixup:
                            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training
                    outputs = model(inputs)
                    if do_mixup:
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                
                if do_mixup:
                    correct += (lam * predicted.eq(targets_a).sum().float()
                              + (1 - lam) * predicted.eq(targets_b).sum().float())
                else:
                    correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{train_loss/(batch_idx+1):.3f}",
                    "acc": f"{100.*correct/total:.2f}%",
                    "lr": f"{current_lr:.6f}"
                })
                
                if (batch_idx + 1) % 100 == 0:
                    clear_memory(device)
            
            # Calculate training metrics
            train_acc = 100. * correct / total
            train_loss = train_loss / len(train_loader)
            
            # Evaluate on test set
            test_acc, test_loss = evaluate_model(model, test_loader, device, criterion)
            
            # Update SWA model if in SWA phase
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_n += 1
            
            # Save history
            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["test_acc"].append(test_acc)
            history["test_loss"].append(test_loss)
            history["learning_rates"].append(current_lr)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.2f}% - "
                f"Test Loss: {test_loss:.4f} - "
                f"Test Acc: {test_acc:.2f}% - "
                f"LR: {current_lr:.6f}"
            )
            
            # Save checkpoint
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'swa_state_dict': swa_model.state_dict() if epoch >= swa_start else None,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'swa_n': swa_n if epoch >= swa_start else 0
            }, is_best)
            
            if is_best:
                logger.info(f'New best accuracy: {best_acc:.2f}%')
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        if epoch >= swa_start:
            logger.info("Updating SWA batch normalization statistics...")
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            accuracy, avg_loss = evaluate_model(swa_model, test_loader, device, criterion)
            logger.info(f"Final SWA Model - Test Loss: {avg_loss:.4f} - Test Acc: {accuracy:.2f}%")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert tensors to Python native types before saving
        serializable_history = convert_tensors_to_python(history)
        json.dump(serializable_history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves plot to {plot_path}")
    
    # Print final results
    logger.info(f"\nTraining completed!")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()

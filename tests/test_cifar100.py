#!/usr/bin/env python3
"""
Test script for evaluating the Wide ResNet on CIFAR100.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from models import get_model, get_dataset
from utils.device import get_device, clear_memory
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def evaluate_model(model, dataloader, device, criterion):
    """Evaluate model accuracy and loss."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

def main():
    # Setup logging
    setup_logging()
    
    # Configuration
    batch_size = 128
    epochs = 200
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = "cifar100_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get datasets
    train_dataset = get_dataset("cifar100", train=True)
    test_dataset = get_dataset("cifar100", train=False)
    
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
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "learning_rates": []
    }
    
    best_acc = 0
    
    # Training loop
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{train_loss/(batch_idx+1):.3f}",
                    "acc": f"{100.*correct/total:.2f}%"
                })
                
                if (batch_idx + 1) % 100 == 0:
                    clear_memory(device)
            
            # Calculate training metrics
            train_acc = 100. * correct / total
            train_loss = train_loss / len(train_loader)
            
            # Evaluate on test set
            test_acc, test_loss = evaluate_model(model, test_loader, device, criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
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
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                }, os.path.join(output_dir, 'best_model.pt'))
                logger.info(f"Saved new best model with accuracy: {test_acc:.2f}%")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
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
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path)
    logger.info(f"Saved training curves to {plot_path}")
    
    # Print final results
    logger.info(f"\nTraining completed!")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()

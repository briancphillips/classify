"""Training script for CIFAR-100 classifier"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from config.config import default_config
from experiment.experiment import Trainer
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_transforms(config):
    """Get training and validation transforms."""
    train_transforms = []
    if config.random_crop:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
    if config.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        if config.normalize else transforms.Lambda(lambda x: x)
    ])
    
    if config.cutout:
        train_transforms.append(transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0
        ))
    
    train_transform = transforms.Compose(train_transforms)
    
    # Validation transforms - just normalize
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        if config.normalize else transforms.Lambda(lambda x: x)
    ])
    
    return train_transform, val_transform

def get_dataloaders(config):
    """Create training and validation dataloaders."""
    train_transform, val_transform = get_transforms(config)
    
    # Load CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader

def create_model(config):
    """Create and initialize the model."""
    if config.model_name == "wrn-28-10":
        from models.wideresnet import WideResNet
        model = WideResNet(
            depth=28,
            num_classes=config.num_classes,
            widen_factor=10,
            dropRate=0.3
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model

def main():
    config = default_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model directory
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    # Create model and move to device
    model = create_model(config)
    model = model.to(device)
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_schedule,
        gamma=config.lr_factor
    )
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config.to_dict()
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Train and evaluate
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.4f} - "
            f"Train Acc: {train_metrics['train_acc']:.4f} - "
            f"Val Loss: {val_metrics['val_loss']:.4f} - "
            f"Val Acc: {val_metrics['val_acc']:.4f}"
        )
        
        # Save checkpoint if improved
        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config.to_dict()
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Regular checkpoint saving
        if (epoch + 1) % config.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_metrics['val_acc'],
                'config': config.to_dict()
            }, checkpoint_path)
            
            # Remove old checkpoints if needed
            checkpoints = sorted(save_dir.glob('checkpoint_epoch_*.pt'))
            if len(checkpoints) > config.keep_last_n:
                os.remove(checkpoints[0])
        
        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Final metrics
    metrics = trainer.get_metrics()
    logger.info("Training completed. Final metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

if __name__ == "__main__":
    main()

"""
Advanced training components for model training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from torch.optim.swa_utils import AveragedModel, SWALR
import math
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from utils.logging import get_logger

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
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)
        mask[:, y:y + self.size, x:x + self.size] = 0
        return img * mask

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
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class AdvancedTrainer:
    """Advanced training with SWA, mixed precision, and advanced augmentations."""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=200,
        swa_start=100,
        swa_lr=0.05,
        grad_clip=0.5,
        mixup_alpha=0.4,
        use_amp=None  # None means auto-detect
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.swa_start = swa_start
        self.grad_clip = grad_clip
        self.mixup_alpha = mixup_alpha
        
        # Setup SWA
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        
        # Setup mixed precision training
        self.use_amp = use_amp if use_amp is not None else (device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            self.autocast_fn = torch.cuda.amp.autocast
        else:
            self.scaler = None
            self.autocast_fn = nullcontext
        
        # Initialize history
        self.history = {
            "train_acc": [], "train_loss": [],
            "val_acc": [], "val_loss": [],
            "learning_rates": []
        }
        
        self.best_acc = 0
        self.swa_n = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            # Apply mixup with probability 0.5
            if random.random() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, self.mixup_alpha, self.device
                )
                
                with self.autocast_fn():
                    outputs = self.model(inputs)
                    loss = mixup_criterion(
                        self.criterion, outputs, targets_a, targets_b, lam
                    )
            else:
                with self.autocast_fn():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if self.mixup_alpha > 0:
                correct += (lam * predicted.eq(targets_a).sum().item()
                          + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss/total:.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return 100. * correct / total, total_loss / len(self.train_loader)

    def evaluate(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        return accuracy, avg_loss

    def train(self):
        try:
            for epoch in range(self.epochs):
                # Training phase
                train_acc, train_loss = self.train_epoch(epoch)
                
                # Evaluation phase
                val_acc, val_loss = self.evaluate()
                
                # Update SWA model if applicable
                if epoch >= self.swa_start:
                    self.swa_model.update_parameters(self.model)
                    self.swa_n += 1
                    self.swa_scheduler.step()
                
                # Log metrics
                self.history["train_acc"].append(train_acc)
                self.history["train_loss"].append(train_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_loss"].append(val_loss)
                self.history["learning_rates"].append(
                    self.optimizer.param_groups[0]["lr"]
                )
                
                # Update best accuracy
                is_best = val_acc > self.best_acc
                self.best_acc = max(val_acc, self.best_acc)
                
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Train Acc: {train_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val Acc: {val_acc:.2f}% - "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                if is_best:
                    logger.info(f"New best accuracy: {self.best_acc:.2f}%")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            if self.swa_n > 0:
                logger.info("Updating SWA batch normalization statistics...")
                torch.optim.swa_utils.update_bn(
                    self.train_loader, self.swa_model, device=self.device
                )
                swa_acc, swa_loss = self.evaluate(self.swa_model)
                logger.info(
                    f"Final SWA Model - "
                    f"Val Loss: {swa_loss:.4f} - "
                    f"Val Acc: {swa_acc:.2f}%"
                )
        
        return self.history, self.best_acc, self.swa_model if self.swa_n > 0 else None

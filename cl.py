# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import torch.nn.functional as F
import requests
import tarfile
import io
import copy
import torch.backends.mps

def extract_features_from_loader(loader, model, device):
    """Extract features using the CNN model's feature extractor."""
    features = []
    labels = []
    
    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting features"):
            inputs = inputs.to(device)
            # Extract features using the model's feature extraction method
            batch_features = model.extract_features(inputs)
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())
    
    # Concatenate all features and labels
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    return features, labels

# Enhanced CIFAR100 Feature Extractor and Classifier
class CIFAR100Classifier(nn.Module):
    """Enhanced CIFAR100 classifier using EfficientNet with custom head."""
    def __init__(self, num_classes=100):
        super(CIFAR100Classifier, self).__init__()
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Modify first conv layer for 32x32 images
        self.backbone.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Get the number of features from the backbone
        num_ftrs = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the new layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
    
    def extract_features(self, x):
        return self.backbone(x)

# New GTSRB Classifier (following similar architecture but with 43 classes)
class GTSRBClassifier(nn.Module):
    def __init__(self):
        super(GTSRBClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 43)  # GTSRB has 43 classes
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        if return_features:
            return features
        x = self.classifier(features.view(features.size(0), -1))
        return x

# Enhanced Imagenette Classifier using Transfer Learning
class ImagenetteClassifier(nn.Module):
    def __init__(self):
        super(ImagenetteClassifier, self).__init__()
        # Use pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        
        # Add new classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )
        
        # Freeze early layers
        for param in list(self.features.parameters())[:-20]:
            param.requires_grad = False
    
    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        if return_features:
            return features
        x = self.fc(features)
        return x

# %%
# Enhanced CIFAR100 transforms with better augmentation
def get_cifar_transforms():
    """Get training and validation transforms for CIFAR100."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # Advanced augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR100 stats
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    return train_transform, val_transform

# New GTSRB transforms
def get_gtsrb_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # GTSRB images are variable size
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3398, 0.3117, 0.3210],
                           std=[0.2755, 0.2647, 0.2712])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3398, 0.3117, 0.3210],
                           std=[0.2755, 0.2647, 0.2712])
    ])
    
    return train_transform, test_transform

# Enhanced Imagenette transforms with better augmentation
def get_imagenette_transforms():
    """Get transforms for Imagenette with enhanced augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def load_cifar100(batch_size=128, num_workers=2, return_datasets=False):
    """Load CIFAR100 dataset using original transforms"""
    train_transform, test_transform = get_cifar_transforms()
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    if return_datasets:
        return train_dataset, test_dataset
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def load_gtsrb(batch_size=128, num_workers=2, return_datasets=False):
    """Load GTSRB dataset"""
    train_transform, test_transform = get_gtsrb_transforms()
    
    train_dataset = torchvision.datasets.GTSRB(
        root='./data', split='train', download=True,
        transform=train_transform
    )
    test_dataset = torchvision.datasets.GTSRB(
        root='./data', split='test', download=True,
        transform=test_transform
    )
    
    if return_datasets:
        return train_dataset, test_dataset
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def load_imagenette(batch_size=128, num_workers=2, return_datasets=False):
    """Load the Imagenette dataset."""
    # Define the data directory
    data_dir = './data/imagenette2'
    
    # Check if dataset exists, if not download it
    if not os.path.exists(data_dir):
        print("Downloading Imagenette dataset...")
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        
        # Download and extract the dataset
        url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
        file.extractall(path='./data')
        print("Dataset downloaded and extracted successfully!")

    # Data transforms
    train_transform, test_transform = get_imagenette_transforms()
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=test_transform
    )
    
    if return_datasets:
        return train_dataset, test_dataset
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def get_dataset_loaders(dataset_name, batch_size=128, num_workers=2, subset_size_per_class=None):
    """
    Get data loaders for the specified dataset with optional balanced subset.
    
    Args:
        dataset_name: Name of the dataset ('cifar100', 'gtsrb', or 'imagenette')
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading
        subset_size_per_class: If specified, create balanced subset with this many samples per class
    """
    if dataset_name.lower() == 'cifar100':
        train_dataset, test_dataset = load_cifar100(batch_size, num_workers, return_datasets=True)
    elif dataset_name.lower() == 'gtsrb':
        train_dataset, test_dataset = load_gtsrb(batch_size, num_workers, return_datasets=True)
    elif dataset_name.lower() == 'imagenette':
        train_dataset, test_dataset = load_imagenette(batch_size, num_workers, return_datasets=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if subset_size_per_class is not None:
        train_dataset = create_balanced_subset(train_dataset, subset_size_per_class)
        # For test set, we'll use a quarter of the training subset size per class
        test_dataset = create_balanced_subset(test_dataset, subset_size_per_class // 4)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Print dataset sizes
    print(f"\nDataset: {dataset_name.upper()}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    if subset_size_per_class:
        print(f"Using subset size of {subset_size_per_class} samples per class")
    print()
    
    return train_loader, test_loader

def create_balanced_subset(dataset, subset_size_per_class):
    """
    Create a balanced subset of the dataset.
    
    Args:
        dataset: PyTorch dataset
        subset_size_per_class: Number of samples per class to include
    
    Returns:
        Subset of the dataset with balanced classes
    """
    if not hasattr(dataset, 'targets'):
        # Convert targets to a list if they're not already
        targets = [y for _, y in dataset]
    else:
        targets = dataset.targets if isinstance(dataset.targets, list) else dataset.targets.tolist()
    
    # Get indices for each class
    classes = sorted(set(targets))
    class_indices = {c: [] for c in classes}
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    # Sample equal number of indices from each class
    selected_indices = []
    for c in classes:
        indices = class_indices[c]
        if len(indices) < subset_size_per_class:
            print(f"Warning: Class {c} has fewer than {subset_size_per_class} samples. Using all available samples.")
            selected_indices.extend(indices)
        else:
            selected = np.random.choice(indices, subset_size_per_class, replace=False)
            selected_indices.extend(selected)
    
    return torch.utils.data.Subset(dataset, selected_indices)

def get_device():
    """
    Determine the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# %%
def train_model(dataset, model, train_loader, test_loader, epochs, device):
    """Train the model with improved training process."""
    criterion = nn.CrossEntropyLoss()
    
    # Cosine annealing learning rate scheduler with lower initial learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(total),
                'acc': 100.*correct/total
            })
        
        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100.*correct/total
        print(f'Validation Accuracy: {acc:.2f}%')
        
        # Save best accuracy
        best_acc = max(best_acc, acc)
        scheduler.step()
    
    print(f'Best validation accuracy: {best_acc:.2f}%')
    return model  # Return only the model

def train_traditional_classifiers(train_features, train_labels, test_features, test_labels):
    """Train and evaluate traditional classifiers with improved parameters."""
    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    # Add PCA to reduce dimensionality while keeping 95% of variance
    pca = PCA(n_components=0.95)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)
    print(f"Reduced feature dimension from {train_features.shape[1]} to {train_features.shape[1]} components")
    
    classifiers = {
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine',
            n_jobs=-1  # Use all CPU cores
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,  # Let trees grow fully
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Use sqrt(n_features) features per split
            class_weight='balanced',
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        ),
        'svm': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            class_weight='balanced',
            cache_size=1000,  # Increase cache size for faster training
            random_state=42
        ),
        'lr': LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        )
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name.upper()}...")
        clf.fit(train_features, train_labels)
        acc = clf.score(test_features, test_labels)
        print(f"{name.upper()} Accuracy: {acc:.4f}")
        results[name] = acc
    
    return results

def train_cifar(model, train_loader, test_loader, epochs=80, device='cuda'):
    """Train CIFAR100 with improved training strategy."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
    
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    # Use different optimizers with adjusted learning rates
    backbone_optimizer = torch.optim.AdamW(backbone_params, lr=0.0001, weight_decay=0.01)
    classifier_optimizer = torch.optim.AdamW(classifier_params, lr=0.001, weight_decay=0.01)
    
    # Cosine annealing with warmup
    backbone_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        backbone_optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4
    )
    
    classifier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        classifier_optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4
    )
    
    best_acc = 0.0
    model = model.to(device)
    
    # Increased patience for longer training
    patience = 15  # Increased from 10
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            backbone_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping for both optimizers
            torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier_params, max_norm=1.0)
            
            backbone_optimizer.step()
            classifier_optimizer.step()
            backbone_scheduler.step()
            classifier_scheduler.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total,
                'lr_backbone': backbone_scheduler.get_last_lr()[0],
                'lr_classifier': classifier_scheduler.get_last_lr()[0]
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss/val_total
        val_acc = 100.*val_correct/val_total
        print(f'\nValidation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if val_acc > best_acc:
                best_acc = val_acc
                print(f'New best accuracy: {best_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    print(f'Final best accuracy: {best_acc:.2f}%')
    return model

def train_gtsrb(model, train_loader, test_loader, epochs=30, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    model = model.to(device)
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Dictionary to store training history
    history = {
        'train_acc': [],
        'val_acc': [],
        'best_acc': 0.0
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 50 == 49:  # More frequent updates due to smaller dataset
                print(f'Loss: {running_loss/50:.3f} | Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        train_acc = 100. * correct / total
        scheduler.step()
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        # Save accuracies to history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_acc'] = best_acc
            print(f'New best accuracy: {best_acc:.2f}%')
    
    print(f'Final best accuracy: {best_acc:.2f}%')
    return model

def train_imagenette(model, train_loader, test_loader, epochs=30, device='cuda'):
    """Train Imagenette with improved training regime."""
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for pretrained and new layers
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "fc" in name:
                new_params.append(param)
            else:
                pretrained_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': 0.0001},
        {'params': new_params, 'lr': 0.001}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_acc = 0.0
    model = model.to(device)
    
    # Dictionary to store training history
    history = {
        'train_acc': [],
        'val_acc': [],
        'best_acc': 0.0
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Save accuracies to history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_acc'] = best_acc
            
        print(f'Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%')
    
    print(f'Final best accuracy: {best_acc:.2f}%')
    return model

def train_model(dataset_name, model, train_loader, test_loader, epochs=30, device='cuda'):
    """Unified function to train any dataset's model"""
    if dataset_name.lower() == 'cifar100':
        model = train_cifar(model, train_loader, test_loader, epochs, device)
    elif dataset_name.lower() == 'gtsrb':
        model = train_gtsrb(model, train_loader, test_loader, epochs, device)
    elif dataset_name.lower() == 'imagenette':
        model = train_imagenette(model, train_loader, test_loader, epochs, device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return model

def get_model(dataset_name):
    """Get the appropriate model for the dataset"""
    if dataset_name.lower() == 'cifar100':
        return CIFAR100Classifier()
    elif dataset_name.lower() == 'gtsrb':
        return GTSRBClassifier()
    elif dataset_name.lower() == 'imagenette':
        return ImagenetteClassifier()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def plot_classifier_comparison(results, output_path):
    """Plot classifier performance across datasets."""
    # Create figure with larger size
    plt.figure(figsize=(15, 8))
    
    # Setup
    datasets = list(results.keys())
    classifiers = ['knn', 'rf', 'svm', 'lr']  # Match the keys used in traditional_results
    
    x = np.arange(len(datasets))  # Label locations
    width = 0.15  # Width of bars
    multiplier = 0
    
    # Plot bars for each classifier
    for classifier in classifiers:
        accuracies = []
        for dataset in datasets:
            # Get classifier accuracy, defaulting to 0 if not found
            try:
                acc = float(results[dataset][classifier])  # Convert to float explicitly
            except (KeyError, TypeError):
                acc = 0.0
            accuracies.append(acc)
        
        offset = width * multiplier
        plt.bar(x + offset, accuracies, width, label=classifier.upper())
        
        # Add value labels on top of bars
        for i, acc in enumerate(accuracies):
            plt.text(x[i] + offset, acc, f'{acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    # Customize the plot
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance Comparison')
    plt.xticks(x + width * (len(classifiers) - 1) / 2, [d.upper() for d in datasets])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    plt.ylim(0, 1.0)
    
    # Save plot
    plt.tight_layout()
    print(f"Saving plot to: {output_path}")  # Debug print
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on specified dataset')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'cifar100', 'gtsrb', 'imagenette'],
                      help='Dataset to use (default: all)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--subset-size-per-class', type=int, default=None, 
                      help='Number of samples per class to use (for balanced subset)')
    args = parser.parse_args()

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'classifier_results.json')
    plot_file = os.path.join(results_dir, 'classifier_comparison.png')

    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    # Load existing results
    try:
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    # Define datasets to process
    datasets_to_run = ['cifar100', 'gtsrb', 'imagenette'] if args.dataset == 'all' else [args.dataset.lower()]
    
    for dataset in datasets_to_run:
        print(f"\n{'='*50}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*50}\n")
        
        # Get data loaders
        train_loader, test_loader = get_dataset_loaders(
            dataset,
            args.batch_size,
            args.num_workers,
            args.subset_size_per_class
        )
        
        # Initialize model
        model = get_model(dataset)
        model = model.to(device)
        
        # Train model
        model = train_model(dataset, model, train_loader, test_loader, args.epochs, device)
        
        # Extract features and train traditional classifiers
        print("\nExtracting features and training traditional classifiers...")
        train_features, train_labels = extract_features_from_loader(train_loader, model, device)
        test_features, test_labels = extract_features_from_loader(test_loader, model, device)
        
        # Train and evaluate traditional classifiers
        results = train_traditional_classifiers(train_features, train_labels, test_features, test_labels)
        
        # Update results for this dataset
        all_results[dataset] = results
        
        # Save results after each dataset (in case of interruption)
        with open(results_file, 'w') as f:
            json.dump(all_results, f)
        
        print(f"\nResults for {dataset} saved to {results_file}")
    
    # Generate final comparison plot
    print("\nGenerating final performance comparison plot...")
    try:
        plot_classifier_comparison(all_results, plot_file)
        print(f"Plot saved successfully to {plot_file}")
    except Exception as e:
        print(f"Error generating plot: {str(e)}")

if __name__ == "__main__":
    main()

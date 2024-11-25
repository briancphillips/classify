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
from typing import Optional, Dict, Tuple, List
from utils import get_device, clear_memory, move_to_device, logger

# Define transforms for each dataset
CIFAR100_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

CIFAR100_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

GTSRB_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]
)

GTSRB_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]
)

IMAGENETTE_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

IMAGENETTE_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_features_from_loader(
    loader: DataLoader, model: nn.Module, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from data loader with proper memory management."""
    if device is None:
        device = get_device()

    features = []
    labels = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(
            tqdm(loader, desc="Extracting features")
        ):
            data = move_to_device(data, device)
            batch_features = model.extract_features(data).cpu()
            features.append(batch_features)
            labels.append(target)

            if (batch_idx + 1) % 10 == 0:
                clear_memory(device)

    return torch.cat(features), torch.cat(labels)


# Updated CIFAR100 Classifier using WideResNet
from torchvision.models.resnet import ResNet, Bottleneck


class CIFAR100Classifier(nn.Module):
    """CIFAR100 classifier using WideResNet-28-10."""

    def __init__(self, num_classes=100):
        super(CIFAR100Classifier, self).__init__()
        # Create a WideResNet with depth=28 and width=10
        self.backbone = torchvision.models.wide_resnet50_2(weights=None)

        # Modify first conv layer and maxpool for CIFAR-100
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        # Adjust the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

        # Initialize weights
        nn.init.kaiming_normal_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        return x


# GTSRB Classifier (unchanged)
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
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 43)  # GTSRB has 43 classes
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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


# Imagenette Classifier (unchanged)
class ImagenetteClassifier(nn.Module):
    def __init__(self):
        super(ImagenetteClassifier, self).__init__()
        # Use pretrained ResNet50 with ImageNet weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)

        # Add new classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10),
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
# Updated CIFAR100 transforms without resizing
def get_cifar_transforms():
    """Get training and validation transforms for CIFAR100."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(
                transforms.AutoAugmentPolicy.CIFAR10
            ),  # Advanced augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),  # CIFAR100 stats
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    return train_transform, val_transform


# New GTSRB transforms (unchanged)
def get_gtsrb_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # GTSRB images are variable size
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3398, 0.3117, 0.3210], std=[0.2755, 0.2647, 0.2712]
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3398, 0.3117, 0.3210], std=[0.2755, 0.2647, 0.2712]
            ),
        ]
    )

    return train_transform, test_transform


# Enhanced Imagenette transforms (unchanged)
def get_imagenette_transforms():
    """Get transforms for Imagenette with enhanced augmentation."""
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def load_cifar100(batch_size=128, num_workers=2, return_datasets=False):
    """Load CIFAR100 dataset using updated transforms"""
    train_transform, test_transform = get_cifar_transforms()

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    if return_datasets:
        return train_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_gtsrb(batch_size=128, num_workers=2, return_datasets=False):
    """Load GTSRB dataset"""
    train_transform, test_transform = get_gtsrb_transforms()

    train_dataset = torchvision.datasets.GTSRB(
        root="./data", split="train", download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.GTSRB(
        root="./data", split="test", download=True, transform=test_transform
    )

    if return_datasets:
        return train_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_imagenette(batch_size=128, num_workers=2, return_datasets=False):
    """Load the Imagenette dataset."""
    # Define the data directory
    data_dir = "./data/imagenette2"

    # Check if dataset exists, if not download it
    if not os.path.exists(data_dir):
        print("Downloading Imagenette dataset...")
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)

        # Download and extract the dataset
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
        file.extractall(path="./data")
        print("Dataset downloaded and extracted successfully!")

    # Data transforms
    train_transform, test_transform = get_imagenette_transforms()

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=test_transform
    )

    if return_datasets:
        return train_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_dataset_loaders(
    dataset_name, batch_size=128, num_workers=2, subset_size_per_class=None
):
    """
    Get data loaders for the specified dataset with optional balanced subset.

    Args:
        dataset_name: Name of the dataset ('cifar100', 'gtsrb', or 'imagenette')
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading
        subset_size_per_class: If specified, create balanced subset with this many samples per class

    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    if dataset_name.lower() == "cifar100":
        train_dataset, test_dataset = load_cifar100(
            batch_size, num_workers, return_datasets=True
        )
    elif dataset_name.lower() == "gtsrb":
        train_dataset, test_dataset = load_gtsrb(
            batch_size, num_workers, return_datasets=True
        )
    elif dataset_name.lower() == "imagenette":
        train_dataset, test_dataset = load_imagenette(
            batch_size, num_workers, return_datasets=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if subset_size_per_class is not None:
        train_dataset = create_balanced_subset(train_dataset, subset_size_per_class)
        # For test set, we'll use a quarter of the training subset size per class
        test_dataset = create_balanced_subset(test_dataset, subset_size_per_class // 4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Print dataset sizes
    print(f"\nDataset: {dataset_name.upper()}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    if subset_size_per_class:
        print(f"Using subset size of {subset_size_per_class} samples per class")
    print()

    return train_loader, test_loader, train_dataset, test_dataset


def create_balanced_subset(dataset, subset_size_per_class):
    """
    Create a balanced subset of the dataset.

    Args:
        dataset: PyTorch dataset
        subset_size_per_class: Number of samples per class to include

    Returns:
        Subset of the dataset with balanced classes
    """
    if not hasattr(dataset, "targets"):
        # Convert targets to a list if they're not already
        targets = [y for _, y in dataset]
    else:
        targets = (
            dataset.targets
            if isinstance(dataset.targets, list)
            else dataset.targets.tolist()
        )

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
            print(
                f"Warning: Class {c} has fewer than {subset_size_per_class} samples. Using all available samples."
            )
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
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %%
# Implement Mixup
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 1.0,
) -> Dict[str, List[float]]:
    """Train model with improved stability and monitoring."""
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            data, target = move_to_device(data, device), move_to_device(target, device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                clear_memory(device)

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        if val_loader is not None:
            val_loss = validate_model(model, val_loader, device)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
            + (f" - Val Loss: {val_loss:.4f}" if val_loader else "")
        )

    return history


def validate_model(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> float:
    """Validate model performance."""
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = move_to_device(data, device), move_to_device(target, device)
            output = model(data)
            val_loss += F.cross_entropy(output, target).item()

    return val_loss / len(val_loader)


def train_traditional_classifiers(
    train_features, train_labels, test_features, test_labels
):
    """Train and evaluate traditional classifiers with improved parameters."""
    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Add PCA to reduce dimensionality while keeping 95% of variance
    pca = PCA(n_components=0.95)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)
    print(f"Reduced feature dimension to {train_features.shape[1]} components")

    classifiers = {
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine",
            n_jobs=-1,  # Use all CPU cores
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,  # Let trees grow fully
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",  # Use sqrt(n_features) features per split
            class_weight="balanced",
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
        ),
        "svm": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            cache_size=1000,  # Increase cache size for faster training
            random_state=42,
        ),
        "lr": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
        ),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name.upper()}...")
        clf.fit(train_features, train_labels)
        acc = clf.score(test_features, test_labels)
        print(f"{name.upper()} Accuracy: {acc:.4f}")
        results[name] = acc

    return results


def get_model(dataset_name):
    """Get the appropriate model for the dataset"""
    if dataset_name.lower() == "cifar100":
        return CIFAR100Classifier()
    elif dataset_name.lower() == "gtsrb":
        return GTSRBClassifier()
    elif dataset_name.lower() == "imagenette":
        return ImagenetteClassifier()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_model(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
) -> None:
    """Save model and training metadata to disk.

    Args:
        model: The PyTorch model to save
        path: Path to save the model to (should end in .pt)
        metadata: Optional dict of metadata to save with the model
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        loss: Optional loss value

    Raises:
        ValueError: If path doesn't end with .pt
    """
    if not path.endswith(".pt"):
        raise ValueError("Checkpoint path must end with .pt extension")

    if metadata is None:
        metadata = {}

    # Add dataset name to metadata based on model class
    if isinstance(model, CIFAR100Classifier):
        metadata["dataset_name"] = "cifar100"
    elif isinstance(model, GTSRBClassifier):
        metadata["dataset_name"] = "gtsrb"
    elif isinstance(model, ImagenetteClassifier):
        metadata["dataset_name"] = "imagenette"
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Prepare checkpoint data
    checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, path)


def load_model(
    path: str, device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict]:
    """Load a model from a checkpoint file.

    Args:
        path: Path to the checkpoint file (.pt)
        device: Optional device to load the model to

    Returns:
        tuple: (model, metadata)
            - model: The loaded PyTorch model
            - metadata: Dictionary containing checkpoint metadata

    Raises:
        ValueError: If checkpoint format is invalid or file not found
    """
    if not path.endswith(".pt"):
        raise ValueError("Checkpoint path must end with .pt extension")

    if not os.path.exists(path):
        raise ValueError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format - expected dictionary")

    if "metadata" not in checkpoint:
        raise ValueError("Invalid checkpoint format - missing metadata")
    metadata = checkpoint["metadata"]

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint format - missing model_state_dict")

    if "dataset_name" not in metadata:
        raise ValueError("Invalid checkpoint format - missing dataset_name in metadata")

    # Create and load model
    model = get_model(metadata["dataset_name"])
    if device is not None:
        model = model.to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, metadata


def plot_classifier_comparison(results, output_path):
    """Plot classifier performance across datasets."""
    # Create figure with larger size
    plt.figure(figsize=(15, 8))

    # Setup
    datasets = list(results.keys())
    classifiers = [
        "knn",
        "rf",
        "svm",
        "lr",
    ]  # Match the keys used in traditional_results

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
            plt.text(
                x[i] + offset, acc, f"{acc:.3f}", ha="center", va="bottom", fontsize=8
            )

        multiplier += 1

    # Customize the plot
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.title("Classifier Performance Comparison")
    plt.xticks(x + width * (len(classifiers) - 1) / 2, [d.upper() for d in datasets])
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    plt.ylim(0, 1.0)

    # Save plot
    plt.tight_layout()
    print(f"Saving plot to: {output_path}")  # Debug print
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a model on specified dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "cifar100", "gtsrb", "imagenette"],
        help="Dataset to use (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loading workers"
    )
    parser.add_argument(
        "--subset-size-per-class",
        type=int,
        default=None,
        help="Number of samples per class to use (for balanced subset)",
    )
    args = parser.parse_args()

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "classifier_results.json")
    plot_file = os.path.join(results_dir, "classifier_comparison.png")

    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    # Load existing results
    try:
        with open(results_file, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    # Define datasets to process
    datasets_to_run = (
        ["cifar100", "gtsrb", "imagenette"]
        if args.dataset == "all"
        else [args.dataset.lower()]
    )

    for dataset in datasets_to_run:
        print(f"\n{'='*50}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*50}\n")

        # Get data loaders
        train_loader, test_loader, train_dataset, test_dataset = get_dataset_loaders(
            dataset, args.batch_size, args.num_workers, args.subset_size_per_class
        )

        # Initialize model
        model = get_model(dataset)
        model = model.to(device)

        # Train model
        model = train_model(
            dataset, model, train_loader, test_loader, args.epochs, device
        )

        # Extract features and train traditional classifiers
        if dataset.lower() == "cifar100":
            print("\nExtracting features and training traditional classifiers...")
            train_features, train_labels = extract_features_from_loader(
                train_loader, model, device
            )
            test_features, test_labels = extract_features_from_loader(
                test_loader, model, device
            )

            # Train and evaluate traditional classifiers
            results = train_traditional_classifiers(
                train_features, train_labels, test_features, test_labels
            )

            # Update results for this dataset
            all_results[dataset] = results

            # Save results after each dataset (in case of interruption)
            with open(results_file, "w") as f:
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

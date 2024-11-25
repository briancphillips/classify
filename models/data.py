import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import requests
import tarfile
import io
from typing import Tuple, Optional
from .transforms import get_transforms
from utils.logging import get_logger

logger = get_logger(__name__)


def get_dataset_loaders(
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 0,
    subset_size_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """Get data loaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset ('cifar100', 'gtsrb', or 'imagenette')
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading
        subset_size_per_class: If specified, create balanced subset with this many samples per class

    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)

    Raises:
        ValueError: If dataset name is not supported
    """
    train_transform, test_transform = get_transforms(dataset_name)

    if dataset_name.lower() == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform
        )
    elif dataset_name.lower() == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(
            root="./data", split="train", download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.GTSRB(
            root="./data", split="test", download=True, transform=test_transform
        )
    elif dataset_name.lower() == "imagenette":
        data_dir = "./data/imagenette2"
        if not os.path.exists(data_dir):
            download_imagenette(data_dir)
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "train"), transform=train_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "val"), transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if subset_size_per_class is not None:
        train_dataset = create_balanced_subset(train_dataset, subset_size_per_class)
        # For test set, we'll use a quarter of the training subset size per class
        test_dataset = create_balanced_subset(test_dataset, subset_size_per_class // 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"\nDataset: {dataset_name.upper()}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    if subset_size_per_class:
        logger.info(f"Using subset size of {subset_size_per_class} samples per class")

    return train_loader, test_loader, train_dataset, test_dataset


def create_balanced_subset(dataset: Dataset, subset_size_per_class: int) -> Dataset:
    """Create a balanced subset of the dataset.

    Args:
        dataset: PyTorch dataset
        subset_size_per_class: Number of samples per class to include

    Returns:
        Dataset: Subset of the dataset with balanced classes
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
            logger.warning(
                f"Class {c} has fewer than {subset_size_per_class} samples. "
                f"Using all available samples."
            )
            selected_indices.extend(indices)
        else:
            selected = np.random.choice(indices, subset_size_per_class, replace=False)
            selected_indices.extend(selected)

    return Subset(dataset, selected_indices)


def download_imagenette(data_dir: str) -> None:
    """Download and extract the Imagenette dataset.

    Args:
        data_dir: Directory to save the dataset
    """
    logger.info("Downloading Imagenette dataset...")
    os.makedirs("./data", exist_ok=True)

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
    file.extractall(path="./data")
    logger.info("Dataset downloaded and extracted successfully!")

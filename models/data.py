import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import os
from typing import Optional, Tuple
import numpy as np
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["get_dataset"]


def get_dataset(
    dataset_name: str,
    train: bool = True,
    subset_size: Optional[int] = None,
    transform: Optional[transforms.Compose] = None,
) -> Dataset:
    """Get dataset by name.

    Args:
        dataset_name: Name of dataset ('cifar100', 'gtsrb', 'imagenette')
        train: Whether to get training or test set
        subset_size: Optional number of samples per class
        transform: Optional transform to apply

    Returns:
        Dataset: The requested dataset
    """
    # Set up base transforms if none provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

    # Get dataset root directory
    data_dir = os.path.join("data", dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    # Load appropriate dataset
    if dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "gtsrb":
        split = "train" if train else "test"
        dataset = datasets.GTSRB(
            root=data_dir, split=split, download=True, transform=transform
        )
    elif dataset_name.lower() == "imagenette":
        split = "train" if train else "val"
        dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, split), transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create subset if requested
    if subset_size is not None:
        # Get indices for each class
        labels = [dataset[i][1] for i in range(len(dataset))]
        unique_labels = sorted(list(set(labels)))

        # Select subset_size samples from each class
        subset_indices = []
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            if len(label_indices) > subset_size:
                selected_indices = np.random.choice(
                    label_indices, subset_size, replace=False
                )
                subset_indices.extend(selected_indices)
            else:
                subset_indices.extend(label_indices)

        dataset = Subset(dataset, subset_indices)
        logger.info(f"\nDataset: {dataset_name.upper()}")
        logger.info(f"Training samples: {len(dataset)}")
        logger.info(f"Test samples: {len(dataset)}")
        logger.info(f"Using subset size of {subset_size} samples per class")

    return dataset

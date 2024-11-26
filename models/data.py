import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import os
from typing import Optional, Tuple
import numpy as np
import requests
import tarfile
from tqdm import tqdm
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["get_dataset"]


def download_imagenette(data_dir: str, split: str = "train"):
    """Download and extract ImageNette dataset."""
    # ImageNette v2-160 URLs
    urls = {
        "train": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    }

    if split not in urls:
        raise ValueError(f"Invalid split: {split}")

    url = urls[split]
    filename = os.path.join(data_dir, os.path.basename(url))

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download file if it doesn't exist
    if not os.path.exists(filename):
        logger.info(f"Downloading ImageNette dataset to {filename}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(filename, "wb") as f, tqdm(
            desc=f"Downloading ImageNette",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

    # Extract file if needed
    if not os.path.exists(os.path.join(data_dir, "imagenette2-160")):
        logger.info(f"Extracting {filename}")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=data_dir)

    # Create train/val symlinks if they don't exist
    src_dir = os.path.join(data_dir, "imagenette2-160")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Remove existing symlinks if they exist
    if os.path.islink(train_dir):
        os.unlink(train_dir)
    if os.path.islink(val_dir):
        os.unlink(val_dir)

    # Create new symlinks
    os.symlink(os.path.join(src_dir, "train"), train_dir)
    os.symlink(os.path.join(src_dir, "val"), val_dir)

    logger.info("ImageNette dataset setup complete")


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
        # Download ImageNette dataset
        download_imagenette(data_dir)
        split = "train" if train else "val"
        split_dir = os.path.join(data_dir, split)
        dataset = datasets.ImageFolder(root=split_dir, transform=transform)
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

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import os
import shutil
import requests
import tarfile
from tqdm import tqdm
from typing import Optional, Tuple
import numpy as np
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["get_dataset"]


def setup_imagenette(data_dir: str) -> None:
    """Set up ImageNette dataset directory structure."""
    # ImageNette v2-160 URL
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    filename = os.path.join(data_dir, os.path.basename(url))

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if dataset is already properly set up
    if os.path.exists(os.path.join(data_dir, "train")) and os.path.exists(
        os.path.join(data_dir, "val")
    ):
        logger.info("ImageNette dataset already set up")
        return

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
    src_dir = os.path.join(data_dir, "imagenette2-160")
    if not os.path.exists(src_dir):
        logger.info(f"Extracting {filename}")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=data_dir)

    # Move directories to correct location
    train_src = os.path.join(src_dir, "train")
    val_src = os.path.join(src_dir, "val")
    train_dst = os.path.join(data_dir, "train")
    val_dst = os.path.join(data_dir, "val")

    if not os.path.exists(train_dst):
        shutil.move(train_src, train_dst)
    if not os.path.exists(val_dst):
        shutil.move(val_src, val_dst)

    # Clean up
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)
    if os.path.exists(filename):
        os.remove(filename)

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
        if dataset_name.lower() == "gtsrb":
            # GTSRB specific transform to handle varying image sizes
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),  # Force resize to fixed dimensions
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ]
            )

    # Get dataset root directory
    data_dir = os.path.join("data", dataset_name.lower())

    # Verify dataset directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory not found: {data_dir}")

    # Load appropriate dataset
    if dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir, train=train, download=False, transform=transform
        )
    elif dataset_name.lower() == "gtsrb":
        split = "train" if train else "test"
        dataset = datasets.GTSRB(
            root=data_dir, split=split, download=True, transform=transform
        )
    elif dataset_name.lower() == "imagenette":
        # Use existing ImageNette directory structure
        split = "train" if train else "val"
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"ImageNette {split} directory not found: {split_dir}")
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

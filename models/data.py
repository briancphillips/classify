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


def cleanup_gtsrb(data_dir: str) -> None:
    """Clean up redundant GTSRB dataset files and directories."""
    import shutil

    # Remove only zip files after dataset is loaded
    for f in os.listdir(data_dir):
        if f.endswith(".zip"):
            zip_path = os.path.join(data_dir, f)
            try:
                os.remove(zip_path)
                logger.info(f"Removed {f}")
            except OSError as e:
                logger.warning(f"Could not remove {f}: {e}")

    logger.info("Cleaned up GTSRB dataset files")


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
        subset_size: Optional number of samples
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
                    transforms.Resize(
                        (32, 32), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
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

    # Create dataset directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Load appropriate dataset
    if dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "gtsrb":
        split = "train" if train else "test"
        try:
            dataset = datasets.GTSRB(
                root=data_dir, split=split, download=True, transform=transform
            )
            # Only clean up zip files after dataset is loaded successfully
            cleanup_gtsrb(data_dir)
        except Exception as e:
            logger.error(f"Error loading GTSRB dataset: {e}")
            raise
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
        # Get all indices
        indices = list(range(len(dataset)))
        
        # Randomly select subset_size samples
        if len(indices) > subset_size:
            subset_indices = np.random.choice(indices, subset_size, replace=False)
        else:
            subset_indices = indices
            
        # Create the subset
        dataset = Subset(dataset, subset_indices)
        logger.info(f"\nDataset: {dataset_name.upper()}")
        logger.info(f"Using subset of {len(dataset)} samples")

    return dataset

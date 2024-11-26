import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional
from tqdm import tqdm
import numpy as np

from config.dataclasses import PoisonConfig, PoisonResult
from attacks import create_poison_attack
from models import train_model, get_model, get_dataset
from utils.device import get_device, clear_memory
from utils.logging import get_logger
from .evaluation import evaluate_model, evaluate_attack
from .visualization import plot_results, plot_attack_comparison

logger = get_logger(__name__)


class PoisonExperiment:
    """Class to manage poisoning experiments."""

    def __init__(
        self,
        dataset_name: str,
        configs: List[PoisonConfig],
        batch_size: int = 128,
        epochs: int = 30,
        learning_rate: float = 0.001,
        num_workers: int = 2,
        subset_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "results",
        checkpoint_dir: str = "checkpoints",
    ):
        """Initialize experiment.

        Args:
            dataset_name: Name of dataset to use
            configs: List of poisoning configurations to run
            batch_size: Batch size for training
            epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            num_workers: Number of workers for data loading
            subset_size: Optional number of samples per class
            device: Optional device to use
            output_dir: Directory to save results
            checkpoint_dir: Directory to save checkpoints
        """
        self.dataset_name = dataset_name
        self.configs = configs
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.device = device if device is not None else get_device()
        self.output_dir = os.path.join(output_dir, dataset_name)
        self.checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Get datasets
        self.train_dataset = get_dataset(
            dataset_name, train=True, subset_size=subset_size
        )
        self.test_dataset = get_dataset(
            dataset_name, train=False, subset_size=subset_size
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False if num_workers == 0 else True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False if num_workers == 0 else True,
        )

        # Create model
        self.model = get_model(dataset_name).to(self.device)

        logger.info(f"Initialized experiment for {dataset_name} dataset")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
        logger.info(f"Running {len(self.configs)} poisoning configurations")

    def run(self) -> List[PoisonResult]:
        """Run all configured poisoning experiments.

        Returns:
            list: Results from all experiments
        """
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        results = []

        # First train clean model
        logger.info("Training clean model...")
        clean_checkpoint_path = os.path.join(self.checkpoint_dir, "clean_model")

        # Try to load clean model checkpoint
        clean_epoch, clean_loss = train_model(
            self.model,
            self.train_loader,
            val_loader=self.test_loader,
            epochs=self.epochs,
            device=self.device,
            learning_rate=self.learning_rate,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name="clean_model",
            resume_training=True,
        )

        # Evaluate clean model with single process
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        clean_accuracy = evaluate_model(self.model, test_loader, self.device)
        logger.info(f"Clean model accuracy: {clean_accuracy:.2f}%")

        # Run each poisoning configuration
        for config in self.configs:
            logger.info(f"\nRunning experiment with config: {config}")
            checkpoint_name = f"poisoned_{config.poison_type.value}"

            try:
                # Create attack instance
                attack = create_poison_attack(config, self.device)

                # Run poisoning attack
                poisoned_dataset, result = attack.poison_dataset(
                    self.train_dataset, self.model
                )

                # Create poisoned data loader with single process
                poisoned_loader = DataLoader(
                    poisoned_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    persistent_workers=False,
                )

                # Train model on poisoned data with checkpoint resumption
                train_model(
                    self.model,
                    poisoned_loader,
                    val_loader=test_loader,  # Use single process test loader
                    epochs=self.epochs,
                    device=self.device,
                    learning_rate=self.learning_rate,
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                    resume_training=True,
                )

                # Evaluate attack
                attack_results = evaluate_attack(
                    self.model,
                    poisoned_loader,
                    test_loader,  # Use single process test loader
                    self.device,
                )
                result.poisoned_accuracy = attack_results["poisoned_accuracy"]
                result.original_accuracy = attack_results["clean_accuracy"]

                # Save results
                result.save(self.output_dir)
                results.append(result)

            except Exception as e:
                logger.error(f"Error during poisoning experiment: {str(e)}")
                continue

            finally:
                # Clear memory
                clear_memory(self.device)

        # Plot results
        if results:
            plot_results(results, self.output_dir)
            plot_attack_comparison(results, self.output_dir)

        return results

#!/usr/bin/env python3
"""
Main script for running data poisoning experiments.
"""

import argparse
import logging
import os
import sys
from typing import List

from config.types import PoisonType
from config.dataclasses import PoisonConfig
from experiment import PoisonExperiment
from utils.device import get_device
from utils.export import export_results
from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger

logger = get_logger(__name__)
error_logger = get_error_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run data poisoning experiments")

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar100", "gtsrb", "imagenette"],
        default="gtsrb",
        help="Dataset to use",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of samples per class to use (default: None, use full dataset)",
    )

    # Attack parameters
    parser.add_argument(
        "--attack",
        type=str,
        choices=["pgd", "ga", "label_flip"],
        default="pgd",
        help="Type of attack to use",
    )
    parser.add_argument(
        "--poison-ratio",
        type=float,
        default=0.1,
        help="Ratio of dataset to poison (default: 0.1)",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class for label flipping attacks",
    )
    parser.add_argument(
        "--source-class",
        type=int,
        default=None,
        help="Source class for targeted label flipping attacks",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs (default: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="wrn-28-10",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="Number of classes in the dataset",
    )

    # Optimizer parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use (default: SGD)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: 5e-4)",
    )

    # Learning rate schedule
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="[60, 120, 160]",
        help="List of epochs to reduce learning rate",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.2,
        help="Factor to reduce learning rate by (default: 0.2)",
    )

    # Advanced training features
    parser.add_argument(
        "--use-amp",
        type=str,
        default="True",
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--use-swa",
        type=str,
        default="True",
        help="Use stochastic weight averaging",
    )
    parser.add_argument(
        "--swa-start",
        type=int,
        default=160,
        help="Epoch to start SWA from (default: 160)",
    )
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=0.05,
        help="SWA learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--use-mixup",
        type=str,
        default="True",
        help="Use mixup data augmentation",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Early stopping minimum delta (default: 0.001)",
    )

    # Data augmentation
    parser.add_argument(
        "--random-crop",
        type=str,
        default="True",
        help="Use random crop augmentation",
    )
    parser.add_argument(
        "--random-horizontal-flip",
        type=str,
        default="True",
        help="Use random horizontal flip augmentation",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="True",
        help="Normalize input images",
    )
    parser.add_argument(
        "--cutout",
        type=str,
        default="True",
        help="Use cutout augmentation",
    )
    parser.add_argument(
        "--cutout-length",
        type=int,
        default=16,
        help="Cutout patch length (default: 16)",
    )
    parser.add_argument(
        "--pin-memory",
        type=str,
        default="True",
        help="Use pinned memory for data loading",
    )

    # Device parameters
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use for training (default: best available)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )

    # Debug parameters
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def create_attack_configs(args) -> List[PoisonConfig]:
    """Create attack configurations based on command line arguments."""
    configs = []

    if args.attack == "pgd":
        configs.append(
            PoisonConfig(
                poison_type=PoisonType.PGD,
                poison_ratio=args.poison_ratio,
                pgd_eps=0.3,
                pgd_alpha=0.01,
                pgd_steps=40,
                random_seed=42,
            )
        )
    elif args.attack == "ga":
        configs.append(
            PoisonConfig(
                poison_type=PoisonType.GRADIENT_ASCENT,
                poison_ratio=args.poison_ratio,
                ga_steps=50,
                ga_iterations=100,
                ga_lr=0.1,
                random_seed=42,
            )
        )
    elif args.attack == "label_flip":
        if args.target_class is not None:
            if args.source_class is not None:
                # Source to target flipping
                configs.append(
                    PoisonConfig(
                        poison_type=PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
                        poison_ratio=args.poison_ratio,
                        source_class=args.source_class,
                        target_class=args.target_class,
                        random_seed=42,
                    )
                )
            else:
                # Random to target flipping
                configs.append(
                    PoisonConfig(
                        poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
                        poison_ratio=args.poison_ratio,
                        target_class=args.target_class,
                        random_seed=42,
                    )
                )
        else:
            # Random to random flipping
            configs.append(
                PoisonConfig(
                    poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
                    poison_ratio=args.poison_ratio,
                    random_seed=42,
                )
            )

    return configs


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create attack configurations
    configs = create_attack_configs(args)

    # Create and run experiment
    experiment = PoisonExperiment(
        dataset_name=args.dataset,
        configs=configs,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=get_device(args.device),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        model=args.model,
        num_classes=args.num_classes,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_factor=args.lr_factor,
        use_amp=args.use_amp.lower() == "true",
        use_swa=args.use_swa.lower() == "true",
        swa_start=args.swa_start,
        swa_lr=args.swa_lr,
        use_mixup=args.use_mixup.lower() == "true",
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        min_delta=args.min_delta,
        random_crop=args.random_crop.lower() == "true",
        random_horizontal_flip=args.random_horizontal_flip.lower() == "true",
        normalize=args.normalize.lower() == "true",
        cutout=args.cutout.lower() == "true",
        cutout_length=args.cutout_length,
        pin_memory=args.pin_memory.lower() == "true",
    )

    try:
        results = experiment.run()
        logger.info("Experiment completed successfully!")

        # Export results to CSV
        csv_filename = os.path.join(args.output_dir, f"{args.dataset}_{args.attack}_results.csv")
        export_results(results, csv_filename)
        logger.info(f"Results exported to {csv_filename}")

        # Log summary of results
        for result in results:
            if isinstance(result, dict):
                # Handle traditional classifier results
                logger.info(f"\nResults for {result['config']['poison_type']}:")
                logger.info(f"Classifier: {result.get('classifier', 'Unknown')}")
                logger.info(f"Poison ratio: {result['config'].get('poison_ratio', 0)}")
                logger.info(f"Accuracy: {result['metrics']['accuracy']:.2f}%")
            else:
                # Handle PoisonResult objects
                logger.info(f"\nResults for {result.config.poison_type.value}:")
                logger.info(f"Poison ratio: {result.config.poison_ratio}")
                logger.info(f"Original accuracy: {result.original_accuracy:.2f}%")
                logger.info(f"Poisoned accuracy: {result.poisoned_accuracy:.2f}%")
                logger.info(f"Attack success rate: {result.poison_success_rate:.2f}%")

    except KeyboardInterrupt:
        error_logger.log_error_msg("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_logger.log_error(e, "Experiment failed")
        logger.error(f"Error details: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

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
        default=30,
        help="Number of epochs (default: 30)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
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
        default=2,
        help="Number of worker processes for data loading (default: 2)",
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

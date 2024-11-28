#!/usr/bin/env python3
"""
Entry point for running data poisoning experiments.
"""

import argparse
import logging
import sys
from typing import Dict, Any

import yaml
from config.types import PoisonType
from config.dataclasses import PoisonConfig
from experiment import PoisonExperiment
from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger

logger = get_logger(__name__)
error_logger = get_error_logger()

def parse_args() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run data poisoning experiments")

    # Required parameters
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--attack", type=str, required=True, help="Attack type to use")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config file")

    return parser.parse_args()

def main():
    """Main function."""
    setup_logging()
    
    try:
        args = parse_args()
        
        # Create poison config
        poison_config = PoisonConfig(
            attack_type=PoisonType(args.attack),
            output_dir=args.output_dir
        )
        
        # Create and run experiment
        experiment = PoisonExperiment(
            dataset_name=args.dataset,
            configs=[poison_config],
            config_path=args.config,
            output_dir=args.output_dir
        )
        experiment.run()
        
    except Exception as e:
        error_logger.exception(f"Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

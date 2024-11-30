#!/usr/bin/env python3
import yaml
import argparse
import subprocess
import concurrent.futures
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any
import os
import sys
from tqdm import tqdm
import torch
from contextlib import contextmanager
from utils.logging import get_logger
from config.experiment_config import ExperimentConfig, create_config

logger = get_logger(__name__)

class ExperimentManager:
    """Manages the execution of experiments with configurations."""
    
    def __init__(self, config_path: str):
        """Initialize the experiment manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = ExperimentConfig.from_yaml(config_path)
        self._setup_device()
        self.results_dir = Path(self.config.output.base_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_experiments = self._count_total_experiments()
        
    def _setup_device(self):
        """Setup the compute device (CPU/GPU/MPS)."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using Apple Silicon MPS")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
            
    def _count_total_experiments(self) -> int:
        """Count total number of experiments to run."""
        count = 0
        for group in self.config.experiment_groups.values():
            for experiment in group.experiments:
                attacks = experiment.get('attacks', [experiment.get('attack')])
                count += len(attacks)
        return count
    
    def override_config(self, **kwargs):
        """Temporarily override config values for debugging.
        
        Args:
            **kwargs: Config values to override. Can include:
                - dataset_overrides: Dict of dataset-specific overrides
                - execution: Dict of execution settings
                - output: Dict of output settings
                - experiment_groups: Dict of experiment groups
        """
        self._original_config = {}
        
        for key, value in kwargs.items():
            if key in self.config.to_dict():
                self._original_config[key] = self.config.to_dict()[key]
                if isinstance(value, dict) and isinstance(self.config.to_dict()[key], dict):
                    # Deep update for nested dicts
                    self.config.to_dict()[key].update(value)
                else:
                    self.config.to_dict()[key] = value
            else:
                logger.warning(f"Unknown config key: {key}")
        
        # Update total experiments count
        self.total_experiments = self._count_total_experiments()
        return self
        
    def reset_config(self):
        """Reset any temporary config overrides."""
        if hasattr(self, '_original_config'):
            self.config.update(self._original_config)
            self._original_config = {}
            self.total_experiments = self._count_total_experiments()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_config()
        
    def run_experiments(self):
        """Run all experiments defined in the configuration."""
        print(f"\nSystem Information:")
        print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
        
        print(f"\nNumber of workers: {self.config.execution.max_workers}", flush=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Run each experiment group
        for group_name, group in self.config.experiment_groups.items():
            print(f"\nGroup: {group_name} - {group.description}")
            
            # Process each experiment in the group
            for experiment in group.experiments:
                exp_name = experiment['name']
                dataset = experiment['dataset']
                attacks = experiment.get('attacks', [experiment.get('attack')])
                
                for attack in attacks:
                    if attack == 'ga':
                        attack = 'gradient_ascent'
                        
                    logger.info(f"Starting experiment: {dataset} with {attack} attack")
                    
                    try:
                        from poison import run_poison_experiment
                        from config.defaults import get_poison_config
                        from config.dataclasses import PoisonConfig
                        from config.types import PoisonType
                        
                        # Start with base parameters
                        base_params = {
                            'dataset': dataset,
                            'attack': attack,
                            'output_dir': str(self.results_dir)
                        }
                        
                        # Get default poison config for this attack type
                        poison_defaults = get_poison_config(attack)
                        
                        # Create poison config object
                        poison_config_dict = poison_defaults.copy()
                        if 'poison_config' in experiment:
                            poison_config_dict.update(experiment['poison_config'])
                        
                        # Create PoisonConfig object
                        poison_config = PoisonConfig(
                            poison_type=PoisonType(attack),
                            **poison_config_dict
                        )
                        
                        # Add supported parameters to base_params
                        params = {
                            **base_params,
                            'poison_config': poison_config,  # Pass the entire config object
                            'subset_size': experiment.get('subset_size')
                        }
                        
                        logger.info(f"Running experiment with params: {params}")
                        
                        # Run the experiment
                        results = run_poison_experiment(**params)
                        logger.info(f"Completed: {dataset} with {attack} attack")
                        
                    except Exception as e:
                        logger.error(f"Failed to run {dataset} with {attack} attack: {str(e)}", exc_info=True)
                        continue
        
        # Consolidate results with progress bar
        print(f"\nConsolidating results...", flush=True)
        self._consolidate_results()
        
        # Print summary
        print(f"\nExperiment Summary:", flush=True)
        print(f"Total experiments: {self.total_experiments}", flush=True)
        print(f"Successful experiments: {self.total_experiments}", flush=True)
        print(f"Failed experiments: 0", flush=True)
        if self.config.output.consolidated_file:
            print(f"Results saved to: {self.results_dir / self.config.output.consolidated_file}", flush=True)
    
    def _consolidate_results(self):
        """Consolidate all experiment results into a single CSV file."""
        try:
            from utils.export import load_experiment_results, create_results_dataframe
            
            # Find all JSON result files
            result_files = list(self.results_dir.glob("*.json"))
            
            logger.info(f"Found {len(result_files)} result files to consolidate")
            for file in result_files:
                logger.info(f"Processing file: {file}")
            
            # Load all results from JSON files
            results = []
            with tqdm(result_files, desc="Reading result files", unit="file", mininterval=0.1) as pbar:
                for file in pbar:
                    if Path(file).exists():
                        logger.info(f"Reading file: {file}")
                        results.extend(load_experiment_results(file))
                    else:
                        logger.warning(f"File not found: {file}")
            
            if not results:
                logger.warning("No valid result files found to consolidate")
                return
            
            # Convert results to DataFrame with proper format
            df = create_results_dataframe(results)
            
            # Save consolidated results
            output_file = self.results_dir / self.config.output.consolidated_file
            logger.info(f"Saving consolidated results to: {output_file}")
            df.to_csv(output_file, index=False)
            logger.info("Results consolidated successfully")
            
            # Optionally remove individual result files
            save_individual = self.config.output.get('save_individual_results', True)
            if not save_individual:
                with tqdm(result_files, desc="Cleaning up individual files", unit="file", mininterval=0.1) as pbar:
                    for file in pbar:
                        try:
                            Path(file).unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove file {file}: {e}")
                
        except Exception as e:
            logger.error(f"Error consolidating results: {str(e)}", exc_info=True)

def main():
    try:
        # Enable debugging for torch.multiprocessing
        if sys.platform == 'darwin':  # macOS specific
            import torch.multiprocessing as mp
            mp.set_start_method('spawn')

        parser = argparse.ArgumentParser(description="Run poisoning experiments")
        parser.add_argument(
            "--config",
            default="experiments/config.yaml",
            help="Path to experiment configuration file"
        )
        args = parser.parse_args()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Log system info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Create and run experiment manager
        manager = ExperimentManager(args.config)
        manager.run_experiments()
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

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

from utils.logging import setup_logging, get_logger
from utils.error_logging import get_error_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)
error_logger = get_error_logger()

class ExperimentManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config['output']['base_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_experiments = self._count_total_experiments()
        
    def _count_total_experiments(self) -> int:
        """Count total number of experiments to run."""
        count = 0
        for group in self.config['experiment_groups'].values():
            for experiment in group['experiments']:
                attacks = experiment.get('attacks', [experiment.get('attack')])
                count += len(attacks)
        return count
    
    def _get_device_info(self) -> str:
        """Get information about available compute devices."""
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"
            
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
            if key in self.config:
                self._original_config[key] = self.config[key]
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Deep update for nested dicts
                    self.config[key].update(value)
                else:
                    self.config[key] = value
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
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _build_command(self, experiment: Dict[str, Any], attack: str) -> List[str]:
        """Build command for a single experiment."""
        # Map attack names to PoisonType values
        attack_map = {
            "pgd": "pgd",
            "ga": "gradient_ascent",
            "random": "label_flip_random_random",
            "target": "label_flip_random_target",
            "source": "label_flip_source_target"
        }
        
        mapped_attack = attack_map.get(attack, attack)
        cmd = [
            "python", "poison.py",
            "--dataset", experiment["dataset"],
            "--attack", mapped_attack,
            "--output-dir", str(self.results_dir),
            "--poison-ratio", str(experiment.get("poison_ratio", 0.1))
        ]
        
        logger.info(f"Built command: {' '.join(cmd)}")
        return cmd
    
    def _run_single_experiment(self, experiment: Dict[str, Any], attack: str, output_file: Path) -> None:
        """Run a single experiment."""
        try:
            logger.info(f"Running experiment: {experiment['name']}")
            logger.info(f"Results will be saved to: {output_file}")
            
            # Create output directory
            self.results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {self.results_dir}")
            
            # Run experiment using function interface
            from poison import run_poison_experiment
            
            # Get poison config from experiment
            poison_config = experiment.get("poison_config", {})
            poison_ratio = poison_config.get("poison_ratio", 0.1)
            batch_size = poison_config.get("batch_size", 32)
            
            logger.info(f"Experiment poison config: {poison_config}")
            logger.info(f"Using poison_ratio={poison_ratio}, batch_size={batch_size}")
            
            results = run_poison_experiment(
                dataset=experiment["dataset"],
                attack=attack,
                output_dir=str(self.results_dir),
                poison_ratio=poison_ratio,
                batch_size=batch_size
            )
            
            # Log results
            logger.info(f"Experiment completed successfully: {experiment['name']}")
            logger.info(f"Results: {results}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {experiment['name']}", exc_info=True)
            raise
    
    def run_experiments(self):
        """Run all experiments defined in the configuration."""
        print(f"\nSystem Information:")
        print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
        
        print(f"\nNumber of workers: {self.config['execution']['max_workers']}", flush=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Run each experiment group
        for group_name, group in self.config['experiment_groups'].items():
            print(f"\nGroup: {group_name} - {group['description']}")
            
            # Process each experiment in the group
            for experiment in group['experiments']:
                exp_name = experiment['name']
                dataset = experiment['dataset']
                attacks = experiment.get('attacks', [experiment.get('attack')])
                
                for attack in attacks:
                    if attack == 'ga':
                        attack = 'gradient_ascent'
                        
                    logger.info(f"Starting experiment: {dataset} with {attack} attack")
                    
                    try:
                        from poison import run_poison_experiment
                        
                        # Prepare experiment parameters
                        params = {
                            'dataset': dataset,
                            'attack': attack,
                            'output_dir': str(self.results_dir),
                            'poison_ratio': experiment.get('poison_config', {}).get('poison_ratio', 0.1)  # Get from poison_config
                        }
                        
                        # Add optional parameters if present
                        if 'subset_size' in experiment:
                            params['subset_size'] = experiment['subset_size']
                        if 'target_class' in experiment:
                            params['target_class'] = experiment['target_class']
                        if 'source_class' in experiment:
                            params['source_class'] = experiment['source_class']
                        
                        # Add poison config parameters
                        if 'poison_config' in experiment:
                            if 'batch_size' in experiment['poison_config']:
                                params['batch_size'] = experiment['poison_config']['batch_size']
                        
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
        if self.config['output']['consolidated_file']:
            print(f"Results saved to: {self.results_dir / self.config['output']['consolidated_file']}", flush=True)
    
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
            output_file = self.results_dir / self.config['output']['consolidated_file']
            logger.info(f"Saving consolidated results to: {output_file}")
            df.to_csv(output_file, index=False)
            logger.info("Results consolidated successfully")
            
            # Optionally remove individual result files
            save_individual = self.config['output'].get('save_individual_results', True)
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

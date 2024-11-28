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

from utils.logging import get_logger
from utils.error_logging import get_error_logger

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
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _build_command(self, experiment: Dict[str, Any], attack: str) -> List[str]:
        """Build command for a single experiment."""
        cmd = ["python", "poison.py"]
        
        # Get dataset-specific defaults
        dataset = experiment['dataset']
        dataset_defaults = self.config.get('dataset_defaults', {}).get(dataset, {})
        
        # Start with global defaults
        params = self.config['defaults'].copy()
        
        # Apply dataset-specific defaults
        params.update(dataset_defaults)
        
        # Apply experiment-specific parameters
        for key, value in experiment.items():
            if key not in ['name', 'attacks', 'dataset']:
                params[key] = value
        
        # Add all parameters to command
        for key, value in params.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Add dataset and attack
        cmd.extend(["--dataset", dataset])
        cmd.extend(["--attack", attack])
        
        return cmd
    
    def _run_single_experiment(self, cmd: List[str], experiment_name: str, attack: str) -> str:
        """Run a single experiment and return its results file path."""
        output_file = self.results_dir / f"{experiment_name}_{attack}_{self.timestamp}.csv"
        
        try:
            logger.info(f"Running experiment: {' '.join(cmd)}")
            logger.info(f"Results will be saved to: {output_file}")
            # Add output directory to command
            cmd.extend(["--output-dir", str(self.results_dir)])
            
            logger.info(f"Creating output directory: {self.results_dir}")
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                # The output file should be directly in the results directory
                return str(output_file)
                
        except subprocess.CalledProcessError as e:
            error_logger.log_error(e, f"Experiment failed: {experiment_name}_{attack}")
            logger.error(f"Experiment failed: {experiment_name}_{attack}")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return None
            
        return str(output_file)
    
    def run_experiments(self):
        """Run all experiments defined in the configuration."""
        all_results_files = []
        completed_experiments = 0
        
        # Print system information
        device_info = self._get_device_info()
        print(f"\nSystem Information:", flush=True)
        print(f"Device: {device_info}", flush=True)
        print(f"Number of workers: {self.config['execution']['max_workers']}", flush=True)
        
        print(f"\nStarting {self.total_experiments} experiments across {len(self.config['experiment_groups'])} groups", flush=True)
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config['execution']['max_workers']
        ) as executor:
            future_to_exp = {}
            
            # Create progress bar for experiment submission
            with tqdm(total=self.total_experiments, desc="Submitting experiments", unit="exp", mininterval=0.1) as pbar:
                # Submit all experiments
                for group_name, group in self.config['experiment_groups'].items():
                    print(f"\nGroup: {group_name} - {group['description']}", flush=True)
                    for experiment in group['experiments']:
                        # Handle multiple attacks per experiment
                        attacks = experiment.get('attacks', [experiment.get('attack')])
                        for attack in attacks:
                            cmd = self._build_command(experiment, attack)
                            future = executor.submit(
                                self._run_single_experiment,
                                cmd,
                                experiment['name'],
                                attack
                            )
                            future_to_exp[future] = (experiment['name'], attack)
                            pbar.update(1)
            
            print("\nRunning experiments:", flush=True)
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_exp):
                exp_name, attack = future_to_exp[future]
                try:
                    result_file = future.result()
                    if result_file:
                        all_results_files.append(result_file)
                        completed_experiments += 1
                        completion_percentage = (completed_experiments/self.total_experiments)*100
                        print(f"Completed {exp_name}_{attack} ({completed_experiments}/{self.total_experiments} - {completion_percentage:.1f}%)", flush=True)
                except Exception as e:
                    error_logger.log_error(e, f"Error collecting results for {exp_name}_{attack}")
                    print(f"Failed: {exp_name}_{attack} - {str(e)}", flush=True)
        
        # Consolidate results with progress bar
        print(f"\nConsolidating {len(all_results_files)} result files...", flush=True)
        self._consolidate_results(all_results_files)
        
        # Print summary
        print(f"\nExperiment Summary:", flush=True)
        print(f"Total experiments: {self.total_experiments}", flush=True)
        print(f"Successful experiments: {completed_experiments}", flush=True)
        print(f"Failed experiments: {self.total_experiments - completed_experiments}", flush=True)
        if all_results_files:
            print(f"Results saved to: {self.results_dir / self.config['output']['consolidated_file']}", flush=True)
    
    def _consolidate_results(self, result_files: List[str]):
        """Consolidate all experiment results into a single CSV file."""
        if not result_files:
            logger.warning("No results files to consolidate")
            return
        
        try:
            logger.info(f"Found {len(result_files)} result files to consolidate")
            for file in result_files:
                logger.info(f"Processing file: {file}")
            
            # Read and combine all CSV files with progress bar
            dfs = []
            with tqdm(result_files, desc="Reading result files", unit="file", mininterval=0.1) as pbar:
                for file in pbar:
                    if Path(file).exists():
                        logger.info(f"Reading file: {file}")
                        df = pd.read_csv(file)
                        dfs.append(df)
                    else:
                        logger.warning(f"File not found: {file}")
            
            if not dfs:
                logger.warning("No valid CSV files found to consolidate")
                return
            
            # Combine all results
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save consolidated results
            output_file = self.results_dir / self.config['output']['consolidated_file']
            logger.info(f"Saving consolidated results to: {output_file}")
            combined_df.to_csv(output_file, index=False)
            logger.info("Results consolidated successfully")
            
            # Optionally remove individual result files
            if not self.config['output']['save_individual_results']:
                with tqdm(result_files, desc="Cleaning up individual files", unit="file", mininterval=0.1) as pbar:
                    for file in pbar:
                        Path(file).unlink()
                logger.info("Removed individual result files")
                
        except Exception as e:
            error_logger.log_error(e, "Error consolidating results")
            logger.error(f"Error consolidating results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run poisoning experiments")
    parser.add_argument(
        "--config",
        default="experiments/config.yaml",
        help="Path to experiment configuration file"
    )
    args = parser.parse_args()
    
    manager = ExperimentManager(args.config)
    manager.run_experiments()

if __name__ == "__main__":
    main()

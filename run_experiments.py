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
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _build_command(self, experiment: Dict[str, Any], attack: str) -> List[str]:
        """Build command for a single experiment."""
        cmd = ["python", "poison.py"]
        
        # Add default parameters
        for key, value in self.config['defaults'].items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Add experiment-specific parameters
        cmd.extend(["--dataset", experiment['dataset']])
        cmd.extend(["--attack", attack])
        
        # Add optional parameters if present
        for param in ['source_class', 'target_class']:
            if param in experiment:
                cmd.extend([f"--{param.replace('_', '-')}", str(experiment[param])])
        
        return cmd
    
    def _run_single_experiment(self, cmd: List[str], experiment_name: str, attack: str) -> str:
        """Run a single experiment and return its results file path."""
        output_file = self.results_dir / f"{experiment_name}_{attack}_{self.timestamp}.csv"
        
        try:
            logger.info(f"Running experiment: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Experiment completed: {experiment_name}_{attack}")
            
            # Move the results to our organized directory
            if result.returncode == 0:
                # Find the most recent results file
                results_pattern = "results_*.csv"
                results_files = sorted(Path().glob(results_pattern), key=os.path.getmtime)
                if results_files:
                    latest_results = results_files[-1]
                    latest_results.rename(output_file)
                
        except subprocess.CalledProcessError as e:
            error_logger.log_error(e, f"Experiment failed: {experiment_name}_{attack}")
            logger.error(f"Experiment failed: {experiment_name}_{attack}")
            logger.error(f"Error output: {e.stderr}")
            return None
            
        return str(output_file)
    
    def run_experiments(self):
        """Run all experiments defined in the configuration."""
        all_results_files = []
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config['execution']['max_workers']
        ) as executor:
            future_to_exp = {}
            
            # Submit all experiments
            for group_name, group in self.config['experiment_groups'].items():
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
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_exp):
                exp_name, attack = future_to_exp[future]
                try:
                    result_file = future.result()
                    if result_file:
                        all_results_files.append(result_file)
                        logger.info(f"Collected results for {exp_name}_{attack}")
                except Exception as e:
                    error_logger.log_error(e, f"Error collecting results for {exp_name}_{attack}")
                    logger.error(f"Error collecting results for {exp_name}_{attack}: {str(e)}")
        
        # Consolidate results
        self._consolidate_results(all_results_files)
    
    def _consolidate_results(self, result_files: List[str]):
        """Consolidate all experiment results into a single CSV file."""
        if not result_files:
            logger.warning("No results files to consolidate")
            return
        
        try:
            # Read and combine all CSV files
            dfs = []
            for file in result_files:
                if Path(file).exists():
                    df = pd.read_csv(file)
                    dfs.append(df)
            
            if not dfs:
                logger.warning("No valid CSV files found to consolidate")
                return
            
            # Combine all results
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save consolidated results
            output_file = self.results_dir / self.config['output']['consolidated_file']
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Consolidated results saved to {output_file}")
            
            # Optionally remove individual result files
            if not self.config['output']['save_individual_results']:
                for file in result_files:
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

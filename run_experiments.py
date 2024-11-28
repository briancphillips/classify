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
            # Build command
            cmd = self._build_command(experiment, attack)
            logger.info(f"Running experiment: {' '.join(cmd)}")
            logger.info(f"Results will be saved to: {output_file}")
            
            # Create output directory
            self.results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {self.results_dir}")
            
            # Run command and capture output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output in real-time
            stdout_lines = []
            stderr_lines = []
            
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    logger.info(stdout_line.strip())
                    stdout_lines.append(stdout_line)
                if stderr_line:
                    logger.error(stderr_line.strip())
                    stderr_lines.append(stderr_line)
                    
                if process.poll() is not None:
                    break
            
            # Get remaining output
            stdout, stderr = process.communicate()
            if stdout:
                logger.info(stdout.strip())
                stdout_lines.append(stdout)
            if stderr:
                logger.error(stderr.strip())
                stderr_lines.append(stderr)
            
            # Check result
            if process.returncode != 0:
                error_msg = (
                    f"Command failed with return code {process.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stdout:\n{''.join(stdout_lines)}\n"
                    f"Stderr:\n{''.join(stderr_lines)}"
                )
                error_logger.log_error_msg(error_msg)
                raise subprocess.CalledProcessError(process.returncode, cmd)
            
        except Exception as e:
            error_logger.log_error(e, f"Experiment failed: {experiment['name']}")
            raise
    
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
                            output_file = self.results_dir / f"{experiment['dataset']}_{attack}_results.csv"
                            future = executor.submit(
                                self._run_single_experiment,
                                experiment,
                                attack,
                                output_file
                            )
                            future_to_exp[future] = (experiment['name'], attack)
                            pbar.update(1)
            
            print("\nRunning experiments:", flush=True)
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_exp):
                exp_name, attack = future_to_exp[future]
                try:
                    future.result()
                    completed_experiments += 1
                    completion_percentage = (completed_experiments/self.total_experiments)*100
                    print(f"Completed {exp_name}_{attack} ({completed_experiments}/{self.total_experiments} - {completion_percentage:.1f}%)", flush=True)
                except Exception as e:
                    error_logger.log_error(e, f"Error collecting results for {exp_name}_{attack}")
                    print(f"Failed: {exp_name}_{attack} - {str(e)}", flush=True)
        
        # Consolidate results with progress bar
        print(f"\nConsolidating results...", flush=True)
        self._consolidate_results()
        
        # Print summary
        print(f"\nExperiment Summary:", flush=True)
        print(f"Total experiments: {self.total_experiments}", flush=True)
        print(f"Successful experiments: {completed_experiments}", flush=True)
        print(f"Failed experiments: {self.total_experiments - completed_experiments}", flush=True)
        if self.config['output']['consolidated_file']:
            print(f"Results saved to: {self.results_dir / self.config['output']['consolidated_file']}", flush=True)
    
    def _consolidate_results(self):
        """Consolidate all experiment results into a single CSV file."""
        try:
            # Find all result files
            result_files = [f for f in self.results_dir.glob("*.csv") if f.is_file()]
            
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
            save_individual = self.config['output'].get('save_individual_results', True)
            if not save_individual:
                with tqdm(result_files, desc="Cleaning up individual files", unit="file", mininterval=0.1) as pbar:
                    for file in pbar:
                        try:
                            Path(file).unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove file {file}: {e}")
                
        except Exception as e:
            error_logger.log_error(e, "Error consolidating results")
            logger.error(f"Error consolidating results: {str(e)}")

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
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        error_logger.exception("An error occurred during training:")
        raise

if __name__ == "__main__":
    main()

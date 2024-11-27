"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from utils.logging import get_logger

logger = get_logger(__name__)

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing metrics
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 5))
    
    # Plot training metrics
    for metric, values in history.items():
        if metric.startswith('train_'):
            plt.plot(values, label=metric)
    
    # Plot validation metrics
    for metric, values in history.items():
        if metric.startswith('val_'):
            plt.plot(values, label=metric, linestyle='--')
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved training history plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_poisoned_results(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot results from poisoning experiments.
    
    Args:
        results: Dictionary containing poisoning results
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Convert results to DataFrame
    data = []
    for exp_name, exp_results in results.items():
        attack_type = exp_results.get('config', {}).get('poison_type', 'unknown')
        
        # Extract metrics
        metrics = {
            'Poison Success Rate': exp_results.get('poison_success_rate', 0),
            'Original Accuracy': exp_results.get('original_accuracy', 0),
            'Poisoned Accuracy': exp_results.get('poisoned_accuracy', 0)
        }
        
        for metric, value in metrics.items():
            data.append({
                'Attack': attack_type,
                'Metric': metric,
                'Value': value  # Values are already in percentages
            })
    
    if not data:
        logger.error("No metrics found in results")
        return
        
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Attack', y='Value', hue='Metric')
    
    plt.title('Poisoning Attack Results')
    plt.xlabel('Attack Type')
    plt.ylabel('Value (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved poisoning results plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_classifier_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot comparison of different classifiers.
    
    Args:
        results: Dictionary containing classifier results
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Convert results to DataFrame
    data = []
    for classifier, metrics in results.items():
        for metric, value in metrics.items():
            data.append({
                'Classifier': classifier,
                'Metric': metric,
                'Value': value
            })
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Classifier', y='Value', hue='Metric')
    
    plt.title('Classifier Comparison')
    plt.xlabel('Classifier')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved classifier comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def load_and_plot_results(
    results_dir: Union[str, Path],
    plot_type: str = 'training',
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Load results from a directory and create appropriate plots.
    
    Args:
        results_dir: Directory containing results
        plot_type: Type of plot to create ('training', 'poisoning', or 'classifier')
        save_dir: Optional directory to save plots
        show: Whether to display the plots
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory {results_dir} does not exist")
        return
    
    # Load results
    results = {}
    for results_file in results_dir.glob('*.json'):
        with open(results_file) as f:
            results[results_file.stem] = json.load(f)
    
    if not results:
        logger.error(f"No results found in {results_dir}")
        return
    
    # Create save directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create appropriate plot
    if plot_type == 'training':
        for exp_name, history in results.items():
            save_path = save_dir / f"{exp_name}_training.png" if save_dir else None
            plot_training_history(history, save_path, show)
    
    elif plot_type == 'poisoning':
        save_path = save_dir / "poisoning_results.png" if save_dir else None
        plot_poisoned_results(results, save_path, show)
    
    elif plot_type == 'classifier':
        save_path = save_dir / "classifier_comparison.png" if save_dir else None
        plot_classifier_comparison(results, save_path, show)
    
    else:
        logger.error(f"Unknown plot type: {plot_type}")

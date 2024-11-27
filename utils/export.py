"""
Utilities for exporting experiment results to CSV format.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_experiment_results(results_dir: Union[str, Path]) -> List[Dict]:
    """
    Load all experiment results from JSON files in a directory.
    
    Args:
        results_dir: Directory containing experiment result JSON files
        
    Returns:
        List of experiment results
    """
    results_dir = Path(results_dir)
    results = []
    
    for results_file in results_dir.glob('*.json'):
        with open(results_file) as f:
            result = json.load(f)
            # Add filename to help track source
            result['source_file'] = results_file.name
            results.append(result)
            
    return results

def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def extract_iteration(filename: str) -> int:
    """Extract iteration number from filename timestamp."""
    try:
        # Assuming filename format: *_YYYYMMDD_HHMMSS.json
        timestamp = filename.split('_')[-2]  # Get YYYYMMDD part
        return int(timestamp[-2:])  # Use last 2 digits as iteration
    except (IndexError, ValueError):
        return 0

def create_results_dataframe(results: List[Union[Dict, 'PoisonResult']], max_classes: int = 100):
    """
    Convert experiment results into a pandas DataFrame with the required format.
    
    Args:
        results: List of experiment results (either dict or PoisonResult objects)
        max_classes: Maximum number of classes to include in per-class metrics
        
    Returns:
        DataFrame containing all experiment results
    """
    rows = []
    for result in results:
        # Helper function to safely get attributes from either dict or object
        def get_value(key, default=None):
            if isinstance(result, dict):
                return result.get(key, default)
            return getattr(result, key, default)
            
        # Get config values
        config = get_value('config', {})
        if not isinstance(config, dict):
            config = config.__dict__ if hasattr(config, '__dict__') else {}
            
        # Handle PoisonType enum
        poison_type = config.get('poison_type', '')
        if hasattr(poison_type, 'value'):
            poison_type = poison_type.value
            
        # Get metrics
        metrics = get_value('metrics', {})
        if not isinstance(metrics, dict):
            metrics = metrics.__dict__ if hasattr(metrics, '__dict__') else {}
            
        # Get current date
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
            
        row = {
            # Required columns in exact order
            'Date': current_date,
            'Iteration': get_value('iteration', 1),
            'Dataset': get_value('dataset_name', ''),  
            'Classifier': get_value('classifier', 'neural_network'),  
            'Model_Architecture': get_value('model_architecture', 'wrn'),
            'Modification_Method': poison_type,
            'Num_Poisoned': int(config.get('poison_ratio', 0.0) * get_value('train_size', 1000)),
            'Poisoned_Classes': str(config.get('source_class', 'all')),
            'Flip_Type': 'N/A',
            'Epochs': get_value('epochs', 2),
            'Batch_Size': get_value('batch_size', 32),
            'Learning_Rate': get_value('learning_rate', 0.1),
            'Weight_Decay': get_value('weight_decay', 0.0005),
            'Optimizer': get_value('optimizer', 'sgd'),
            'Train_Size': get_value('train_size', 1000),
            'Test_Size': get_value('test_size', 1000),
            'Num_Classes': get_value('num_classes', 100),
            'Training_Time': metrics.get('training_time', 0.0),
            'Inference_Time': metrics.get('inference_time', 0.0),
            'Total_Time': metrics.get('total_time', 0.0),
            'Original_Accuracy': metrics.get('original_accuracy', 0.0),
            'Poisoned_Accuracy': metrics.get('poisoned_accuracy', 0.0),
            'Poison_Success_Rate': metrics.get('attack_success_rate', 0.0),
            'Clean_Test_Accuracy': metrics.get('clean_accuracy', 0.0),
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1-Score': metrics.get('f1_score', 0.0),
            'Latency': metrics.get('avg_batch_time', 0.0),
            'Final_Train_Loss': metrics.get('final_train_loss', 0.0),
            'Final_Test_Loss': metrics.get('final_val_loss', 0.0),
            'Best_Train_Loss': metrics.get('best_train_loss', 0.0),
            'Best_Test_Loss': metrics.get('best_val_loss', 0.0),
        }
        
        # Add per-class accuracies
        class_accuracies = metrics.get('class_accuracies', {})
        if isinstance(class_accuracies, dict):
            for i in range(100):  # Always include all 100 classes
                row[f'Class_{i}_Accuracy'] = class_accuracies.get(str(i), 0.0)
        else:
            for i in range(100):
                row[f'Class_{i}_Accuracy'] = 0.0
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def export_results_to_csv(
    results_dirs: List[Union[str, Path]],
    output_file: Union[str, Path] = 'all_results.csv',
    max_classes: int = 100
) -> None:
    """
    Export all experiment results to a CSV file.
    
    Args:
        results_dirs: List of directories containing result files
        output_file: Path to output CSV file
        max_classes: Maximum number of classes to include
    """
    all_results = []
    
    # Load results from all directories
    for results_dir in results_dirs:
        results = load_experiment_results(results_dir)
        all_results.extend(results)
    
    # Convert to DataFrame
    df = create_results_dataframe(all_results, max_classes)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} results to {output_file}")

def export_results(results, output_file: Union[str, Path]):
    """
    Export experiment results directly to CSV.
    
    Args:
        results: List of experiment results (PoisonResult objects or dicts)
        output_file: Path to output CSV file
    """
    df = create_results_dataframe(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Exported {len(df)} results to {output_file}")

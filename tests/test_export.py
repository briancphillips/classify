"""
Test export functionality.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.export import (
    load_experiment_results,
    create_results_dataframe,
    export_results_to_csv
)

def test_load_experiment_results():
    """Test loading experiment results from JSON files."""
    results_dir = project_root / "results" / "cifar100"
    results = load_experiment_results(results_dir)
    
    assert len(results) > 0
    assert isinstance(results[0], dict)
    assert 'source_file' in results[0]

def test_create_results_dataframe():
    """Test creating DataFrame from experiment results."""
    results_dir = project_root / "results" / "cifar100"
    results = load_experiment_results(results_dir)
    df = create_results_dataframe(results)
    
    # Check required columns
    required_columns = [
        'Iteration', 'Dataset', 'Classifier', 'Modification_Method',
        'Num_Poisoned', 'Poisoned_Classes', 'Flip_Type', 'Accuracy',
        'Precision', 'Recall', 'F1-Score', 'Latency'
    ]
    for col in required_columns:
        assert col in df.columns
    
    # Check class accuracy columns
    for i in range(100):
        assert f'Class_{i}_Accuracy' in df.columns

def test_export_results_to_csv():
    """Test exporting results to CSV file."""
    results_dirs = [
        project_root / "results" / "cifar100",
        project_root / "results" / "gtsrb"
    ]
    output_file = project_root / "results" / "test_all_results.csv"
    
    # Export results
    export_results_to_csv(results_dirs, output_file)
    
    # Check if file exists and has content
    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) > 0
    
    # Clean up
    output_file.unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

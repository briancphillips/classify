#!/usr/bin/env python3

import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def load_all_results(base_dir: str) -> Dict[str, List[Dict]]:
    """Load results from all dataset directories."""
    all_results = {}
    
    # Get all subdirectories in the results directory
    for item in os.listdir(base_dir):
        dataset_dir = os.path.join(base_dir, item)
        if os.path.isdir(dataset_dir):
            # Load results from this dataset directory
            dataset_results = []
            for filepath in glob.glob(os.path.join(dataset_dir, "poison_results_*.json")):
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    dataset_results.append(result)
            
            if dataset_results:  # Only add if we found results
                all_results[item] = dataset_results
    
    return all_results

def simplify_label(label: str) -> str:
    """Convert complex label to simple word."""
    # Extract just the attack type from labels like "pgd_0.1_clean" or "label_flip_random_random_0.1_poisoned"
    if label == "clean_baseline":
        return "Clean"
    
    attack_type = label.split('_')[0]
    
    # Map to simpler names
    mapping = {
        'pgd': 'PGD',
        'ga': 'Gradient Ascent',
        'label': 'LabelFlip'  # For label_flip attacks
    }
    return mapping.get(attack_type, attack_type)

def set_plot_style():
    """Set the global plot style for better visualization."""
    # Set clean, modern style parameters manually
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    # Modern color palette
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#2ecc71',  # Green
        '#3498db',  # Blue
        '#e74c3c',  # Red
        '#f1c40f',  # Yellow
        '#9b59b6',  # Purple
        '#1abc9c'   # Turquoise
    ])

def create_attack_label(attack_type: str, poison_ratio: float = None) -> str:
    """Create a formatted attack label with poison ratio if applicable."""
    if attack_type == "Clean":
        return "Clean"
    return f"{attack_type}\n({poison_ratio*100:.0f}%)" if poison_ratio is not None else attack_type

def plot_combined_classifier_comparison(all_results: Dict[str, List[Dict]], output_dir: str):
    """Create a combined plot showing classifier accuracies across all datasets."""
    set_plot_style()
    data = []
    
    for dataset_name, results in all_results.items():
        # First, add clean baseline from the first result's clean accuracy
        if results:
            for clf_name, acc in results[0]['classifier_results_clean'].items():
                data.append({
                    'Classifier': clf_name.upper(),
                    'Accuracy': acc,
                    'Dataset': dataset_name,
                    'Type': 'clean_baseline',
                    'Simple_Type': 'Clean',
                    'Display_Label': 'Clean'
                })
        
        for result in results:
            attack_type = result['config']['poison_type']
            poison_ratio = result['config']['poison_ratio']
            
            # Add poisoned dataset results only (clean baseline already added)
            for clf_name, acc in result['classifier_results_poisoned'].items():
                simple_type = simplify_label(attack_type)
                data.append({
                    'Classifier': clf_name.upper(),
                    'Accuracy': acc,
                    'Dataset': dataset_name,
                    'Type': f"{attack_type}_{poison_ratio}_poisoned",
                    'Simple_Type': simple_type,
                    'Display_Label': create_attack_label(simple_type, poison_ratio)
                })
    
    df = pd.DataFrame(data)
    
    # Sort the data to ensure Clean baseline appears first
    df['Simple_Type'] = pd.Categorical(df['Simple_Type'], 
                                     categories=['Clean'] + sorted(list(set(df['Simple_Type']) - {'Clean'})), 
                                     ordered=True)
    df = df.sort_values('Simple_Type')
    
    # Create the plot with subplots for each dataset
    num_datasets = len(all_results)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(15, 6*num_datasets))
    if num_datasets == 1:
        axes = [axes]
    
    for ax, (dataset_name, _) in zip(axes, all_results.items()):
        # Filter data for this dataset
        dataset_df = df[df['Dataset'] == dataset_name]
        
        # Create the subplot with custom style
        bars = plt.bar(dataset_df['Display_Label'], dataset_df['Accuracy'], ax=ax, alpha=0.8)
        
        # Remove the top line of bars
        for patch in ax.patches:
            patch.set_edgecolor('none')
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        # Customize the subplot
        ax.set_title(f'{dataset_name} Dataset Performance', fontsize=14, pad=20)
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(dataset_df['Accuracy']) * 1.15)  # Add 15% padding for labels
    
    plt.tight_layout()
    
    # Save the plot with high DPI
    plot_path = os.path.join(output_dir, 'combined_classifier_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attack_effectiveness(all_results: Dict[str, List[Dict]], output_dir: str):
    """Create a plot showing the effectiveness of different attacks across datasets."""
    set_plot_style()
    data = []
    
    for dataset_name, results in all_results.items():
        # Add clean baseline
        if results:
            clean_acc = results[0]['original_accuracy']
            data.append({
                'Dataset': dataset_name,
                'Attack': 'clean_baseline',
                'Simple_Attack': 'Clean',
                'Display_Label': 'Clean',
                'Original Accuracy': clean_acc,
                'Poisoned Accuracy': clean_acc,
                'Success Rate': 0
            })
        
        for result in results:
            attack_type = result['config']['poison_type']
            poison_ratio = result['config']['poison_ratio']
            simple_type = simplify_label(attack_type)
            
            data.append({
                'Dataset': dataset_name,
                'Attack': f"{attack_type}_{poison_ratio}",
                'Simple_Attack': simple_type,
                'Display_Label': create_attack_label(simple_type, poison_ratio),
                'Original Accuracy': result['original_accuracy'],
                'Poisoned Accuracy': result['poisoned_accuracy'],
                'Success Rate': result['poison_success_rate']
            })
    
    df = pd.DataFrame(data)
    
    # Sort to ensure Clean baseline appears first
    df['Simple_Attack'] = pd.Categorical(df['Simple_Attack'], 
                                       categories=['Clean'] + sorted(list(set(df['Simple_Attack']) - {'Clean'})), 
                                       ordered=True)
    df = df.sort_values('Simple_Attack')
    
    # Create subplots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    metrics = ['Original Accuracy', 'Poisoned Accuracy', 'Success Rate']
    colors = ['#2ecc71', '#e74c3c', '#f1c40f']  # Green, Red, Yellow
    
    for ax, metric, color in zip(axes, metrics, colors):
        bars = plt.bar(df['Display_Label'], df[metric], ax=ax, alpha=0.8, color=color)
        
        # Remove the top line of bars
        for patch in ax.patches:
            patch.set_edgecolor('none')
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        ax.set_title(f'{metric} by Attack Type and Dataset', fontsize=14, pad=20)
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(df[metric]) * 1.15)  # Add 15% padding for labels
    
    plt.tight_layout()
    
    # Save the plot with high DPI
    plot_path = os.path.join(output_dir, 'attack_effectiveness.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_classifier_robustness(all_results: Dict[str, List[Dict]], output_dir: str):
    """Create a plot showing classifier robustness against different attacks across datasets."""
    set_plot_style()
    data = []
    classifiers = ['knn', 'rf', 'svm', 'lr']
    
    for dataset_name, results in all_results.items():
        # Add clean baseline with 100% robustness
        if results:
            for clf in classifiers:
                data.append({
                    'Dataset': dataset_name,
                    'Classifier': clf.upper(),
                    'Attack': 'clean_baseline',
                    'Simple_Attack': 'Clean',
                    'Display_Label': 'Clean',
                    'Robustness': 100  # Clean baseline has perfect robustness by definition
                })
        
        for result in results:
            attack_type = result['config']['poison_type']
            poison_ratio = result['config']['poison_ratio']
            simple_type = simplify_label(attack_type)
            
            for clf in classifiers:
                clean_acc = result['classifier_results_clean'].get(clf, 0)
                poisoned_acc = result['classifier_results_poisoned'].get(clf, 0)
                robustness = (poisoned_acc / clean_acc * 100) if clean_acc > 0 else 0
                
                data.append({
                    'Dataset': dataset_name,
                    'Classifier': clf.upper(),
                    'Attack': f"{attack_type}_{poison_ratio}",
                    'Simple_Attack': simple_type,
                    'Display_Label': create_attack_label(simple_type, poison_ratio),
                    'Robustness': robustness
                })
    
    df = pd.DataFrame(data)
    
    # Sort to ensure Clean baseline appears first
    df['Simple_Attack'] = pd.Categorical(df['Simple_Attack'], 
                                       categories=['Clean'] + sorted(list(set(df['Simple_Attack']) - {'Clean'})), 
                                       ordered=True)
    df = df.sort_values('Simple_Attack')
    
    # Create subplot for each dataset
    num_datasets = len(all_results)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(15, 6*num_datasets))
    if num_datasets == 1:
        axes = [axes]
    
    for ax, (dataset_name, _) in zip(axes, all_results.items()):
        # Filter data for this dataset
        dataset_df = df[df['Dataset'] == dataset_name]
        
        # Create the subplot with custom style
        bars = plt.bar(dataset_df['Display_Label'], dataset_df['Robustness'], ax=ax, alpha=0.8)
        
        # Remove the top line of bars
        for patch in ax.patches:
            patch.set_edgecolor('none')
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        # Customize the subplot
        ax.set_title(f'{dataset_name} Classifier Robustness', fontsize=14, pad=20)
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Robustness Score (%)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(dataset_df['Robustness']) * 1.15)  # Add 15% padding for labels
    
    plt.tight_layout()
    
    # Save the plot with high DPI
    plot_path = os.path.join(output_dir, 'classifier_robustness.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize poison attack results across datasets')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Base directory containing dataset-specific result directories')
    parser.add_argument('--output-dir', type=str, default='analysis_plots',
                      help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results from all datasets
    all_results = load_all_results(args.results_dir)
    if not all_results:
        print(f"No result files found in any subdirectories of {args.results_dir}")
        return
    
    print(f"Loaded results from {len(all_results)} datasets: {', '.join(all_results.keys())}")
    
    # Create plots
    plot_combined_classifier_comparison(all_results, args.output_dir)
    plot_attack_effectiveness(all_results, args.output_dir)
    plot_classifier_robustness(all_results, args.output_dir)

if __name__ == "__main__":
    main()

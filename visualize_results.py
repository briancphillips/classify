#!/usr/bin/env python3

import os
import json
import glob
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict

def load_results(results_dir: str) -> List[Dict]:
    """Load all JSON result files from the specified directory."""
    results = []
    for filepath in glob.glob(os.path.join(results_dir, "poison_results_*.json")):
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    return results

def plot_combined_classifier_comparison(results: List[Dict], output_dir: str):
    """Create a combined plot showing classifier accuracies across all datasets."""
    # Prepare data for plotting
    data = []
    for result in results:
        attack_type = result['config']['poison_type']
        poison_ratio = result['config']['poison_ratio']
        
        # Add clean dataset results
        for clf_name, acc in result['classifier_results_clean'].items():
            data.append({
                'Classifier': clf_name.upper(),
                'Accuracy': acc,
                'Dataset': f"{attack_type}_{poison_ratio}_clean"
            })
        
        # Add poisoned dataset results
        for clf_name, acc in result['classifier_results_poisoned'].items():
            data.append({
                'Classifier': clf_name.upper(),
                'Accuracy': acc,
                'Dataset': f"{attack_type}_{poison_ratio}_poisoned"
            })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Classifier')
    
    # Customize the plot
    plt.title('Classifier Performance Across All Datasets', fontsize=14, pad=20)
    plt.xlabel('Dataset (Attack_Type_Ratio_Status)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'combined_classifier_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined classifier comparison plot to {plot_path}")

def plot_attack_effectiveness(results: List[Dict], output_dir: str):
    """Create a plot showing the effectiveness of different attacks."""
    data = []
    for result in results:
        attack_type = result['config']['poison_type']
        poison_ratio = result['config']['poison_ratio']
        
        data.append({
            'Attack': f"{attack_type}_{poison_ratio}",
            'Original Accuracy': result['original_accuracy'],
            'Poisoned Accuracy': result['poisoned_accuracy'],
            'Success Rate': result['poison_success_rate']
        })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    x = range(len(df))
    width = 0.25
    
    plt.bar([i - width for i in x], df['Original Accuracy'], width, label='Original Accuracy', color='blue', alpha=0.7)
    plt.bar(x, df['Poisoned Accuracy'], width, label='Poisoned Accuracy', color='red', alpha=0.7)
    plt.bar([i + width for i in x], df['Success Rate'], width, label='Success Rate', color='green', alpha=0.7)
    
    plt.xlabel('Attack Type and Ratio')
    plt.ylabel('Percentage')
    plt.title('Attack Effectiveness Comparison')
    plt.xticks(x, df['Attack'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'attack_effectiveness.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved attack effectiveness plot to {plot_path}")

def plot_classifier_robustness(results: List[Dict], output_dir: str):
    """Create a plot showing classifier robustness against different attacks."""
    data = []
    classifiers = ['knn', 'rf', 'svm', 'lr']
    
    for result in results:
        attack_type = result['config']['poison_type']
        poison_ratio = result['config']['poison_ratio']
        
        for clf in classifiers:
            clean_acc = result['classifier_results_clean'].get(clf, 0)
            poisoned_acc = result['classifier_results_poisoned'].get(clf, 0)
            robustness = (poisoned_acc / clean_acc * 100) if clean_acc > 0 else 0
            
            data.append({
                'Classifier': clf.upper(),
                'Attack': f"{attack_type}_{poison_ratio}",
                'Robustness': robustness
            })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Attack', y='Robustness', hue='Classifier')
    
    plt.title('Classifier Robustness Against Different Attacks')
    plt.xlabel('Attack Type and Ratio')
    plt.ylabel('Robustness Score (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'classifier_robustness.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved classifier robustness plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize poison attack results')
    parser.add_argument('--results-dir', type=str, default='poison_results',
                      help='Directory containing JSON result files')
    parser.add_argument('--output-dir', type=str, default='poison_plots',
                      help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Create plots
    plot_combined_classifier_comparison(results, args.output_dir)
    plot_attack_effectiveness(results, args.output_dir)
    plot_classifier_robustness(results, args.output_dir)

if __name__ == "__main__":
    main()

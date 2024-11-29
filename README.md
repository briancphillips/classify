# Data Poisoning Framework for Image Classification

A modular framework for experimenting with data poisoning attacks on image classification models.

## Features

- Support for multiple datasets:
  - CIFAR-100
  - GTSRB (German Traffic Sign Recognition Benchmark)
  - Imagenette
- Multiple poisoning attack types:
  - PGD (Projected Gradient Descent)
  - GA (Gradient Ascent)
  - Label Flipping (random-to-random, random-to-target, source-to-target)
- Comprehensive evaluation and visualization
- Checkpoint support for interrupted training
- Support for CUDA, MPS (Apple Silicon), and CPU
- Experiment management with YAML configuration
- Automated results consolidation and analysis

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd classify4
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The framework uses YAML configuration files to define experiments. The main entry point is `run_experiments.py`:

```bash
python run_experiments.py --config experiments/config.yaml
```

### Configuration

Experiments are defined in YAML files. Example configuration:

```yaml
# Output configuration
output:
  base_dir: "results"
  save_model: true
  consolidated_file: "all_results.csv"

# Experiment groups
experiment_groups:
  basic_comparison:
    description: "Basic comparison of all attacks"
    experiments:
      - name: cifar100_all_attacks
        dataset: cifar100
        attacks: [pgd, ga, label_flip]
        poison_ratio: 0.1
```

### Example Configurations

1. Basic PGD attack on CIFAR-100:
```yaml
experiments:
  - name: cifar100_pgd
    dataset: cifar100
    attacks: [pgd]
    poison_ratio: 0.1
```

2. Gradient ascent with smaller subset:
```yaml
experiments:
  - name: cifar100_ga_small
    dataset: cifar100
    attacks: [ga]
    subset_size: 100
```

3. Label flipping with target class:
```yaml
experiments:
  - name: cifar100_label_flip
    dataset: cifar100
    attacks: [label_flip]
    target_class: 0
    poison_ratio: 0.1
```

### Configuration Parameters

#### Dataset Parameters
- `dataset`: Dataset to use (`cifar100`, `gtsrb`, `imagenette`)
- `subset_size`: Number of samples per class (optional)

#### Attack Parameters
- `attacks`: List of attacks to run (`pgd`, `ga`, `label_flip`)
- `poison_ratio`: Ratio of dataset to poison (default: 0.1)
- `target_class`: Target class for label flipping attacks
- `source_class`: Source class for targeted label flipping

#### Output Parameters
- `base_dir`: Directory to save results (default: results)
- `save_model`: Whether to save trained models (default: true)
- `consolidated_file`: Name of consolidated results file (default: all_results.csv)

## Running Experiments

To run a full set of experiments:

```bash
python run_experiments.py --config experiments/config.yaml
```

The experiment runner will:
1. Execute all experiments defined in the config file
2. Save individual results as JSON files
3. Consolidate all results into a single CSV file (`all_results.csv`)

### Experiment Results

Results are saved in two formats:

1. **Individual JSON Files**: 
   - One per experiment: `{dataset}_{attack}_results.json`
   - Contains detailed metrics and configuration
   - Useful for debugging and detailed analysis

2. **Consolidated CSV**:
   - All experiments combined: `all_results.csv`
   - Standardized format with columns:
     - **Metadata**: Date, Iteration
     - **Dataset Info**: Dataset, Train_Size, Test_Size, Num_Classes
     - **Model Info**: Classifier, Model_Architecture
     - **Attack Details**: 
       - Modification_Method
       - Num_Poisoned
       - Poisoned_Classes
       - Flip_Type
     - **Training Parameters**:
       - Epochs, Batch_Size
       - Learning_Rate, Weight_Decay
       - Optimizer
     - **Performance Metrics**:
       - Original_Accuracy
       - Poisoned_Accuracy
       - Poison_Success_Rate
       - Clean_Test_Accuracy
       - Precision, Recall, F1-Score
     - **Time & Resource Usage**:
       - Training_Time
       - Inference_Time
       - Total_Time
       - Latency (avg batch time)
     - **Loss Values**:
       - Final_Train_Loss
       - Final_Test_Loss
       - Best_Train_Loss
       - Best_Test_Loss
     - **Per-Class Performance**:
       - Class_0_Accuracy through Class_99_Accuracy
   - Ideal for analysis and plotting

## Project Structure

```
.
├── attacks/              # Attack implementations
│   ├── base.py          # Base attack class
│   ├── pgd.py           # PGD attack
│   ├── gradient_ascent.py # Gradient ascent attack
│   └── label_flip.py    # Label flipping attacks
├── config/              # Configuration classes
│   ├── types.py         # Enums and types
│   └── dataclasses.py   # Config dataclasses
├── experiment/          # Experiment management
│   ├── experiment.py    # Main experiment class
│   ├── evaluation.py    # Evaluation utilities
│   └── visualization.py # Plotting utilities
├── models/              # Model implementations
│   ├── architectures.py # Neural network architectures
│   ├── transforms.py    # Data transforms
│   ├── training.py      # Training utilities
│   └── data.py         # Dataset loading
├── utils/              # Utility functions
│   ├── device.py       # Device management
│   └── logging.py      # Logging setup
├── run_experiments.py  # Main script
└── requirements.txt    # Python dependencies
```

## Results

Results and checkpoints are saved in the following directories:

- `results/`: Experiment results and plots
- `checkpoints/`: Model checkpoints
- `logs/`: Log files

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

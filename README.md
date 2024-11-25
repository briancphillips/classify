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

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd classify2
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python poison.py --dataset <dataset> --attack <attack_type> [options]
```

### Example Commands

1. Basic PGD attack on GTSRB:

```bash
python poison.py --dataset gtsrb --attack pgd --poison-ratio 0.1 --epochs 10
```

2. Gradient ascent attack with smaller subset:

```bash
python poison.py --dataset cifar100 --attack ga --subset-size 100 --epochs 5
```

3. Label flipping attack with target class:

```bash
python poison.py --dataset gtsrb --attack label_flip --target-class 0 --poison-ratio 0.1
```

4. Targeted label flipping from one class to another:

```bash
python poison.py --dataset gtsrb --attack label_flip --source-class 1 --target-class 0
```

### Command Line Arguments

#### Dataset Parameters

- `--dataset`: Dataset to use (`cifar100`, `gtsrb`, `imagenette`)
- `--subset-size`: Number of samples per class (optional)

#### Attack Parameters

- `--attack`: Attack type (`pgd`, `ga`, `label_flip`)
- `--poison-ratio`: Ratio of dataset to poison (default: 0.1)
- `--target-class`: Target class for label flipping attacks
- `--source-class`: Source class for targeted label flipping

#### Training Parameters

- `--epochs`: Number of training epochs (default: 30)
- `--learning-rate`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 128)
- `--num-workers`: Number of data loading workers (default: 2)

#### Device Parameters

- `--device`: Device to use (`cuda`, `mps`, `cpu`)

#### Output Parameters

- `--output-dir`: Directory to save results (default: results)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)
- `--debug`: Enable debug logging

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
├── poison.py           # Main script
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

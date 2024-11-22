# Data Poisoning Framework

This framework implements various data poisoning attacks on image classification models, with support for CIFAR100, GTSRB, and Imagenette datasets.

## Model Architectures

### CIFAR100
- WideResNet-50-2 backbone
- Modified for 32x32 input size
- 100 output classes
- Custom initialization
- Improved training strategy with mixup

### GTSRB (German Traffic Sign Recognition Benchmark)
- Custom CNN architecture with 4 blocks
- 43 output classes
- Specialized for traffic sign recognition
- Custom initialization

### Imagenette
- Pretrained ResNet50 backbone
- Transfer learning (frozen early layers)
- Custom classifier head
- 10 output classes

## Available Attacks

### 1. Projected Gradient Descent (PGD)
```python
config = PoisonConfig(
    poison_type=PoisonType.PGD,
    poison_ratio=0.1,        # Percentage of dataset to poison (0.0 to 1.0)
    pgd_eps=0.3,            # Maximum perturbation size
    pgd_alpha=0.01,         # Step size for each iteration
    pgd_steps=40,           # Number of iterations
    random_seed=42          # Optional random seed
)
```

### 2. Genetic Algorithm (GA)
```python
config = PoisonConfig(
    poison_type=PoisonType.GA,
    poison_ratio=0.1,           # Percentage of dataset to poison
    ga_pop_size=50,            # Population size
    ga_generations=100,        # Number of generations
    ga_mutation_rate=0.1,      # Mutation probability
    random_seed=42             # Optional random seed
)
```

### 3. Label Flipping Attacks
```python
# Random to Random
config = PoisonConfig(
    poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_RANDOM,
    poison_ratio=0.1,
    random_seed=42
)

# Random to Target
config = PoisonConfig(
    poison_type=PoisonType.LABEL_FLIP_RANDOM_TO_TARGET,
    poison_ratio=0.1,
    target_class=1,           # Target class to flip to
    random_seed=42
)

# Source to Target
config = PoisonConfig(
    poison_type=PoisonType.LABEL_FLIP_SOURCE_TO_TARGET,
    poison_ratio=0.1,
    source_class=0,          # Source class to flip from
    target_class=1,          # Target class to flip to
    random_seed=42
)
```

## Running Experiments

### Basic Usage
```bash
# Run default experiment (CIFAR100 with PGD attack)
python poison.py
```

### Custom Configuration
Create a Python script with your configuration:

```python
from poison import PoisonConfig, PoisonType, PoisonExperiment
from models import get_model
import torchvision

# 1. Create model and load dataset
model = get_model('cifar100')  # or 'gtsrb' or 'imagenette'
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True,
    transform=CIFAR100_TRANSFORM_TRAIN
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True,
    transform=CIFAR100_TRANSFORM_TEST
)

# 2. Configure attacks
configs = [
    PoisonConfig(
        poison_type=PoisonType.PGD,
        poison_ratio=0.1,
        pgd_eps=0.3
    ),
    PoisonConfig(
        poison_type=PoisonType.GA,
        poison_ratio=0.1,
        ga_pop_size=50
    )
]

# 3. Run experiment
experiment = PoisonExperiment(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    configs=configs,
    output_dir="results/cifar100",
    checkpoint_dir="checkpoints/cifar100"
)

# 4. Train clean model and save checkpoint
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
experiment.train_model(
    train_loader,
    epochs=30,
    learning_rate=0.001,
    checkpoint_name="clean_model"
)

# 5. Run poisoning experiments
experiment.run_experiments()
```

## Output Structure

```
classify/
├── checkpoints/
│   └── cifar100/
│       └── clean_model.pt       # Saved model checkpoints
├── results/
│   └── cifar100/
│       └── poison_results_*.json # Experiment results
```

## Results Format
Results are saved as JSON files containing:
- Original model accuracy
- Poisoned model accuracy
- Attack success rate
- Configuration parameters
- Timestamp

## Dependencies
Required packages are listed in `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tqdm>=4.62.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

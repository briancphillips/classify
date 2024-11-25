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

## Model Functions

### Loading Models

```python
from models import get_model, save_model, load_model

# Get a new model instance
model = get_model('cifar100')  # or 'gtsrb' or 'imagenette'

# Save a trained model with metadata
save_model(
    model,
    'checkpoints/cifar100/my_model.pt',
    metadata={'custom_info': 'value'},
    optimizer=optimizer,      # Optional: save optimizer state
    epoch=current_epoch,     # Optional: save epoch number
    loss=current_loss        # Optional: save loss value
)

# Load a saved model (optionally specify device)
model, metadata = load_model('checkpoints/cifar100/my_model.pt', device='cuda')
```

### Checkpoint Management

The framework provides robust checkpoint handling with three types of checkpoints:

- Latest checkpoint: `{name}_latest.pt`
- Best checkpoint: `{name}_best.pt` (lowest validation loss)
- Emergency checkpoint: `{name}_emergency.pt` (saved during OOM or errors)

Each checkpoint contains:

- Model state dict
- Metadata (including dataset name)
- Optional optimizer state
- Optional epoch number and loss value

### Data Transforms

```python
from models import (
    CIFAR100_TRANSFORM_TRAIN, CIFAR100_TRANSFORM_TEST,
    GTSRB_TRANSFORM_TRAIN, GTSRB_TRANSFORM_TEST,
    IMAGENETTE_TRANSFORM_TRAIN, IMAGENETTE_TRANSFORM_TEST
)

# Use pre-defined transforms
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True,
    transform=CIFAR100_TRANSFORM_TRAIN
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False,
    transform=CIFAR100_TRANSFORM_TEST
)
```

### Model Architecture Details

#### CIFAR100 Model

- Input: 32x32x3 RGB images
- Backbone: WideResNet-50-2 with modified first layer
- Feature dimension: 2048
- Output: 100 classes
- Training: SGD with momentum, learning rate scheduling

#### GTSRB Model

- Input: 32x32x3 RGB images
- Architecture:
  - 4 convolutional blocks with batch normalization
  - Global average pooling
  - 2 fully connected layers
- Output: 43 classes
- Training: Adam optimizer, reduced learning rate on plateau

#### Imagenette Model

- Input: 224x224x3 RGB images
- Backbone: Pretrained ResNet50
- Modifications:
  - Frozen backbone layers
  - Custom classifier head
  - Adaptive pooling for flexible input sizes
- Output: 10 classes
- Training: SGD with momentum, cosine annealing scheduler

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

### 2. Gradient Ascent (GA)

```python
config = PoisonConfig(
    poison_type=PoisonType.GA,
    poison_ratio=0.1,           # Percentage of dataset to poison
    ga_pop_size=50,            # Number of gradient steps
    ga_generations=100,        # Number of iterations
    ga_mutation_rate=0.1,      # Learning rate
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

### Command Line Usage

```bash
# Basic usage (defaults to CIFAR100 with PGD attack)
python poison.py

# Run PGD attack on CIFAR100 with custom parameters
python poison.py --dataset cifar100 --attack pgd \
    --poison-ratio 0.2 --pgd-eps 0.4 --pgd-steps 50

# Run Genetic Algorithm attack
python poison.py --dataset cifar100 --attack ga \
    --poison-ratio 0.1 --ga-pop-size 100 --ga-generations 200

# Run Label Flipping (random to random)
python poison.py --dataset cifar100 --attack label_flip_random_random \
    --poison-ratio 0.1

# Run Label Flipping (random to target)
python poison.py --dataset cifar100 --attack label_flip_random_target \
    --poison-ratio 0.1 --target-class 1

# Run Label Flipping (source to target)
python poison.py --dataset cifar100 --attack label_flip_source_target \
    --poison-ratio 0.1 --source-class 0 --target-class 1

# Custom training parameters
python poison.py --epochs 50 --learning-rate 0.0005 --batch-size 64

# Custom output directories
python poison.py --output-dir "my_results" --checkpoint-dir "my_checkpoints"
```

### Available Arguments

```
Dataset selection:
  --dataset {cifar100,gtsrb,imagenette}
                        Dataset to use (default: cifar100)

Attack configuration:
  --attack {pgd,ga,label_flip_random_random,label_flip_random_target,label_flip_source_target}
                        Attack type (default: pgd)
  --poison-ratio FLOAT  Poison ratio (default: 0.1)
  --random-seed INT     Random seed (default: 42)

PGD attack parameters:
  --pgd-eps FLOAT      Maximum perturbation size (default: 0.3)
  --pgd-alpha FLOAT    Step size for each iteration (default: 0.01)
  --pgd-steps INT      Number of iterations (default: 40)

Genetic Algorithm parameters:
  --ga-pop-size INT    Population size (default: 50)
  --ga-generations INT Number of generations (default: 100)
  --ga-mutation-rate FLOAT
                      Mutation rate (default: 0.1)

Label Flipping parameters:
  --source-class INT   Source class for label flipping (default: None)
  --target-class INT   Target class for label flipping (default: None)

Training parameters:
  --epochs INT         Number of epochs (default: 30)
  --learning-rate FLOAT
                      Learning rate (default: 0.001)
  --batch-size INT     Batch size (default: 128)

Output parameters:
  --output-dir DIR     Output directory (default: results/[dataset])
  --checkpoint-dir DIR Checkpoint directory (default: checkpoints/[dataset])
  --checkpoint-name STR
                      Name for model checkpoint (default: clean_model)
```

## Clean Training with models.py

For training clean models without any poisoning, you can use `models.py` directly:

```bash
python models.py [args]
```

### Command Line Arguments

```
Dataset selection:
  --dataset {cifar100,gtsrb,imagenette}
                        Dataset to use (default: cifar100)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 128)
  --epochs EPOCHS       Number of training epochs (default: varies by dataset)
                       - CIFAR100: 200 epochs
                       - GTSRB: 30 epochs
                       - Imagenette: 30 epochs
  --learning-rate LR   Initial learning rate (default: varies by dataset)
                       - CIFAR100: 0.1
                       - GTSRB: 0.001
                       - Imagenette: 0.01
  --num-workers NUM_WORKERS
                        Number of data loading workers (default: 4)
  --device DEVICE      Device to use (cuda, mps, or cpu)

Output options:
  --checkpoint-dir DIR Path to save model checkpoints (default: checkpoints/[dataset])
  --results-dir DIR   Path to save training results (default: results/[dataset])
```

### Examples

1. Train CIFAR100 model:

```bash
python models.py --dataset cifar100 --epochs 200
```

2. Train GTSRB model with custom batch size:

```bash
python models.py --dataset gtsrb --batch-size 64
```

3. Train Imagenette model on CPU:

```bash
python models.py --dataset imagenette --device cpu
```

Each model will be saved in the checkpoints directory with training metrics and can be loaded later using the functions described in the Model Functions section.

## Custom Configuration

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

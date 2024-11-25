import torchvision.transforms as transforms

# CIFAR100 transforms
CIFAR100_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

CIFAR100_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

# GTSRB transforms
GTSRB_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]
)

GTSRB_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]
)

# Imagenette transforms
IMAGENETTE_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

IMAGENETTE_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_transforms(dataset_name: str):
    """Get the appropriate transforms for the dataset.

    Args:
        dataset_name: Name of the dataset ('cifar100', 'gtsrb', or 'imagenette')

    Returns:
        tuple: (train_transform, test_transform)

    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name.lower() == "cifar100":
        return CIFAR100_TRANSFORM_TRAIN, CIFAR100_TRANSFORM_TEST
    elif dataset_name.lower() == "gtsrb":
        return GTSRB_TRANSFORM_TRAIN, GTSRB_TRANSFORM_TEST
    elif dataset_name.lower() == "imagenette":
        return IMAGENETTE_TRANSFORM_TRAIN, IMAGENETTE_TRANSFORM_TEST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

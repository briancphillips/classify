import torch
import torch.nn as nn
import torchvision
from torchvision import models
from utils.logging import get_logger

logger = get_logger(__name__)


class CIFAR100Classifier(nn.Module):
    """CIFAR100 classifier using WideResNet-28-10."""

    def __init__(self, num_classes=100):
        super(CIFAR100Classifier, self).__init__()
        # Create a WideResNet with depth=28 and width=10
        self.backbone = torchvision.models.wide_resnet50_2(weights=None)

        # Modify first conv layer and maxpool for CIFAR-100
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        # Adjust the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        return x


class GTSRBClassifier(nn.Module):
    """GTSRB classifier with custom architecture."""

    def __init__(self):
        super(GTSRBClassifier, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 43)  # GTSRB has 43 classes
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        if return_features:
            return features
        x = self.classifier(features.view(features.size(0), -1))
        return x


class ImagenetteClassifier(nn.Module):
    """Imagenette classifier using pretrained ResNet50."""

    def __init__(self):
        super(ImagenetteClassifier, self).__init__()
        # Use pretrained ResNet50 with ImageNet weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)

        # Add new classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10),  # Imagenette has 10 classes
        )

        # Freeze early layers
        for param in list(self.features.parameters())[:-20]:
            param.requires_grad = False

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        if return_features:
            return features
        x = self.fc(features)
        return x


def get_model(dataset_name: str) -> nn.Module:
    """Get the appropriate model for the dataset.

    Args:
        dataset_name: Name of the dataset ('cifar100', 'gtsrb', or 'imagenette')

    Returns:
        nn.Module: Initialized model

    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name.lower() == "cifar100":
        logger.info("Creating CIFAR100 classifier")
        return CIFAR100Classifier()
    elif dataset_name.lower() == "gtsrb":
        logger.info("Creating GTSRB classifier")
        return GTSRBClassifier()
    elif dataset_name.lower() == "imagenette":
        logger.info("Creating Imagenette classifier")
        return ImagenetteClassifier()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

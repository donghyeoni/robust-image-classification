"""ResNet-18 model factory.

Every experiment in this project uses the same architecture: a torchvision
ResNet-18 whose first convolution (``conv1``) is re-stemmed to accept a
configurable number of input channels (1, 2, 3 or 4) and whose classifier
head (``fc``) is replaced with a linear layer producing ``num_classes`` logits.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def build_resnet18(in_channels: int = 3, num_classes: int = 4,
                   pretrained: bool = False) -> nn.Module:
    """Build a ResNet-18 re-stemmed for ``in_channels`` inputs.

    Parameters
    ----------
    in_channels:
        Number of channels of the input tensor. The original ResNet-18 stem
        expects 3; this factory rebuilds ``conv1`` for 1/2/3/4 channels while
        keeping the ``kernel_size=7, stride=2, padding=3, bias=False`` stem
        used across the notebooks.
    num_classes:
        Number of output classes (4 animal classes by default).
    pretrained:
        If True, load ImageNet weights before re-stemming. The original
        notebooks trained from scratch (``models.resnet18()``), so this
        defaults to False to preserve that behaviour.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
    # Kept as a Sequential to match the checkpoints produced by the notebooks.
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model

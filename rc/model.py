from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def build_resnet18(in_channels: int = 3, num_classes: int = 4,
                   pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model

from __future__ import annotations
import torch.nn as nn
from torchvision.models import resnet18

def ResNet18(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    m = resnet18(num_classes=num_classes)
    if in_ch != 3:
        # replace first conv
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return m

from __future__ import annotations
import torch.nn as nn
from torchvision.models import vgg16

def VGG16(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    m = vgg16(num_classes=num_classes)
    if in_ch != 3:
        # replace first conv layer
        first = m.features[0]
        m.features[0] = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
                                  stride=first.stride, padding=first.padding)
    return m

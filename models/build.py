from __future__ import annotations
from typing import Tuple
import torch.nn as nn

from .lenet import LeNet
from .resnet import ResNet18
from .resnet18_cifar import ResNet18CIFAR
from .resnet20 import ResNet20
from .vgg import VGG16

def build_model(name: str, num_classes: int, in_ch: int) -> nn.Module:
    n = name.lower()
    if n == "lenet":
        return LeNet(in_ch=in_ch, num_classes=num_classes)
    if n in ("resnet18", "resnet"):
        return ResNet18(num_classes=num_classes, in_ch=in_ch)
    if n in ("resnet18_cifar", "resnet_cifar18", "cifar_resnet18"):
        return ResNet18CIFAR(num_classes=num_classes, in_ch=in_ch)
    if n in ("resnet20",):
        return ResNet20(num_classes=num_classes, in_ch=in_ch)
    if n in ("vgg16", "vgg"):
        return VGG16(num_classes=num_classes, in_ch=in_ch)
    raise ValueError(f"Unknown model: {name}")

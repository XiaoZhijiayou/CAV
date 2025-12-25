from __future__ import annotations

# CIFAR-style ResNet18: 3x3 conv1, no maxpool, shortcut naming, linear head.
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_ch=3, base_planes=64):
        super().__init__()
        self.in_planes = base_planes
        self.conv1 = nn.Conv2d(in_ch, base_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_planes)
        self.layer1 = self._make_layer(block, base_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(base_planes * 8, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)


def ResNet18CIFAR(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return ResNetCIFAR18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_ch=in_ch)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder

@dataclass
class DataConfig:
    dataset: str
    data_root: str
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 32

def _cifar_transform(train: bool, image_size: int):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])

def _mnist_transform(train: bool):
    return transforms.Compose([transforms.ToTensor()])

def _try_build_gtsrb(root: Path, train: bool, image_size: int):
    # Try torchvision.datasets.GTSRB if available in this torchvision version
    try:
        ds = datasets.GTSRB(
            root=str(root),
            split="train" if train else "test",
            download=False,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]),
        )
        return ds
    except Exception:
        pass

    # Try common local structures
    # 1) ImageFolder-style: gtsrb/train/<cls>/*.png
    split_dir = root / ("train" if train else "test")
    if split_dir.exists() and any(p.is_dir() for p in split_dir.iterdir()):
        return ImageFolder(
            root=str(split_dir),
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]),
        )

    raise FileNotFoundError(
        f"GTSRB dataset not found under {root}. "
        f"Expected either torchvision structure, or ImageFolder style: {root}/train/<cls>/*.png"
    )

def build_dataloaders(cfg: DataConfig):
    root = Path(cfg.data_root)
    name = cfg.dataset.lower()

    if name in ("cifar10", "cifar-10"):
        train_set = datasets.CIFAR10(
            root=str(root), train=True, download=False, transform=_cifar_transform(True, cfg.image_size)
        )
        test_set = datasets.CIFAR10(
            root=str(root), train=False, download=False, transform=_cifar_transform(False, cfg.image_size)
        )
        num_classes = 10
        in_ch = 3

    elif name in ("cifar100", "cifar-100"):
        train_set = datasets.CIFAR100(
            root=str(root), train=True, download=False, transform=_cifar_transform(True, cfg.image_size)
        )
        test_set = datasets.CIFAR100(
            root=str(root), train=False, download=False, transform=_cifar_transform(False, cfg.image_size)
        )
        num_classes = 100
        in_ch = 3

    elif name in ("mnist",):
        train_set = datasets.MNIST(
            root=str(root), train=True, download=False, transform=_mnist_transform(True)
        )
        test_set = datasets.MNIST(
            root=str(root), train=False, download=False, transform=_mnist_transform(False)
        )
        num_classes = 10
        in_ch = 1

    elif name in ("gtsrb",):
        train_set = _try_build_gtsrb(root / "gtsrb", train=True, image_size=cfg.image_size)
        test_set = _try_build_gtsrb(root / "gtsrb", train=False, image_size=cfg.image_size)
        num_classes = 43
        in_ch = 3

    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, test_loader, num_classes, in_ch

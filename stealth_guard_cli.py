from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.ckpt import load_checkpoint
from src.auth.stealth_guard import StealthGuard
from models.build import build_model


def parse_args():
    p = argparse.ArgumentParser("StealthGuard embed/verify")
    default_workers = 0 if os.name == "nt" else 2
    p.add_argument("--mode", type=str, required=True, choices=["embed", "verify"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=default_workers)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--download", action="store_true")
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--qim_step", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()

_DATASET_SPECS = {
    "cifar10": {
        "dataset": datasets.CIFAR10,
        "num_classes": 10,
        "channels": 3,
        "input_size": 32,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "cifar100": {
        "dataset": datasets.CIFAR100,
        "num_classes": 100,
        "channels": 3,
        "input_size": 32,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
    },
    "gtsrb": {
        "dataset": datasets.GTSRB,
        "num_classes": 43,
        "channels": 3,
        "input_size": 32,
        "mean": [0.3403, 0.3121, 0.3214],
        "std": [0.2724, 0.2608, 0.2669],
    },
    "mnist": {
        "dataset": datasets.MNIST,
        "num_classes": 10,
        "channels": 1,
        "input_size": 28,
        "mean": [0.1307],
        "std": [0.3081],
    },
}

_DATASET_ALIASES = {
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "cifar-100": "cifar100",
    "cifar_100": "cifar100",
}

def _canonical_dataset(name: str) -> str:
    key = name.lower()
    return _DATASET_ALIASES.get(key, key)

def _dataset_spec(name: str) -> dict:
    key = _canonical_dataset(name)
    if key not in _DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {name}")
    return _DATASET_SPECS[key]

def _target_size_for_model(model_name: str, dataset_name: str) -> int:
    spec = _dataset_spec(dataset_name)
    if model_name.lower() == "lenet":
        return 28
    return spec["input_size"]

def _build_transforms(dataset_name: str, train: bool, target_size: Optional[int]) -> transforms.Compose:
    spec = _dataset_spec(dataset_name)
    mean = spec["mean"]
    std = spec["std"]
    name = _canonical_dataset(dataset_name)
    ops: List = []

    if target_size is not None and target_size != spec["input_size"]:
        ops.append(transforms.Resize((target_size, target_size)))
    elif name.startswith("cifar") and train:
        ops.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    elif name == "gtsrb":
        ops.append(transforms.Resize((spec["input_size"], spec["input_size"])))
        if train:
            ops.append(transforms.RandomRotation(15))
    elif name == "mnist" and target_size is not None and target_size != spec["input_size"]:
        ops.append(transforms.Resize((target_size, target_size)))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(ops)

def _build_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    train: bool,
    download: bool,
    target_size: Optional[int],
) -> Tuple[DataLoader, int, int]:
    spec = _dataset_spec(dataset_name)
    name = _canonical_dataset(dataset_name)
    transform = _build_transforms(name, train=train, target_size=target_size)
    ds_kwargs = {"root": data_root, "download": download, "transform": transform}
    if name == "gtsrb":
        split = "train" if train else "test"
        dataset = spec["dataset"](split=split, **ds_kwargs)
    else:
        dataset = spec["dataset"](train=train, **ds_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader, spec["num_classes"], spec["channels"]

def _extract_state_dict(ck: dict) -> dict:
    if "model_state" in ck:
        return ck["model_state"]
    if "state_dict" in ck:
        return ck["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck
    raise KeyError("Checkpoint does not contain model_state/state_dict.")


def _default_layers(model: nn.Module) -> list[str]:
    layers: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and name:
            layers.append(name)
    return layers


def main():
    args = parse_args()
    ck = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg", {})
    model_name = args.model or cfg.get("model", "")
    dataset_name = args.dataset or cfg.get("dataset", "")
    if not model_name:
        raise ValueError("Missing model name. Provide --model or include cfg.model in checkpoint.")
    if not dataset_name:
        raise ValueError("Missing dataset name. Provide --dataset or include cfg.dataset in checkpoint.")

    train_split = args.split == "train"
    target_size = _target_size_for_model(model_name, dataset_name)
    loader, num_classes, in_ch = _build_dataloader(
        dataset_name,
        batch_size=args.batch_size,
        data_root=args.data_root,
        train=train_split,
        num_workers=args.num_workers,
        download=args.download,
        target_size=target_size,
    )
    model = build_model(
        model_name,
        num_classes=num_classes,
        in_ch=in_ch,
    ).to(args.device)
    model.load_state_dict(_extract_state_dict(ck), strict=True)
    model.eval()
    if not cfg:
        cfg = {
            "model": model_name,
            "dataset": dataset_name,
            "num_classes": num_classes,
            "in_ch": in_ch,
        }
    else:
        cfg.setdefault("model", model_name)
        cfg.setdefault("dataset", dataset_name)
        cfg.setdefault("num_classes", num_classes)
        cfg.setdefault("in_ch", in_ch)

    if args.layers:
        target_layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    else:
        target_layers = _default_layers(model)
    if not target_layers:
        raise RuntimeError("No target layers found.")

    guard = StealthGuard(model, target_layers=target_layers, qim_step=args.qim_step)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "embed":
        if not args.out:
            raise ValueError("Embed mode requires --out path.")
        info = guard.embed(loader, criterion, return_indices=True)
        indices = info.pop("indices", {})
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "cfg": cfg,
            "stealth_guard": {
                "indices": indices,
                "qim_step": guard.qim_step,
                "vip_ratio": guard.vip_ratio,
                "carrier_ratio": guard.carrier_ratio,
                "ratio": guard.ratio,
            },
        }, str(out_path))
        print(json.dumps(info, indent=2))
        print(f"[OK] saved StealthGuard model to: {out_path}")
    else:
        sg_meta = ck.get("stealth_guard", {})
        indices = sg_meta.get("indices")
        res = guard.verify_and_locate(loader, criterion, indices_by_layer=indices)
        res["used_saved_indices"] = bool(indices)
        print("PASS" if res["ok"] else "FAIL")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

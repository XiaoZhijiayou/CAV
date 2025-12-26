from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.build import build_model
from src.utils.ckpt import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser("Compare base vs watermarked checkpoints")
    p.add_argument("--base_ckpt", type=str, required=True)
    p.add_argument("--wm_ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=128)
    default_workers = 0 if os.name == "nt" else 2
    p.add_argument("--num_workers", type=int, default=default_workers)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--download", action="store_true")
    p.add_argument("--log", type=str, default="")
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


def _build_eval_transform(dataset_name: str, target_size: Optional[int]) -> transforms.Compose:
    spec = _dataset_spec(dataset_name)
    ops: List = []
    if target_size is not None and target_size != spec["input_size"]:
        ops.append(transforms.Resize((target_size, target_size)))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    return transforms.Compose(ops)


def _build_dataset(
    dataset_name: str,
    data_root: str,
    train: bool,
    download: bool,
    transform: transforms.Compose,
):
    name = _canonical_dataset(dataset_name)
    spec = _dataset_spec(name)
    if name == "gtsrb":
        split = "train" if train else "test"
        return spec["dataset"](root=data_root, split=split, download=download, transform=transform)
    return spec["dataset"](root=data_root, train=train, download=download, transform=transform)


def _build_loaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    download: bool,
    target_size: Optional[int],
) -> tuple[Optional[DataLoader], DataLoader, int, int]:
    spec = _dataset_spec(dataset_name)
    transform = _build_eval_transform(dataset_name, target_size)
    train_ds = _build_dataset(dataset_name, data_root, train=True, download=download, transform=transform)
    test_ds = _build_dataset(dataset_name, data_root, train=False, download=download, transform=transform)

    if val_split > 0.0:
        n_total = len(train_ds)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(seed)
        _train_subset, val_subset = random_split(train_ds, [n_train, n_val], generator=gen)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_loader = None

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_loader, test_loader, spec["num_classes"], spec["channels"]


def _extract_state_dict(ck: dict) -> dict:
    if "model_state" in ck:
        return ck["model_state"]
    if "state_dict" in ck:
        return ck["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck
    raise KeyError("Checkpoint does not contain model_state/state_dict.")


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str, max_batches: int) -> dict:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return {
        "acc": correct / max(total, 1),
        "loss": total_loss / max(total, 1),
        "samples": total,
    }


def _compare_state_dict_bits(a: dict, b: dict) -> dict:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    common = sorted(keys_a & keys_b)
    missing = sorted(keys_a - keys_b)
    extra = sorted(keys_b - keys_a)
    skipped = []

    total_bits = 0
    diff_bits = 0
    total_elems = 0
    diff_elems = 0

    for k in common:
        ta = a[k]
        tb = b[k]
        if not (torch.is_floating_point(ta) and torch.is_floating_point(tb)):
            continue
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            skipped.append(k)
            continue
        ta = ta.detach().cpu().contiguous()
        tb = tb.detach().cpu().contiguous()
        elem_size = ta.element_size()
        a_bytes = ta.numpy().view(np.uint8).reshape(-1, elem_size)
        b_bytes = tb.numpy().view(np.uint8).reshape(-1, elem_size)
        xor = np.bitwise_xor(a_bytes, b_bytes)
        diff_bits += int(np.unpackbits(xor).sum())
        total_bits += xor.size * 8
        diff_elems += int(np.any(a_bytes != b_bytes, axis=1).sum())
        total_elems += a_bytes.shape[0]

    return {
        "total_bits": int(total_bits),
        "diff_bits": int(diff_bits),
        "diff_bits_ratio": float(diff_bits) / max(float(total_bits), 1.0),
        "total_elements": int(total_elems),
        "diff_elements": int(diff_elems),
        "diff_elements_ratio": float(diff_elems) / max(float(total_elems), 1.0),
        "missing_keys": missing,
        "extra_keys": extra,
        "skipped_keys": skipped,
    }


def _default_log_path(base_path: Path, wm_path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base_path.parent / f"compare_{base_path.stem}_vs_{wm_path.stem}_{ts}.json"


def main():
    args = parse_args()
    base_ck = load_checkpoint(args.base_ckpt, map_location="cpu")
    wm_ck = load_checkpoint(args.wm_ckpt, map_location="cpu")

    cfg = base_ck.get("cfg", {})
    model_name = args.model or cfg.get("model", "")
    if not model_name:
        raise ValueError("Missing model name. Provide --model or include cfg.model in checkpoint.")

    val_loader, test_loader, num_classes, in_ch = _build_loaders(
        args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        download=args.download,
        target_size=_target_size_for_model(model_name, args.dataset),
    )

    model_base = build_model(model_name, num_classes=num_classes, in_ch=in_ch).to(args.device)
    model_wm = build_model(model_name, num_classes=num_classes, in_ch=in_ch).to(args.device)
    model_base.load_state_dict(_extract_state_dict(base_ck), strict=True)
    model_wm.load_state_dict(_extract_state_dict(wm_ck), strict=True)

    metrics = {"base": {}, "watermarked": {}}
    if val_loader is not None:
        metrics["base"]["val"] = _evaluate(model_base, val_loader, args.device, args.max_batches)
        metrics["watermarked"]["val"] = _evaluate(model_wm, val_loader, args.device, args.max_batches)
    metrics["base"]["test"] = _evaluate(model_base, test_loader, args.device, args.max_batches)
    metrics["watermarked"]["test"] = _evaluate(model_wm, test_loader, args.device, args.max_batches)

    diff = _compare_state_dict_bits(_extract_state_dict(base_ck), _extract_state_dict(wm_ck))

    log_path = Path(args.log) if args.log else _default_log_path(Path(args.base_ckpt), Path(args.wm_ckpt))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "base_ckpt": args.base_ckpt,
        "wm_ckpt": args.wm_ckpt,
        "model": model_name,
        "dataset": args.dataset,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_split": args.val_split,
        "seed": args.seed,
        "max_batches": args.max_batches,
        "metrics": metrics,
        "weight_diff": diff,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[OK] saved log to: {log_path}")


if __name__ == "__main__":
    main()

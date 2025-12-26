from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import prune as prune_api

from models.build import build_model
from src.utils.ckpt import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser("Random pruning attack on a checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--num_classes", type=int, default=-1)
    p.add_argument("--in_ch", type=int, default=-1)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--include_layers", type=str, default="")
    p.add_argument("--log", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


_DATASET_SPECS = {
    "cifar10": {"num_classes": 10, "channels": 3},
    "cifar100": {"num_classes": 100, "channels": 3},
    "gtsrb": {"num_classes": 43, "channels": 3},
    "mnist": {"num_classes": 10, "channels": 1},
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


def _dataset_spec(name: str) -> Optional[dict]:
    if not name:
        return None
    key = _canonical_dataset(name)
    return _DATASET_SPECS.get(key)


def _extract_state_dict(ck: dict) -> dict:
    if "model_state" in ck:
        return ck["model_state"]
    if "state_dict" in ck:
        return ck["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck
    raise KeyError("Checkpoint does not contain model_state/state_dict.")


def _resolve_model_io(args, cfg: dict) -> tuple[str, int, int]:
    model_name = args.model or cfg.get("model", "")
    dataset_name = args.dataset or cfg.get("dataset", "")
    num_classes = args.num_classes if args.num_classes > 0 else cfg.get("num_classes", -1)
    in_ch = args.in_ch if args.in_ch > 0 else cfg.get("in_ch", -1)
    spec = _dataset_spec(dataset_name)
    if num_classes < 1 and spec:
        num_classes = spec["num_classes"]
    if in_ch < 1 and spec:
        in_ch = spec["channels"]
    if not model_name:
        raise ValueError("Missing model name. Provide --model or include cfg.model in checkpoint.")
    if num_classes < 1 or in_ch < 1:
        raise ValueError("Missing num_classes/in_ch. Provide --dataset or explicit --num_classes/--in_ch.")
    return model_name, num_classes, in_ch


def _parse_include_layers(value: str) -> Optional[set[str]]:
    if not value:
        return None
    if value.strip().lower() == "all":
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return set(items) if items else None


def _random_prune(model: nn.Module, ratio: float, seed: int, include_layers: Optional[set[str]]) -> dict:
    torch.manual_seed(seed)
    total = 0
    pruned = 0
    layers = []
    matched_layers = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if include_layers is not None and name not in include_layers:
            continue
        w = getattr(module, "weight", None)
        if w is None or not torch.is_floating_point(w):
            continue
        prune_api.random_unstructured(module, name="weight", amount=ratio)
        mask = module.weight_mask.detach()
        count = mask.numel()
        pruned_count = int((mask == 0).sum().item())
        prune_api.remove(module, "weight")
        matched_layers.append(name)
        total += count
        pruned += pruned_count
        layers.append({
            "layer": name,
            "total_params": count,
            "pruned_params": pruned_count,
            "pruned_ratio": pruned_count / max(count, 1),
        })
    missing_layers = sorted((include_layers or set()) - set(matched_layers))
    return {
        "total_params": total,
        "pruned_params": pruned,
        "pruned_ratio": pruned / max(total, 1),
        "layers": layers,
        "matched_layers": matched_layers,
        "missing_layers": missing_layers,
    }


def _default_log_path(out_path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return out_path.parent / f"prune_attack_{ts}.json"


def main():
    args = parse_args()
    if not (0.0 <= args.ratio <= 1.0):
        raise ValueError("--ratio must be in [0, 1].")

    ck = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg", {})
    model_name, num_classes, in_ch = _resolve_model_io(args, cfg)
    model = build_model(model_name, num_classes=num_classes, in_ch=in_ch).to(args.device)
    model.load_state_dict(_extract_state_dict(ck), strict=True)
    model.eval()

    include_layers = _parse_include_layers(args.include_layers)
    stats = _random_prune(model, ratio=args.ratio, seed=args.seed, include_layers=include_layers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_pack = {"model_state": model.state_dict(), "cfg": cfg}
    if "cav_loc" in ck:
        out_pack["cav_loc"] = ck["cav_loc"]
    torch.save(out_pack, str(out_path))

    log_path = Path(args.log) if args.log else _default_log_path(out_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "input_ckpt": args.ckpt,
        "output_ckpt": str(out_path),
        "model": model_name,
        "dataset": args.dataset or cfg.get("dataset", ""),
        "ratio": args.ratio,
        "seed": args.seed,
        "include_layers": args.include_layers or "all",
        "device": args.device,
        "stats": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[OK] saved attacked model to: {out_path}")
    print(f"[OK] saved log to: {log_path}")


if __name__ == "__main__":
    main()

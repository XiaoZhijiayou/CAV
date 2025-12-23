from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Checkpoint:
    model_state: dict
    optim_state: dict | None
    epoch: int
    best_metric: float
    cfg: dict

def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": ckpt.model_state,
        "optim_state": ckpt.optim_state,
        "epoch": ckpt.epoch,
        "best_metric": ckpt.best_metric,
        "cfg": ckpt.cfg,
    }, str(path))

def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(str(path), map_location=map_location)

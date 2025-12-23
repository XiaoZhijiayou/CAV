from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    device: str = "cpu"

def train_one_epoch(model, loader: DataLoader, optim, device: str) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    acc = correct / max(total, 1)
    return {"acc": acc, "loss": total_loss / max(total, 1)}

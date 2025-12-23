from __future__ import annotations
import torch

@torch.no_grad()
def random_noise_tamper(model, rate: float = 0.01, sigma: float = 1e-2) -> None:
    params = [p for p in model.parameters() if p.requires_grad and p.dtype == torch.float32]
    flat = torch.cat([p.view(-1) for p in params])
    n = flat.numel()
    k = max(1, int(rate * n))
    idx = torch.randperm(n, device=flat.device)[:k]
    flat[idx] += sigma * torch.randn_like(flat[idx])

@torch.no_grad()
def zero_out_last_layer(model) -> None:
    # crude tamper: zero out last linear layer weights
    for name, p in model.named_parameters():
        if name.endswith("weight") and ("fc" in name or "classifier.6" in name or "linear" in name):
            p.zero_()

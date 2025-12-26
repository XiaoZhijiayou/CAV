from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class StealthGuard:
    def __init__(self, model: nn.Module, target_layers: List[str], qim_step: float = 1e-4):
        self.model = model
        self.target_layers = list(target_layers)
        self.qim_step = float(qim_step)
        self.ratio = 0.1
        self.vip_ratio = 0.1
        self.carrier_ratio = 0.5
        self.error_threshold = 0.05
        self.top_k = 5
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def _get_param(self, layer_name: str) -> nn.Parameter:
        named_params = dict(self.model.named_parameters())
        if layer_name in named_params:
            return named_params[layer_name]
        named_modules = dict(self.model.named_modules())
        if layer_name in named_modules:
            mod = named_modules[layer_name]
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                return mod.weight
        raise KeyError(f"Target layer not found: {layer_name}")

    def _get_batch(self, data_loader):
        try:
            batch = next(iter(data_loader))
        except StopIteration as e:
            raise ValueError("Empty data_loader.") from e
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("data_loader must yield (inputs, targets).")

    def _calculate_importance(self, param: nn.Parameter, data_loader, criterion) -> torch.Tensor:
        self.model = self.model.to(self.device).eval()
        self.model.zero_grad(set_to_none=True)
        x, y = self._get_batch(data_loader)
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.enable_grad():
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
        if param.grad is None:
            raise RuntimeError("No gradient for parameter. Ensure it participates in forward.")
        return param.grad.detach().abs()

    def _generate_phi(self, m: int, n: int, seed_key: str) -> torch.Tensor:
        seed = hash(seed_key)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(m, n, generator=gen, device=self.device)

    def _qim_embed(self, values: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        delta = self.qim_step
        q = torch.round(values / delta) * delta
        offset = (bits.float() * 2.0 - 1.0) * (delta / 4.0)
        return q + offset

    def _qim_extract(self, values: torch.Tensor) -> torch.Tensor:
        delta = self.qim_step
        q = torch.round(values / delta) * delta
        return (values - q >= 0).to(torch.int8)

    def _coerce_indices(self, idx, device: torch.device) -> torch.Tensor:
        if isinstance(idx, torch.Tensor):
            return idx.to(device=device, dtype=torch.long)
        return torch.as_tensor(idx, device=device, dtype=torch.long)

    def _split_indices(self, importance: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = importance.view(-1)
        n_total = flat.numel()
        vip_count = max(1, int(n_total * self.vip_ratio))
        carrier_count = max(1, int(n_total * self.carrier_ratio))
        sorted_idx = torch.argsort(flat, descending=True)
        vip_idx = sorted_idx[:vip_count]
        carrier_idx = sorted_idx[-carrier_count:]
        return {
            "vip_idx": vip_idx,
            "carrier_idx": carrier_idx,
            "vip_count": vip_count,
            "carrier_count": carrier_count,
        }

    def embed(self, data_loader, criterion, return_indices: bool = False) -> Dict[str, object]:
        info: Dict[str, object] = {"layers": []}
        indices_by_layer: Dict[str, Dict[str, object]] = {}
        for layer_name in self.target_layers:
            param = self._get_param(layer_name)
            if not param.dtype.is_floating_point:
                continue
            importance = self._calculate_importance(param, data_loader, criterion)
            split = self._split_indices(importance)
            vip_idx = split["vip_idx"]
            carrier_idx = split["carrier_idx"]

            vip_vals = param.detach().view(-1)[vip_idx].float()
            m = max(1, int(vip_vals.numel() * self.ratio))
            m = min(m, carrier_idx.numel())
            if m <= 0:
                continue

            if return_indices:
                indices_by_layer[layer_name] = {
                    "vip": vip_idx.detach().cpu().to(torch.int32),
                    "carrier": carrier_idx.detach().cpu().to(torch.int32),
                    "m": int(m),
                }

            phi = self._generate_phi(m, vip_vals.numel(), seed_key=layer_name)
            y = torch.matmul(phi, vip_vals)
            bits = (y >= 0).to(torch.int8)

            carrier_vals = param.detach().view(-1)[carrier_idx[:m]].float()
            embedded_vals = self._qim_embed(carrier_vals, bits)
            with torch.no_grad():
                flat_param = param.view(-1)
                flat_param[carrier_idx[:m]] = embedded_vals.to(flat_param.dtype)

            info["layers"].append({
                "layer": layer_name,
                "vip_count": int(split["vip_count"]),
                "carrier_count": int(split["carrier_count"]),
                "embedded_bits": int(m),
            })
        if return_indices:
            info["indices"] = indices_by_layer
        return info

    def verify_and_locate(
        self,
        data_loader,
        criterion,
        indices_by_layer: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        results: List[Dict[str, object]] = []
        overall_ok = True
        for layer_name in self.target_layers:
            param = self._get_param(layer_name)
            if not param.dtype.is_floating_point:
                continue

            saved = None
            if indices_by_layer is not None and layer_name in indices_by_layer:
                saved = indices_by_layer[layer_name]
                vip_raw = saved.get("vip")
                carrier_raw = saved.get("carrier")
                if vip_raw is None or carrier_raw is None:
                    saved = None

            if saved is None:
                importance = self._calculate_importance(param, data_loader, criterion)
                split = self._split_indices(importance)
                vip_idx = split["vip_idx"]
                carrier_idx = split["carrier_idx"]
            else:
                vip_idx = self._coerce_indices(saved["vip"], param.device)
                carrier_idx = self._coerce_indices(saved["carrier"], param.device)

            vip_vals = param.detach().view(-1)[vip_idx].float()
            if saved is None:
                m = max(1, int(vip_vals.numel() * self.ratio))
            else:
                m = int(saved.get("m", 0))
                if m <= 0:
                    m = max(1, int(vip_vals.numel() * self.ratio))
            m = min(m, carrier_idx.numel())
            if m <= 0:
                continue

            phi = self._generate_phi(m, vip_vals.numel(), seed_key=layer_name)
            y_current = torch.matmul(phi, vip_vals)
            bits_current = (y_current >= 0).to(torch.int8)

            carrier_vals = param.detach().view(-1)[carrier_idx[:m]].float()
            bits_extracted = self._qim_extract(carrier_vals)

            diff = bits_extracted != bits_current
            error_rate = diff.float().mean().item()
            layer_info: Dict[str, object] = {
                "layer": layer_name,
                "error_rate": error_rate,
            }
            if error_rate > self.error_threshold:
                error_indices = torch.nonzero(diff, as_tuple=False).view(-1)
                if error_indices.numel() > 0:
                    relevant_phi = phi[error_indices]
                    suspect_scores = torch.sum(torch.abs(relevant_phi), dim=0)
                    top_k = min(self.top_k, suspect_scores.numel())
                    top_scores, top_indices = torch.topk(suspect_scores, k=top_k, largest=True)
                    global_vip_indices = vip_idx[top_indices]
                    layer_info["tampered_indices"] = global_vip_indices.detach().cpu().tolist()
                    layer_info["tampered_scores"] = top_scores.detach().cpu().tolist()
                overall_ok = False
            results.append(layer_info)
        return {"ok": overall_ok, "layers": results}

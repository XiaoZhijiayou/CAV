from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import zlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.misc import sha256, to_bits, bits_to_bytes

PAYLOAD_MAGIC = b"CAVB"
PAYLOAD_VERSION = 1
PAYLOAD_FMT_BASE = "<4sBBHBBfBHH9s32s"
PAYLOAD_BASE_LEN = struct.calcsize(PAYLOAD_FMT_BASE)
PAYLOAD_LEN = PAYLOAD_BASE_LEN + 4


@dataclass
class SCConfig:
    key: bytes
    device: str = "cpu"
    probe_n: int = 64
    probe_shape: Tuple[int, int, int] = (3, 32, 32)
    top_m: int = 8
    sv_k: int = 4
    quant_scale: float = 100.0
    lsb_bits: int = 1
    embed_param_names: Tuple[str, ...] = ("fc.weight", "classifier.6.weight", "linear.weight")
    max_floats_per_param: int = 4096


class CAVSelfContainedAuth:
    def __init__(self, cfg: SCConfig):
        if not cfg.key:
            raise ValueError("SCConfig.key is required.")
        if cfg.lsb_bits != 1:
            raise ValueError("Self-contained mode requires lsb_bits=1 for fixed payload size.")
        self.cfg = cfg

    def _module_groups(self, model: nn.Module) -> Tuple[List[str], Dict[str, List[Tuple[str, torch.Tensor]]]]:
        groups: Dict[str, List[Tuple[str, torch.Tensor]]] = {}
        for name, tensor in model.state_dict().items():
            module_id = name.rsplit(".", 1)[0] if "." in name else name
            groups.setdefault(module_id, []).append((name, tensor))
        for module_id in groups:
            groups[module_id].sort(key=lambda x: x[0])
        module_ids = sorted(groups.keys())
        return module_ids, groups

    def _param_commitment(self, module_id: str, entries: List[Tuple[str, torch.Tensor]]) -> bytes:
        h = hashlib.sha256()
        h.update(self.cfg.key)
        h.update(b"|PARAM|")
        h.update(module_id.encode("utf-8"))
        for name, tensor in entries:
            t = tensor.detach().cpu().contiguous()
            name_b = name.encode("utf-8")
            dtype_b = str(t.dtype).encode("utf-8")
            shape_b = np.array(t.shape, dtype="<i8").tobytes()
            data_b = t.numpy().tobytes()
            h.update(b"|")
            h.update(name_b)
            h.update(b"\0")
            h.update(dtype_b)
            h.update(b"\0")
            h.update(shape_b)
            h.update(b"\0")
            h.update(data_b)
        return h.digest()

    def _cav_hashes(self, model: nn.Module, layer_names: List[str], in_ch: int) -> Dict[str, bytes]:
        model = model.to(self.cfg.device).eval()
        x_probe = self._make_probes(in_ch=in_ch)
        named = dict(model.named_modules())
        out: Dict[str, bytes] = {}

        for lname in layer_names:
            layer = named[lname]
            acts: List[torch.Tensor] = []

            def fwd_hook(_m, _inp, out_t):
                acts.append(out_t)

            h = layer.register_forward_hook(fwd_hook)
            with torch.enable_grad():
                logits = model(x_probe)
                y = self._fixed_target(logits)
                obj = logits[torch.arange(logits.size(0), device=logits.device), y].sum()
                model.zero_grad(set_to_none=True)
                obj.backward()
            h.remove()

            W = layer.weight
            gW = W.grad
            if gW is None:
                raise RuntimeError(f"No grad for layer {lname}. Is it used in forward?")

            if isinstance(layer, nn.Conv2d):
                score = gW.detach().abs().mean(dim=[2, 3]).mean(dim=1)
                out_c = W.size(0)
            else:
                score = gW.detach().abs().mean(dim=1)
                out_c = W.size(0)

            m = min(self.cfg.top_m, out_c)
            top_idx = torch.topk(score, k=m, largest=True).indices.sort().values

            W_sel = W.detach()[top_idx]
            mat = W_sel.reshape(m, -1) if isinstance(layer, nn.Conv2d) else W_sel
            svals = torch.linalg.svdvals(mat.float())
            k = min(self.cfg.sv_k, svals.numel())
            s_top = svals[:k]
            s_top = s_top / (s_top.sum() + 1e-12)

            a = acts[0].detach()
            if a.dim() == 4:
                a_sel = a[:, top_idx, :, :].float()
                a_mu = a_sel.mean(dim=(0, 2, 3))
                a_std = a_sel.std(dim=(0, 2, 3), unbiased=False)
            else:
                a_sel = a[:, top_idx].float()
                a_mu = a_sel.mean(dim=0)
                a_std = a_sel.std(dim=0, unbiased=False)

            fp = torch.cat([s_top, a_mu, a_std], dim=0)
            fp_q = torch.clamp((fp * self.cfg.quant_scale).round(), -32768, 32767).short().cpu().numpy()
            payload = self.cfg.key + b"|CAV|" + lname.encode("utf-8") + b"|" + fp_q.tobytes()
            out[lname] = sha256(payload)

        return out

    def _make_probes(self, in_ch: int) -> torch.Tensor:
        C, H, W = self.cfg.probe_shape
        if in_ch != C:
            C = in_ch
        xs = []
        for i in range(self.cfg.probe_n):
            seed = int.from_bytes(sha256(self.cfg.key + b"|PROBE|" + i.to_bytes(4, "big"))[:8], "big")
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(1, C, H, W, generator=g)
            xs.append(torch.tanh(x))
        x_probe = torch.cat(xs, dim=0)
        return x_probe.to(self.cfg.device)

    def _fixed_target(self, logits: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        seed = int.from_bytes(sha256(self.cfg.key + b"|Y|")[:4], "big")
        y_fixed = int(seed % num_classes)
        return torch.full((logits.size(0),), y_fixed, device=logits.device, dtype=torch.long)

    def _leaf_hash(self, module_id: str, h_param: bytes, h_cav: bytes) -> bytes:
        payload = self.cfg.key + b"|LEAF|" + module_id.encode("utf-8") + b"|" + h_param + h_cav
        return sha256(payload)

    def _merkle_root(self, leaves: List[bytes]) -> bytes:
        if not leaves:
            return sha256(self.cfg.key + b"|EMPTY")
        level = leaves[:]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            nxt = []
            for i in range(0, len(level), 2):
                payload = self.cfg.key + b"|NODE|" + level[i] + level[i + 1]
                nxt.append(sha256(payload))
            level = nxt
        return level[0]

    def compute_root(self, model: nn.Module, in_ch: int) -> bytes:
        module_ids, groups = self._module_groups(model)
        cav_layers = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        cav_hashes = self._cav_hashes(model, cav_layers, in_ch=in_ch)
        leaves: List[bytes] = []
        for module_id in module_ids:
            h_param = self._param_commitment(module_id, groups[module_id])
            h_cav = cav_hashes.get(module_id)
            if h_cav is None:
                h_cav = sha256(self.cfg.key + b"|CAV|EMPTY|" + module_id.encode("utf-8"))
            leaves.append(self._leaf_hash(module_id, h_param, h_cav))
        return self._merkle_root(leaves)

    def pack_payload(self, root: bytes, in_ch: int) -> bytes:
        C, H, W = self.cfg.probe_shape
        if in_ch != C:
            C = in_ch
        reserved = b"\0" * 9
        base = struct.pack(
            PAYLOAD_FMT_BASE,
            PAYLOAD_MAGIC,
            PAYLOAD_VERSION,
            0,
            self.cfg.probe_n,
            self.cfg.top_m,
            self.cfg.sv_k,
            float(self.cfg.quant_scale),
            int(C),
            int(H),
            int(W),
            reserved,
            root,
        )
        crc = zlib.crc32(base) & 0xFFFFFFFF
        return base + struct.pack("<I", crc)

    def unpack_payload(self, payload: bytes) -> Tuple[SCConfig, bytes, Dict[str, object]]:
        if len(payload) != PAYLOAD_LEN:
            raise ValueError(f"Invalid payload length: {len(payload)}")
        base = payload[:PAYLOAD_BASE_LEN]
        crc_expected = struct.unpack("<I", payload[PAYLOAD_BASE_LEN:])[0]
        crc_calc = zlib.crc32(base) & 0xFFFFFFFF
        magic, version, flags, probe_n, top_m, sv_k, quant_scale, probe_c, probe_h, probe_w, _reserved, root = struct.unpack(
            PAYLOAD_FMT_BASE, base
        )
        magic_ok = magic == PAYLOAD_MAGIC
        crc_ok = crc_calc == crc_expected
        cfg = SCConfig(
            key=self.cfg.key,
            device=self.cfg.device,
            probe_n=int(probe_n),
            probe_shape=(int(probe_c), int(probe_h), int(probe_w)),
            top_m=int(top_m),
            sv_k=int(sv_k),
            quant_scale=float(quant_scale),
            lsb_bits=self.cfg.lsb_bits,
            embed_param_names=self.cfg.embed_param_names,
            max_floats_per_param=self.cfg.max_floats_per_param,
        )
        meta = {
            "magic_ok": magic_ok,
            "crc_ok": crc_ok,
            "version": int(version),
            "flags": int(flags),
        }
        return cfg, root, meta

    def _select_carriers(self, model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
        named = dict(model.named_parameters())
        carriers: List[Tuple[str, torch.Tensor]] = []
        used = set()
        for name in self.cfg.embed_param_names:
            if name in named and named[name].dtype == torch.float32:
                carriers.append((name, named[name]))
                used.add(name)
        for name in sorted(named.keys()):
            if name in used:
                continue
            p = named[name]
            if p.dtype == torch.float32:
                carriers.append((name, p))
        if not carriers:
            raise RuntimeError("No float32 parameters found for embedding.")
        return carriers

    def _carrier_caps(self, carriers: List[Tuple[str, torch.Tensor]]) -> List[int]:
        caps = []
        for _name, p in carriers:
            caps.append(min(p.numel(), self.cfg.max_floats_per_param))
        return caps

    def _build_mapping(self, carrier_names: List[str], carrier_caps: List[int]) -> List[Tuple[int, int, int]]:
        mapping: List[Tuple[int, int, int]] = []
        for pi, cap in enumerate(carrier_caps):
            for off in range(cap):
                for bi in range(self.cfg.lsb_bits):
                    mapping.append((pi, off, bi))
        seed_payload = "|".join(carrier_names).encode("utf-8") + b"|" + ",".join(str(c) for c in carrier_caps).encode("utf-8")
        seed = int.from_bytes(sha256(self.cfg.key + b"|MAP|" + seed_payload)[:8], "big")
        rng = np.random.default_rng(seed)
        rng.shuffle(mapping)
        return mapping

    def _embed_bits_carriers(
        self,
        carriers: List[Tuple[str, torch.Tensor]],
        mapping: List[Tuple[int, int, int]],
        bits: List[int],
    ) -> int:
        arrays = []
        for _name, tensor in carriers:
            arr = tensor.detach().cpu().contiguous().numpy()
            u32 = arr.view(np.uint32).ravel()
            arrays.append((arr, u32))
        m = min(len(bits), len(mapping))
        for i in range(m):
            pi, off, bi = mapping[i]
            u32 = arrays[pi][1]
            mask = 1 << bi
            if bits[i] == 1:
                u32[off] |= mask
            else:
                u32[off] &= ~mask
        for (_name, tensor), (arr, _u32) in zip(carriers, arrays):
            tensor.data.copy_(torch.from_numpy(arr).view_as(tensor))
        return m

    def _extract_bits_carriers(
        self,
        carriers: List[Tuple[str, torch.Tensor]],
        mapping: List[Tuple[int, int, int]],
        nbits: int,
    ) -> List[int]:
        arrays = []
        for _name, tensor in carriers:
            arr = tensor.detach().cpu().contiguous().numpy()
            u32 = arr.view(np.uint32).ravel()
            arrays.append(u32)
        m = min(nbits, len(mapping))
        out = []
        for i in range(m):
            pi, off, bi = mapping[i]
            out.append(int((arrays[pi][off] >> bi) & 1))
        return out

    def embed_inplace(self, model: nn.Module, in_ch: int) -> Dict[str, object]:
        model = model.to(self.cfg.device).eval()
        root = self.compute_root(model, in_ch=in_ch)
        payload = self.pack_payload(root, in_ch=in_ch)
        bits = to_bits(payload)
        carriers = self._select_carriers(model)
        carrier_names = [n for n, _ in carriers]
        carrier_caps = self._carrier_caps(carriers)
        mapping = self._build_mapping(carrier_names, carrier_caps)
        embedded = self._embed_bits_carriers(carriers, mapping, bits)
        capacity = len(mapping)
        if embedded < len(bits):
            raise RuntimeError(f"Carrier capacity insufficient: need {len(bits)} bits, embedded {embedded} bits")
        return {
            "root_hex": root.hex(),
            "carrier_params": carrier_names,
            "payload_bits": len(bits),
            "capacity_bits": capacity,
        }

    def verify(self, model: nn.Module, in_ch: int) -> Dict[str, object]:
        model = model.to(self.cfg.device).eval()
        carriers = self._select_carriers(model)
        carrier_names = [n for n, _ in carriers]
        carrier_caps = self._carrier_caps(carriers)
        mapping = self._build_mapping(carrier_names, carrier_caps)
        capacity = len(mapping)
        if capacity < PAYLOAD_LEN * 8:
            return {
                "ok": False,
                "error": "carrier capacity insufficient for payload",
                "capacity_bits": capacity,
                "carrier_params": carrier_names,
            }
        bits = self._extract_bits_carriers(carriers, mapping, PAYLOAD_LEN * 8)
        payload = bits_to_bytes(bits[: PAYLOAD_LEN * 8])
        cfg2, root_emb, meta = self.unpack_payload(payload)
        auth2 = CAVSelfContainedAuth(cfg2)
        try:
            root_calc = auth2.compute_root(model, in_ch=cfg2.probe_shape[0])
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "carrier_params": carrier_names,
                "magic_ok": meta["magic_ok"],
                "crc_ok": meta["crc_ok"],
            }
        ok = meta["magic_ok"] and meta["crc_ok"] and (root_calc == root_emb)
        return {
            "ok": ok,
            "magic_ok": meta["magic_ok"],
            "crc_ok": meta["crc_ok"],
            "version": meta["version"],
            "root_emb_hex": root_emb.hex(),
            "root_calc_hex": root_calc.hex(),
            "carrier_params": carrier_names,
            "capacity_bits": capacity,
        }

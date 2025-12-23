from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from ..utils.crypto import hkdf_sha256, key_id_from_key
from ..utils.misc import sha256, to_bits, bits_to_bytes, hamming

HAMMING_DATA_POS_1511 = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
HAMMING_PARITY_POS_1511 = [1, 2, 4, 8]

@dataclass
class CAVConfig:
    seed: int = 1234
    key: bytes | None = None
    key_id: str = ""
    version: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # probe set
    probe_n: int = 64
    probe_shape: Tuple[int, int, int] = (3, 32, 32)

    # canonical selection
    top_m: int = 8
    target_mode: str = "top1"  # "top1" or "fixed"
    fixed_target: int = 0

    # fingerprint
    sv_k: int = 4
    quant_scale: float = 100.0
    sample_w: int = 32
    sample_quant_scale: float = 100.0

    # embedding
    embed_param_names: Tuple[str, ...] = ("fc.weight", "classifier.6.weight", "linear.weight")
    max_embed_params: int = 4
    lsb_bits: int = 1
    redundancy: int = 3
    max_floats_per_param: int = 4096
    ecc_scheme: str = "hamming1511"

    # decision
    ham_thr: int = 8

class CAVMerkleAuth:
    def __init__(self, cfg: CAVConfig):
        self.cfg = cfg
        if not cfg.key:
            raise ValueError("CAVConfig.key is required. Provide key bytes from a secure source.")
        if not cfg.key_id:
            cfg.key_id = key_id_from_key(cfg.key)
        self._derive_keys(cfg.key)
        self._set_seed(cfg.seed)

    def _derive_keys(self, key: bytes) -> None:
        salt = sha256(b"CAV-MerkleAuth-v2")
        self.k_fp = hkdf_sha256(key, salt, b"fp", 32)
        self.k_merkle = hkdf_sha256(key, salt, b"merkle", 32)
        self.k_header = hkdf_sha256(key, salt, b"header", 32)
        self.k_embed = hkdf_sha256(key, salt, b"embed", 32)
        self.k_sample = hkdf_sha256(key, salt, b"sample", 32)

    def _set_seed(self, seed: int) -> None:
        import random, os
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def make_probe(self, in_ch: int) -> torch.Tensor:
        C, H, W = self.cfg.probe_shape
        assert C in (1, 3), "probe_shape channel must be 1 or 3"
        if in_ch != C:
            # adjust probe channel for MNIST-like models
            C = in_ch
        g = torch.Generator().manual_seed(self.cfg.seed)
        x = torch.randn(self.cfg.probe_n, C, H, W, generator=g)
        return x.to(self.cfg.device)

    def _select_target(self, logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.target_mode == "fixed":
            return torch.full((logits.size(0),), self.cfg.fixed_target, device=logits.device, dtype=torch.long)
        return logits.argmax(dim=1)

    def _iter_fp_layers(self, model: nn.Module) -> List[str]:
        return [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    def canonical_fingerprints(self, model: nn.Module, fp_layers: List[str], in_ch: int) -> Dict[str, np.ndarray]:
        model = model.to(self.cfg.device).eval()
        x_probe = self.make_probe(in_ch=in_ch)

        named = dict(model.named_modules())
        feats: Dict[str, np.ndarray] = {}

        for lname in fp_layers:
            layer = named[lname]
            acts = []

            def fwd_hook(_m, _inp, out):
                acts.append(out)

            h = layer.register_forward_hook(fwd_hook)
            logits = model(x_probe)
            y = self._select_target(logits)
            obj = logits[torch.arange(logits.size(0), device=logits.device), y].sum()

            model.zero_grad(set_to_none=True)
            obj.backward()
            h.remove()

            W = layer.weight
            gW = W.grad
            if gW is None:
                raise RuntimeError(f"No grad for layer {lname}. Is it used in forward?")

            # importance by weight gradient magnitude (output channel / neuron importance)
            if isinstance(layer, nn.Conv2d):
                score = gW.detach().abs().mean(dim=(1, 2, 3))
                out_c = W.size(0)
            else:
                score = gW.detach().abs().mean(dim=1)
                out_c = W.size(0)

            m = min(self.cfg.top_m, out_c)
            top_idx = torch.topk(score, k=m, largest=True).indices.sort().values

            # weight spectral feature
            W_sel = W.detach()[top_idx]
            mat = W_sel.reshape(m, -1) if isinstance(layer, nn.Conv2d) else W_sel
            svals = torch.linalg.svdvals(mat.float())
            k = min(self.cfg.sv_k, svals.numel())
            s_top = svals[:k]
            s_top = s_top / (s_top.sum() + 1e-12)

            # activation stats
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

            if self.cfg.sample_w > 0:
                w_flat = W.detach().view(-1)
                sample_n = min(self.cfg.sample_w, w_flat.numel())
                if sample_n > 0:
                    seed = int.from_bytes(sha256(self.k_sample + b"|" + lname.encode("utf-8"))[:8], "big")
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(w_flat.numel(), size=sample_n, replace=False)
                    w_sample = w_flat[idx].float().cpu().numpy()
                    w_q = np.clip(
                        np.round(w_sample * self.cfg.sample_quant_scale), -32768, 32767
                    ).astype(np.int16)
                    fp_q = np.concatenate([fp_q, w_q], axis=0)
            feats[lname] = fp_q

        return feats

    def layer_hashes(self, fps: Dict[str, np.ndarray]) -> Dict[str, bytes]:
        out: Dict[str, bytes] = {}
        for lname, arr in fps.items():
            payload = self.k_merkle + b"|" + lname.encode("utf-8") + b"|" + arr.tobytes()
            out[lname] = sha256(payload)
        return out

    def param_hashes(self, model: nn.Module) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, p in model.named_parameters():
            t = p.detach().cpu().contiguous()
            dtype = str(t.dtype).encode("utf-8")
            shape = ",".join(str(x) for x in t.shape).encode("utf-8")
            payload = (
                self.k_merkle
                + b"|PARAM|"
                + name.encode("utf-8")
                + b"|"
                + dtype
                + b"|"
                + shape
                + b"|"
                + t.view(torch.uint8).numpy().tobytes()
            )
            out[name] = sha256(payload).hex()
        return out

    def merkle_root(self, hashes: List[bytes]) -> bytes:
        if len(hashes) == 0:
            return sha256(self.k_merkle + b"|EMPTY")
        level = hashes[:]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            nxt = []
            for i in range(0, len(level), 2):
                nxt.append(sha256(self.k_merkle + b"|" + level[i] + level[i + 1]))
            level = nxt
        return level[0]

    def compute_root(self, model: nn.Module, in_ch: int, fp_layers: Optional[List[str]] = None) -> Tuple[bytes, Dict[str, str], List[str]]:
        if fp_layers is None:
            fp_layers = self._iter_fp_layers(model)
        fps = self.canonical_fingerprints(model, fp_layers, in_ch=in_ch)
        lh = self.layer_hashes(fps)
        ordered = [lh[k] for k in sorted(lh.keys())]
        root = self.merkle_root(ordered)
        return root, {k: v.hex() for k, v in lh.items()}, fp_layers

    # --- payload / ecc ---
    def _header_bytes(self) -> bytes:
        payload = f"V{self.cfg.version}|{self.cfg.key_id}".encode("utf-8")
        return sha256(self.k_header + b"|HDR|" + payload)[:4]

    def _cfg_hash_bytes(self, fp_layers: List[str]) -> bytes:
        cfg_payload = {
            "version": self.cfg.version,
            "key_id": self.cfg.key_id,
            "probe_n": self.cfg.probe_n,
            "probe_shape": list(self.cfg.probe_shape),
            "top_m": self.cfg.top_m,
            "target_mode": self.cfg.target_mode,
            "fixed_target": self.cfg.fixed_target,
            "sv_k": self.cfg.sv_k,
            "quant_scale": self.cfg.quant_scale,
            "sample_w": self.cfg.sample_w,
            "sample_quant_scale": self.cfg.sample_quant_scale,
            "redundancy": self.cfg.redundancy,
            "lsb_bits": self.cfg.lsb_bits,
            "max_floats_per_param": self.cfg.max_floats_per_param,
            "max_embed_params": self.cfg.max_embed_params,
            "embed_param_names": list(self.cfg.embed_param_names),
            "ecc_scheme": self.cfg.ecc_scheme,
            "ham_thr": self.cfg.ham_thr,
            "fp_layers": fp_layers,
        }
        blob = json.dumps(cfg_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return sha256(self.k_header + b"|CFG|" + blob)[:4]

    def _build_payload_bits(self, root: bytes, cfg_hash: bytes) -> List[int]:
        header = self._header_bytes()  # 32-bit
        payload = header + cfg_hash + root
        bits = to_bits(payload)

        root_bits = to_bits(root)
        for _ in range(self.cfg.redundancy):
            bits += root_bits
        return bits

    def _hamming1511_encode(self, bits: List[int]) -> List[int]:
        out: List[int] = []
        for i in range(0, len(bits), 11):
            blk = bits[i:i+11]
            if len(blk) < 11:
                blk = blk + [0] * (11 - len(blk))
            code = [0] * 16
            for pos, bit in zip(HAMMING_DATA_POS_1511, blk):
                code[pos] = bit & 1
            for p in HAMMING_PARITY_POS_1511:
                parity = 0
                for pos in range(1, 16):
                    if pos & p:
                        parity ^= code[pos]
                code[p] = parity
            out.extend(code[1:])
        return out

    def _hamming1511_decode(self, bits: List[int]) -> Tuple[List[int], Dict[str, int]]:
        out: List[int] = []
        corrected = 0
        uncorrectable = 0
        total_blocks = 0
        for i in range(0, len(bits), 15):
            blk = bits[i:i+15]
            if len(blk) < 15:
                blk = blk + [0] * (15 - len(blk))
            total_blocks += 1
            code = [0] + [b & 1 for b in blk]
            syndrome = 0
            for p in HAMMING_PARITY_POS_1511:
                parity = 0
                for pos in range(1, 16):
                    if pos & p:
                        parity ^= code[pos]
                if parity:
                    syndrome |= p
            if syndrome != 0:
                if syndrome <= 15:
                    code[syndrome] ^= 1
                    corrected += 1
                syndrome2 = 0
                for p in HAMMING_PARITY_POS_1511:
                    parity = 0
                    for pos in range(1, 16):
                        if pos & p:
                            parity ^= code[pos]
                    if parity:
                        syndrome2 |= p
                if syndrome2 != 0:
                    uncorrectable += 1
            for pos in HAMMING_DATA_POS_1511:
                out.append(code[pos])
        return out, {
            "total_blocks": total_blocks,
            "corrected_blocks": corrected,
            "uncorrectable_blocks": uncorrectable,
        }

    def _ecc_encode(self, bits: List[int]) -> Tuple[List[int], Dict[str, int]]:
        scheme = self.cfg.ecc_scheme.lower()
        if scheme == "none":
            return bits, {"scheme": "none"}
        if scheme == "hamming1511":
            enc = self._hamming1511_encode(bits)
            return enc, {"scheme": "hamming1511"}
        raise ValueError(f"Unsupported ecc_scheme: {self.cfg.ecc_scheme}")

    def _ecc_decode(self, bits: List[int], payload_bits: int) -> Tuple[List[int], Dict[str, int]]:
        scheme = self.cfg.ecc_scheme.lower()
        if scheme == "none":
            return bits[:payload_bits], {"scheme": "none"}
        if scheme == "hamming1511":
            data, stats = self._hamming1511_decode(bits)
            data = data[:payload_bits]
            stats["scheme"] = "hamming1511"
            return data, stats
        raise ValueError(f"Unsupported ecc_scheme: {self.cfg.ecc_scheme}")

    # --- float32 LSB embedding ---
    def _select_carrier_params(self, model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
        named = dict(model.named_parameters())
        carriers: List[Tuple[str, torch.Tensor]] = []
        used = set()
        for name in self.cfg.embed_param_names:
            if name in named and named[name].dtype == torch.float32:
                carriers.append((name, named[name]))
                used.add(name)
        if len(carriers) < self.cfg.max_embed_params:
            for name in sorted(named.keys()):
                if name in used:
                    continue
                p = named[name]
                if p.dtype != torch.float32:
                    continue
                carriers.append((name, p))
                if len(carriers) >= self.cfg.max_embed_params:
                    break
        if not carriers:
            raise RuntimeError("No float32 parameters found for embedding.")
        return carriers

    def _carriers_from_names(self, model: nn.Module, names: List[str]) -> List[Tuple[str, torch.Tensor]]:
        named = dict(model.named_parameters())
        carriers: List[Tuple[str, torch.Tensor]] = []
        for name in names:
            if name not in named:
                raise RuntimeError(f"Carrier param missing: {name}")
            p = named[name]
            if p.dtype != torch.float32:
                raise RuntimeError(f"Carrier param not float32: {name}")
            carriers.append((name, p))
        return carriers

    def _carrier_caps(self, carriers: List[Tuple[str, torch.Tensor]]) -> List[int]:
        caps = []
        for _name, p in carriers:
            caps.append(min(p.numel(), self.cfg.max_floats_per_param))
        return caps

    def _build_embed_mapping(self, carrier_names: List[str], carrier_caps: List[int]) -> List[Tuple[int, int, int]]:
        if self.cfg.lsb_bits < 1 or self.cfg.lsb_bits > 2:
            raise ValueError("lsb_bits should be 1 or 2 for this implementation.")
        mapping: List[Tuple[int, int, int]] = []
        for pi, cap in enumerate(carrier_caps):
            for off in range(cap):
                for bi in range(self.cfg.lsb_bits):
                    mapping.append((pi, off, bi))
        seed_payload = "|".join(carrier_names).encode("utf-8")
        seed = int.from_bytes(sha256(self.k_embed + b"|" + seed_payload)[:8], "big")
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
            arr = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
            u32 = arr.view(np.uint32)
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
        for (_name, tensor), (arr, u32) in zip(carriers, arrays):
            arr[:] = u32.view(np.uint8)
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
            arr = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
            u32 = arr.view(np.uint32)
            arrays.append(u32)
        m = min(nbits, len(mapping))
        out = []
        for i in range(m):
            pi, off, bi = mapping[i]
            out.append(int((arrays[pi][off] >> bi) & 1))
        return out

    # --- API ---
    @torch.no_grad()
    def embed(self, model: nn.Module, in_ch: int) -> Dict[str, object]:
        model = model.to(self.cfg.device).eval()
        root, layer_hashes_hex, fp_layers = self.compute_root(model, in_ch=in_ch)
        cfg_hash = self._cfg_hash_bytes(fp_layers)
        payload_bits = self._build_payload_bits(root, cfg_hash)
        bits, ecc_meta = self._ecc_encode(payload_bits)

        carriers = self._select_carrier_params(model)
        carrier_names = [n for n, _ in carriers]
        carrier_caps = self._carrier_caps(carriers)
        mapping = self._build_embed_mapping(carrier_names, carrier_caps)
        embedded = self._embed_bits_carriers(carriers, mapping, bits)
        if embedded < len(bits):
            raise RuntimeError(f"Carrier capacity insufficient: need {len(bits)} bits, embedded {embedded} bits")

        param_hashes = self.param_hashes(model)

        cfg_info = {
            "probe_n": self.cfg.probe_n,
            "probe_shape": list(self.cfg.probe_shape),
            "top_m": self.cfg.top_m,
            "target_mode": self.cfg.target_mode,
            "fixed_target": self.cfg.fixed_target,
            "sv_k": self.cfg.sv_k,
            "quant_scale": self.cfg.quant_scale,
            "sample_w": self.cfg.sample_w,
            "sample_quant_scale": self.cfg.sample_quant_scale,
            "redundancy": self.cfg.redundancy,
            "lsb_bits": self.cfg.lsb_bits,
            "max_floats_per_param": self.cfg.max_floats_per_param,
            "max_embed_params": self.cfg.max_embed_params,
            "embed_param_names": list(self.cfg.embed_param_names),
            "ecc_scheme": self.cfg.ecc_scheme,
            "ham_thr": self.cfg.ham_thr,
        }

        return {
            "auth_version": self.cfg.version,
            "key_id": self.cfg.key_id,
            "root_hex": root.hex(),
            "header_hex": self._header_bytes().hex(),
            "cfg_hash_hex": cfg_hash.hex(),
            "fp_layers": fp_layers,
            "layer_hashes": layer_hashes_hex,
            "param_hashes": param_hashes,
            "carrier_params": carrier_names,
            "carrier_caps": carrier_caps,
            "payload_bits": len(payload_bits),
            "encoded_bits": len(bits),
            "embedded_bits": embedded,
            "ecc": ecc_meta,
            "cfg": cfg_info,
        }

    @torch.no_grad()
    def verify(self, model: nn.Module, meta: Dict[str, object], in_ch: int) -> Dict[str, object]:
        model = model.to(self.cfg.device).eval()
        fp_layers = meta["fp_layers"]
        root_ref_hex = meta["root_hex"]
        ref_layer_hash = meta["layer_hashes"]
        payload_bits = int(meta["payload_bits"])
        encoded_bits = int(meta["encoded_bits"])
        carrier_names = list(meta["carrier_params"])
        carrier_caps_ref = list(meta.get("carrier_caps", []))

        calc_root, calc_layer_hash, _ = self.compute_root(model, in_ch=in_ch, fp_layers=fp_layers)

        carriers = self._carriers_from_names(model, carrier_names)
        carrier_caps = self._carrier_caps(carriers)
        mapping = self._build_embed_mapping(carrier_names, carrier_caps)
        capacity_ok = encoded_bits <= len(mapping)
        if not capacity_ok:
            return {
                "ok": False,
                "error": "carrier capacity insufficient for encoded_bits",
                "encoded_bits": encoded_bits,
                "capacity_bits": len(mapping),
                "carrier_params": carrier_names,
            }
        bits = self._extract_bits_carriers(carriers, mapping, encoded_bits)
        data_bits, ecc_stats = self._ecc_decode(bits, payload_bits)

        header = data_bits[:32]
        cfg_hash_bits = data_bits[32:64]
        root_bits0 = data_bits[64:64+256]
        rep = data_bits[64+256:]

        votes = [root_bits0]
        for i in range(self.cfg.redundancy):
            seg = rep[i*256:(i+1)*256]
            if len(seg) == 256:
                votes.append(seg)

        rec_root_bits = []
        for j in range(256):
            s = sum(v[j] for v in votes)
            rec_root_bits.append(1 if s >= (len(votes)/2.0) else 0)
        emb_root = bits_to_bytes(rec_root_bits)

        header_ok = header == to_bits(self._header_bytes())
        cfg_hash_ok = cfg_hash_bits == to_bits(self._cfg_hash_bytes(fp_layers))
        d = hamming(to_bits(calc_root), to_bits(emb_root))
        ok = (d <= self.cfg.ham_thr) and header_ok and cfg_hash_ok

        mism_layers = [k for k, v in calc_layer_hash.items() if (k in ref_layer_hash and v != ref_layer_hash[k])]

        mism_params: List[str] = []
        missing_params: List[str] = []
        extra_params: List[str] = []
        ref_param_hash = meta.get("param_hashes")
        if isinstance(ref_param_hash, dict):
            calc_param_hash = self.param_hashes(model)
            for name, h in calc_param_hash.items():
                if name in ref_param_hash and h != ref_param_hash[name]:
                    mism_params.append(name)
            missing_params = [k for k in ref_param_hash.keys() if k not in calc_param_hash]
            extra_params = [k for k in calc_param_hash.keys() if k not in ref_param_hash]

        return {
            "ok": ok,
            "hamming_root": d,
            "calc_root_hex": calc_root.hex(),
            "emb_root_hex": emb_root.hex(),
            "ref_root_hex": root_ref_hex,
            "carrier_params": carrier_names,
            "carrier_caps_match": (carrier_caps_ref == carrier_caps) if carrier_caps_ref else True,
            "ecc": ecc_stats,
            "header_ok": header_ok,
            "cfg_hash_ok": cfg_hash_ok,
            "mism_layers": mism_layers,
            "layer_mismatch_count": len(mism_layers),
            "mism_params": mism_params,
            "param_mismatch_count": len(mism_params),
            "missing_params": missing_params,
            "extra_params": extra_params,
        }

    @staticmethod
    def cfg_from_meta(meta: Dict[str, object], key: bytes, device: str) -> CAVConfig:
        cfg = meta["cfg"]
        return CAVConfig(
            key=key,
            key_id=key_id_from_key(key),
            version=int(meta.get("auth_version", 2)),
            device=device,
            probe_n=int(cfg.get("probe_n", 64)),
            probe_shape=tuple(cfg.get("probe_shape", (3, 32, 32))),
            top_m=int(cfg.get("top_m", 8)),
            target_mode=str(cfg.get("target_mode", "top1")),
            fixed_target=int(cfg.get("fixed_target", 0)),
            sv_k=int(cfg.get("sv_k", 4)),
            quant_scale=float(cfg.get("quant_scale", 100.0)),
            sample_w=int(cfg.get("sample_w", 0)),
            sample_quant_scale=float(cfg.get("sample_quant_scale", 100.0)),
            redundancy=int(cfg.get("redundancy", 3)),
            lsb_bits=int(cfg.get("lsb_bits", 1)),
            max_floats_per_param=int(cfg.get("max_floats_per_param", 4096)),
            max_embed_params=int(cfg.get("max_embed_params", 4)),
            embed_param_names=tuple(cfg.get("embed_param_names", ("fc.weight", "classifier.6.weight", "linear.weight"))),
            ecc_scheme=str(cfg.get("ecc_scheme", "hamming1511")),
            ham_thr=int(cfg.get("ham_thr", 8)),
        )

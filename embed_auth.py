from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from src.utils.ckpt import load_checkpoint
from src.utils.crypto import key_id_from_key, load_key_bytes, sign_meta_ed25519
from src.models.build import build_model
from src.auth.cav_merkle_auth import CAVConfig, CAVMerkleAuth

def parse_args():
    p = argparse.ArgumentParser("Embed CAV-MerkleAuth into a checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--meta_out", type=str, default="")
    p.add_argument("--key_hex", type=str, default="")
    p.add_argument("--key_file", type=str, default="")
    p.add_argument("--key_env", type=str, default="CAV_AUTH_KEY_HEX")
    p.add_argument("--key_id", type=str, default="")
    p.add_argument("--privkey", type=str, default="")
    p.add_argument("--no_sign", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--probe_n", type=int, default=64)
    p.add_argument("--top_m", type=int, default=8)
    p.add_argument("--sv_k", type=int, default=4)
    p.add_argument("--lsb_bits", type=int, default=1)
    p.add_argument("--redundancy", type=int, default=3)
    p.add_argument("--sample_w", type=int, default=32)
    p.add_argument("--sample_quant_scale", type=float, default=100.0)
    p.add_argument("--max_embed_params", type=int, default=4)
    p.add_argument("--max_floats_per_param", type=int, default=4096)
    p.add_argument("--ecc_scheme", type=str, default="hamming1511", choices=["hamming1511", "none"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    key = load_key_bytes(args.key_hex or None, args.key_file or None, args.key_env)
    key_id = args.key_id or key_id_from_key(key)
    ck = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    model = build_model(cfg["model"], num_classes=cfg["num_classes"], in_ch=cfg["in_ch"])
    model.load_state_dict(ck["model_state"], strict=True)

    acfg = CAVConfig(
        key=key,
        key_id=key_id,
        seed=args.seed,
        probe_n=args.probe_n,
        top_m=args.top_m,
        sv_k=args.sv_k,
        lsb_bits=args.lsb_bits,
        redundancy=args.redundancy,
        sample_w=args.sample_w,
        sample_quant_scale=args.sample_quant_scale,
        max_embed_params=args.max_embed_params,
        max_floats_per_param=args.max_floats_per_param,
        ecc_scheme=args.ecc_scheme,
        device=args.device,
    )
    auth = CAVMerkleAuth(acfg)
    meta = auth.embed(model, in_ch=cfg["in_ch"])

    if not args.no_sign:
        if not args.privkey:
            raise ValueError("--privkey is required for signing unless --no_sign is set.")
        meta["signature_alg"] = "ed25519"
        meta["signature"] = sign_meta_ed25519(meta, args.privkey)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "auth_meta": meta,
    }, str(out_path))

    meta_path = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".auth_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] saved model with auth to: {out_path}")
    print(f"[OK] saved meta to: {meta_path}")

if __name__ == "__main__":
    main()

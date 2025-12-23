from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from src.models.build import build_model
from src.auth.cav_merkle_auth import CAVMerkleAuth
from src.auth.attacks import random_noise_tamper, zero_out_last_layer
from src.utils.crypto import load_key_bytes, verify_meta_ed25519, key_id_from_key

def parse_args():
    p = argparse.ArgumentParser("Attack a model and verify auth")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--key_hex", type=str, default="")
    p.add_argument("--key_file", type=str, default="")
    p.add_argument("--key_env", type=str, default="CAV_AUTH_KEY_HEX")
    p.add_argument("--pubkey", type=str, default="")
    p.add_argument("--allow_unsigned", action="store_true")
    p.add_argument("--attack", type=str, required=True, choices=["random_noise","zero_last"])
    p.add_argument("--rate", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    pack = torch.load(args.ckpt, map_location="cpu")
    cfg = pack["cfg"]
    model = build_model(cfg["model"], num_classes=cfg["num_classes"], in_ch=cfg["in_ch"])
    model.load_state_dict(pack["model_state"], strict=True)
    model = model.to(args.device)

    if args.attack == "random_noise":
        random_noise_tamper(model, rate=args.rate, sigma=args.sigma)
    elif args.attack == "zero_last":
        zero_out_last_layer(model)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    key = load_key_bytes(args.key_hex or None, args.key_file or None, args.key_env)
    sig_ok = None
    if "signature" in meta and not args.allow_unsigned:
        if not args.pubkey:
            raise ValueError("--pubkey is required to verify signature.")
        sig_ok = verify_meta_ed25519(meta, meta["signature"], args.pubkey)
    elif "signature" not in meta and not args.allow_unsigned:
        raise ValueError("Missing signature. Use --allow_unsigned to bypass verification.")

    acfg = CAVMerkleAuth.cfg_from_meta(meta, key=key, device=args.device)
    auth = CAVMerkleAuth(acfg)
    res = auth.verify(model, meta=meta, in_ch=cfg["in_ch"])
    res["signature_ok"] = sig_ok
    res["key_id_match"] = (meta.get("key_id") == key_id_from_key(key))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

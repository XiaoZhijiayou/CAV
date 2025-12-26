from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

from src.utils.ckpt import load_checkpoint
from src.utils.crypto import load_key_bytes
try:
    from models.build import build_model
except ImportError:
    from src.models.build import build_model
from src.auth.cav_selfcontained_auth import SCConfig, CAVSelfContainedAuth


def parse_args():
    p = argparse.ArgumentParser("Embed self-contained CAV auth into a checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--key_hex", type=str, default="")
    p.add_argument("--key_file", type=str, default="")
    p.add_argument("--key_env", type=str, default="CAV_AUTH_KEY_HEX")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--num_classes", type=int, default=-1)
    p.add_argument("--in_ch", type=int, default=-1)
    p.add_argument("--probe_n", type=int, default=64)
    p.add_argument("--probe_h", type=int, default=32)
    p.add_argument("--probe_w", type=int, default=32)
    p.add_argument("--top_m", type=int, default=8)
    p.add_argument("--sv_k", type=int, default=4)
    p.add_argument("--quant_scale", type=float, default=100.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def _extract_state_dict(ck: dict) -> dict:
    if "model_state" in ck:
        return ck["model_state"]
    if "state_dict" in ck:
        return ck["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck
    raise KeyError("Checkpoint does not contain model_state/state_dict.")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    key = load_key_bytes(args.key_hex or None, args.key_file or None, args.key_env)
    ck = load_checkpoint(args.ckpt, map_location="cpu")
    if "cfg" in ck:
        cfg = ck["cfg"]
        model_name = cfg["model"]
        num_classes = cfg["num_classes"]
        in_ch = cfg["in_ch"]
    else:
        if not args.model or args.num_classes < 1 or args.in_ch < 1:
            raise ValueError("Checkpoint missing cfg. Provide --model --num_classes --in_ch.")
        cfg = {
            "model": args.model,
            "num_classes": args.num_classes,
            "in_ch": args.in_ch,
        }
        model_name = args.model
        num_classes = args.num_classes
        in_ch = args.in_ch
    model = build_model(model_name, num_classes=num_classes, in_ch=in_ch)
    model.load_state_dict(_extract_state_dict(ck), strict=True)

    scfg = SCConfig(
        key=key,
        device=args.device,
        probe_n=args.probe_n,
        probe_shape=(int(in_ch), args.probe_h, args.probe_w),
        top_m=args.top_m,
        sv_k=args.sv_k,
        quant_scale=args.quant_scale,
    )
    auth = CAVSelfContainedAuth(scfg)
    info = auth.embed_inplace(model, in_ch=in_ch)
    _root_loc, module_ids, leaf_map = auth.compute_root_and_leaves(model, in_ch=in_ch)
    _param_ids, param_map = auth.compute_param_hashes(model)
    cav_loc = {
        "version": 2,
        "module_ids": module_ids,
        "leaf_hex": [leaf_map[mid].hex() for mid in module_ids],
        "param_hex": [param_map.get(mid, b"").hex() for mid in module_ids],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "cav_loc": cav_loc,
    }, str(out_path))

    logger.info("embedded: root_hex=%s", info["root_hex"])
    logger.info("embedded: carriers=%d payload_bits=%d capacity_bits=%d", len(info["carrier_params"]), info["payload_bits"], info["capacity_bits"])
    print(f"[OK] saved self-contained auth model to: {out_path}")


if __name__ == "__main__":
    main()

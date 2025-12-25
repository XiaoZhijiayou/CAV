from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

from src.utils.ckpt import load_checkpoint
from src.utils.crypto import load_key_bytes
from src.models.build import build_model
from src.auth.cav_selfcontained_auth import SCConfig, CAVSelfContainedAuth


def parse_args():
    p = argparse.ArgumentParser("Embed self-contained CAV auth into a checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--key_hex", type=str, default="")
    p.add_argument("--key_file", type=str, default="")
    p.add_argument("--key_env", type=str, default="CAV_AUTH_KEY_HEX")
    p.add_argument("--probe_n", type=int, default=64)
    p.add_argument("--probe_h", type=int, default=32)
    p.add_argument("--probe_w", type=int, default=32)
    p.add_argument("--top_m", type=int, default=8)
    p.add_argument("--sv_k", type=int, default=4)
    p.add_argument("--quant_scale", type=float, default=100.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    key = load_key_bytes(args.key_hex or None, args.key_file or None, args.key_env)
    ck = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    model = build_model(cfg["model"], num_classes=cfg["num_classes"], in_ch=cfg["in_ch"])
    model.load_state_dict(ck["model_state"], strict=True)

    scfg = SCConfig(
        key=key,
        device=args.device,
        probe_n=args.probe_n,
        probe_shape=(int(cfg["in_ch"]), args.probe_h, args.probe_w),
        top_m=args.top_m,
        sv_k=args.sv_k,
        quant_scale=args.quant_scale,
    )
    auth = CAVSelfContainedAuth(scfg)
    info = auth.embed_inplace(model, in_ch=cfg["in_ch"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
    }, str(out_path))

    logger.info("embedded: root_hex=%s", info["root_hex"])
    logger.info("embedded: carrier=%s payload_bits=%d capacity_bits=%d", info["carrier_param"], info["payload_bits"], info["capacity_bits"])
    print(f"[OK] saved self-contained auth model to: {out_path}")


if __name__ == "__main__":
    main()

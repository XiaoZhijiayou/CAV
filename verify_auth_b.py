from __future__ import annotations

import argparse
import json
import logging
import torch

from src.models.build import build_model
from src.utils.crypto import load_key_bytes
from src.auth.cav_selfcontained_auth import SCConfig, CAVSelfContainedAuth


def parse_args():
    p = argparse.ArgumentParser("Verify self-contained CAV auth")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--key_hex", type=str, default="")
    p.add_argument("--key_file", type=str, default="")
    p.add_argument("--key_env", type=str, default="CAV_AUTH_KEY_HEX")
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
    pack = torch.load(args.ckpt, map_location="cpu")
    cfg = pack["cfg"]
    model = build_model(cfg["model"], num_classes=cfg["num_classes"], in_ch=cfg["in_ch"])
    model.load_state_dict(pack["model_state"], strict=True)

    scfg = SCConfig(key=key, device=args.device)
    auth = CAVSelfContainedAuth(scfg)
    res = auth.verify(model, in_ch=cfg["in_ch"])
    if "carrier_params" in res:
        logger.info("carriers=%d capacity_bits=%s", len(res["carrier_params"]), res.get("capacity_bits"))
    print("PASS" if res.get("ok") else "FAIL")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

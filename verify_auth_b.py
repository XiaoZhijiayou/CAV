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
    p.add_argument("--model", type=str, default="")
    p.add_argument("--num_classes", type=int, default=-1)
    p.add_argument("--in_ch", type=int, default=-1)
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
    pack = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in pack:
        cfg = pack["cfg"]
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
    model.load_state_dict(_extract_state_dict(pack), strict=True)

    scfg = SCConfig(key=key, device=args.device)
    auth = CAVSelfContainedAuth(scfg)
    res = auth.verify(model, in_ch=in_ch)
    if "carrier_params" in res:
        logger.info("carriers=%d capacity_bits=%s", len(res["carrier_params"]), res.get("capacity_bits"))
    print("PASS" if res.get("ok") else "FAIL")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

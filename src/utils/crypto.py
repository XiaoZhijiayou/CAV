from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    if length <= 0:
        raise ValueError("length must be positive")
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out = b""
    t = b""
    counter = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]


def key_id_from_key(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:8]


def load_key_bytes(
    key_hex: str | None,
    key_file: str | Path | None,
    key_env: str = "CAV_AUTH_KEY_HEX",
) -> bytes:
    if key_hex:
        return bytes.fromhex(key_hex)
    if key_file:
        data = Path(key_file).read_bytes()
        try:
            text = data.decode("utf-8").strip()
            if text and all(c in "0123456789abcdefABCDEF" for c in text) and len(text) % 2 == 0:
                return bytes.fromhex(text)
        except UnicodeDecodeError:
            pass
        return data
    env = os.getenv(key_env, "").strip()
    if env:
        return bytes.fromhex(env)
    raise ValueError("No key material provided. Use --key_hex, --key_file, or CAV_AUTH_KEY_HEX.")


def canonical_json(data: Dict[str, Any], exclude: Iterable[str] = ()) -> bytes:
    filtered = {k: v for k, v in data.items() if k not in exclude}
    return json.dumps(filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sign_meta_ed25519(meta: Dict[str, Any], private_key_path: str | Path) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key_data = Path(private_key_path).read_bytes()
    key = load_pem_private_key(key_data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Private key is not Ed25519.")
    msg = canonical_json(meta, exclude=("signature", "signature_alg"))
    sig = key.sign(msg)
    return sig.hex()


def verify_meta_ed25519(meta: Dict[str, Any], signature_hex: str, public_key_path: str | Path) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    key_data = Path(public_key_path).read_bytes()
    key = load_pem_public_key(key_data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Public key is not Ed25519.")
    msg = canonical_json(meta, exclude=("signature", "signature_alg"))
    sig = bytes.fromhex(signature_hex)
    try:
        key.verify(sig, msg)
        return True
    except Exception:
        return False

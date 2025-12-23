from __future__ import annotations
import hashlib
from typing import List

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def to_bits(b: bytes) -> List[int]:
    out: List[int] = []
    for x in b:
        for i in range(8):
            out.append((x >> (7 - i)) & 1)
    return out

def bits_to_bytes(bits: List[int]) -> bytes:
    assert len(bits) % 8 == 0
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | int(bits[i + j])
        out.append(v)
    return bytes(out)

def hamming(a: List[int], b: List[int]) -> int:
    return sum(int(x != y) for x, y in zip(a, b))

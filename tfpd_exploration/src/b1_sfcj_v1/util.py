"""Hashing, JSON receipts, and small numeric helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return str(value)
        return value
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_ready(obj), sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")


def ieee_positive_zero(value) -> bool:
    import torch

    tensor = value.detach() if hasattr(value, "detach") else torch.as_tensor(value)
    if tuple(tensor.shape) != ():
        return False
    if not bool(torch.equal(tensor, torch.zeros_like(tensor))):
        return False
    return not bool(torch.signbit(tensor).item())

"""Byte hashes for consumer / basis / normalizer arrays."""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import torch


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return array.tobytes()


def array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def state_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    chunks: list[bytes] = []
    for key in sorted(state):
        chunks.append(str(key).encode("utf-8"))
        chunks.append(tensor_bytes(state[key]))
    return b"".join(chunks)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return sha256_bytes(state_bytes(state))


def state_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {str(key): sha256_bytes(tensor_bytes(value)) for key, value in state.items()}


def consumer_state(student) -> dict[str, torch.Tensor]:
    payload = {
        **{f"decoder.{key}": value for key, value in student.decoder.state_dict().items()},
        **{f"id_encoder.{key}": value for key, value in student.id_encoder.state_dict().items()},
        "carrier_projection_weight": student.carrier_projection_weight.detach().clone(),
    }
    return {key: value.detach().cpu().clone() for key, value in payload.items()}


def learned_state(student, basis) -> dict[str, torch.Tensor]:
    return {
        **{f"student.{key}": value.detach().cpu().clone() for key, value in student.state_dict().items()},
        **{f"basis.{key}": value.detach().cpu().clone() for key, value in basis.state_dict().items()},
    }

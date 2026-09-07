"""Pure and model-level primitives for anchored H1 cross-record memory."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .plan import CHUNK_LENGTH, IDENTITY_LENGTH, SUPPORT_MEMBERS, UNITS, WINDOW


class H1CrossRecordError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1CrossRecordError(message)


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    require(array.ndim >= 1 and np.isfinite(array).all(), "digest array must be finite and non-scalar")
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def query_member(neural: Any) -> np.ndarray:
    from h1_causal_activity_completion_v1.core import resample_activity_member

    value = np.ascontiguousarray(np.asarray(neural), dtype=np.float32)
    require(value.ndim == 2 and value.shape[0] >= CHUNK_LENGTH and value.shape[1] == UNITS, "query stream geometry drift")
    require(np.isfinite(value).all(), "query stream is nonfinite")
    return resample_activity_member(value[:CHUNK_LENGTH], output_length=IDENTITY_LENGTH)


def identity_states(net: Any, support: Any, chunk: Any, carrier: Any, *, device: str) -> dict[str, Any]:
    """Return static, native-growing and post-MLP-growing identities.

    The post-MLP state applies the complete learned post network to each member
    before the member reduction.  It is intentionally OOD relative to the
    frozen native C1 path and is therefore never deployed without anchoring.
    """

    import torch

    support_np = np.ascontiguousarray(np.asarray(support), dtype=np.float32)
    chunk_np = np.ascontiguousarray(np.asarray(chunk), dtype=np.float32)
    carrier_np = np.ascontiguousarray(np.asarray(carrier), dtype=np.float32)
    require(support_np.shape == (SUPPORT_MEMBERS, IDENTITY_LENGTH, UNITS), "M3 support geometry drift")
    require(chunk_np.shape == (IDENTITY_LENGTH, UNITS), "query member geometry drift")
    require(carrier_np.shape == (UNITS, 4), "carrier geometry drift")
    require(np.isfinite(support_np).all() and np.isfinite(chunk_np).all() and np.isfinite(carrier_np).all(), "identity inputs are nonfinite")
    all_members = np.concatenate((support_np, chunk_np[None]), axis=0)
    tensor = torch.as_tensor(all_members, dtype=torch.float32, device=device)
    temporal = tensor.permute(0, 2, 1)
    fixed = torch.as_tensor(carrier_np, dtype=torch.float32, device=device)
    effective = torch.zeros_like(fixed) if net.zero_carrier else fixed
    with torch.inference_mode():
        encoded = net.carrier_pre_pool(temporal)
        h0 = net.carrier_post_pool(torch.cat((encoded[:SUPPORT_MEMBERS].mean(dim=0), effective), dim=-1))
        hn = net.carrier_post_pool(torch.cat((encoded.mean(dim=0), effective), dim=-1))
        repeated = effective.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        hp = net.carrier_post_pool(torch.cat((encoded, repeated), dim=-1)).mean(dim=0)
    require(tuple(h0.shape) == tuple(hn.shape) == tuple(hp.shape) == (UNITS, WINDOW), "identity output geometry drift")
    require(bool(torch.isfinite(h0).all() and torch.isfinite(hn).all() and torch.isfinite(hp).all()), "identity output is nonfinite")
    return {"static": h0, "native": hn, "post": hp}


def anchored_identity(static: Any, candidate: Any, gate: float) -> Any:
    """Exact +0 branch followed by the declared anchored residual."""

    import torch

    value = float(gate)
    require(value in (0.0, 0.05, 0.10, 0.20, 0.50, 1.0), "gate drift")
    require(isinstance(static, torch.Tensor) and isinstance(candidate, torch.Tensor), "identity tensor type drift")
    require(tuple(static.shape) == tuple(candidate.shape) == (UNITS, WINDOW), "anchored identity shape drift")
    if value == 0.0:
        return static
    result = static + value * (candidate - static)
    require(bool(torch.isfinite(result).all()), "anchored identity is nonfinite")
    return result


def stream_window(neural: np.ndarray, endpoint: int) -> np.ndarray:
    values = np.asarray(neural, dtype=np.float32)
    require(values.ndim == 2 and values.shape[1] == UNITS, "stream neural geometry drift")
    require(type(endpoint) is int and 0 <= endpoint < values.shape[0], "stream endpoint drift")
    output = np.zeros((WINDOW, UNITS), dtype=np.float32)
    start = max(0, endpoint - WINDOW + 1)
    suffix = values[start : endpoint + 1]
    output[-len(suffix) :] = suffix
    return output


__all__ = (
    "H1CrossRecordError", "anchored_identity", "array_sha256", "identity_states",
    "query_member", "require", "stream_window",
)

